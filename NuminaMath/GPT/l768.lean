import Mathlib

namespace fraction_of_visitors_l768_768953

variable (V E U : ℕ)
variable (H1 : E = U)
variable (H2 : 600 - E - 150 = 450)

theorem fraction_of_visitors (H3 : 600 = E + 150 + 450) : (450 : ℚ) / 600 = (3 : ℚ) / 4 :=
by
  apply sorry

end fraction_of_visitors_l768_768953


namespace dodecahedron_edge_probability_l768_768359

theorem dodecahedron_edge_probability :
  ∀ (V E : ℕ), 
  V = 20 → 
  ((∀ v ∈ finset.range V, 3 = 3) → -- condition representing each of the 20 vertices is connected to 3 other vertices
  ∃ (p : ℚ), p = 3 / 19) :=
begin
  intros,
  use 3 / 19,
  split,
  sorry
end

end dodecahedron_edge_probability_l768_768359


namespace volume_ratio_of_spheres_l768_768317

theorem volume_ratio_of_spheres 
  (r1 r2 : ℝ) 
  (h_surface_area : (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 16) : 
  (4 / 3 * Real.pi * r1^3) / (4 / 3 * Real.pi * r2^3) = 1 / 64 :=
by 
  sorry

end volume_ratio_of_spheres_l768_768317


namespace idalina_land_l768_768284

noncomputable section

open real

-- Definition of problem constants
def area_ABC : ℝ := 120
def CD : ℝ := 10
def DE : ℝ := 10
def AC : ℝ := 20

-- Definition of main proof for total area calculation and distance CF calculation
theorem idalina_land (
    h1 : CD = 10,
    h2 : DE = 10,
    h3 : AC = 20,
    h4 : area_ABC = 120
) : (let area_ACDE := 150
           total_area := 270
           target_part_area := 135
           area_ACF := 15
           distance_CF := 1.5 in
        area_ACDE + area_ABC = total_area ∧
        AC / 2 * distance_CF = area_ACF) := by {
    sorry
}

end idalina_land_l768_768284


namespace linear_eq_k_l768_768550

theorem linear_eq_k (k : ℝ) : (k - 3) * x ^ (|k| - 2) + 5 = k - 4 → |k| = 3 → k ≠ 3 → k = -3 :=
by
  intros h1 h2 h3
  sorry

end linear_eq_k_l768_768550


namespace total_items_deleted_l768_768813

-- Define the initial conditions
def initial_apps : Nat := 17
def initial_files : Nat := 21
def remaining_apps : Nat := 3
def remaining_files : Nat := 7
def transferred_files : Nat := 4

-- Prove the total number of deleted items
theorem total_items_deleted : (initial_apps - remaining_apps) + (initial_files - (remaining_files + transferred_files)) = 24 :=
by
  sorry

end total_items_deleted_l768_768813


namespace factorize_expression_l768_768827

theorem factorize_expression (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := 
by 
  sorry

end factorize_expression_l768_768827


namespace range_of_a_l768_768062

def p (a : ℝ) : Prop :=
  (1 : ℝ)^2 - 2 * (1 : ℝ) + a < 0 ∧
  (2 : ℝ)^2 - 2 * (2 : ℝ) + a > 0

def q (a : ℝ) : Prop :=
  (2 * a - 3)^2 - 4 > 0

theorem range_of_a (a : ℝ) : ¬ (p a ∧ q a) ∧ (p a ∨ q a) ↔ (a ∈ Iio 0 ∪ Icc (1 / 2) 1 ∪ Ioi (5 / 2)) :=
sorry

end range_of_a_l768_768062


namespace triangle_perimeter_l768_768839

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def A : ℝ × ℝ := (-3, 1)
def B : ℝ × ℝ := (-3, 7)
def C : ℝ × ℝ := (2, 4)

def d_AB := distance A B
def d_BC := distance B C
def d_CA := distance C A

noncomputable def perimeter := d_AB + d_BC + d_CA

theorem triangle_perimeter :
  perimeter = 6 + 2 * Real.sqrt 34 :=
by
  sorry

end triangle_perimeter_l768_768839


namespace num_four_digit_integers_with_3_and_6_l768_768108

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end num_four_digit_integers_with_3_and_6_l768_768108


namespace plate_acceleration_magnitude_plate_acceleration_direction_l768_768371

variable (R r : ℝ) (m g : ℝ) (alpha : ℝ)

def plate_conditions := (R = 1) ∧ (r = 0.5) ∧ (m = 75) ∧ (alpha = Real.arccos 0.82) ∧ (g = 10)

theorem plate_acceleration_magnitude (h : plate_conditions R r m g alpha) : 
  let a := g * Real.sqrt ((1 - Real.cos alpha) / 2) 
  in a = 3 := by
  sorry

theorem plate_acceleration_direction (h : plate_conditions R r m g alpha) : 
  let direction := Real.arcsin 0.2 
  in direction = (alpha / 2) := by
  sorry

end plate_acceleration_magnitude_plate_acceleration_direction_l768_768371


namespace find_day_for_balance_l768_768936

-- Define the initial conditions and variables
def initialEarnings : ℤ := 20
def secondDaySpending : ℤ := 15
variables (X Y : ℤ)

-- Define the function for net balance on day D
def netBalance (D : ℤ) : ℤ :=
  initialEarnings + (D - 1) * X - (secondDaySpending + (D - 2) * Y)

-- The main theorem proving the day D for net balance of Rs. 60
theorem find_day_for_balance (X Y : ℤ) : ∃ D : ℤ, netBalance X Y D = 60 → 55 = (D + 1) * (X - Y) :=
by
  sorry

end find_day_for_balance_l768_768936


namespace count_routes_A_to_B_l768_768807

def num_routes_3x2_grid_no_consecutive_moves : ℕ := 2

theorem count_routes_A_to_B :
  ∃ routes : ℕ, routes = num_routes_3x2_grid_no_consecutive_moves :=
begin
  sorry
end

end count_routes_A_to_B_l768_768807


namespace population_approximation_l768_768822

theorem population_approximation :
  ∀ (P : ℕ → ℕ), (P 2000 = 400) → (∀ n, P (2000 + 20 * n) = P (2000 + 20 * (n - 1)) * 4) → P 2060 ≈ 12800 :=
by
  sorry

end population_approximation_l768_768822


namespace z_conjugate_sum_l768_768629

noncomputable def z : ℂ := (1 + complex.I) / (1 - complex.I + complex.I^2)

theorem z_conjugate_sum : z + conj z = -2 := by
  sorry

end z_conjugate_sum_l768_768629


namespace product_units_digit_of_five_consecutive_l768_768716

theorem product_units_digit_of_five_consecutive (n : ℕ) : 
  ((n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10) = 0 := 
sorry

end product_units_digit_of_five_consecutive_l768_768716


namespace inequality_proof_l768_768224

open Real

theorem inequality_proof (x y : ℝ) (hx : x > 1/2) (hy : y > 1) : 
  (4 * x^2) / (y - 1) + (y^2) / (2 * x - 1) ≥ 8 := 
by
  sorry

end inequality_proof_l768_768224


namespace part1_composite_eq_part1_find_a_b_part2_conditions_part3_fixed_point_l768_768572

-- Definition of the problem
def composite_function (k1 k2 : ℝ) (b1 b2 : ℝ) : (ℝ → ℝ) := λ x, (k1 + k2) * x + b1 * b2

-- Statement of the problem (a) part 1
theorem part1_composite_eq : composite_function 3 (-4) 2 3 = λ x, (-1) * x + 6 := 
  sorry

-- Statement for part 1
theorem part1_find_a_b (a b : ℝ) :
  composite_function a (-1) (-2) b = λ x, 3 * x + 2 → a = 4 ∧ b = -1 :=
  sorry

-- Statement for part 2
theorem part2_conditions (k b : ℝ) :
  (λ x, (-1 + k) * x - 3 * b) = composite_function (-1) k b (-3) → 
  (-1 + k < 0) ∧ (-3 * b > 0) → 
  k < 1 ∧ b < 0 :=
  sorry

-- Statement for part 3
theorem part3_fixed_point (m : ℝ) :
  composite_function (-2) (3 * m) m (-6) = λ x, m * (3 * x - 6) - 2 * x → 
  ∀ m, (2, -4) ∈ set_of_points (λ x, m * (3 * x - 6) - 2 * x) :=
  sorry

end part1_composite_eq_part1_find_a_b_part2_conditions_part3_fixed_point_l768_768572


namespace gcd_calculation_l768_768709

theorem gcd_calculation : 
  Nat.gcd (111^2 + 222^2 + 333^2) (110^2 + 221^2 + 334^2) = 3 := 
by
  sorry

end gcd_calculation_l768_768709


namespace sum_x_coords_congruences_l768_768649

theorem sum_x_coords_congruences : 
  ∃ x1 x2 : ℤ, (0 ≤ x1) ∧ (x1 < 13) ∧ (0 ≤ x2) ∧ (x2 < 13) ∧ 
  ((∃ y : ℤ, (y ≡ 3 * x1 + 4 [MOD 13]) ∧ (y ≡ x1^2 + x1 + 1 [MOD 13])) ∧
   (∃ y : ℤ, (y ≡ 3 * x2 + 4 [MOD 13]) ∧ (y ≡ x2^2 + x2 + 1 [MOD 13]))) ∧
  (x1 + x2 = 15) := 
sorry

end sum_x_coords_congruences_l768_768649


namespace tickets_sold_correctly_l768_768391

theorem tickets_sold_correctly :
  let total := 620
  let cost_per_ticket := 4
  let tickets_sold := 155
  total / cost_per_ticket = tickets_sold :=
by
  sorry

end tickets_sold_correctly_l768_768391


namespace roots_condition_l768_768854

theorem roots_condition (a x1 x2 : ℝ) 
  (quad_eq : ∀ x, x^2 - 2 * a * x - (1 / a^2) = 0) 
  (root_eq : x1^4 + x2^4 = 16 + 8 * Real.sqrt 2) : 
  a = Real.sqrt[8] (1 / 8) ∨ a = -Real.sqrt[8] (1 / 8) :=
sorry

end roots_condition_l768_768854


namespace odd_function_f_2_eq_2_l768_768864

noncomputable def f (x : ℝ) : ℝ := 
if x < 0 then x^2 + 3 * x else -(if -x < 0 then (-x)^2 + 3 * (-x) else x^2 + 3 * x)

theorem odd_function_f_2_eq_2 : f 2 = 2 :=
by
  -- sorry will be used to skip the actual proof
  sorry

end odd_function_f_2_eq_2_l768_768864


namespace part_a_part_b_l768_768762

def is_balanced (N : ℕ) : Prop :=
  N = 1 ∨ (N > 1 ∧ (∃ l : list ℕ, (∀ p ∈ l, Nat.Prime p) ∧ l.length % 2 = 0 ∧ N = l.prod))

def P (a b x : ℕ) : ℕ := (x + a) * (x + b)

theorem part_a : 
  ∃ a b : ℕ, a ≠ b ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 50 → is_balanced (P a b n)) :=
sorry

theorem part_b (a b : ℕ) (h : ∀ n : ℕ, is_balanced (P a b n)) : 
  a = b :=
sorry

end part_a_part_b_l768_768762


namespace vacant_seats_l768_768585

theorem vacant_seats (total_seats : ℕ) (filled_percent vacant_percent : ℚ) 
  (h_total : total_seats = 600)
  (h_filled_percent : filled_percent = 75)
  (h_vacant_percent : vacant_percent = 100 - filled_percent)
  (h_vacant_percent_25 : vacant_percent = 25) :
  (25 / 100) * 600 = 150 :=
by 
  -- this is the final answer we want to prove, replace with sorry to skip the proof just for statement validation
  sorry

end vacant_seats_l768_768585


namespace morgan_total_pens_l768_768244

def initial_red_pens : Nat := 65
def initial_blue_pens : Nat := 45
def initial_black_pens : Nat := 58
def initial_green_pens : Nat := 36
def initial_purple_pens : Nat := 27

def red_pens_given_away : Nat := 15
def blue_pens_given_away : Nat := 20
def green_pens_given_away : Nat := 10

def black_pens_bought : Nat := 12
def purple_pens_bought : Nat := 5

def final_red_pens : Nat := initial_red_pens - red_pens_given_away
def final_blue_pens : Nat := initial_blue_pens - blue_pens_given_away
def final_black_pens : Nat := initial_black_pens + black_pens_bought
def final_green_pens : Nat := initial_green_pens - green_pens_given_away
def final_purple_pens : Nat := initial_purple_pens + purple_pens_bought

def total_pens : Nat := final_red_pens + final_blue_pens + final_black_pens + final_green_pens + final_purple_pens

theorem morgan_total_pens : total_pens = 203 := 
by
  -- final_red_pens = 50
  -- final_blue_pens = 25
  -- final_black_pens = 70
  -- final_green_pens = 26
  -- final_purple_pens = 32
  -- Therefore, total_pens = 203
  sorry

end morgan_total_pens_l768_768244


namespace exists_set_with_reciprocal_sum_one_l768_768493

theorem exists_set_with_reciprocal_sum_one (n : ℕ) (h_pos : n > 0) (h_exception : n ≠ 2) :
  ∃ (S : Finset ℕ), S.card = n ∧ (∀ x ∈ S, x ≤ n^2) ∧ (∑ x in S, (1 : ℚ) / x) = 1 := 
sorry

end exists_set_with_reciprocal_sum_one_l768_768493


namespace expected_value_area_std_deviation_area_l768_768418

section

variables (L W : ℝ) (σ_L σ_W : ℝ) (L_measurement W_measurement : ℝ → ℝ)
variable independent_measurements : independence_indicator L_measurement W_measurement

-- Given conditions
def length := 2 
def width := 1
def std_dev_length := 0.003
def std_dev_width := 0.002

-- Expected area value
def expected_area : ℝ := 2

-- Standard deviation of the area in cm²
def std_dev_area_cm := 41

theorem expected_value_area :
  (∀ L W, L = length → W = width → L_measurement(0) = std_dev_length → W_measurement(0) = std_dev_width) →
  E(L * W) = expected_area := 
sorry

theorem std_deviation_area :
  (∀ L W, L = length → W = width → L_measurement(0) = std_dev_length → W_measurement(0) = std_dev_width) →
  sqrt (Var(L * W)) = std_dev_area_cm / 100 :=
sorry
end

end expected_value_area_std_deviation_area_l768_768418


namespace restore_original_expression_l768_768177

-- Define the altered product and correct restored products
def original_expression_1 := 4 * 5 * 4 * 7 * 4
def original_expression_2 := 4 * 7 * 4 * 5 * 4
def altered_product := 2247
def corrected_product := 2240

-- Statement that proves the corrected restored product given the altered product
theorem restore_original_expression :
  (4 * 5 * 4 * 7 * 4 = corrected_product ∨ 4 * 7 * 4 * 5 * 4 = corrected_product) :=
sorry

end restore_original_expression_l768_768177


namespace area_of_triangle_l768_768021

theorem area_of_triangle (a b c : ℝ) (h₁ : a = 7) (h₂ : b = 8) (h₃ : c = 9) : 
  let s := (a + b + c) / 2 in 
  sqrt (s * (s - a) * (s - b) * (s - c)) = 12 * sqrt 5 :=
by
  sorry

end area_of_triangle_l768_768021


namespace max_distinct_numbers_in_table_l768_768185

theorem max_distinct_numbers_in_table 
  (table : ℕ → ℕ → ℕ)
  (h_dim : ∀ i j, i < 75 → j < 75)
  (h_row : ∀ i, i < 75 → (finset.image (table i) (finset.range 75)).card ≥ 15)
  (h_three_rows : ∀ i, i < 73 → 
    (finset.bUnion (finset.range 3)
      (λ k, finset.image (table (i + k)) (finset.range 75))).card ≤ 25) :
  ∃ distinct_numbers, distinct_numbers = 385 ∧ 
  ∀ (i j : ℕ), i < 75 → j < 75 → table i j ∈ distinct_numbers :=
sorry

end max_distinct_numbers_in_table_l768_768185


namespace part1_part2_part3_l768_768084

-- Part (1)
theorem part1 (f g : ℝ → ℝ) (m : ℝ) (n : ℝ) (h_m : m = 1)
  (h_f : ∀ x, f x = Real.log x)
  (h_g : ∀ x, g x = (m * (x + n)) / (x + 1)) :
  (∀ x, (f' x) * (g' x) | x = 1 = -1) → n = 5 := 
sorry

-- Part (2)
theorem part2 (f g : ℝ → ℝ) (m n : ℝ) (h_m_pos : m > 0)
  (h_f : ∀ x, f x = Real.log x)
  (h_g : ∀ x, g x = (m * (x + n)) / (x + 1))
  (y : ℝ → ℝ) (h_y : ∀ x, y x = f x - g x) :
  ¬MonotonicOn (Set.Ioi 0) y → m - n > 3 :=
sorry

-- Part (3)
theorem part3 (f g : ℝ → ℝ) (k : ℝ)
  (h_f : ∀ x, f x = Real.log x + k / x)
  (h_g : ∀ x, g x = Real.exp x / x)
  (domain : Set ℝ) (h_domain : ∀ x, x ∈ domain → x > 1 / 2) :
  (∀ x ∈ domain, f x < g x) → k ≤ 1 :=
sorry

end part1_part2_part3_l768_768084


namespace price_of_pants_l768_768241

theorem price_of_pants
  (P S H : ℝ)
  (h1 : P + S + H = 340)
  (h2 : S = (3 / 4) * P)
  (h3 : H = P + 10) :
  P = 120 :=
by
  sorry

end price_of_pants_l768_768241


namespace hypotenuse_of_30_60_90_triangle_l768_768652

theorem hypotenuse_of_30_60_90_triangle (l : ℝ) (θ : ℝ) 
  (leg_condition : l = 15) (angle_condition : θ = 30) : 
  ∃ h : ℝ, h = 30 :=
by
  use 2 * l
  rw [leg_condition, angle_condition]
  trivial

end hypotenuse_of_30_60_90_triangle_l768_768652


namespace miles_wed_thurs_proof_l768_768614

-- Define the conditions
def mileage_rate := 0.36
def miles_monday := 18
def miles_tuesday := 26
def miles_friday := 16
def total_reimbursement := 36

-- Define the total miles driven on Monday, Tuesday, and Friday
def total_miles_mon_tue_fri := miles_monday + miles_tuesday + miles_friday

-- Define the total miles for the entire week
def total_miles_week := total_reimbursement / mileage_rate

-- Calculate the miles driven on Wednesday and Thursday combined
def miles_wed_thurs := total_miles_week - total_miles_mon_tue_fri

theorem miles_wed_thurs_proof : miles_wed_thurs = 40 := by
  -- Proof not required, just provide the statement for now
  sorry

end miles_wed_thurs_proof_l768_768614


namespace arithmetic_mean_of_integers_from_neg6_to_7_l768_768703

theorem arithmetic_mean_of_integers_from_neg6_to_7 :
  let range := List.range' (-6) 14 in
  let mean := (range.sum : ℚ) / (range.length : ℚ) in
  mean = 0.5 := by
  -- Let's define the range of integers from -6 to 7 inclusive
  let range := List.range' (-6) 14
  -- Let the sum of this range be S
  let S : ℚ := (range.sum)
  -- Let the number of elements in this range be N
  let N : ℚ := (range.length)
  -- The mean of this range is S/N
  let mean := S / N
  -- We assert that this mean is equal to 0.5
  have h_correct_mean : S / N = 0.5 := sorry
  exact h_correct_mean

end arithmetic_mean_of_integers_from_neg6_to_7_l768_768703


namespace triangle_area_ADC_l768_768962

-- Define the basic elements of the triangle and the points
theorem triangle_area_ADC (A B C D : Type) [triangle ABC]
  (h_right_angle : ∠B = 90) 
  (h_d_ratio : BD / DC = 5 / 2)
  (area_ABD : area 'ABD = 35) :
  area 'ADC = 14 :=
sorry

end triangle_area_ADC_l768_768962


namespace deepak_age_is_21_l768_768316

noncomputable def DeepakCurrentAge (x : ℕ) : Prop :=
  let Rahul := 4 * x
  let Deepak := 3 * x
  let Karan := 5 * x
  Rahul + 6 = 34 ∧
  (Rahul + 6) / 7 = (Deepak + 6) / 5 ∧ (Rahul + 6) / 7 = (Karan + 6) / 9 → 
  Deepak = 21

theorem deepak_age_is_21 : ∃ x : ℕ, DeepakCurrentAge x :=
by
  use 7
  sorry

end deepak_age_is_21_l768_768316


namespace last_two_digits_of_sum_of_factorials_l768_768378

theorem last_two_digits_of_sum_of_factorials : 
  let sum := (1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9! + 10!) % 100 in 
  sum = 13 :=
by
  -- Detailed proof steps would go here
  sorry

end last_two_digits_of_sum_of_factorials_l768_768378


namespace range_of_m_l768_768998

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) :
  (∀ x ∈ set.Icc 1 3, f m x < -m + 4) ↔ m < 5 / 7 := by
sorry

end range_of_m_l768_768998


namespace fraction_addition_l768_768729

variable {w x y : ℝ}

theorem fraction_addition (h1 : w / x = 1 / 3) (h2 : w / y = 3 / 4) : (x + y) / y = 13 / 4 := by
  sorry

end fraction_addition_l768_768729


namespace a_perp_b_l768_768190

variables {Point : Type} [AffineSpace Point]
variables (a b : Line_3D Point) (α β : Plane_3D Point)

-- Conditions
def line_perpendicular_to_plane (a : Line_3D Point) (α : Plane_3D Point) : Prop := ∀ p: Point, p ∈ a → ∀ q: Point, q ∈ α → (p -ᵥ q).orthogonal (normal_vector α)
def line_in_plane (b : Line_3D Point) (β : Plane_3D Point) : Prop := ∀ p: Point, p ∈ b → p ∈ β
def planes_parallel (α β : Plane_3D Point) : Prop := ∃ v : Vector_3D Point, normal_vector α = v ∧ normal_vector β = v

-- Given conditions
axiom α_parallel_β : planes_parallel α β
axiom a_perp_α : line_perpendicular_to_plane a α
axiom b_in_β : line_in_plane b β

-- Conclusion
theorem a_perp_b : ∀ p q: Point, p ∈ a → q ∈ b → (p -ᵥ q).orthogonal  (normal_vector α) :=
sorry

end a_perp_b_l768_768190


namespace cookies_per_batch_l768_768917

-- Define the necessary conditions
def total_chips : ℕ := 81
def batches : ℕ := 3
def chips_per_cookie : ℕ := 9

-- Theorem stating the number of cookies per batch
theorem cookies_per_batch : (total_chips / batches) / chips_per_cookie = 3 :=
by
  -- Here would be the proof, but we use sorry as placeholder
  sorry

end cookies_per_batch_l768_768917


namespace covering_condition_l768_768999

def is_hollow_rectangle (m n : ℕ) : Prop :=
  ∃ (s : list (ℕ × ℕ)), s.length = (m * n) / 6 ∧
  ∀ x, x ∈ s → x.1 < m ∧ x.2 < n

theorem covering_condition (m n : ℕ) (h : m * n % 12 = 0) :
  is_hollow_rectangle m n :=
sorry

end covering_condition_l768_768999


namespace greatest_number_of_problems_missed_l768_768784

theorem greatest_number_of_problems_missed 
    (total_problems : ℕ) (passing_percentage : ℝ) (max_missed : ℕ) :
    total_problems = 40 →
    passing_percentage = 0.85 →
    max_missed = total_problems - ⌈total_problems * passing_percentage⌉ →
    max_missed = 6 :=
by
  intros h1 h2 h3
  sorry

end greatest_number_of_problems_missed_l768_768784


namespace count_four_digit_integers_with_3_and_6_l768_768111

theorem count_four_digit_integers_with_3_and_6 : 
  (∃ (count : ℕ), count = 16 ∧ 
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → 
  (∀ i : ℕ, i < 4 → (n / (10 ^ i)) % 10 = 3 ∨ (n / (10 ^ i)) % 10 = 6) ↔ n ∈ {3333, 3366, 3633, 3666, 6333, 6366, 6633, 6666}) :=
by
  have h : 2 ^ 4 = 16 := by norm_num
  use 16
  split
  · exact h
  · sorry

end count_four_digit_integers_with_3_and_6_l768_768111


namespace find_parabola_standard_equation_l768_768074

-- Define the conditions
def hyperbola : Prop := ∀ (x y : ℝ), 16 * x^2 - 9 * y^2 = 144

def focus_of_parabola (x y : ℝ) : Prop :=
  (x, y) = (-3, 0)

def passes_through_point (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  p = (2, -4) → x = 2 ∧ y = -4

noncomputable def parabola_eq : Prop :=
  ∃ (p : ℝ), (y^2 = 2 * p * x ∨ x^2 = -y) 

theorem find_parabola_standard_equation :
  hyperbola →
  (∃ (x y : ℝ), focus_of_parabola x y ∧ passes_through_point (2, -4) x y) →
  parabola_eq :=
by
  -- hyperbola condition
  assume h_hyperbola,
  -- focus of the parabola and passes through point condition
  assume ⟨x, y, h_focus, h_passes⟩,
  sorry

end find_parabola_standard_equation_l768_768074


namespace muffins_milk_proportion_l768_768342

theorem muffins_milk_proportion :
  ∀ (liters_milk : ℝ) (muffins : ℕ) (liters_to_cups : ℝ), 
  muffins = 24 → liters_milk = 3 → liters_to_cups = 4 →
  let cups := liters_milk * liters_to_cups in
  (∀ (smaller_batch : ℕ), smaller_batch = 6 →
  (cups / muffins) * smaller_batch = 3) :=
begin
  intros liters_milk muffins liters_to_cups h_muffins h_liters h_conversion smaller_batch h_smaller_batch,
  sorry,
end

end muffins_milk_proportion_l768_768342


namespace four_digit_3_or_6_l768_768116

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end four_digit_3_or_6_l768_768116


namespace eval_floor_ceil_l768_768013

theorem eval_floor_ceil : ⌊-3.75⌋ + ⌈34.2⌉ + 1 / 2 = 31.5 := by
  sorry

end eval_floor_ceil_l768_768013


namespace N_inequality_l768_768232

noncomputable def N (a1 a2 a3 : ℕ) : ℕ :=
∑ (x1 : ℕ) in (finset.range (2 * a1 + 1)).filter (λ x1, x1 > 0 ∧ x1 ∣ a1), -- Summation over the range we're interested in
∑ (x2 : ℕ) in (finset.range (2 * a2 + 1)).filter (λ x2, x2 > 0 ∧ x2 ∣ a2), -- Summation over the range we're interested in
∑ (x3 : ℕ) in (finset.range (2 * a3 + 1)).filter (λ x3, x3 > 0 ∧ x3 ∣ a3), -- Summation over the range we're interested in
if (a1/x1 + a2/x2 + a3/x3 = 1) then 1 else 0

theorem N_inequality (a1 a2 a3 : ℕ) (h : a1 ≥ a2 ∧ a2 ≥ a3) :
  N a1 a2 a3 ≤ 6 * a1 * a2 * (3 + Real.log (2 * a1)) := 
sorry

end N_inequality_l768_768232


namespace probability_is_two_over_nine_l768_768722

def is_inside_circle (m n : ℕ) : Prop :=
  m^2 + n^2 < 17

def valid_points := 
  {(m, n) | m ∈ {1, 2, 3, 4, 5, 6} ∧ n ∈ {1, 2, 3, 4, 5, 6} ∧ is_inside_circle m n}

def total_points := 
  {(m, n) | m ∈ {1, 2, 3, 4, 5, 6} ∧ n ∈ {1, 2, 3, 4, 5, 6}}

theorem probability_is_two_over_nine : 
  (valid_points.to_finset.card : ℚ) / (total_points.to_finset.card : ℚ) = 2 / 9 := 
by
  sorry

end probability_is_two_over_nine_l768_768722


namespace linear_equation_m_not_eq_4_l768_768282

theorem linear_equation_m_not_eq_4 (m x y : ℝ) :
  (m * x + 3 * y = 4 * x - 1) → m ≠ 4 :=
by
  sorry

end linear_equation_m_not_eq_4_l768_768282


namespace transform_graph_2cos2x_l768_768692

theorem transform_graph_2cos2x (x : ℝ) : 
  (∀ x, 2 * cos x ^ 2 = 2 * (1 / 2 + cos (2 * x) / 2)) → 
  (∀ x, 1 + cos x = 1 + cos x) → 
  (∀ y, 1 + cos (2 * (x / 2)) = y) :=
by 
  sorry

end transform_graph_2cos2x_l768_768692


namespace green_light_probability_l768_768451

-- Define the durations of the red, green, and yellow lights
def red_light_duration : ℕ := 30
def green_light_duration : ℕ := 25
def yellow_light_duration : ℕ := 5

-- Define the total cycle time
def total_cycle_time : ℕ := red_light_duration + green_light_duration + yellow_light_duration

-- Define the expected probability
def expected_probability : ℚ := 5 / 12

-- Prove the probability of seeing a green light equals the expected_probability
theorem green_light_probability :
  (green_light_duration : ℚ) / (total_cycle_time : ℚ) = expected_probability :=
by
  sorry

end green_light_probability_l768_768451


namespace max_students_test_l768_768768

noncomputable def max_students (questions options : ℕ) : ℕ :=
  if questions = 4 ∧ options = 3 then 9 else 0

theorem max_students_test :
  ∃ n : ℕ, 
  (∀ (students : List (Array (Fin 4) (Fin 3))),
    (∀ (s1 s2 s3 : Fin students.length),
      s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3 →
      ∃ q : Fin 4,
      students[s1][q] ≠ students[s2][q]
      ∧ students[s1][q] ≠ students[s3][q]
      ∧ students[s2][q] ≠ students[s3][q])
    ) →
  n = max_students 4 3 :=
sorry

end max_students_test_l768_768768


namespace min_ω_value_l768_768083

noncomputable def ω_min : ℕ := 2

theorem min_ω_value (ω : ℕ) (h1 : ω ∈ (Set.mem (Set.Ico 1 (ω + 1)))) :
  (∀ x : ℝ, y = cos (ω * x - π / 3) → axis_of_symmetry y (π / 6)) → 
  ω = ω_min :=
sorry

end min_ω_value_l768_768083


namespace solution_for_x_l768_768225

theorem solution_for_x : ∀ (x : ℚ), (10 * x^2 + 9 * x - 2 = 0) ∧ (30 * x^2 + 59 * x - 6 = 0) → x = 1 / 5 :=
by
  sorry

end solution_for_x_l768_768225


namespace Jane_exercises_days_per_week_l768_768969

theorem Jane_exercises_days_per_week 
  (goal_hours_per_day : ℕ)
  (weeks : ℕ)
  (total_hours : ℕ)
  (exercise_days_per_week : ℕ) 
  (h_goal : goal_hours_per_day = 1)
  (h_weeks : weeks = 8)
  (h_total_hours : total_hours = 40)
  (h_exercise_hours_weekly : total_hours / weeks = exercise_days_per_week) :
  exercise_days_per_week = 5 :=
by
  sorry

end Jane_exercises_days_per_week_l768_768969


namespace oil_price_reduction_l768_768434

/-- 
Given:
1. The reduced price per kg of oil is Rs. 60.
2. A 30% reduction in price from the original price P results in the reduced price.
3. Rs. 1800 is spent to buy the oil.

Prove:
The housewife can obtain 9 kgs more oil with the reduced price.
-/
theorem oil_price_reduction
  (P : ℝ)
  (h1 : 0.7 * P = 60)
  (h2 : 1800 / 60 - 1800 / P = 9) :
  1800 / 60 - 1800 / P = 9 := 
begin
  exact h2
end

end oil_price_reduction_l768_768434


namespace number_of_true_propositions_l768_768680

theorem number_of_true_propositions : 
  let prop1 := False
  let prop2 := True
  let prop3 := True
  (nat_add (nat_add (cond prop1 1 0) (cond prop2 1 0)) (cond prop3 1 0)) = 2
:= 
by
  sorry

end number_of_true_propositions_l768_768680


namespace area_of_rectangle_is_sqrt_1935_l768_768432

noncomputable def area_of_rectangle (ABCD : Type) [metric_space ABCD] 
  (A B C D : ABCD) (L L' : line ABCD) (DB : segment ABCD)
  (DE EF FB : ℝ) (hDE : DE = 3) (hEF : EF = 4) (hFB : FB = 5)
  (hL : L ∋ A) (hL' : L' ∋ C) (hL_perp : L ⟂ DB) (hL'_perp : L' ⟂ DB) : ℝ :=
by
  have hDB : DB.length = 12 := by simp [hDE, hEF, hFB]
  sorry

theorem area_of_rectangle_is_sqrt_1935 (ABCD : Type) [metric_space ABCD] 
  (A B C D : ABCD) (L L' : line ABCD) (DB : segment ABCD)
  (DE EF FB : ℝ) (hDE : DE = 3) (hEF : EF = 4) (hFB : FB = 5)
  (hL : L ∋ A) (hL' : L' ∋ C) (hL_perp : L ⟂ DB) (hL'_perp : L' ⟂ DB) :
  area_of_rectangle ABCD A B C D L L' DB DE EF FB hDE hEF hFB hL hL' hL_perp hL'_perp = √1935 :=
by
  simp [area_of_rectangle]
  sorry

end area_of_rectangle_is_sqrt_1935_l768_768432


namespace total_pears_l768_768971

noncomputable def Jason_pears : ℝ := 46
noncomputable def Keith_pears : ℝ := 47
noncomputable def Mike_pears : ℝ := 12
noncomputable def Sarah_pears : ℝ := 32.5
noncomputable def Emma_pears : ℝ := (2 / 3) * Mike_pears
noncomputable def James_pears : ℝ := (2 * Sarah_pears) - 3

theorem total_pears :
  Jason_pears + Keith_pears + Mike_pears + Sarah_pears + Emma_pears + James_pears = 207.5 :=
by
  sorry

end total_pears_l768_768971


namespace sum_of_consecutive_integers_product_336_l768_768308

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l768_768308


namespace range_of_a_l768_768087

-- Definitions capturing the given conditions
variables (a b c : ℝ)

-- Conditions are stated as assumptions
def condition1 := a^2 - b * c - 8 * a + 7 = 0
def condition2 := b^2 + c^2 + b * c - 6 * a + 6 = 0

-- The mathematically equivalent proof problem
theorem range_of_a (h1 : condition1 a b c) (h2 : condition2 a b c) : 1 ≤ a ∧ a ≤ 9 := 
sorry

end range_of_a_l768_768087


namespace Natasha_descend_time_l768_768647

-- Definition of the given conditions
variables [Natasha_time_to_top: ℝ] [t: ℝ]
variables (Natasha_whole_avg_speed: ℝ) (Natasha_climb_avg_speed: ℝ)

noncomputable def Natasha_total_distance :=
  2 * Natasha_climb_avg_speed * Natasha_time_to_top

-- Statement of the proof problem to show the time taken to descend the hill
theorem Natasha_descend_time (h1: Natasha_time_to_top = 4)
  (h2: Natasha_climb_avg_speed = 1.5)
  (h3: Natasha_whole_avg_speed = 2) : t = 2 :=
by 
  -- Needed for handling real numbers since we are only presenting statement
  sorry

end Natasha_descend_time_l768_768647


namespace equilateral_triangle_complex_l768_768882

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry

lemma equal_magnitudes (z1 z2 z3 : ℂ) : z1.abs = z2.abs ∧ z2.abs = z3.abs := sorry

lemma product_sum_zero (z1 z2 z3 : ℂ) : z1 * z2 + z2 * z3 + z3 * z1 = 0 := sorry

theorem equilateral_triangle_complex (z1 z2 z3 : ℂ) 
  (h1 : z1.abs = z2.abs) (h2 : z2.abs = z3.abs) 
  (h3 : z1 * z2 + z2 * z3 + z3 * z1 = 0) : 
  (is_equilateral_triangle z1 z2 z3) := sorry

end equilateral_triangle_complex_l768_768882


namespace num_partition_sets_correct_l768_768592

noncomputable def num_partition_sets (n : ℕ) : ℕ :=
  2^(n-1) - 1

theorem num_partition_sets_correct (n : ℕ) (hn : n ≥ 2) : 
  num_partition_sets n = 2^(n-1) - 1 := 
by sorry

end num_partition_sets_correct_l768_768592


namespace consecutive_integers_sum_l768_768292

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l768_768292


namespace fibonacci_invariant_abs_difference_l768_768254

-- Given the sequence defined by the recurrence relation
def mArithmetical_fibonacci (u_n : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, u_n n = u_n (n - 2) + u_n (n - 1)

theorem fibonacci_invariant_abs_difference (u : ℕ → ℤ) 
  (h : mArithmetical_fibonacci u) :
  ∃ c : ℤ, ∀ n : ℕ, |u (n - 1) * u (n + 2) - u n * u (n + 1)| = c := 
sorry

end fibonacci_invariant_abs_difference_l768_768254


namespace vector_magnitude_proof_l768_768913

open Real

-- Define a structure for vectors in the plane
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

-- Define vector magnitude (norm) function
def Vector2D.norm (v : Vector2D) : ℝ :=
  sqrt (v.x^2 + v.y^2)

-- Define vector subtraction
def Vector2D.sub (v1 v2 : Vector2D) : Vector2D :=
  { x := v1.x - 2 * v2.x, y := v1.y - 2 * v2.y }

-- Vectors and their properties as given in the conditions
variable (a b : Vector2D)
variable (angle : ℝ)
variable (ha : Vector2D.norm a = 1)
variable (hb : Vector2D.norm b = 1 / 2)
variable (hangle : angle = π / 3)

-- The proof statement
theorem vector_magnitude_proof : Vector2D.norm (Vector2D.sub a b) = 1 :=
  sorry

end vector_magnitude_proof_l768_768913


namespace min_magnitude_diff_l768_768046

open Real

-- Definitions of the vectors
def a (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)
def b (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 2 * t - 1, 0)

-- The magnitude of the difference between vectors b and a
def magnitude_diff (t : ℝ) : ℝ :=
  let diff := (1 - t - 2, 2 * t - 1 - t, 0 - t)
  sqrt ((diff.1)^2 + (diff.2)^2 + (diff.3)^2)

theorem min_magnitude_diff : ∀ t : ℝ, magnitude_diff t ≥ sqrt 2 := by
  intro t
  sorry

end min_magnitude_diff_l768_768046


namespace percentage_increase_sale_l768_768388

theorem percentage_increase_sale (P S : ℝ) (hP : 0 < P) (hS : 0 < S) :
  let new_price := 0.65 * P
  let original_revenue := P * S
  let new_revenue := 1.17 * original_revenue
  let percentage_increase := 80 / 100
  let new_sales := S * (1 + percentage_increase)
  new_price * new_sales = new_revenue :=
by
  sorry

end percentage_increase_sale_l768_768388


namespace sum_of_no_solution_primes_l768_768009

theorem sum_of_no_solution_primes :
  ∑ p in { p : ℕ | p.prime ∧ ¬(∃ x : ℤ, 5 * (5 * x + 2) ≡ 3 [MOD p]) }, p = 5 :=
by
  sorry

end sum_of_no_solution_primes_l768_768009


namespace price_increase_profit_relation_proof_price_decrease_profit_relation_proof_max_profit_price_increase_l768_768950

def cost_price : ℝ := 40
def initial_price : ℝ := 60
def initial_sales_volume : ℕ := 300
def sales_decrease_rate (x : ℕ) : ℕ := 10 * x
def sales_increase_rate (a : ℕ) : ℕ := 20 * a

noncomputable def price_increase_proft_relation (x : ℕ) : ℝ :=
  -10 * (x : ℝ)^2 + 100 * (x : ℝ) + 6000

theorem price_increase_profit_relation_proof (x : ℕ) (h : 0 ≤ x ∧ x ≤ 30) :
  price_increase_proft_relation x = -10 * (x : ℝ)^2 + 100 * (x : ℝ) + 6000 := sorry

noncomputable def price_decrease_profit_relation (a : ℕ) : ℝ :=
  -20 * (a : ℝ)^2 + 100 * (a : ℝ) + 6000

theorem price_decrease_profit_relation_proof (a : ℕ) :
  price_decrease_profit_relation a = -20 * (a : ℝ)^2 + 100 * (a : ℝ) + 6000 := sorry

theorem max_profit_price_increase :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 30 ∧ price_increase_proft_relation x = 6250 := sorry

end price_increase_profit_relation_proof_price_decrease_profit_relation_proof_max_profit_price_increase_l768_768950


namespace trains_cross_time_l768_768368

-- Definitions of lengths, speeds, and conversion factor
def len_first_train : ℝ := 250
def len_second_train : ℝ := 350
def speed_first_train : ℝ := 80
def speed_second_train : ℝ := 60
def kmph_to_mps : ℝ := 5 / 18

-- Relative speed calculation
def relative_speed := (speed_first_train + speed_second_train) * kmph_to_mps

-- Total distance calculation
def total_distance := len_first_train + len_second_train

-- Proof statement of the time taken to cross each other
theorem trains_cross_time : (total_distance / relative_speed) ≈ 15.432 :=
by sorry

end trains_cross_time_l768_768368


namespace gcd_280_2155_l768_768286

theorem gcd_280_2155 : Nat.gcd 280 2155 = 35 := 
sorry

end gcd_280_2155_l768_768286


namespace quadratic_real_roots_range_l768_768157

-- Given conditions and definitions
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def equation_has_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c ≥ 0

-- Problem translated into a Lean statement
theorem quadratic_real_roots_range (m : ℝ) :
  equation_has_real_roots 1 (-2) (-m) ↔ m ≥ -1 :=
by
  sorry

end quadratic_real_roots_range_l768_768157


namespace imaginary_unit_problem_l768_768868

variable {a b : ℝ}

theorem imaginary_unit_problem (h : i * (a + i) = b + 2 * i) : a + b = 1 :=
sorry

end imaginary_unit_problem_l768_768868


namespace collinear_if_and_only_if_concyclic_l768_768869

noncomputable def convex_quadrilateral 
  (E B C D O : Type) 
  [O ∈ (EBCD : convex_quadrilateral)] : Prop := sorry

noncomputable def orthocenter 
  (H1 H2 : Type) 
  [H1 ∈ (convex_quadrilateral.orthocenter E O D)] 
  [H2 ∈ (convex_quadrilateral.orthocenter B O C)] : Prop := sorry

noncomputable def intersection 
  (BE CD A : Type) : Prop := sorry

theorem collinear_if_and_only_if_concyclic 
  {E B C D O A H1 H2 : Type}
  [convex_quadrilateral E B C D O]
  [orthocenter H1 H2]
  [intersection BE CD A] :
  (collinear A H1 H2) ↔ (concyclic E B C D) := sorry

end collinear_if_and_only_if_concyclic_l768_768869


namespace probability_of_opening_on_third_attempt_l768_768430

-- Define the scenario
def scenario : Type := 
  { keys : ℕ // keys = 5 } ∧ 
  { open_keys : ℕ // open_keys = 2 }

-- Define the probability of successfully opening the door exactly on the third attempt
def probability_success_on_third_attempt (s : scenario) : ℚ :=
  1 / 5

-- Statement of the problem in Lean
theorem probability_of_opening_on_third_attempt (s : scenario) :
  probability_success_on_third_attempt s = 1 / 5 :=
sorry

end probability_of_opening_on_third_attempt_l768_768430


namespace sphere_radius_l768_768422

theorem sphere_radius (d_sphere_shadow r_post h_post: ℝ) (par_sun: Prop) (tan_rel: Prop) :
    d_sphere_shadow = 15 ->
    r_post = 1.5 ->
    h_post = 3 ->
    par_sun ->
    tan_rel ->
    let r := d_sphere_shadow * (r_post / h_post) in
    r = 7.5 :=
by 
  sorry

end sphere_radius_l768_768422


namespace count_four_digit_integers_with_3_and_6_l768_768115

theorem count_four_digit_integers_with_3_and_6 : 
  (∃ (count : ℕ), count = 16 ∧ 
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → 
  (∀ i : ℕ, i < 4 → (n / (10 ^ i)) % 10 = 3 ∨ (n / (10 ^ i)) % 10 = 6) ↔ n ∈ {3333, 3366, 3633, 3666, 6333, 6366, 6633, 6666}) :=
by
  have h : 2 ^ 4 = 16 := by norm_num
  use 16
  split
  · exact h
  · sorry

end count_four_digit_integers_with_3_and_6_l768_768115


namespace ram_shyam_weight_increase_by_approx_l768_768326

noncomputable def ram_shyam_total_weight_increase_percent : Real :=
let total_weight_initial : Real := 6 * (82.8 / (6.6 + 5)) + 5 * (82.8 / (6.6 + 5))
let ram_new_weight : Real := 6 * (82.8 / (6.6 + 5)) * 1.10
let shyam_new_weight : Real := 5 * (82.8 / (6.6 + 5)) * 1.21
let total_weight_new : Real := ram_new_weight + shyam_new_weight
let weight_increase : Real := total_weight_new - total_weight_initial
(weight_increase / total_weight_initial) * 100

theorem ram_shyam_weight_increase_by_approx:
    ram_shyam_total_weight_increase_percent ≈ 14.99 := 
begin 
  sorry
end

end ram_shyam_weight_increase_by_approx_l768_768326


namespace intersection_M_N_eq_l768_768888

variables (f : ℝ → ℝ)
variables (φ : ℝ → ℝ)
variables (f_odd : ∀ x, f(-x) = -f(x))
variables (f_strict_mono_neg : ∀ x y, x < y → y < 0 → f(x) < f(y))
variables (f_at_neg_one : f(-1) = 0)
variables (M : set ℝ := { m | ∀ x ∈ set.Icc 0 (real.pi / 2), φ x < 0 })
variables (N : set ℝ := { m | ∀ x ∈ set.Icc 0 (real.pi / 2), f (φ x) < 0 })

def φ_definition (m : ℝ) : ℝ → ℝ :=
  λ x, real.sin x ^ 2 + m * real.cos x - 2 * m

theorem intersection_M_N_eq :
  M ∩ N = { m | 4 - 2 * real.sqrt 2 < m } :=
sorry

end intersection_M_N_eq_l768_768888


namespace area_of_region_l768_768380

theorem area_of_region (x y : ℝ) : (x^2 + y^2 + 6 * x - 8 * y = 1) → (π * 26) = 26 * π :=
by
  intro h
  sorry

end area_of_region_l768_768380


namespace redistribute_oil_l768_768330

def total_boxes (trucks1 trucks2 boxes1 boxes2 : Nat) :=
  (trucks1 * boxes1) + (trucks2 * boxes2)

def total_containers (boxes containers_per_box : Nat) :=
  boxes * containers_per_box

def containers_per_truck (total_containers trucks : Nat) :=
  total_containers / trucks

theorem redistribute_oil :
  ∀ (trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks : Nat),
  trucks1 = 7 →
  trucks2 = 5 →
  boxes1 = 20 →
  boxes2 = 12 →
  containers_per_box = 8 →
  total_trucks = 10 →
  containers_per_truck (total_containers (total_boxes trucks1 trucks2 boxes1 boxes2) containers_per_box) total_trucks = 160 :=
by
  intros trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks
  intros h_trucks1 h_trucks2 h_boxes1 h_boxes2 h_containers_per_box h_total_trucks
  sorry

end redistribute_oil_l768_768330


namespace find_value_of_f_l768_768900

noncomputable def f (ω φ : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem find_value_of_f (ω φ : ℝ) (h_symmetry : ∀ x : ℝ, f ω φ (π/4 + x) = f ω φ (π/4 - x)) :
  f ω φ (π/4) = 2 ∨ f ω φ (π/4) = -2 := 
sorry

end find_value_of_f_l768_768900


namespace max_distinct_numbers_in_table_l768_768184

open Nat

theorem max_distinct_numbers_in_table : 
  ∀ (M : Matrix (Fin 75) (Fin 75) ℕ), 
  (∀ i : Fin 75, (Finset.card (Finset.image (λ j => M i j) Finset.univ)) ≥ 15) →
  (∀ i : Fin 73, (Finset.card (Finset.bUnion (Finset.Icc i (i+2 : Fin 75)) (λ k => Finset.image (λ j => M k j) Finset.univ)) ≤ 25) →
  ∃ N : ℕ, 
  Finset.card (Finset.bUnion Finset.univ (λ i => Finset.image (λ j => M i j) Finset.univ)) = 385 :=
by
  sorry

end max_distinct_numbers_in_table_l768_768184


namespace correct_statement_l768_768443

noncomputable def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  let rec b_aux (m : ℕ) :=
    match m with
    | 0     => 0
    | m + 1 => 1 + 1 / (α m + b_aux m)
  b_aux n

theorem correct_statement (α : ℕ → ℕ) (h : ∀ k, α k > 0) : b 4 α < b 7 α :=
by sorry

end correct_statement_l768_768443


namespace quadratic_roots_real_l768_768853

theorem quadratic_roots_real (k : ℝ) : 
  (∃ x : ℝ, x^2 - k * x + k^2 - 1 = 0) ↔ (k ∈ set.Icc (-(2 * real.sqrt 3 / 3)) (2 * real.sqrt 3 / 3)) := 
sorry

end quadratic_roots_real_l768_768853


namespace leftmost_three_digits_of_largest_N_l768_768616

open Nat

noncomputable def largest_N_with_property : ℕ :=
  let squares := [16, 25, 36, 49, 64, 81]
  let possible_sequences := 
    [81649, 1649, 3649, 49, 649, 25]
  possible_sequences.maximum'.get! sorry

theorem leftmost_three_digits_of_largest_N : 
  let N := largest_N_with_property in
  Nat.digits 10 N = [8, 1, 6] :=
by
  sorry

end leftmost_three_digits_of_largest_N_l768_768616


namespace sum_bound_l768_768867

open Real

noncomputable def problem (n : ℕ) (h : n ≥ 2) (x : Fin n → ℝ) : ℝ :=
  | ∑ i in Finset.range n, ∑ j in Finset.range (i + 1), x j | / (i + 1) 

theorem sum_bound 
  (n : ℕ) (hn : n ≥ 2)
  (x : Fin n → ℝ)
  (h_sum_abs : ∑ i in Finset.range n, |x i| = 1)
  (h_sum : ∑ i in Finset.range n, x i = 0) :
  | problem n hn x | ≤ 1/2 - 1/(2*n) :=
sorry

end sum_bound_l768_768867


namespace find_area_of_triangle_l768_768497

noncomputable def area_of_triangle_tangent_lines : ℝ := 
  let intersection_point := (1 : ℝ, 1 : ℝ)
  let tangent_line_1 := λ x : ℝ, -x + 2
  let tangent_line_2 := λ x : ℝ, 2 * x - 1
  let x_intercept_line_1 := 2
  let x_intercept_line_2 := 1 / 2
  1/2 * (x_intercept_line_1 - x_intercept_line_2) * 1

theorem find_area_of_triangle :
  let intersection_point := (1 : ℝ, 1 : ℝ)
  let tangent_line_1 := λ x : ℝ, -x + 2
  let tangent_line_2 := λ x : ℝ, 2 * x - 1
  let x_intercept_line_1 := 2
  let x_intercept_line_2 := 1 / 2
  
  (1 / 2 * (x_intercept_line_1 - x_intercept_line_2) * 1 = 3 / 4) := 
by
  let intersection_point := (1 : ℝ, 1 : ℝ)
  let tangent_line_1 := λ x : ℝ, -x + 2
  let tangent_line_2 := λ x : ℝ, 2 * x - 1
  let x_intercept_line_1 := 2
  let x_intercept_line_2 := 1 / 2
  
  have area : 1 / 2 * (x_intercept_line_1 - x_intercept_line_2) * 1 = 3 / 4 := sorry
  exact area

end find_area_of_triangle_l768_768497


namespace min_correct_all_four_l768_768583

def total_questions : ℕ := 15
def correct_xiaoxi : ℕ := 11
def correct_xiaofei : ℕ := 12
def correct_xiaomei : ℕ := 13
def correct_xiaoyang : ℕ := 14

theorem min_correct_all_four : 
(∀ total_questions correct_xiaoxi correct_xiaofei correct_xiaomei correct_xiaoyang, 
  total_questions = 15 → correct_xiaoxi = 11 → 
  correct_xiaofei = 12 → correct_xiaomei = 13 → 
  correct_xiaoyang = 14 → 
  ∃ k : ℕ, k = 5 ∧ 
    k = total_questions - ((total_questions - correct_xiaoxi) + 
    (total_questions - correct_xiaofei) + 
    (total_questions - correct_xiaomei) + 
    (total_questions - correct_xiaoyang)) / 4) := 
sorry

end min_correct_all_four_l768_768583


namespace prove_propositions_count_l768_768875

def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := sorry

-- Given conditions
axiom h1 : ∀ x, f(x) + (deriv g)(x) = 10
axiom h2 : ∀ x, f(x) - (deriv g)(4-x) = 10
axiom h3 : ∀ x, g(-x) = g(x)

-- Prove the conclusion
theorem prove_propositions_count :
  (f(1) + f(3) = 20 ∧ f(4) = 10 ∧ f(2022) = 10)
  → f(-1) = f(-3)
  → 3 = 3 :=
sorry

end prove_propositions_count_l768_768875


namespace no_nat_number_satisfies_l768_768607

theorem no_nat_number_satisfies (n : ℕ) : ¬ ((n^2 + 6 * n + 2019) % 100 = 0) :=
sorry

end no_nat_number_satisfies_l768_768607


namespace equilateral_octagon_has_side_length_p_plus_q_plus_r_l768_768957

open Real

noncomputable def octagon_side_length (AE FB : ℝ) (AB BC : ℝ) : ℝ :=
  let s := (-9 + sqrt 163)
  s

theorem equilateral_octagon_has_side_length :
  ∀ (AE FB : ℝ) (AB BC : ℝ),
    AE = FB →
    AE < 5 →
    AB = 10 →
    BC = 8 →
    octagon_side_length AE FB AB BC = -9 + sqrt 163 :=
by
  intros AE FB AB BC h1 h2 h3 h4
  unfold octagon_side_length
  sorry

theorem p_plus_q_plus_r :
  ∀ (AE FB : ℝ) (AB BC : ℝ),
    AE = FB →
    AE < 5 →
    AB = 10 →
    BC = 8 →
    let s := octagon_side_length AE FB AB BC
    let p := -9
    let q := 1
    let r := 163
    p + q + r = 155 :=
by
  intros AE FB AB BC h1 h2 h3 h4
  let s := octagon_side_length AE FB AB BC
  let p := -9
  let q := 1
  let r := 163
  show p + q + r = 155
  calc
    p + q + r = (-9) + 1 + 163 := by rfl
          ... = 155            := by norm_num

end equilateral_octagon_has_side_length_p_plus_q_plus_r_l768_768957


namespace num_four_digit_pos_integers_l768_768131

theorem num_four_digit_pos_integers : 
  ∃ n : ℕ, (n = 16) ∧ ∀ k : ℕ, (1000 ≤ k ∧ k < 10000 ∧ 
  ∀ d ∈ [k.digits 10], d = 3 ∨ d = 6) := sorry

end num_four_digit_pos_integers_l768_768131


namespace number_of_integers_in_interval_l768_768560

theorem number_of_integers_in_interval :
  let n : ℤ := by sorry,
  let pi : ℝ := by sorry
in ∀ n, -5 * pi ≤ (n : ℝ) ∧ (n : ℝ) ≤ 15 * pi → 63 :=
sorry

end number_of_integers_in_interval_l768_768560


namespace num_prime_divisors_of_50_fac_l768_768142

-- Define the set of all prime numbers less than or equal to 50.
def primes_le_50 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

-- Define the factorial function.
noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the number of prime divisors of n.
noncomputable def num_prime_divisors (n : ℕ) : ℕ :=
(set.count (λ p, p ∣ n) primes_le_50)

-- The theorem statement.
theorem num_prime_divisors_of_50_fac : num_prime_divisors (factorial 50) = 15 :=
by
  sorry

end num_prime_divisors_of_50_fac_l768_768142


namespace union_M_N_l768_768089

def M := {1, 2}
def N := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_M_N :
  M ∪ N = {1, 2, 3} := 
sorry

end union_M_N_l768_768089


namespace triangle_area_l768_768599

noncomputable def sqrt3 := Real.sqrt 3

/-- For a triangle ABC with sides a = sqrt(3), b = 1, and angle B = 30 degrees,
  the possible areas of the triangle are sqrt(3)/2 or sqrt(3)/4 -/
theorem triangle_area (
  (a : ℝ) (b : ℝ) (B : ℝ)
  (ha : a = sqrt3) (hb : b = 1) (hB : B = π / 6)
) :
  ∃ S, (S = sqrt3 / 2 ∨ S = sqrt3 / 4) :=
sorry

end triangle_area_l768_768599


namespace greatest_number_of_bouquets_l768_768262

def sara_red_flowers : ℕ := 16
def sara_yellow_flowers : ℕ := 24

theorem greatest_number_of_bouquets : Nat.gcd sara_red_flowers sara_yellow_flowers = 8 := by
  rfl

end greatest_number_of_bouquets_l768_768262


namespace lateral_surface_area_of_cone_l768_768756

-- Defining a cone with given properties
structure Cone :=
(base_radius : ℝ)
(sector_radius : ℝ)

-- Given conditions as part of the problem
def given_cone : Cone := {base_radius := 4, sector_radius := 5}

-- The theorem statement representing the problem
theorem lateral_surface_area_of_cone (c : Cone) (h_base : c.base_radius = 4) (h_sector : c.sector_radius = 5) :
  ∃ A, A = 20 * Real.pi := by
sory

end lateral_surface_area_of_cone_l768_768756


namespace probability_two_vertices_endpoints_l768_768354

theorem probability_two_vertices_endpoints (V E : Type) [Fintype V] [DecidableEq V] 
  (dodecahedron : Graph V E) (h1 : Fintype.card V = 20)
  (h2 : ∀ v : V, Fintype.card (dodecahedron.neighbors v) = 3)
  (h3 : Fintype.card E = 30) :
  (∃ A B : V, A ≠ B ∧ (A, B) ∈ dodecahedron.edgeSet) → 
  (∃ p : ℚ, p = 3/19) := 
sorry

end probability_two_vertices_endpoints_l768_768354


namespace arithmetic_sequence_count_eq_45_geometric_sequence_count_eq_17_l768_768471

-- Definition of the conditions
def is_three_digit_integer (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_arithmetic_sequence (h t u : ℕ) : Prop := 
  t - h = u - t ¬ and u - t constitutes the common difference

def is_geometric_sequence (h t u : ℕ) : Prop := 
  (t / h ) = (u / t ) ¬ and (u / t) constitutes the common ration

-- The proof problem statements
theorem arithmetic_sequence_count_eq_45 : 
  ∃ (count : ℕ), count = 45 ∧ ∀ n, is_three_digit_integer n → 
  (∃ h t u, (n = 100 * h + 10 * t + u ∧ is_arithmetic_sequence h t u)).count = count := 
sorry

theorem geometric_sequence_count_eq_17 : 
  ∃ (count : ℕ), count = 17 ∧ ∀ n, is_three_digit_integer n → 
  (∃ h t u, (n = 100 * h + 10 * t + u ∧ is_geometric_sequence h t u)).count = count := 
sorry

end arithmetic_sequence_count_eq_45_geometric_sequence_count_eq_17_l768_768471


namespace consecutive_integers_sum_l768_768291

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l768_768291


namespace constant_term_expansion_l768_768961

theorem constant_term_expansion : 
  (∃ k: ℤ, x^(k + 2): ℂ) ∧ (4^(-x) - 1) * (2^x - 3)^5 = (∃ k: ℂ) -27 := 
by
  sorry

end constant_term_expansion_l768_768961


namespace b4_lt_b7_l768_768440

noncomputable def b : ℕ → ℝ
| 1       := 1 + 1 / α 1
| (n + 1) := 1 + 1 / (α 1 + b n)

theorem b4_lt_b7 (α : ℕ → ℝ) (hα : ∀ k, α k > 0) : b α 4 < b α 7 :=
by { sorry }

end b4_lt_b7_l768_768440


namespace consecutive_integers_product_l768_768299

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l768_768299


namespace part_a_part_b_l768_768251

-- Definitions for the given problem conditions for Part (a)
variables (A B C D P T M N : Type)
variables (angle_ABC : ℝ)
variables (dist_MP dist_NT dist_BD : ℝ)

-- Definitions for Part (a)
-- Point D lies on the side AC of triangle ABC
def point_D_on_AC (A B C D : Type) : Prop := sorry
-- A circle with diameter BD intersects sides AB and BC at points P and T respectively
def circle_diameter_BD_intersections (A B C D P T : Type) : Prop := sorry
-- Points M and N are the midpoints of segments AD and CD respectively
def midpoints_M_N (A D M C N : Type) (dist_MP : ℝ) (dist_NT : ℝ) : Prop := sorry
-- PM is parallel to TN
def PM_parallel_TN (P M T N : Type) : Prop := sorry

-- Part (a): Prove that ∠ABC = 90°
theorem part_a (A B C D P T M N : Type)
  (angle_ABC : ℝ)
  (h1 : point_D_on_AC A B C D)
  (h2 : circle_diameter_BD_intersections A B C D P T)
  (h3 : midpoints_M_N A D M C N dist_MP dist_NT)
  (h4 : PM_parallel_TN P M T N) :
  angle_ABC = 90 := 
sorry

-- Definitions for the additional conditions in Part (b)
-- MP = 1/2
def MP_length (dist_MP : ℝ) : Prop := dist_MP = 1/2
-- NT = 2
def NT_length (dist_NT : ℝ) : Prop := dist_NT = 2
-- BD = sqrt(3)
def BD_length (dist_BD : ℝ) : Prop := dist_BD = sqrt 3

-- Part (b): Prove the area of triangle ABC is (5√13) / (3√2)
theorem part_b (A B C D P T M N : Type)
  (dist_MP dist_NT dist_BD : ℝ)
  (area_ABC : ℝ)
  (h1 : point_D_on_AC A B C D)
  (h2 : circle_diameter_BD_intersections A B C D P T)
  (h3 : midpoints_M_N A D M C N dist_MP dist_NT)
  (h4 : PM_parallel_TN P M T N)
  (h5 : MP_length dist_MP)
  (h6 : NT_length dist_NT)
  (h7 : BD_length dist_BD) :
  area_ABC = (5 * sqrt 13) / (3 * sqrt 2) := 
sorry

end part_a_part_b_l768_768251


namespace pumpkins_total_weight_l768_768242

-- Define the weights of the pumpkins as given in the conditions
def first_pumpkin_weight : ℝ := 4
def second_pumpkin_weight : ℝ := 8.7

-- Prove that the total weight of the two pumpkins is 12.7 pounds
theorem pumpkins_total_weight : first_pumpkin_weight + second_pumpkin_weight = 12.7 := by
  sorry

end pumpkins_total_weight_l768_768242


namespace solve_for_x_l768_768845

theorem solve_for_x (x : ℝ) (h : sqrt (x + 3) = 7) : x = 46 :=
by {
  -- proof will be here
  sorry
}

end solve_for_x_l768_768845


namespace percent_of_dollar_is_37_l768_768041

variable (coins_value_in_cents : ℕ)
variable (percent_of_one_dollar : ℕ)

def value_of_pennies : ℕ := 2 * 1
def value_of_nickels : ℕ := 3 * 5
def value_of_dimes : ℕ := 2 * 10

def total_coin_value : ℕ := value_of_pennies + value_of_nickels + value_of_dimes

theorem percent_of_dollar_is_37
  (h1 : total_coin_value = coins_value_in_cents)
  (h2 : percent_of_one_dollar = (coins_value_in_cents * 100) / 100) : 
  percent_of_one_dollar = 37 := 
by
  sorry

end percent_of_dollar_is_37_l768_768041


namespace containers_per_truck_l768_768335

theorem containers_per_truck (trucks1 boxes1 trucks2 boxes2 boxes_to_containers total_trucks : ℕ)
  (h1 : trucks1 = 7) 
  (h2 : boxes1 = 20) 
  (h3 : trucks2 = 5) 
  (h4 : boxes2 = 12) 
  (h5 : boxes_to_containers = 8) 
  (h6 : total_trucks = 10) :
  (((trucks1 * boxes1) + (trucks2 * boxes2)) * boxes_to_containers) / total_trucks = 160 := 
sorry

end containers_per_truck_l768_768335


namespace john_made_47000_from_car_l768_768972

def cost_to_fix_before_discount := 20000
def discount := 0.20
def prize := 70000
def keep_percentage := 0.90

def cost_to_fix_after_discount := cost_to_fix_before_discount - (discount * cost_to_fix_before_discount)
def prize_kept := keep_percentage * prize
def money_made := prize_kept - cost_to_fix_after_discount

theorem john_made_47000_from_car : money_made = 47000 := by
  sorry

end john_made_47000_from_car_l768_768972


namespace probability_of_green_face_is_3_over_10_l768_768273

theorem probability_of_green_face_is_3_over_10 :
  let total_faces := 10
  let green_faces := 3
  ∃ (p : ℚ), p = (green_faces : ℚ) / (total_faces : ℚ) ∧ p = 3/10 :=
by
  let total_faces := 10
  let green_faces := 3
  use (green_faces : ℚ) / (total_faces : ℚ)
  split
  · refl
  · norm_num
  · unfold nat.cast sorry

end probability_of_green_face_is_3_over_10_l768_768273


namespace greatest_possible_int_diff_l768_768630

theorem greatest_possible_int_diff (x a y b : ℝ) 
    (hx : 3 < x ∧ x < 4) 
    (ha : 4 < a ∧ a < x) 
    (hy : 6 < y ∧ y < 8) 
    (hb : 8 < b ∧ b < y) 
    (h_ineq : a^2 + b^2 > x^2 + y^2) : 
    abs (⌊x⌋ - ⌈y⌉) = 2 :=
sorry

end greatest_possible_int_diff_l768_768630


namespace number_of_a_l768_768055

theorem number_of_a (h : ∃ a : ℝ, ∃! x : ℝ, |x^2 + 2 * a * x + 3 * a| ≤ 2) : 
  ∃! a : ℝ, ∃! x : ℝ, |x^2 + 2 * a * x + 3 * a| ≤ 2 :=
sorry

end number_of_a_l768_768055


namespace files_remaining_l768_768376

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h_music : music_files = 13) 
  (h_video : video_files = 30) 
  (h_deleted : deleted_files = 10) : 
  (music_files + video_files - deleted_files) = 33 :=
by
  sorry

end files_remaining_l768_768376


namespace sequence_count_l768_768006

theorem sequence_count :
  ∃ (a : ℕ → ℕ), 
  (a 10 = 3 * a 1) ∧ 
  (a 2 + a 8 = 2 * a 5) ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ 9 → (a (i + 1) = 1 + a i ∨ a (i + 1) = 2 + a i)) ∧ 
  (number_of_valid_sequences a = 80) :=
sorry

end sequence_count_l768_768006


namespace smallest_n_for_special_quadruple_l768_768032

-- Given a finite set of integers from 1 to 999
def S := {n : ℕ | n ∈ Set.Icc 1 999}

-- Function that determines if there are four distinct elements a, b, c, d such that a + 2b + 3c = d
def exists_special_quadruple (s : Set ℕ) : Prop :=
  ∃ a b c d ∈ s, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + 2*b + 3*c = d

-- Theorem stating the smallest n such that any subset of n distinct elements from {1,2,...,999} 
-- contains such a quadruple is 835.
theorem smallest_n_for_special_quadruple : ∀ (n : ℕ), 
  (n > 834 → ∀ (s : Set ℕ), s ⊆ S → (Finset.card s = n) → exists_special_quadruple s) ∧ 
  (∀ (s : Set ℕ), s ⊆ S → (Finset.card s = 834) → ¬exists_special_quadruple s) :=
sorry

end smallest_n_for_special_quadruple_l768_768032


namespace fx_increasing_on_R_fx_max_min_on_interval_l768_768552

def f (x : ℝ) := 3 * x + 2

theorem fx_increasing_on_R : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
by
  intro x1 x2 h
  dsimp [f]
  linarith

theorem fx_max_min_on_interval :
  ∃ max min, (max = f (-2)) ∧ (min = f (-3)) :=
by
  use [f (-2), f (-3)]
  dsimp [f]
  split;
  simp;
  linarith

end fx_increasing_on_R_fx_max_min_on_interval_l768_768552


namespace true_proposition_count_l768_768778

-- Define each of the propositions
def Proposition1 : Prop :=
  "From a uniformly moving product assembly line, a quality inspector takes one product every 10 minutes for a certain indicator test. Such sampling is stratified sampling."

def Proposition2 : Prop :=
  "The stronger the linear correlation between two random variables, the closer the absolute value of the correlation coefficient is to 1."

def Proposition3 : Prop :=
  "If the variance of the data \(x_1, x_2, x_3, \ldots, x_n\) is 1, then the variance of \(2x_1, 2x_2, 2x_3, \ldots, 2x_n\) is 2."

def Proposition4 : Prop :=
  "For the observed value \(k\) of the random variable \(k^2\) for categorical variables \(X\) and \(Y\), the smaller \(k\) is, the more confident we can be that \(X\) is related to \(Y\)."

-- Define the main theorem and the conditions
theorem true_proposition_count :
  (¬ Proposition1) ∧ Proposition2 ∧ (¬ Proposition3) ∧ (¬ Proposition4) → ∃! p, p = Proposition2 :=
by sorry

end true_proposition_count_l768_768778


namespace triangle_angles_l768_768571

-- Definitions and Conditions
variables {α β γ : Type} [linear_ordered_field α] [ordered_ring β] [ring γ]

-- A right triangle ABC with ∠C = 90° and the incircle touches BC, CA, and AB at D, E, and F respectively.
def right_triangle (A B C : α) :=
  ∃ (D E F : α), ∠C = 90 ∧ incircle_tangent A B C D E F

-- Prove that the angles of the triangle formed by points of tangency of an incircle in a right triangle are one right angle and two equal acute angles.
theorem triangle_angles (A B C D E F : α)
  (h : right_triangle A B C) : ∠DEF = 90 ∧ ∠EDF = 45 ∧ ∠FED = 45 :=
by 
  sorry

end triangle_angles_l768_768571


namespace cookies_per_batch_l768_768918

-- Define the necessary conditions
def total_chips : ℕ := 81
def batches : ℕ := 3
def chips_per_cookie : ℕ := 9

-- Theorem stating the number of cookies per batch
theorem cookies_per_batch : (total_chips / batches) / chips_per_cookie = 3 :=
by
  -- Here would be the proof, but we use sorry as placeholder
  sorry

end cookies_per_batch_l768_768918


namespace probability_endpoints_of_edge_l768_768365

noncomputable def num_vertices : ℕ := 12
noncomputable def edges_per_vertex : ℕ := 3

theorem probability_endpoints_of_edge :
  let total_ways := Nat.choose num_vertices 2,
      total_edges := (num_vertices * edges_per_vertex) / 2,
      probability := total_edges / total_ways in
  probability = 3 / 11 := by
  sorry

end probability_endpoints_of_edge_l768_768365


namespace seq_value_2008_l768_768633

def seq_condition (n : ℕ) : Prop :=
  (v (n + 1) - v n = 4) ∧ v 1 % 4 = 1 ∧ ∀ k, (∃ m, v k = 4 * m + n)

theorem seq_value_2008 :
  ∀ v : ℕ → ℕ, seq_condition v 2008 → v 2008 = 3703 :=
by
  sorry

end seq_value_2008_l768_768633


namespace derivative_u_l768_768491

noncomputable def u (x : ℝ) : ℝ :=
  let z := Real.sin x
  let y := x^2
  Real.exp (z - 2 * y)

theorem derivative_u (x : ℝ) :
  deriv u x = Real.exp (Real.sin x - 2 * x^2) * (Real.cos x - 4 * x) :=
by
  sorry

end derivative_u_l768_768491


namespace exist_ranking_l768_768582

theorem exist_ranking (competitors : Type)
  (judge_rank : competitors → competitors → Bool)
  (cond_intransitive : ∀ (A B C : competitors) (X Y Z : ℕ), 
     (judge_rank A B ∧ judge_rank B C ∧ judge_rank C A) → False) :
  ∃ ranking : list competitors, 
    ∀ A B, A ∈ ranking → B ∈ ranking → list.index_of A ranking < list.index_of B ranking →
      (∃ judges_count : ℕ, 
        (judges_count ≥ 50) ∧ 
        (judges_count = ∑ j : ℕ in finset.range 100, if judge_rank A B then 1 else 0)) :=
begin
  sorry,
end

end exist_ranking_l768_768582


namespace graph_rotation_180_l768_768682

theorem graph_rotation_180 (x : ℝ) : 
  let G := λ x, exp x
  let G' := λ x, -exp x
  ∀ (x : ℝ), G' x = G (-x) :=
by 
  sorry

end graph_rotation_180_l768_768682


namespace tan_identity_l768_768846

theorem tan_identity : 
  ∀ (A B C : ℝ), A + B + C = 180 ∧ A = 40 ∧ B = 60 ∧ C = 80 →
  (tan (A * (π / 180)) * tan (B * (π / 180)) * tan (C * (π / 180))) / 
  (tan (A * (π / 180)) + tan (B * (π / 180)) + tan (C * (π / 180))) = 1 :=
by
  sorry

end tan_identity_l768_768846


namespace broken_necklaces_l768_768453

theorem broken_necklaces (initial_necklaces : ℕ) (new_necklaces : ℕ)
                        (given_away : ℕ) (current_necklaces : ℕ)
                        (had_55_necklaces : initial_necklaces + new_necklaces = 55)
                        (given_away_necklaces : 55 - given_away = 40)
                        (current_necklaces_eq : 40 - current_necklaces = 3) :
  initial_necklaces = 50 ∧ new_necklaces = 5 ∧ given_away = 15 ∧ current_necklaces = 37 :=
by
  { split;
    { sorry } }

end broken_necklaces_l768_768453


namespace hyperbola_asymptotes_l768_768555

theorem hyperbola_asymptotes (a : ℝ) (x y : ℝ)
  (h₁ : a > 0)
  (h₂ : ∀ (x y : ℝ), x^2 / a^2 - y^2 = 1)
  (h₃ : 2 * real.sqrt ((a^2 + 1) : ℝ) = 4)
  : y = real.sqrt(3) / 3 * x ∨ y = - real.sqrt(3) / 3 * x :=
begin
  sorry
end

end hyperbola_asymptotes_l768_768555


namespace harry_cookies_batch_l768_768919

theorem harry_cookies_batch
  (total_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips = 81)
  (batches = 3)
  (chips_per_cookie = 9) :
  (total_chips / batches) / chips_per_cookie = 3 := by
  sorry

end harry_cookies_batch_l768_768919


namespace redistribute_oil_l768_768329

def total_boxes (trucks1 trucks2 boxes1 boxes2 : Nat) :=
  (trucks1 * boxes1) + (trucks2 * boxes2)

def total_containers (boxes containers_per_box : Nat) :=
  boxes * containers_per_box

def containers_per_truck (total_containers trucks : Nat) :=
  total_containers / trucks

theorem redistribute_oil :
  ∀ (trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks : Nat),
  trucks1 = 7 →
  trucks2 = 5 →
  boxes1 = 20 →
  boxes2 = 12 →
  containers_per_box = 8 →
  total_trucks = 10 →
  containers_per_truck (total_containers (total_boxes trucks1 trucks2 boxes1 boxes2) containers_per_box) total_trucks = 160 :=
by
  intros trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks
  intros h_trucks1 h_trucks2 h_boxes1 h_boxes2 h_containers_per_box h_total_trucks
  sorry

end redistribute_oil_l768_768329


namespace arithmetic_mean_of_integers_from_neg6_to_7_l768_768707

theorem arithmetic_mean_of_integers_from_neg6_to_7 : 
  (list.sum (list.range (7 + 1 + 6) ∘ λ n => n - 6) : ℚ) / (7 - (-6) + 1 : ℚ) = 0.5 := 
by
  sorry

end arithmetic_mean_of_integers_from_neg6_to_7_l768_768707


namespace arithmetic_sequence_evaluation_l768_768717

theorem arithmetic_sequence_evaluation :
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 337 :=
by 
-- Proof omitted
sorry

end arithmetic_sequence_evaluation_l768_768717


namespace sum_log_divisors_eight_pow_l768_768690

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem sum_log_divisors_eight_pow (n : ℕ) :
  (∑ k in Finset.range (3 * n + 1), log10 (2 ^ k)) = 1092 → n = 9 := by
  sorry

end sum_log_divisors_eight_pow_l768_768690


namespace probability_endpoints_of_edge_l768_768362

noncomputable def num_vertices : ℕ := 12
noncomputable def edges_per_vertex : ℕ := 3

theorem probability_endpoints_of_edge :
  let total_ways := Nat.choose num_vertices 2,
      total_edges := (num_vertices * edges_per_vertex) / 2,
      probability := total_edges / total_ways in
  probability = 3 / 11 := by
  sorry

end probability_endpoints_of_edge_l768_768362


namespace probability_of_at_most_two_heads_l768_768732

theorem probability_of_at_most_two_heads (h : ∀ coin : fin 3, prob (coin = H) = 1/2) : 
  prob (at_most_two_heads) = 7/8 := 
by
  -- Definitions for the probability space of coin tosses
  let outcomes := {('HHH, 'HHT, 'HTH, 'HTT, 'THH, 'THT, 'TTH, 'TTT)}
  -- Define what "at_most_two_heads" means for this set of outcomes
  let at_most_two_heads := {o ∈ outcomes | num_heads o ≤ 2}
  sorry

end probability_of_at_most_two_heads_l768_768732


namespace parabola_c_value_l768_768429

theorem parabola_c_value {b c : ℝ} :
  (1:ℝ)^2 + b * (1:ℝ) + c = 2 → 
  (4:ℝ)^2 + b * (4:ℝ) + c = 5 → 
  (7:ℝ)^2 + b * (7:ℝ) + c = 2 →
  c = 9 :=
by
  intros h1 h2 h3
  sorry

end parabola_c_value_l768_768429


namespace nonagon_diagonals_l768_768097

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nonagon_diagonals : number_of_diagonals 9 = 27 := 
by
  sorry

end nonagon_diagonals_l768_768097


namespace equilateral_triangle_complex_l768_768883

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry

lemma equal_magnitudes (z1 z2 z3 : ℂ) : z1.abs = z2.abs ∧ z2.abs = z3.abs := sorry

lemma product_sum_zero (z1 z2 z3 : ℂ) : z1 * z2 + z2 * z3 + z3 * z1 = 0 := sorry

theorem equilateral_triangle_complex (z1 z2 z3 : ℂ) 
  (h1 : z1.abs = z2.abs) (h2 : z2.abs = z3.abs) 
  (h3 : z1 * z2 + z2 * z3 + z3 * z1 = 0) : 
  (is_equilateral_triangle z1 z2 z3) := sorry

end equilateral_triangle_complex_l768_768883


namespace exp_sum_l768_768566

theorem exp_sum (a x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(2 * x + 3 * y) = 108 :=
sorry

end exp_sum_l768_768566


namespace hypotenuse_of_30_degree_triangle_l768_768655

noncomputable theory

-- Define the conditions
def leg_length : ℝ := 15
def angle_opposite_leg : ℝ := 30

-- Define the expected result
def expected_hypotenuse : ℝ := 30

-- The theorem to be proved
theorem hypotenuse_of_30_degree_triangle (leg_length = 15) (angle_opposite_leg = 30) : 
  (2 * leg_length = expected_hypotenuse) :=
by
  sorry

end hypotenuse_of_30_degree_triangle_l768_768655


namespace find_x_l768_768094

noncomputable def a : ℝ × ℝ := (3, 4)
noncomputable def b : ℝ × ℝ := (2, 1)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_x (x : ℝ) (h : dot_product (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2) = 0) : x = -3 :=
  sorry

end find_x_l768_768094


namespace tan_2α_value_l768_768540

-- Define the conditions
variable (α : ℝ)
variable (hα1 : (sin (2 * α)) = - (sqrt 3) * (cos α))
variable (hα2 : α ∈ Ioo (-π / 2) 0)

-- Prove the target statement
theorem tan_2α_value : tan (2 * α) = sqrt 3 := 
sorry

end tan_2α_value_l768_768540


namespace numberOfTruePropositions_l768_768991

noncomputable def prop1 := ∀ (α β : Plane) (m n : Line), 
  (α ∥ β) ∧ (m ⊆ β) ∧ (n ⊆ α) → ¬(m ∥ n)

noncomputable def prop2 := ∀ (α β : Plane) (m n : Line), 
  (α ∥ β) ∧ (m ⊥ β) ∧ (n ∥ α) → (m ⊥ n)

noncomputable def prop3 := ∀ (α β : Plane) (m n : Line), 
  (α ⊥ β) ∧ (m ⊥ α) ∧ (n ∥ β) → ¬(m ∥ n)

noncomputable def prop4 := ∀ (α β : Plane) (m n : Line), 
  (α ⊥ β) ∧ (m ⊥ α) ∧ (n ⊥ β) → (m ⊥ n)

theorem numberOfTruePropositions : 
  (prop1, prop2, prop3, prop4) →
  (2 : ℕ) := 
begin
  sorry
end

end numberOfTruePropositions_l768_768991


namespace triangles_congruent_by_medians_l768_768257

theorem triangles_congruent_by_medians
  (A B C A1 B1 C1 M M1 N N1 K K1 O O1 : Type)
  [AddGroup A]
  [AddGroup B]
  [AddGroup C]
  [AddGroup A1]
  [AddGroup B1]
  [AddGroup C1]
  [AddGroup M]
  [AddGroup M1]
  [AddGroup N]
  [AddGroup N1]
  [AddGroup K]
  [AddGroup K1]
  [AddGroup O]
  [AddGroup O1]
  (AM A1M1 : M)
  (BN B1N1 : N)
  (CK C1K1 : K)
  (median_AM : AM = A1M1)
  (median_BN : BN = B1N1)
  (median_CK : CK = C1K1) :
  \(\triangle ABC \cong \triangle A_1B_1C_1\) := sorry

end triangles_congruent_by_medians_l768_768257


namespace arithmetic_mean_of_integers_from_neg6_to_7_l768_768704

theorem arithmetic_mean_of_integers_from_neg6_to_7 :
  let range := List.range' (-6) 14 in
  let mean := (range.sum : ℚ) / (range.length : ℚ) in
  mean = 0.5 := by
  -- Let's define the range of integers from -6 to 7 inclusive
  let range := List.range' (-6) 14
  -- Let the sum of this range be S
  let S : ℚ := (range.sum)
  -- Let the number of elements in this range be N
  let N : ℚ := (range.length)
  -- The mean of this range is S/N
  let mean := S / N
  -- We assert that this mean is equal to 0.5
  have h_correct_mean : S / N = 0.5 := sorry
  exact h_correct_mean

end arithmetic_mean_of_integers_from_neg6_to_7_l768_768704


namespace ln_inequality_l768_768258

open Real

theorem ln_inequality (x : ℝ) (hx : 0 < x) : ln (x + 1) ≥ x - 1 / 2 * x^2 := sorry

end ln_inequality_l768_768258


namespace four_digit_3_or_6_l768_768120

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end four_digit_3_or_6_l768_768120


namespace dodecahedron_edge_probability_l768_768350

def dodecahedron_vertices : ℕ := 20

def vertex_connections : ℕ := 3

theorem dodecahedron_edge_probability :
  (∃ (u v : fin dodecahedron_vertices), u ≠ v ∧ u.1 < vertex_connections → 
    (Pr (u, v) = 3 / (dodecahedron_vertices - 1))) :=
sorry

end dodecahedron_edge_probability_l768_768350


namespace required_percentage_to_win_election_l768_768179

theorem required_percentage_to_win_election:
  let G := 0.01 * 6000 in
  let W := G + 3000 in
  let V := 6000 in
  (W / V) * 100 = 51 :=
by
  let G : ℝ := 0.01 * 6000
  let W : ℝ := G + 3000
  let V : ℝ := 6000
  show (W / V) * 100 = 51
  sorry

end required_percentage_to_win_election_l768_768179


namespace greatest_integer_y_l768_768710

-- Define the fraction and inequality condition
def inequality_condition (y : ℤ) : Prop := 8 * 17 > 11 * y

-- Prove the greatest integer y satisfying the condition is 12
theorem greatest_integer_y : ∃ y : ℤ, inequality_condition y ∧ (∀ z : ℤ, inequality_condition z → z ≤ y) ∧ y = 12 :=
by
  exists 12
  sorry

end greatest_integer_y_l768_768710


namespace bob_cleaning_time_l768_768776

theorem bob_cleaning_time (alice_time : ℝ) (h1 : alice_time = 40) 
                          (ratio : ℝ) (h2 : ratio = 3 / 8) : 
                          (bob_time : ℝ) (h3 : bob_time = ratio * alice_time) :
                          bob_time = 15 :=
by 
  sorry

end bob_cleaning_time_l768_768776


namespace polynomial_expansion_properties_l768_768860

theorem polynomial_expansion_properties : 
  let a := (λ x, (1 - x)^9) in
  let coeffs := (λ x, coeffs_of_polynomial x) in
  let a_0 := coeffs a 0 in
  let a_1 := coeffs a 1 in
  let a_2 := coeffs a 2 in
  let a_3 := coeffs a 3 in
  let a_4 := coeffs a 4 in
  let a_5 := coeffs a 5 in
  let a_6 := coeffs a 6 in
  let a_7 := coeffs a 7 in
  let a_8 := coeffs a 8 in
  let a_9 := coeffs a 9 in
  (a_0 = 1) ∧ 
  (a_1 + a_3 + a_5 + a_7 + a_9 = -256) ∧ 
  (2 * a_1 + 2^2 * a_2 + 2^3 * a_3 + 2^4 * a_4 + 2^5 * a_5 + 2^6 * a_6 + 2^7 * a_7 + 2^8 * a_8 + 2^9 * a_9 = -2) :=
by sorry

end polynomial_expansion_properties_l768_768860


namespace plate_acceleration_l768_768370

noncomputable def cylindrical_roller_acceleration
  (R : ℝ) (r : ℝ) (m : ℝ) (alpha : ℝ) (g : ℝ) : Prop :=
  let a := g * Real.sqrt((1 - Real.cos(alpha)) / 2) in
  ∃ (accel : ℝ) (dir : ℝ), accel = a ∧ dir = Real.arcsin(0.2)

theorem plate_acceleration:
  cylindrical_roller_acceleration
    1.0  -- R = 1 m
    0.5  -- r = 0.5 m
    75.0 -- m = 75 kg
    (Real.arccos(0.82))  -- α = arccos(0.82)
    10.0  -- g = 10 m/s^2
:=
sorry

end plate_acceleration_l768_768370


namespace f_at_2_l768_768541

-- Definitions based on conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + x
  else sorry  -- We do not need to define for x >= 0 based on conditions

-- Theorem statement
theorem f_at_2 : is_even f → f 2 = 2 :=
by
  sorry

end f_at_2_l768_768541


namespace max_subsets_no_containment_l768_768848

theorem max_subsets_no_containment (n m : ℕ) (h₁ : n ≥ 4)
  (h₂ : ∀ i, 1 ≤ i → i ≤ m → ∃ A : finset ℕ, finset.card A = i)
  (h₃ : ∀ i j : ℕ, 1 ≤ i → i ≤ m → 1 ≤ j → j ≤ m → i ≠ j → ∀ A B : finset ℕ, A ∈ (λ i, finset.univ.filter (λ A, finset.card A = i) i) → B ∈ (λ i, finset.univ.filter (λ A, finset.card A = j) j) → ¬ A ⊆ B) :
  m = n - 2 :=
by
  sorry

end max_subsets_no_containment_l768_768848


namespace power_of_q_in_product_l768_768154

theorem power_of_q_in_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (n : ℕ) (h : Nat.numDivisors (p^3 * q^n) = 28) : n = 6 :=
by 
  sorry

end power_of_q_in_product_l768_768154


namespace graphs_are_ellipse_l768_768866

theorem graphs_are_ellipse (a b : ℝ) (h : a * b ≠ 0) :
  ∃ (E : set (ℝ × ℝ)), (∀ (x y : ℝ), (a * x - y + b = 0 ∧ b * x^2 + a * y^2 = a * b) → (x, y) ∈ E) ∧
  (is_ellipse E) :=
sorry

end graphs_are_ellipse_l768_768866


namespace num_four_digit_integers_with_3_and_6_l768_768106

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end num_four_digit_integers_with_3_and_6_l768_768106


namespace dodecahedron_edge_probability_l768_768349

def dodecahedron_vertices : ℕ := 20

def vertex_connections : ℕ := 3

theorem dodecahedron_edge_probability :
  (∃ (u v : fin dodecahedron_vertices), u ≠ v ∧ u.1 < vertex_connections → 
    (Pr (u, v) = 3 / (dodecahedron_vertices - 1))) :=
sorry

end dodecahedron_edge_probability_l768_768349


namespace sum_of_possible_values_eq_one_l768_768932

theorem sum_of_possible_values_eq_one :
  (∃ x : ℝ, (x + 3) * (x - 4) = 12) →
  (∑ x in ({x | (x + 3) * (x - 4) = 12}), x) = 1 :=
begin
  sorry
end

end sum_of_possible_values_eq_one_l768_768932


namespace smallest_integer_in_consecutive_set_l768_768180

theorem smallest_integer_in_consecutive_set :
  ∃ (n : ℤ), (∀ m, m ∈ {n, n+1, n+2, n+3, n+4, n+5, n+6} → m ≤ (2 * (n + 3))) ∧ n = 0 :=
sorry

end smallest_integer_in_consecutive_set_l768_768180


namespace snowdrift_ratio_l768_768407

theorem snowdrift_ratio
  (depth_first_day : ℕ := 20)
  (depth_second_day : ℕ)
  (h1 : depth_second_day + 24 = 34)
  (h2 : depth_second_day = 10) :
  depth_second_day / depth_first_day = 1 / 2 := by
  sorry

end snowdrift_ratio_l768_768407


namespace fraction_of_triangle_area_l768_768699

theorem fraction_of_triangle_area {A B C K : Type} 
  [ordered_field K]
  (triangle_ABC : ∀ {a b c : K}, triangle A B C)
  (K_cond_internal : ∀ (K : K), homothety_center_ratio A B C K (- (1 / 2)) → is_internal A B K)
  (K_cond_external : ∀ (K : K), homothety_center_ratio A B C K (- (1 / 2)) → is_external C K) :
  area_fraction_K A B C K = (1 / 9) :=
sorry

end fraction_of_triangle_area_l768_768699


namespace min_contribution_l768_768567

theorem min_contribution (h1 : ∑ x in (range 12), x = 20) (h2 : ∀ x : ℝ, x ≥ 1) (h3 : ∀ x : ℝ, x ≤ 9) : 
  ∃ x : ℝ, x = 1 := 
sorry

end min_contribution_l768_768567


namespace min_positive_period_pi_max_value_of_f_l768_768076

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) - 2 * (sin x)^2

theorem min_positive_period_pi : ∀ x : ℝ, f (x + π) = f x := by
  -- To be proved
  sorry

theorem max_value_of_f :
  (∀ x : ℝ, f x ≤ 0) ∧
  (∀ x : ℝ, f x = 0 ↔ ∃ k : ℤ, x = (π / 4) + k * π) := by
  -- To be proved
  sorry

end min_positive_period_pi_max_value_of_f_l768_768076


namespace total_balls_estimation_l768_768956

theorem total_balls_estimation 
  (num_red_balls : ℕ)
  (total_trials : ℕ)
  (red_ball_draws : ℕ)
  (red_ball_ratio : ℚ)
  (total_balls_estimate : ℕ)
  (h1 : num_red_balls = 5)
  (h2 : total_trials = 80)
  (h3 : red_ball_draws = 20)
  (h4 : red_ball_ratio = 1 / 4)
  (h5 : red_ball_ratio = red_ball_draws / total_trials)
  (h6 : red_ball_ratio = num_red_balls / total_balls_estimate)
  : total_balls_estimate = 20 := 
sorry

end total_balls_estimation_l768_768956


namespace positive_integers_divisible_by_4_5_and_6_less_than_300_l768_768930

open Nat

theorem positive_integers_divisible_by_4_5_and_6_less_than_300 : 
    ∃ n : ℕ, n = 5 ∧ ∀ m, m < 300 → (m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0) → (m % 60 = 0) :=
by
  sorry

end positive_integers_divisible_by_4_5_and_6_less_than_300_l768_768930


namespace sum_infinite_geometric_series_l768_768532

noncomputable def geometric_series (a : ℝ) (n : ℕ) : ℝ :=
(1 / 3^n) + a

theorem sum_infinite_geometric_series (a : ℝ) (h : ∀ n : ℕ, S n = geometric_series a n) :
  (∑' n, geometric_series (-1) n) = -1 :=
by
  sorry

end sum_infinite_geometric_series_l768_768532


namespace compare_fractions_l768_768805

theorem compare_fractions : - (1 + 3 / 5) < -1.5 := 
by
  sorry

end compare_fractions_l768_768805


namespace product_divisible_by_8_probability_l768_768374

theorem product_divisible_by_8_probability :
  let dice := ℕ → ℕ
  let outcomes := {n | n ∈ {1, 2, 3, 4, 5, 6}}
  let product_div_by_8 (rolls : fin 8 → ℕ) := 
    (∏ i, rolls i) % 8 = 0
  let probability (P : fin 8 → ℕ → Prop) :=
    let enum := {rolls | ∀ i, P i (rolls i)}
    (fintype.card {rolls | ∀ i, P i (rolls i)} : ℝ) / (6 ^ 8)
  probability (λ _ n, n ∈ outcomes ∧ product_div_by_8) = 13 / 16 :=
sorry

end product_divisible_by_8_probability_l768_768374


namespace integer_root_count_l768_768277

theorem integer_root_count (b c d e f : ℤ) :
  ∃ (n : ℕ), n ∈ {0, 1, 2, 3, 5} ∧ ∀ (root : ℤ), (root = 0 ∨ root ∈ {-(b + root^4 + c * root^3 + d * root^2 + e * root + f)}) :=
sorry

end integer_root_count_l768_768277


namespace team_B_independent_days_l768_768753

-- Definitions and conditions of the problem
def team_A_daily_rate : ℝ := 1 / 20
def team_A_worked_days : ℕ := 4
def total_days_reduced : ℕ := 10
def total_project_days : ℕ := 20
def together_work_days : ℕ := total_project_days - total_days_reduced - team_A_worked_days

-- The proof statement
theorem team_B_independent_days (x : ℝ) :
  (team_A_worked_days * team_A_daily_rate + (team_A_daily_rate + 1 / x) * together_work_days = 1) → 
  (x = 12) :=
begin
  sorry
end

end team_B_independent_days_l768_768753


namespace linear_function_in_quadrants_l768_768870

section LinearFunctionQuadrants

variable (m : ℝ)

def passesThroughQuadrants (m : ℝ) : Prop :=
  (m + 1 > 0) ∧ (m - 1 > 0)

theorem linear_function_in_quadrants (h : passesThroughQuadrants m) : m > 1 :=
by sorry

end LinearFunctionQuadrants

end linear_function_in_quadrants_l768_768870


namespace trucks_have_160_containers_per_truck_l768_768332

noncomputable def containers_per_truck: ℕ :=
  let boxes1 := 7 * 20
  let boxes2 := 5 * 12
  let total_boxes := boxes1 + boxes2
  let total_containers := total_boxes * 8
  let trucks := 10
  total_containers / trucks

theorem trucks_have_160_containers_per_truck:
  containers_per_truck = 160 :=
by
  sorry

end trucks_have_160_containers_per_truck_l768_768332


namespace min_S_n_value_l768_768529

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_n (f : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range (n + 1), f i

noncomputable def b_sequence (a : ℕ → ℤ) (n : ℕ) : ℕ → ℤ :=
  λ n, a n * a (n + 1) * a (n + 2)

noncomputable def S_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  sum_n (b_sequence a) n

theorem min_S_n_value (a : ℕ → ℤ) (h1 : arithmetic_sequence a)
  (h2 : a 1 < 0)
  (h3 : sum_n a 100 = 0) :
  (∃ n, (n = 48 ∨ n = 50) ∧ (∀ m, S_n a m >= S_n a n)) :=
sorry

end min_S_n_value_l768_768529


namespace length_of_ED_l768_768178

variable (AF CE AE Area ED : ℝ)
variable hAF : AF = 30
variable hCE : CE = 40
variable hAE : AE = 120
variable hArea : Area = 7200

theorem length_of_ED : ED = 20 :=
by
  sorry

end length_of_ED_l768_768178


namespace number_of_dispatch_plans_l768_768855

theorem number_of_dispatch_plans :
  (∃ (students : Finset ℕ) (communities : Finset ℕ),
    students.card = 4 ∧ communities.card = 3 ∧
    ∀ (f : students → communities), surjective f) → 
  ∃ (dispatch_plans : ℕ), dispatch_plans = 36 :=
by
  sorry

end number_of_dispatch_plans_l768_768855


namespace projection_a_on_b_l768_768044

-- Define the vectors a and b
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, 1)

-- Function to calculate the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to calculate the norm of a vector
def norm (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Statement of the problem: 
theorem projection_a_on_b : 
  (dot_product a b) / (norm b) = 2 * real.sqrt 5 :=
by
  sorry

end projection_a_on_b_l768_768044


namespace inequality_solution_l768_768726

theorem inequality_solution (x : ℝ) (hx : 0 < x) : 
  (0.4 ^ ((Real.log3 (3 / x)) * (Real.log3 (3 * x))) > 6.25 ^ (Real.log3 (x^2 + 2))) ↔ 
  (x ∈ set.Ioo 0 ((1 : ℝ) / 3) ∪ set.Ioi 243) :=
by {
  sorry
}

end inequality_solution_l768_768726


namespace angle_between_a_c_l768_768997

-- Definitions of the vectors and their properties
variables {ℝ : Type*} [NormedField ℝ] [NormedSpace ℝ (EuclideanSpace ℝ (Fin 3))] {a b c : EuclideanSpace ℝ (Fin 3)}
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1)
variables (h_cross : a × (b × c) = (b - 2 • c) / 2)
variables (h_lin_indep : LinearIndependent ℝ ![a, b, c])

open Real

-- Statement of the problem
theorem angle_between_a_c : angle a c = π / 3 :=
by
  sorry

end angle_between_a_c_l768_768997


namespace volume_of_pyramid_l768_768403

namespace VolumeProof

def isCube (P Q R S T U : ℝ×ℝ×ℝ) (sideLength : ℝ) : Prop :=
  (P.x.toReal = 0 ∧ P.y.toReal = 0 ∧ P.z.toReal = 0) ∧
  (Q.x.toReal = sideLength ∧ Q.y.toReal = 0 ∧ Q.z.toReal = 0) ∧
  (R.x.toReal = sideLength ∧ R.y.toReal = sideLength ∧ R.z.toReal = 0) ∧
  (S.x.toReal = 0 ∧ S.y.toReal = sideLength ∧ S.z.toReal = 0) ∧
  (T.x.toReal = 0 ∧ T.y.toReal = 0 ∧ T.z.toReal = sideLength) ∧
  (U.x.toReal = sideLength ∧ U.y.toReal = 0 ∧ U.z.toReal = sideLength)

theorem volume_of_pyramid (P Q R S T U : ℝ×ℝ×ℝ) (u : ℝ) :
  isCube P Q R S T U (real_root.cbrt u) → u = 8 → 
  let sideLength := real_root.cbrt u in 
  let baseArea := (sideLength ^ 2) / 2 in 
  let height := sideLength in
  (1 / 3) * baseArea * height = 4 / 3 := by
   intro hCube vCube
   sorry

end VolumeProof

end volume_of_pyramid_l768_768403


namespace bouquet_cost_with_50_roses_l768_768450

def base_cost : ℝ := 5
def dozen_roses_cost : ℝ := 25

def variable_cost_per_rose := dozen_roses_cost - base_cost
def variable_cost_for_dozen_roses := variable_cost_per_rose
def cost_per_rose := variable_cost_for_dozen_roses / 12

theorem bouquet_cost_with_50_roses :
  base_cost + 50 * cost_per_rose = 88.33 :=
by
  unfold base_cost dozen_roses_cost variable_cost_per_rose variable_cost_for_dozen_roses cost_per_rose
  sorry

end bouquet_cost_with_50_roses_l768_768450


namespace ternary_1021_to_decimal_l768_768466

-- Define the function to convert a ternary string to decimal
def ternary_to_decimal (n : String) : Nat :=
  n.foldr (fun c acc => acc * 3 + (c.toNat - '0'.toNat)) 0

-- The statement to prove
theorem ternary_1021_to_decimal : ternary_to_decimal "1021" = 34 := by
  sorry

end ternary_1021_to_decimal_l768_768466


namespace find_AB_l768_768189

-- Define the given conditions in the problem
variable {A B C : Type} [EuclideanSpace ℝ (P : Type) [AddCommGroup P] [Module ℝ P]]
variable (AC BC AB : ℝ)

-- Given conditions
def tan_A_eq : Prop := (BC / AC = 3 / 5)
def AC_val : Prop := (AC = 10)
def angle_C_eq : Prop := (right_triangle A B C)

-- Statement of the problem
theorem find_AB (h1 : tan_A_eq BC AC) (h2 : AC_val AC) (h3 : angle_C_eq) : AB = 2 * sqrt 34 :=
  sorry

end find_AB_l768_768189


namespace isabella_hair_length_l768_768608

theorem isabella_hair_length (final_length growth_length initial_length : ℕ) 
  (h1 : final_length = 24) 
  (h2 : growth_length = 6) 
  (h3 : final_length = initial_length + growth_length) : 
  initial_length = 18 :=
by
  sorry

end isabella_hair_length_l768_768608


namespace monotonic_intervals_max_min_values_l768_768043

noncomputable def F (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + (7/3)

theorem monotonic_intervals :
  (∀ x, -1 < x ∧ x < 0 → F x < F (x + 1)) ∧
  (∀ x, 0 < x ∧ x < 4 → F x > F (x + 1)) ∧
  (∀ x, 4 < x → F x < F (x + 1)) :=
sorry

theorem max_min_values :
  ∃ (x_max x_min : ℝ), x_max ∈ set.Icc 1 5 ∧ x_min ∈ set.Icc 1 5 ∧
  (F x_max = 2/3) ∧ (F x_min = -25/3) :=
sorry

end monotonic_intervals_max_min_values_l768_768043


namespace solve_for_s_l768_768814

def F (a b c : ℝ) : ℝ := a * b^c

theorem solve_for_s (s : ℝ) (h : 0 < s) : F(s, s, 2) = 256 ↔ s = 2^(8/3) :=
by
  sorry

end solve_for_s_l768_768814


namespace quadratic_root_k_l768_768163

theorem quadratic_root_k (k : ℝ) : (x = -1) → (x^2 - 2x + k - 1 = 0) → k = -2 :=
by {
  sorry
}

end quadratic_root_k_l768_768163


namespace pool_filling_time_l768_768399

theorem pool_filling_time
  (R : ℝ) 
  (h₁ : 0 < R) -- condition: rate of second pipe is positive
  (h₂ : 1.25 * R * 5 = 1) -- condition: both pipes together fill the pool in 5 hours
  : (1.25 * R)^(-1) = 9 := 
by 
  sorry

end pool_filling_time_l768_768399


namespace AM_MD_ratio_l768_768965

variables (A B C D N M : Point)
variables (AD BC : Line)
hypothesis (h_trap: is_trapezoid A B C D)
hypothesis (h_angle_A: ∠A = 60)
hypothesis (h_angle_D: ∠D = 30)
hypothesis (H_rat: BN / NC  = 2)
hypothesis (H_area: MN.area_divides (ABCD) = 1/2)
hypothesis (h_perp: ⊥ MN AD)
hypothesis (h_perp2: ⊥ MN BC)

theorem AM_MD_ratio : AM / MD = 3 / 4 := 
  sorry

end AM_MD_ratio_l768_768965


namespace count_four_digit_integers_with_3_and_6_l768_768114

theorem count_four_digit_integers_with_3_and_6 : 
  (∃ (count : ℕ), count = 16 ∧ 
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → 
  (∀ i : ℕ, i < 4 → (n / (10 ^ i)) % 10 = 3 ∨ (n / (10 ^ i)) % 10 = 6) ↔ n ∈ {3333, 3366, 3633, 3666, 6333, 6366, 6633, 6666}) :=
by
  have h : 2 ^ 4 = 16 := by norm_num
  use 16
  split
  · exact h
  · sorry

end count_four_digit_integers_with_3_and_6_l768_768114


namespace raspberry_soda_probability_l768_768672

def binomial_probability (n k : ℕ) (p q : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * (q ^ (n - k))

theorem raspberry_soda_probability :
  binomial_probability 7 3 (3 / 4) (1 / 4) = 945 / 16384 :=
sorry

end raspberry_soda_probability_l768_768672


namespace auto_fin_credit_extension_l768_768452

-- Given conditions as definitions
def total_consumer_installment_credit : ℝ := 475 -- in billion dollars
def proportion_automobile_credit : ℝ := 0.36
def proportion_ext_by_auto_fin : ℝ := 1 / 3

-- Define total automobile installment credit
def total_auto_installment_credit : ℝ := proportion_automobile_credit * total_consumer_installment_credit

-- Define credit extended by automobile finance companies
def credit_ext_by_auto_fin : ℝ := proportion_ext_by_auto_fin * total_auto_installment_credit

-- Theorem to prove the required credit extended by automobile finance companies
theorem auto_fin_credit_extension : credit_ext_by_auto_fin = 57 := by
  -- We use 'sorry' here to skip the actual proof
  sorry

end auto_fin_credit_extension_l768_768452


namespace curve_C_equation_angle_condition_T_l768_768534

-- Part I: Circle and ellipse definitions
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9
def curve_C (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1 ∧ x ≠ -2

-- The proof problem for Part I
theorem curve_C_equation :
    ∀ (x y : ℝ), curve_C x y → 
    ((∃ r : ℝ, (x + 1)^2 + y^2 = 1 ∧ (x - 1)^2 + y^2 = 9 
      ∧ r + 1 + (3 - r) = 4) → 
    (x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2)) :=
  sorry

-- Part II: Line intersection and angle condition
def line_y (x : ℝ) (k : ℝ) : ℝ := k * (x - 1)
def point_R (x1 y1 t : ℝ) : Prop := ∃ k : ℝ, curve_C x1 y1 ∧ y1 = k * (x1 - 1)
def point_S (x2 y2 t : ℝ) : Prop := ∃ k : ℝ, curve_C x2 y2 ∧ y2 = k * (x2 - 1)
def angle_condition (OTS OTR : Prop) : Prop := OTS = OTR

-- The proof problem for Part II
theorem angle_condition_T :
    ∃ T : ℝ × ℝ, T.2 = 0 ∧ T.1 = 4 ∧
    (∀ (x1 y1 x2 y2 k : ℝ), 
      (curve_C x1 y1 → curve_C x2 y2 → line_y x1 k = y1 → line_y x2 k = y2 → 
      (∃ t : ℝ, ∠OTR = ∠OTS) → 
      angle_condition (angle O (T, 0) S) (angle O (T, 0) R))) :=
  sorry

end curve_C_equation_angle_condition_T_l768_768534


namespace solve_trig_problem_l768_768051

noncomputable def trig_problem (α : ℝ) : Prop :=
  α ∈ (Set.Ioo 0 (Real.pi / 2)) ∪ Set.Ioo (Real.pi / 2) Real.pi ∧
  ∃ r : ℝ, r ≠ 0 ∧ Real.sin α * r = Real.sin (2 * α) ∧ Real.sin (2 * α) * r = Real.sin (4 * α)

theorem solve_trig_problem (α : ℝ) (h : trig_problem α) : α = 2 * Real.pi / 3 :=
by
  sorry

end solve_trig_problem_l768_768051


namespace three_digit_numbers_with_2_or_8_l768_768144

theorem three_digit_numbers_with_2_or_8 : 
  (∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
    (∃ d, (d = 8 ∨ d = 2) ∧ (d ∈ list.digits n))) →
  ∑ n in (finset.range (999 + 1)).filter (λ n, 100 ≤ n ∧ (∃ d, (d = 8 ∨ d = 2) ∧ (d ∈ list.digits n))), 1 = 452 :=
by sorry

end three_digit_numbers_with_2_or_8_l768_768144


namespace azambuja_part_a_azambuja_part_b_l768_768787

def operations_allowed (q : ℚ) : Prop :=
  q + 1 = q ∨ q - 1 = q ∨ (q ≠ 1/2 ∧ (q - 1)/(2*q - 1) = q)

theorem azambuja_part_a : ¬ ∃ (steps : ℕ → ℚ), steps 0 = 0 ∧
  ∀ n, operations_allowed (steps n) → steps (n+1) = steps n ∧
  steps (n+1) = 1/2018 :=
sorry

theorem azambuja_part_b : ∃ (initial_q : ℚ), initial_q = 1/2018 ∧ 
  (gcd initial_q.num initial_q.den = 1 ∧ ¬ (odd initial_q.num ∧ odd initial_q.den)) :=
sorry

end azambuja_part_a_azambuja_part_b_l768_768787


namespace soda_cost_l768_768213

variable (b s f : ℝ)

noncomputable def keegan_equation : Prop :=
  3 * b + 2 * s + f = 975

noncomputable def alex_equation : Prop :=
  2 * b + 3 * s + f = 900

theorem soda_cost (h1 : keegan_equation b s f) (h2 : alex_equation b s f) : s = 18.75 :=
by
  sorry

end soda_cost_l768_768213


namespace ball_hits_ground_in_2_72_seconds_l768_768786

noncomputable def height_at_time (t : ℝ) : ℝ :=
  -16 * t^2 - 30 * t + 200

theorem ball_hits_ground_in_2_72_seconds :
  ∃ t : ℝ, t = 2.72 ∧ height_at_time t = 0 :=
by
  use 2.72
  sorry

end ball_hits_ground_in_2_72_seconds_l768_768786


namespace smallest_four_digit_divisible_by_34_l768_768385

/-- Define a four-digit number. -/
def is_four_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000

/-- Define a number to be divisible by another number. -/
def divisible_by (n k : ℕ) : Prop :=
  k ∣ n

/-- Prove that the smallest four-digit number divisible by 34 is 1020. -/
theorem smallest_four_digit_divisible_by_34 : ∃ n : ℕ, is_four_digit n ∧ divisible_by n 34 ∧ 
    (∀ m : ℕ, is_four_digit m → divisible_by m 34 → n ≤ m) :=
  sorry

end smallest_four_digit_divisible_by_34_l768_768385


namespace prime_pair_probability_even_sum_l768_768166

open Finset

-- Conditions given in the problem
def firstEightPrimes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Main statement of the problem to prove
theorem prime_pair_probability_even_sum : 
  let total_pairs := (firstEightPrimes.choose 2).card in
  let odd_pairs := (firstEightPrimes.filter (λ x, x ≠ 2)).card in
  total_pairs > 0 → 
  (total_pairs - odd_pairs) / total_pairs = 3 / 4 :=
by
  intros total_pairs odd_pairs h
  sorry

end prime_pair_probability_even_sum_l768_768166


namespace initial_alloy_weight_l768_768780

theorem initial_alloy_weight
  (x : ℝ)  -- Weight of the initial alloy in ounces
  (h1 : 0.80 * (x + 24) = 0.50 * x + 24)  -- Equation derived from conditions
: x = 16 := 
sorry

end initial_alloy_weight_l768_768780


namespace find_radius_of_circle_l768_768056

theorem find_radius_of_circle
  (a b R : ℝ)
  (h1 : R^2 = a * b) :
  R = Real.sqrt (a * b) :=
by
  sorry

end find_radius_of_circle_l768_768056


namespace max_distinct_numbers_in_table_l768_768183

open Nat

theorem max_distinct_numbers_in_table : 
  ∀ (M : Matrix (Fin 75) (Fin 75) ℕ), 
  (∀ i : Fin 75, (Finset.card (Finset.image (λ j => M i j) Finset.univ)) ≥ 15) →
  (∀ i : Fin 73, (Finset.card (Finset.bUnion (Finset.Icc i (i+2 : Fin 75)) (λ k => Finset.image (λ j => M k j) Finset.univ)) ≤ 25) →
  ∃ N : ℕ, 
  Finset.card (Finset.bUnion Finset.univ (λ i => Finset.image (λ j => M i j) Finset.univ)) = 385 :=
by
  sorry

end max_distinct_numbers_in_table_l768_768183


namespace find_sixth_number_l768_768731

theorem find_sixth_number 
  (A : ℕ → ℝ)
  (h1 : ((A 1 + A 2 + A 3 + A 4 + A 5 + A 6 + A 7 + A 8 + A 9 + A 10 + A 11) / 11 = 60))
  (h2 : ((A 1 + A 2 + A 3 + A 4 + A 5 + A 6) / 6 = 58))
  (h3 : ((A 6 + A 7 + A 8 + A 9 + A 10 + A 11) / 6 = 65)) 
  : A 6 = 78 :=
by
  sorry

end find_sixth_number_l768_768731


namespace total_puppies_count_l768_768390

def first_week_puppies : Nat := 20
def second_week_puppies : Nat := 2 * first_week_puppies / 5
def third_week_puppies : Nat := 3 * second_week_puppies / 8
def fourth_week_puppies : Nat := 2 * second_week_puppies
def fifth_week_puppies : Nat := first_week_puppies + 10
def sixth_week_puppies : Nat := 2 * third_week_puppies - 5
def seventh_week_puppies : Nat := 2 * sixth_week_puppies
def eighth_week_puppies : Nat := 5 * seventh_week_puppies / 6 / 1 -- Assuming rounding down to nearest whole number

def total_puppies : Nat :=
  first_week_puppies + second_week_puppies + third_week_puppies +
  fourth_week_puppies + fifth_week_puppies + sixth_week_puppies +
  seventh_week_puppies + eighth_week_puppies

theorem total_puppies_count : total_puppies = 81 := by
  sorry

end total_puppies_count_l768_768390


namespace smallest_positive_integer_in_form_l768_768714

theorem smallest_positive_integer_in_form :
  ∃ (m n p : ℤ), 1234 * m + 56789 * n + 345 * p = 1 := sorry

end smallest_positive_integer_in_form_l768_768714


namespace no_such_function_exists_l768_768474

theorem no_such_function_exists :
  ¬(∃ f : ℝ → ℝ, (∀ x : ℝ, 2 * f (|cos x|) = f (|sin x|) + |sin x|) ∧ (∀ t, t ∈ Icc (-1:ℝ) (1:ℝ) → f t = f t)) :=
by
  sorry

end no_such_function_exists_l768_768474


namespace sufficient_not_necessary_l768_768445

variable (a b : ℝ)

theorem sufficient_not_necessary (h : a > b + 1) : a > b :=
by {
  exact lt_trans (by linarith) (by linarith),
}

end sufficient_not_necessary_l768_768445


namespace dodecahedron_edge_probability_l768_768348

def dodecahedron_vertices : ℕ := 20

def vertex_connections : ℕ := 3

theorem dodecahedron_edge_probability :
  (∃ (u v : fin dodecahedron_vertices), u ≠ v ∧ u.1 < vertex_connections → 
    (Pr (u, v) = 3 / (dodecahedron_vertices - 1))) :=
sorry

end dodecahedron_edge_probability_l768_768348


namespace negation_equiv_l768_768290

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, cos x > sin x - 1

-- Define the negation of the proposition p
def neg_p : Prop := ∃ x : ℝ, cos x ≤ sin x - 1

-- The theorem stating the equivalence between the negation of p and neg_p
theorem negation_equiv : ¬ p ↔ neg_p :=
by
  sorry

end negation_equiv_l768_768290


namespace misha_smartphone_original_price_l768_768982

theorem misha_smartphone_original_price :
  ∃ (a b c d : ℕ), (a ≠ 0) ∧ (d ≠ 0) ∧ (∀ n : ℕ, n = 1000 * a + 100 * b + 10 * c + d → 
    1.2 * n = 1000 * d + 100 * c + 10 * b + a) ∧ 
    (1000 * a + 100 * b + 10 * c + d = 4995) := 
sorry

end misha_smartphone_original_price_l768_768982


namespace bananas_per_friend_l768_768611

theorem bananas_per_friend (total_bananas : ℤ) (total_friends : ℤ) (H1 : total_bananas = 21) (H2 : total_friends = 3) : 
  total_bananas / total_friends = 7 :=
by
  sorry

end bananas_per_friend_l768_768611


namespace tangent_line_at_x_1_increasing_implies_a_log_n_gt_sum_of_reciprocals_l768_768901

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x

-- Given the conditions
variable (a : ℝ) (h_a_pos : 0 < a)

-- Part (1) tangent line when a = 1
theorem tangent_line_at_x_1 : 
  f 1 1 = 0 := sorry

-- Part (2) if f is increasing on [2, +∞), then a ≥ 1/2
theorem increasing_implies_a : 
  (∀ x ≥ 2, 0 ≤ (x - 1) / (x^2)) → a ≥ 1 / 2 := 
  sorry

-- Part (3) for any positive integer n > 1, ln n > 1/2 + 1/3 + ... + 1/n
theorem log_n_gt_sum_of_reciprocals (n : ℕ) (h_n_gt_1 : 1 < n) :
  Real.log n > ∑ i in Finset.range (n + 1) \ {0, 1}, (1 : ℝ) / i  := 
  sorry

end tangent_line_at_x_1_increasing_implies_a_log_n_gt_sum_of_reciprocals_l768_768901


namespace fish_population_after_three_months_l768_768176

theorem fish_population_after_three_months 
  (initial_tagged : ℕ)   -- 1000 initial tagged fish
  (caught_second_time : ℕ) -- 500 fish were caught again
  (tagged_second_time : ℕ) -- 7 fish were found tagged in the second catch
  (migration_rate : ℕ) -- 150 new fish migrate per month
  (death_rate : ℕ) -- 200 fish die per month
  (months : ℕ)    -- 3 months
  (initial_population : ℝ) -- Initial number of fish in the reserve
  (reserve_ratio : ℝ) -- Ratio of tagged fish in reserve
  (approx_number : ℝ) -- Approximate number of fish in reserve after three months
  : initial_tagged = 1000 → 
    caught_second_time = 500 → 
    tagged_second_time = 7 → 
    migration_rate = 150 → 
    death_rate = 200 → 
    months = 3 → 
    reserve_ratio = (tagged_second_time / caught_second_time : ℝ) → 
    approx_number = initial_population - 3 * (death_rate - migration_rate) → 
    initial_population / (initial_population - 150 : ℝ) = 1000 / (initial_population - 150 : ℝ) →
    approx_number ≈ 71429 := 
by 
  intros 
    h1 h2 h3 h4 h5 h6 h7 h8 h9 
  have h10 : initial_population / (initial_population - 150) = 1000 / (initial_population - 150) := by assumption
  have h11 : initial_population = 71578.57 := sorry 
  exact h8

end fish_population_after_three_months_l768_768176


namespace ordered_pair_exists_l768_768151

noncomputable def find_p_q (p q : ℚ) : Prop :=
  let f := λ x : ℚ, x^5 - x^4 + x^3 - p * x^2 + q * x - 8 in
  f (-3) = 0 ∧ f 2 = 0

theorem ordered_pair_exists : find_p_q (-67/3) (-158/3) :=
by {
  sorry
}

end ordered_pair_exists_l768_768151


namespace geometric_series_sum_l768_768001

theorem geometric_series_sum :
  let a := 2
  let r := -2
  let n := 10
  let Sn := (a : ℚ) * (r^n - 1) / (r - 1)
  Sn = 2050 / 3 :=
by
  sorry

end geometric_series_sum_l768_768001


namespace complex_fraction_value_l768_768893

theorem complex_fraction_value (a b : ℝ) (h : (i - 2) / (1 + i) = a + b * i) : a + b = 1 :=
by
  sorry

end complex_fraction_value_l768_768893


namespace sin_value_l768_768539

theorem sin_value (x : ℝ) (h : (sec x + tan x) = 5 / 4) : sin x = 9 / 41 :=
sorry

end sin_value_l768_768539


namespace num_four_digit_pos_integers_l768_768133

theorem num_four_digit_pos_integers : 
  ∃ n : ℕ, (n = 16) ∧ ∀ k : ℕ, (1000 ≤ k ∧ k < 10000 ∧ 
  ∀ d ∈ [k.digits 10], d = 3 ∨ d = 6) := sorry

end num_four_digit_pos_integers_l768_768133


namespace total_pears_picked_l768_768777

def pears_Alyssa : ℕ := 42
def pears_Nancy : ℕ := 17

theorem total_pears_picked : pears_Alyssa + pears_Nancy = 59 :=
by sorry

end total_pears_picked_l768_768777


namespace min_value_of_function_l768_768050

theorem min_value_of_function (x : ℝ) (h : x > 5 / 4) : 
  ∃ ymin : ℝ, ymin = 7 ∧ ∀ y : ℝ, y = 4 * x + 1 / (4 * x - 5) → y ≥ ymin := 
sorry

end min_value_of_function_l768_768050


namespace temperature_decrease_representation_l768_768945

def temperature_increase (rise : ℤ) : ℤ := rise

def temperature_decrease (fall : ℤ) : ℤ := -fall

theorem temperature_decrease_representation :
  (temperature_increase 6 = 6) →
  temperature_decrease 2 = -2 :=
by
  assume h1 : temperature_increase 6 = 6
  sorry

end temperature_decrease_representation_l768_768945


namespace num_true_propositions_l768_768877

variables {ℝ : Type*} [real ℝ]

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry
def g' : ℝ → ℝ := sorry

-- Conditions
axiom h_deriv : ∀ x : ℝ, has_deriv_at g (g' x) x
axiom h_even : ∀ x : ℝ, g(x) = g(-x)
axiom h_eq1 : ∀ x : ℝ, f(x) + g'(x) = 10
axiom h_eq2 : ∀ x : ℝ, f(x) - g'(4 - x) = 10

-- Propositions
def p1 := f(1) + f(3) = 20
def p2 := f(4) = 10
def p3 := f(-1) = f(-3)
def p4 := f(2022) = 10

-- The number of true propositions
theorem num_true_propositions : (p1 ∧ p2 ∧ p4) ∧ ¬ p3 := sorry

end num_true_propositions_l768_768877


namespace shortest_side_of_right_triangle_l768_768769

theorem shortest_side_of_right_triangle (a b : ℝ) (h : a = 9 ∧ b = 12) : ∃ c : ℝ, (c = min a b) ∧ c = 9 :=
by
  sorry

end shortest_side_of_right_triangle_l768_768769


namespace exists_untouched_edge_l768_768438

noncomputable def problem_statement (n : ℕ) : Prop :=
  let vertices := List.range (2 * n)
  let edges := vertices.zip (vertices.tail ++ [vertices.head])
  ∀ (moves : List (ℕ × ℕ)),
    (∀ edge ∈ edges, edge ∈ moves) →
    (∀ (i j : ℕ), i < j → (i, j) ∈ moves → (j, i) ∉ moves) →
    (∀ token ∈ vertices, token ∈ (moves.map fst) ∧ token ∈ (moves.map snd)) →
    (∃ edge ∈ edges, edge ∉ moves)

theorem exists_untouched_edge (n : ℕ) : problem_statement n :=
  sorry

end exists_untouched_edge_l768_768438


namespace sum_of_naturals_leq_5_l768_768034

theorem sum_of_naturals_leq_5 : (∑ i in Finset.range 6, i) = 15 := by
  sorry

end sum_of_naturals_leq_5_l768_768034


namespace plywood_perimeter_difference_l768_768744

theorem plywood_perimeter_difference :
  ∃ (p_max p_min : ℕ),
  let plywood_width := 6,
      plywood_height := 9,
      n := 3,
      possible_dimensions := [(6, 3), (2, 9)],
      perimeters := possible_dimensions.map (λ dim, dim.1 + dim.1 + dim.2 + dim.2) in
  p_max = perimeters.maximum ∧
  p_min = perimeters.minimum ∧
  p_max - p_min = 4 :=
sorry

end plywood_perimeter_difference_l768_768744


namespace range_of_log_transformed_l768_768315

theorem range_of_log_transformed:
  (∀ x : ℝ, x > 9 -> ∃ y : ℝ, (y = 1 + log 3 x) ∧ y > 3) ∧ 
  (∀ y : ℝ, y > 3 -> ∃ x : ℝ, x > 9 ∧ y = 1 + log 3 x): 
  sorry

end range_of_log_transformed_l768_768315


namespace range_of_a_l768_768881

variable (m a : ℝ)

def proposition_p : Prop := m^2 - 7 * a * m + 12 * a^2 < 0
def proposition_q : Prop := (1 < m ∧ m < 3 / 2)
def sufficient_condition : Prop := ¬proposition_q → ¬proposition_p

theorem range_of_a (h1 : 0 < a) (h2 : sufficient_condition) : 
  (1 / 3 ≤ a) ∧ (a ≤ 3 / 8) := sorry

end range_of_a_l768_768881


namespace pole_not_perpendicular_l768_768763

theorem pole_not_perpendicular
  (height : ℝ) (dist_ground : ℝ) (cable_length : ℝ) 
  (h1 : height = 1.4) (h2 : dist_ground = 2) (h3 : cable_length = 2.5) : 
  height * height + dist_ground * dist_ground ≠ cable_length * cable_length :=
by 
  rw [h1, h2, h3]
  -- 1.4^2 + 2^2 = 1.96 + 4 = 5.96
  -- 2.5^2 = 6.25
  norm_num
  -- 5.96 ≠ 6.25
  exact ne_of_lt (by norm_num)

end pole_not_perpendicular_l768_768763


namespace percentage_seeds_from_dandelions_l768_768459

def Carla_sunflowers := 6
def Carla_dandelions := 8
def seeds_per_sunflower := 9
def seeds_per_dandelion := 12

theorem percentage_seeds_from_dandelions :
  96 / 150 * 100 = 64 := by
  sorry

end percentage_seeds_from_dandelions_l768_768459


namespace find_g3_l768_768222

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g3 (g_condition: ∀ x : ℝ, x ≠ 0 → g(x) - 3 * g(1 / x) = 3^x) : 
  g(3) = -3.5 :=
by
  sorry

end find_g3_l768_768222


namespace arithmetic_sequence_sum_eq_l768_768182

theorem arithmetic_sequence_sum_eq {
  -- Conditions
  (a : ℕ → ℤ) -- a is a function that gives the terms of the sequence
  (d : ℤ) (h₀ : d ≠ 0)
  (h₁ : ∀ n, a n = (n - 1) * d) (h₂ : a 1 = 0)
  (h₃ : (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = a k)
  -- Goal
  (k : ℕ) : 
  k = 22 :=
sorry

end arithmetic_sequence_sum_eq_l768_768182


namespace speedster_convertibles_l768_768411

noncomputable def total_inventory (not_speedsters : Nat) (fraction_not_speedsters : ℝ) : ℝ :=
  (not_speedsters : ℝ) / fraction_not_speedsters

noncomputable def number_speedsters (total_inventory : ℝ) (fraction_speedsters : ℝ) : ℝ :=
  total_inventory * fraction_speedsters

noncomputable def number_convertibles (number_speedsters : ℝ) (fraction_convertibles : ℝ) : ℝ :=
  number_speedsters * fraction_convertibles

theorem speedster_convertibles : (not_speedsters = 30) ∧ (fraction_not_speedsters = 2 / 3) ∧ (fraction_speedsters = 1 / 3) ∧ (fraction_convertibles = 4 / 5) →
  number_convertibles (number_speedsters (total_inventory not_speedsters fraction_not_speedsters) fraction_speedsters) fraction_convertibles = 12 :=
by
  intros h
  sorry

end speedster_convertibles_l768_768411


namespace circle_numbers_problem_l768_768406

theorem circle_numbers_problem :
  ∃ (numbers : ℕ → ℚ), (∀ i, numbers i = |numbers (i + 1 % 30) - numbers (i + 2 % 30)|) ∧ (finset.sum (finset.range 30) numbers = 1) ∧ (∀ i : fin 30, numbers i = (if i % 3 == 0 then 1/20 else if i % 3 == 1 then 1/20 else 0)) :=
sorry

end circle_numbers_problem_l768_768406


namespace new_average_score_l768_768675

theorem new_average_score (n : ℕ) (initial_avg : ℕ) (grace_marks : ℕ) (h1 : n = 35) (h2 : initial_avg = 37) (h3 : grace_marks = 3) : initial_avg + grace_marks = 40 := by
  sorry

end new_average_score_l768_768675


namespace prove_propositions_count_l768_768876

def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := sorry

-- Given conditions
axiom h1 : ∀ x, f(x) + (deriv g)(x) = 10
axiom h2 : ∀ x, f(x) - (deriv g)(4-x) = 10
axiom h3 : ∀ x, g(-x) = g(x)

-- Prove the conclusion
theorem prove_propositions_count :
  (f(1) + f(3) = 20 ∧ f(4) = 10 ∧ f(2022) = 10)
  → f(-1) = f(-3)
  → 3 = 3 :=
sorry

end prove_propositions_count_l768_768876


namespace correct_statement_l768_768442

noncomputable def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  let rec b_aux (m : ℕ) :=
    match m with
    | 0     => 0
    | m + 1 => 1 + 1 / (α m + b_aux m)
  b_aux n

theorem correct_statement (α : ℕ → ℕ) (h : ∀ k, α k > 0) : b 4 α < b 7 α :=
by sorry

end correct_statement_l768_768442


namespace count_four_digit_integers_with_3_and_6_l768_768110

theorem count_four_digit_integers_with_3_and_6 : 
  (∃ (count : ℕ), count = 16 ∧ 
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → 
  (∀ i : ℕ, i < 4 → (n / (10 ^ i)) % 10 = 3 ∨ (n / (10 ^ i)) % 10 = 6) ↔ n ∈ {3333, 3366, 3633, 3666, 6333, 6366, 6633, 6666}) :=
by
  have h : 2 ^ 4 = 16 := by norm_num
  use 16
  split
  · exact h
  · sorry

end count_four_digit_integers_with_3_and_6_l768_768110


namespace polar_equation_of_C_length_of_AB_l768_768964

-- Definitions from the conditions
def line_parametric_eq (α t : ℝ) : ℝ × ℝ :=
  (2 + t * cos α, sqrt 3 + t * sin α)

def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * cos θ, sin θ)

def polar_eq_of_C (ρ θ : ℝ) : Prop :=
  ρ^2 = 4 / (1 + 3 * sin θ ^ 2)

def line_segment_length (t1 t2 : ℝ) : ℝ :=
  abs (t1 - t2)

-- Theorem Statements
theorem polar_equation_of_C (ρ θ : ℝ) :
  polar_eq_of_C ρ θ :=
sorry

theorem length_of_AB (α : ℝ) (t1 t2 : ℝ) (hα: α = π / 3)
  (ht1t2: 13 * t1^2 + 56 * t1 + 48 = 0 ∧ 13 * t2^2 + 56 * t2 + 48 = 0) :
  line_segment_length t1 t2 = 8 * sqrt 10 / 13 :=
sorry

end polar_equation_of_C_length_of_AB_l768_768964


namespace train_speed_kmph_l768_768439

-- The conditions
def speed_m_s : ℝ := 52.5042
def conversion_factor : ℝ := 3.6

-- The theorem we need to prove
theorem train_speed_kmph : speed_m_s * conversion_factor = 189.01512 := 
  sorry

end train_speed_kmph_l768_768439


namespace arithmetic_sequence_specific_values_l768_768060

variable {a : ℕ → ℤ} -- define the arithmetic sequence as a function from natural numbers to integers

-- Given conditions
axiom a_3 : a 3 = 5
axiom a_5 : a 5 = 3

-- Define the common difference and general term formula for the arithmetic sequence
def d := a 3 - a 2
def a_n (n : ℕ) : ℤ := a 1 + (n - 1) * d

-- Define the sum of the first n terms
def S_n (n : ℕ) : ℤ := n * (a 1 + a n) / 2

-- Prove the specific values for a_n and S_7
theorem arithmetic_sequence_specific_values : (a_n n = 8 - n) ∧ (S_n 7 = 28) :=
by
  sorry

end arithmetic_sequence_specific_values_l768_768060


namespace dodecahedron_edge_probability_l768_768347

def dodecahedron_vertices : ℕ := 20

def vertex_connections : ℕ := 3

theorem dodecahedron_edge_probability :
  (∃ (u v : fin dodecahedron_vertices), u ≠ v ∧ u.1 < vertex_connections → 
    (Pr (u, v) = 3 / (dodecahedron_vertices - 1))) :=
sorry

end dodecahedron_edge_probability_l768_768347


namespace nested_sqrt_eval_l768_768823

theorem nested_sqrt_eval : ∀ x : ℝ, x = Real.sqrt (24 + x) → x = 6 :=
begin
  sorry
end

end nested_sqrt_eval_l768_768823


namespace square_wiring_insufficient_for_11_meters_l768_768203

theorem square_wiring_insufficient_for_11_meters (a b c d : Point)
  (h_ab : dist a b = 4)
  (h_bc : dist b c = 4)
  (h_cd : dist c d = 4)
  (h_da : dist d a = 4) :
  ¬∃ (G : Graph), G.connected ∧ G.spanning [a, b, c, d] ∧ G.total_edge_length ≤ 11 := sorry

end square_wiring_insufficient_for_11_meters_l768_768203


namespace find_m_l768_768619

-- Definition of vectors in terms of the condition
def vec_a (m : ℝ) : ℝ × ℝ := (2 * m + 1, m)
def vec_b (m : ℝ) : ℝ × ℝ := (1, m)

-- Condition that vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2) = 0

-- Problem statement: find m such that vec_a is perpendicular to vec_b
theorem find_m (m : ℝ) (h : perpendicular (vec_a m) (vec_b m)) : m = -1 := by
  sorry

end find_m_l768_768619


namespace retailer_paid_market_price_for_40_pens_l768_768765

/-- 
Given:
1. A retailer buys 40 pens at the market price.
2. The retailer sells these pens giving a discount of 1%.
3. The retailer's profit is 9.999999999999996%.
Prove: The retailer paid the market price for 40 pens.
-/
theorem retailer_paid_market_price_for_40_pens :
  ∃ (pens_bought : ℕ) (discount : ℝ) (profit_percentage : ℝ), 
    pens_bought = 40 ∧ 
    discount = 0.01 ∧ 
    profit_percentage = 9.999999999999996 / 100 ∧ 
    pens_bought = 40 :=
by 
  use 40
  use 0.01
  use 9.999999999999996 / 100
  simp
  sorry

end retailer_paid_market_price_for_40_pens_l768_768765


namespace num_four_digit_integers_with_3_and_6_l768_768105

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end num_four_digit_integers_with_3_and_6_l768_768105


namespace Jordan_total_listens_by_end_of_year_l768_768976

theorem Jordan_total_listens_by_end_of_year :
  ∀ (initial_listens m : ℕ), initial_listens = 60000 → m = 3 →
  (∀ i : ℕ, i < m → ∀ current_listens : ℕ, current_listens = (2^i) * initial_listens → 
  ∑ i in finset.range(m), (2^i * initial_listens)) + initial_listens = 900000 :=
by sorry

end Jordan_total_listens_by_end_of_year_l768_768976


namespace problem_l768_768519

theorem problem (x : ℝ) (h : x + x⁻¹ = 5) : x^2 + x⁻² = 23 :=
sorry

end problem_l768_768519


namespace trig_identity_l768_768071

theorem trig_identity (θ : ℝ)
  (h1 : ∃ k : ℤ, θ = 2 * k * π + arctan (3))
  (h2 : 0 ≤ θ ∧ θ < 2 * π ∨ π ≤ θ ∧ θ < 3 * π) :
  sin (2 * θ + π / 3) = (3 - 4 * sqrt 3) / 10 := by
  sorry

end trig_identity_l768_768071


namespace geom_seq_product_l768_768197

-- Given conditions
variables (a : ℕ → ℝ)
variable (r : ℝ)
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom a1_eq_1 : a 1 = 1
axiom a10_eq_3 : a 10 = 3

-- Proof goal
theorem geom_seq_product : a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 :=  
sorry

end geom_seq_product_l768_768197


namespace triangle_AFC_perimeter_l768_768600

-- Definitions for the geometry problem
structure Triangle where
  A B C : Point
  BC : ℝ
  AC : ℝ

structure Midpoint (D : Point) (A B : Point) : Prop :=
  (midpoint_def : dist A D = dist B D)

structure Perpendicular (DF : Line) (AB : Line) : Prop :=
  (perpendicular_def : angle DF AB = π / 2)

structure Intersection (DF : Line) (BC : Line) (F : Point) :=
  (intersection_def : F ∈ DF ∩ BC)

-- Assumption: We have triangle ABC with given conditions
def triangle_ABC : Triangle := {
  A := A, B := B, C := C, BC := 18, AC := 9
}

-- Midpoint and geometric properties
def D : Point := midpoint A B
instance : Midpoint D A B := {
  midpoint_def := sorry -- Proof/definition omitted for brevity
}

def DF : Line := perpendicular_line D A B
instance : Perpendicular DF (line A B) := {
  perpendicular_def := sorry -- Proof/definition omitted for brevity
}

def F : Point := intersection_point DF (line B C)
instance : Intersection DF (line B C) F := {
  intersection_def := sorry -- Proof/definition omitted for brevity
}

-- Proof statement
theorem triangle_AFC_perimeter 
  (h1 : D = midpoint A B)
  (h2 : DF ⊥ line A B)
  (h3 : F ∈ line B C)
  (h4 : triangle_ABC.BC = 18)
  (h5 : triangle_ABC.AC = 9) : 
  perimeter (triangle A F C) = 27 := sorry

end triangle_AFC_perimeter_l768_768600


namespace solve_for_x_l768_768844

theorem solve_for_x (x : ℝ) (h : sqrt (x + 3) = 7) : x = 46 :=
by {
  -- proof will be here
  sorry
}

end solve_for_x_l768_768844


namespace AM_GM_inequality_l768_768091

theorem AM_GM_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3 : ℝ) :=
begin
  sorry
end

end AM_GM_inequality_l768_768091


namespace six_point_circle_product_one_l768_768413

theorem six_point_circle_product_one
  (A B C P Q R S T U : Type)
  (circle : Type)
  (intersects_BC : circle ∩ (line_segment B C) = {P, Q})
  (intersects_CA : circle ∩ (line_segment C A) = {R, S})
  (intersects_AB : circle ∩ (line_segment A B) = {T, U}) :
  (dist A T / dist T B) * (dist A U / dist U B) * (dist B P / dist P C) * (dist B Q / dist Q C) * (dist C R / dist R A) * (dist C S / dist S A) = 1 := 
sorry

end six_point_circle_product_one_l768_768413


namespace total_possible_rankings_l768_768821

-- Define the players
inductive Player
| P | Q | R | S

-- Define the tournament results
inductive Result
| win | lose

-- Define Saturday's match outcomes
structure SaturdayOutcome :=
(P_vs_Q: Result)
(R_vs_S: Result)

-- Function to compute the number of possible tournament ranking sequences
noncomputable def countTournamentSequences : Nat :=
  let saturdayOutcomes: List SaturdayOutcome :=
    [ {P_vs_Q := Result.win, R_vs_S := Result.win}
    , {P_vs_Q := Result.win, R_vs_S := Result.lose}
    , {P_vs_Q := Result.lose, R_vs_S := Result.win}
    , {P_vs_Q := Result.lose, R_vs_S := Result.lose}
    ]
  let sundayPermutations (outcome : SaturdayOutcome) : Nat :=
    2 * 2  -- 2 permutations for 1st and 2nd places * 2 permutations for 3rd and 4th places per each outcome
  saturdayOutcomes.foldl (fun acc outcome => acc + sundayPermutations outcome) 0

-- Define the theorem to prove the total number of permutations
theorem total_possible_rankings : countTournamentSequences = 8 :=
by
  -- Proof steps here (proof omitted)
  sorry

end total_possible_rankings_l768_768821


namespace points_lie_on_parabola_l768_768850

def curve (u : ℝ) : ℝ × ℝ :=
  let x := 3^u - 2
  let y := 9^u - 6 * 3^u + 4
  (x, y)

theorem points_lie_on_parabola : ∀ u : ℝ, ∃ a b c : ℝ, (curve u).snd = a * (curve u).fst ^ 2 + b * (curve u).fst + c :=
by
  intro u
  use 1, -2, 4
  sorry

end points_lie_on_parabola_l768_768850


namespace range_of_function_l768_768689

noncomputable def function_range (x : ℝ) : ℝ :=
    (1 / 2) ^ (-x^2 + 2 * x)

theorem range_of_function : 
    (Set.range function_range) = Set.Ici (1 / 2) :=
by
    sorry

end range_of_function_l768_768689


namespace root_one_value_of_m_real_roots_range_of_m_l768_768524

variables {m x : ℝ}

-- Part 1: Prove that if 1 is a root of 'mx^2 - 4x + 1 = 0', then m = 3
theorem root_one_value_of_m (h : m * 1^2 - 4 * 1 + 1 = 0) : m = 3 :=
  by sorry

-- Part 2: Prove that 'mx^2 - 4x + 1 = 0' has real roots iff 'm ≤ 4 ∧ m ≠ 0'
theorem real_roots_range_of_m : (∃ x : ℝ, m * x^2 - 4 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 0) :=
  by sorry

end root_one_value_of_m_real_roots_range_of_m_l768_768524


namespace part_I_part_II_l768_768577

variables {A B C a b c : ℝ} (S : ℝ)

axiom triangle_conditions :
  c * sin B = sqrt 3 * b * cos C ∧
  a^2 - c^2 = 2 * b^2 ∧
  S = 21 * sqrt 3 ∧
  C = π / 3 ∧
  S = 1/2 * a * b * sin C

theorem part_I : C = π / 3 :=
by
  apply triangle_conditions
  exact and.left (triangle_conditions S)

theorem part_II : b = 2 * sqrt 7 :=
by
  apply triangle_conditions
  -- Rest of the proof
  sorry

end part_I_part_II_l768_768577


namespace hours_per_day_l768_768448

-- Conditions
def days_worked : ℝ := 3
def total_hours_worked : ℝ := 7.5

-- Theorem to prove the number of hours worked each day
theorem hours_per_day : total_hours_worked / days_worked = 2.5 :=
by
  sorry

end hours_per_day_l768_768448


namespace total_cost_of_chairs_l768_768789

theorem total_cost_of_chairs (original_price : ℝ) (discount_rate : ℝ) (extra_discount_rate : ℝ) (threshold : ℕ) (num_chairs : ℕ) (total_cost : ℝ) :
  original_price = 20 → 
  discount_rate = 0.25 → 
  extra_discount_rate = 1/3 → 
  threshold = 5 → 
  num_chairs = 8 →
  total_cost = 105 :=
by 
  intros h1 h2 h3 h4 h5 
  have discounted_price : ℝ := original_price * (1 - discount_rate),
  have initial_total : ℝ := discounted_price * num_chairs,
  have extra_discount : ℝ := if num_chairs > threshold then (num_chairs - threshold) * (discounted_price * extra_discount_rate) else 0,
  have final_total : ℝ := initial_total - extra_discount,
  rw [h1, h2, h4, h5] at *,
  rw final_total,
  have base_discount_price : ℝ := 20 * 0.75,
  have base_total : ℝ := base_discount_price * 8,
  have extra_price : ℝ := if 8 > 5 then (8 - 5) * base_discount_price / 3 else 0,
  have total : ℝ := base_total - extra_price,
  simp, norm_num at total,
  exact total

end total_cost_of_chairs_l768_768789


namespace sum_bases_exponents_max_product_l768_768215

theorem sum_bases_exponents_max_product (A : ℕ) (hA : A = 3 ^ 670 * 2 ^ 2) : 
    (3 + 2 + 670 + 2 = 677) := by
  sorry

end sum_bases_exponents_max_product_l768_768215


namespace Q_nonneg_l768_768626

noncomputable def P : Polynomial ℝ := sorry
def n : ℕ := sorry
def Q : Polynomial ℝ := ∑ i in Finset.range (n + 1), P.derivative^[i]

theorem Q_nonneg (x : ℝ) (hP : ∀ x : ℝ, 0 ≤ P.eval x) :
  0 ≤ Q.eval x :=
sorry

end Q_nonneg_l768_768626


namespace arithmetic_mean_of_integers_from_neg6_to_7_l768_768708

theorem arithmetic_mean_of_integers_from_neg6_to_7 : 
  (list.sum (list.range (7 + 1 + 6) ∘ λ n => n - 6) : ℚ) / (7 - (-6) + 1 : ℚ) = 0.5 := 
by
  sorry

end arithmetic_mean_of_integers_from_neg6_to_7_l768_768708


namespace math_problem_l768_768596

-- Define the polar curve C
def curve_C (rho theta : ℝ) : Prop :=
  rho = 2 * real.cos theta

-- Define the parametric line l
def line_l (x y t : ℝ) : Prop :=
  (x = real.sqrt 3 * t) ∧ (y = -1 + t)

-- Define the point P in polar coordinates and its conversion to rectangular coordinates
def point_P_polar : ℝ × ℝ :=
  (1, 3 * real.pi / 2)

def point_P_rectangular (P : ℝ × ℝ) : Prop :=
  let x := P.1 * real.cos P.2 in
  let y := P.1 * real.sin P.2 in
  (x = 0) ∧ (y = -1)

-- Define the rectangular equation of curve C
def curve_C_rect (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * x

-- Define the general equation of line l
def line_l_general (x y : ℝ) : Prop :=
  x - real.sqrt 3 * y - real.sqrt 3 = 0

-- Define the distance calculation |PA| and |PB|
def distance_PA (P A : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

def distance_PB (P B : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)

-- Define the value calculation
def value (PA PB : ℝ) : ℝ :=
  (PA + 1) * (PB + 1)

-- Main theorem
theorem math_problem :
  ∀ t P A B : ℝ × ℝ,
  point_P_rectangular P →
  curve_C_rect A.1 A.2 ∧ line_l_general A.1 A.2 →
  curve_C_rect B.1 B.2 ∧ line_l_general B.1 B.2 →
  value (distance_PA P A) (distance_PB P B) = 3 + real.sqrt 3 :=
by
  sorry

end math_problem_l768_768596


namespace problem_solution_l768_768852

def divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n + 1))

def divisorSum (n : ℕ) : ℕ :=
  (divisors n).filter (λ d, d ≠ n).sum

def doubleDivisorSum (n : ℕ) : ℕ :=
  divisorSum (divisorSum n)

theorem problem_solution : doubleDivisorSum 8 = 1 :=
by
  -- Insert the proof here
  sorry

end problem_solution_l768_768852


namespace minimize_sum_of_distances_l768_768880

-- Define a type for representing points on a line.
def line_point : Type := ℝ

-- Define the points on the line.
variables (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : line_point)

-- Define an arbitrary point Q on the line.
variable (Q : line_point)

-- Define the function t to be the sum of undirected distances from Q to each Q_i.
noncomputable def t (Q : line_point) : ℝ :=
  abs (Q - Q1) + abs (Q - Q2) + abs (Q - Q3) + abs (Q - Q4) + abs (Q - Q5) + 
  abs (Q - Q6) + abs (Q - Q7) + abs (Q - Q8) + abs (Q - Q9)

-- State the theorem: The sum of distances t is minimized when Q is at Q5.
theorem minimize_sum_of_distances : ∀ Q : line_point, t Q5 ≤ t Q :=
by sorry

end minimize_sum_of_distances_l768_768880


namespace number_of_valid_numbers_l768_768925

def digits := {2, 0, 2, 3}

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def uses_given_digits (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  1000 * a + 100 * b + 10 * c + d = n

def is_greater_than_2000 (n : ℕ) : Prop := n > 2000

theorem number_of_valid_numbers : 
  ∃ (count : ℕ), count = 6 ∧ 
  ∀ n, is_four_digit n ∧ is_greater_than_2000 n ∧ uses_given_digits n ↔ n = count := 
sorry

end number_of_valid_numbers_l768_768925


namespace sum_of_consecutive_integers_product_336_l768_768305

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l768_768305


namespace complement_intersection_correct_l768_768908

def U : Set ℝ := Set.univ

def B : Set ℝ := { x : ℝ | (1 / 2) ^ x ≤ 1 }

def A : Set ℝ := { x : ℝ | x ≥ 2 }

def C_U_A : Set ℝ := { x : ℝ | x < 2 }

def complement_intersection (U A B : Set ℝ) : Set ℝ :=
  (C_U_A ∩ B)

theorem complement_intersection_correct : 
  complement_intersection U A B = { x : ℝ | 0 ≤ x ∧ x < 2 } :=
by
  -- Proof here
  sorry

end complement_intersection_correct_l768_768908


namespace area_triangle_BCD_132_l768_768959

theorem area_triangle_BCD_132 (area_ABC : ℝ) (length_AC : ℝ) (length_CD : ℝ) (h_area_ABC : area_ABC = 36) (h_length_AC : length_AC = 9) (h_length_CD : length_CD = 33) :
  let h := 8 in
  let area_BCD := 1 / 2 * length_CD * h in
  area_BCD = 132 :=
by
  let h := 8
  let area_BCD := 1 / 2 * length_CD * h
  have h_area_BCD : area_BCD = 132
  { -- Calculate and verify the area of BCD
    have proof_ht := calc
      1 / 2 * length_CD * h = 1 / 2 * 33 * 8 : by rw [h_length_CD]
      ... = 132 : by norm_num,
    exact proof_ht },
  exact h_area_BCD

end area_triangle_BCD_132_l768_768959


namespace badgers_win_at_least_four_games_l768_768435

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
(nat.choose n k) * p^k * (1 - p)^(n - k)

theorem badgers_win_at_least_four_games :
  let p := 0.5 in
  let n := 7 in
  (binomial_probability n 4 p + binomial_probability n 5 p +
   binomial_probability n 6 p + binomial_probability n 7 p) = 0.5 := 
by
  sorry

end badgers_win_at_least_four_games_l768_768435


namespace angle_D_and_DC_equal_l768_768147

variables {A B C D : Type} [euclidean_geometry A B C D]

-- Given conditions
variables 
  (h_congruent : congruent (triangle A B C) (triangle A D C))
  (h_AB_AD : AB = AD)
  (h_angle_B : ∠ABC = 70)
  (h_BC : BC = 3)

-- Theorem to prove
theorem angle_D_and_DC_equal :
  ∠ADC = 70 ∧ DC = 3 :=
by
  sorry

end angle_D_and_DC_equal_l768_768147


namespace solve_fraction_equation_l768_768495

theorem solve_fraction_equation :
  {x : ℝ | (1 / (x^2 + 13 * x - 12) + 1 / (x^2 + 4 * x - 12) + 1 / (x^2 - 15 * x - 12) = 0)} =
  {1, -12, 12, -1} :=
by
  sorry

end solve_fraction_equation_l768_768495


namespace find_m_l768_768192

def circle_center : ℝ × ℝ := (1, -2)

def line_eq (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x - y + m = 0

def distance_point_line (x1 y1 : ℝ) (m : ℝ) : ℝ :=
  |x1 - y1 + m| / Real.sqrt 2

theorem find_m (m : ℝ) (h1 : distance_point_line 1 -2 m = Real.sqrt 2) :
  m = -1 ∨ m = -5 :=
sorry

end find_m_l768_768192


namespace none_of_these_true_l768_768072

def op_star (a b : ℕ) := b ^ a -- Define the binary operation

theorem none_of_these_true :
  ¬ (∀ a b : ℕ, 0 < a ∧ 0 < b → op_star a b = op_star b a) ∧
  ¬ (∀ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c → op_star a (op_star b c) = op_star (op_star a b) c) ∧
  ¬ (∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n → (op_star a b) ^ n = op_star n (op_star a b)) ∧
  ¬ (∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n → op_star a (b ^ n) = op_star n (op_star b a)) :=
sorry

end none_of_these_true_l768_768072


namespace remainder_polynomial_l768_768808

theorem remainder_polynomial (p : ℚ[X]) (h1 : eval 2 p = 3) (h2 : eval 3 p = -4) :
  ∃ q : ℚ[X], p = q * (X - 2) * (X - 3) + (-7 * X + 17) :=
sorry

end remainder_polynomial_l768_768808


namespace no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018_l768_768011

theorem no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018 (m n : ℕ) : ¬ (m^2 = n^2 + 2018) :=
sorry

end no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018_l768_768011


namespace instantaneous_velocity_at_t2_l768_768721

def displacement (t : ℝ) : ℝ := 2 * (1 - t) ^ 2

theorem instantaneous_velocity_at_t2 :
  (deriv displacement 2) = 4 :=
by
  sorry

end instantaneous_velocity_at_t2_l768_768721


namespace perpendicular_lines_value_of_a_l768_768911

theorem perpendicular_lines_value_of_a (a : ℝ) (ax_y_2a : ℝ → ℝ → Prop) (hc1 : ax_y_2a = λ x y, ax - y + 2a = 0)
    (2a_1x_ay_a : ℝ → ℝ → Prop) (hc2 : 2a_1x_ay_a = λ x y, (2a - 1)x + ay + a = 0)
    (perpendicular : ∀ x y, ax_y_2a x y → 2a_1x_ay_a x y → (a * (2a - 1) + (-1) * a = 0)) :
  a = 0 ∨ a = 1 :=
by {
  sorry
}

end perpendicular_lines_value_of_a_l768_768911


namespace num_four_digit_pos_integers_l768_768130

theorem num_four_digit_pos_integers : 
  ∃ n : ℕ, (n = 16) ∧ ∀ k : ℕ, (1000 ≤ k ∧ k < 10000 ∧ 
  ∀ d ∈ [k.digits 10], d = 3 ∨ d = 6) := sorry

end num_four_digit_pos_integers_l768_768130


namespace factorize_expression_l768_768017

theorem factorize_expression (a b : ℝ) : 2 * a ^ 2 - 8 * b ^ 2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by
  sorry

end factorize_expression_l768_768017


namespace complex_simplify_l768_768268

theorem complex_simplify :
  3 * (4 - 2 * complex.i) + 2 * complex.i * (3 + complex.i) = 10 :=
by
  sorry

end complex_simplify_l768_768268


namespace arithmetic_sequences_ration_correct_l768_768092

theorem arithmetic_sequences_ration_correct 
  (a b : ℕ → ℝ) 
  (S T : ℕ → ℝ)
  (h_SeqA : ∀ n, S n = (0 to n-1).sum (a i))
  (h_SeqB : ∀ n, T n = (0 to n-1).sum (b i))
  (h_Rel : ∀ n, T n ≠ 0 → S n / T n = n / (n + 7)) :
  a 7 / b 7 = 13 / 20 := 
by 
  sorry

end arithmetic_sequences_ration_correct_l768_768092


namespace printers_time_l768_768724

-- Conditions
def rate_A : ℝ := 35 / 60
def rate_B : ℝ := rate_A + 3
def total_pages : ℝ := 35
def combined_rate : ℝ := rate_A + rate_B

-- Target to prove
theorem printers_time (rate_A rate_B total_pages combined_rate : ℝ) : 
  combined_rate = rate_A + rate_B → total_pages / combined_rate = 8.4 :=
sorry

end printers_time_l768_768724


namespace opposite_quotient_l768_768573

theorem opposite_quotient {a b : ℝ} (h1 : a ≠ b) (h2 : a = -b) : a / b = -1 := 
sorry

end opposite_quotient_l768_768573


namespace arithmetic_sequence_inequality_l768_768937

theorem arithmetic_sequence_inequality 
  (a b c : ℝ) 
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : b - a = d)
  (h3 : c - b = d) :
  ¬ (a^3 * b + b^3 * c + c^3 * a ≥ a^4 + b^4 + c^4) :=
sorry

end arithmetic_sequence_inequality_l768_768937


namespace cos_sum_sub_diff_l768_768489

theorem cos_sum_sub_diff (a b : ℝ) :
  cos (a + b) - cos (a - b) = -2 * sin a * sin b :=
sorry

end cos_sum_sub_diff_l768_768489


namespace unique_nonnegative_sequence_l768_768010

open BigOperators

def is_strictly_increasing (s : List ℕ) : Prop :=
  ∀ i j, i < j → s.nth i < s.nth j

theorem unique_nonnegative_sequence :
  ∃ k : ℕ, k = 204 ∧ ∃ s : List ℕ, is_strictly_increasing s ∧ (∀ x ∈ s, 0 ≤ x) ∧ (∑ i in s.to_finset, 2^i = (2^385 + 1) / (2^11 + 1)) :=
  by 
  sorry

end unique_nonnegative_sequence_l768_768010


namespace clyde_picked_correct_number_of_cobs_l768_768751

-- Definitions based on conditions
def bushel_weight : ℝ := 56
def ear_weight : ℝ := 0.5
def bushels_picked : ℝ := 2

-- Calculation based on conditions
def cobs_per_bushel (bushel_weight ear_weight : ℝ) : ℝ := bushel_weight / ear_weight
def total_cobs (bushels_picked cobs_per_bushel : ℝ) : ℝ := bushels_picked * cobs_per_bushel

-- Theorem statement
theorem clyde_picked_correct_number_of_cobs :
  total_cobs bushels_picked (cobs_per_bushel bushel_weight ear_weight) = 224 := 
by
  sorry

end clyde_picked_correct_number_of_cobs_l768_768751


namespace schedule_courses_non_consecutive_l768_768145
-- Import necessary Lean libraries

-- Lean statement representing the described problem
theorem schedule_courses_non_consecutive : 
  ∃ (f : ℕ → ℕ), (∀ i, i ∈ finset.range 4 → f i ∈ finset.range 7) ∧ 
                 (∀ i j, i < j ∧ i ∈ finset.range 4 ∧ j ∈ finset.range 4 → f i < f j) ∧ 
                 f 0 > 0 ∧ f 3 < 6 ∧ ∀ i, i ∈ finset.range 3 → f (i + 1) > f i + 1 :=
  sorry

end schedule_courses_non_consecutive_l768_768145


namespace hypotenuse_of_30_degree_triangle_l768_768654

noncomputable theory

-- Define the conditions
def leg_length : ℝ := 15
def angle_opposite_leg : ℝ := 30

-- Define the expected result
def expected_hypotenuse : ℝ := 30

-- The theorem to be proved
theorem hypotenuse_of_30_degree_triangle (leg_length = 15) (angle_opposite_leg = 30) : 
  (2 * leg_length = expected_hypotenuse) :=
by
  sorry

end hypotenuse_of_30_degree_triangle_l768_768654


namespace alternating_harmonic_even_sum_l768_768887

theorem alternating_harmonic_even_sum (n : ℕ) (h_even : Even n) (h_pos : 0 < n) :
  (Finset.range n).sum (λ k, (-1 : ℚ) ^ k * (1 / (k + 1))) = 2 * (Finset.Ico n (2 * n)).sum (λ k, 1 / (k + 1)) :=
sorry

end alternating_harmonic_even_sum_l768_768887


namespace problem_l768_768934

variables (x : ℝ)

-- Define the condition
def condition (x : ℝ) : Prop :=
  0.3 * (0.2 * x) = 24

-- Define the target statement
def target (x : ℝ) : Prop :=
  0.2 * (0.3 * x) = 24

-- The theorem we want to prove
theorem problem (x : ℝ) (h : condition x) : target x :=
sorry

end problem_l768_768934


namespace cost_per_chicken_l768_768250

-- Definitions for conditions
def totalBirds : ℕ := 15
def ducks : ℕ := totalBirds / 3
def chickens : ℕ := totalBirds - ducks
def feed_cost : ℕ := 20

-- Theorem stating the cost per chicken
theorem cost_per_chicken : (feed_cost / chickens) = 2 := by
  sorry

end cost_per_chicken_l768_768250


namespace no_fixed_points_l768_768851

def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f(x) = x

theorem no_fixed_points (f : ℝ → ℝ) (h : f = λ x, x^2 + 1) : ¬∃ x : ℝ, is_fixed_point f x := by
  sorry

end no_fixed_points_l768_768851


namespace car_b_travel_time_l768_768800

def car_a_speed := 50  -- in km/hr
def car_a_time := 6  -- in hours
def car_b_speed := 100  -- in km/hr
def distance_ratio := 3 -- ratio of distances covered by Car A and Car B

theorem car_b_travel_time : 
  let distance_a := car_a_speed * car_a_time in
  let distance_b := distance_a / distance_ratio in 
  let time_b := distance_b / car_b_speed in 
  time_b = 1 :=
by
  sorry

end car_b_travel_time_l768_768800


namespace car_and_cyclist_speeds_and_meeting_point_l768_768410

/-- 
(1) Distance between points $A$ and $B$ is $80 \mathrm{~km}$.
(2) After one hour, the distance between them reduces to $24 \mathrm{~km}$.
(3) The cyclist takes a 1-hour rest but they meet $90$ minutes after their departure.
-/
def initial_distance : ℝ := 80 -- km
def distance_after_one_hour : ℝ := 24 -- km apart after 1 hour
def cyclist_rest_duration : ℝ := 1 -- hour
def meeting_time : ℝ := 1.5 -- hours (90 minutes after departure)

def car_speed : ℝ := 40 -- km/hr
def cyclist_speed : ℝ := 16 -- km/hr

theorem car_and_cyclist_speeds_and_meeting_point :
  initial_distance = 80 → 
  distance_after_one_hour = 24 → 
  cyclist_rest_duration = 1 → 
  meeting_time = 1.5 → 
  car_speed = 40 ∧ cyclist_speed = 16 ∧ meeting_point_from_A = 60 ∧ meeting_point_from_B = 20 :=
by
  sorry

end car_and_cyclist_speeds_and_meeting_point_l768_768410


namespace hyperbola_focal_length_l768_768500

theorem hyperbola_focal_length (λ : ℝ) (A : ℝ × ℝ) (hA : A = (-3, 3 * Real.sqrt 2))
  (h_equiv : λ = -1 / 8) :
  let a := Real.sqrt 2,
      b := Real.sqrt (9 / 8),
      c := Real.sqrt (a^2 + b^2) in
  2 * c = 5 * Real.sqrt 2 / 2 :=
by sorry

end hyperbola_focal_length_l768_768500


namespace simplify_expression_l768_768664

theorem simplify_expression (x : ℝ) : (x + 1) ^ 2 + x * (x - 2) = 2 * x ^ 2 + 1 :=
by
  sorry

end simplify_expression_l768_768664


namespace expression_for_f_intervals_where_monotonically_increasing_values_of_a_and_b_l768_768067

-- Define conditions and functions
variables (x : ℝ) (a ω φ : ℝ)

def OA := (2 * a * (cos ((ω * x + φ) / 2))^2, 1)
def OB := (1, sqrt 3 * a * sin (ω * x + φ) - a)
def f := OA.1 * OB.1 + OA.2 * OB.2

-- Given conditions
variable (a_ne_0 : a ≠ 0)
variable (ω_gt_0 : ω > 0)
variable (phi_interval : 0 < φ ∧ φ < π / 2)

-- Proof problems
theorem expression_for_f : 
  (∀ x, f x = 2 * a * sin (2 * x + π / 3)) :=
sorry

theorem intervals_where_monotonically_increasing : 
  (a > 0) → (∀ k : ℤ, ∀ x, k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12 ↔ 
  ∀ x', 2 * x’ + π / 3 = x ↔ f x' < f (x' + 1)) :=
sorry

theorem values_of_a_and_b :
  (x ∈ Icc 0 (π / 2)) → 
  ((∀ x, f x + 2 = 0 ∧ f x - sqrt 3 = 0) ↔ ((a = -1 ∧ b = 2 - sqrt 3) ∨ (a = 1 ∧ b = sqrt 3 - 1))) :=
sorry

end expression_for_f_intervals_where_monotonically_increasing_values_of_a_and_b_l768_768067


namespace integral_semicircle_minus_sin_l768_768461

open Real

noncomputable def semicircle_integral := ∫ x in -1..1, sqrt (1 - x^2)
noncomputable def sine_integral := ∫ x in -1..1, sin x

theorem integral_semicircle_minus_sin :
  (2 * semicircle_integral - sine_integral) = π :=
by
  have semi_circle_area : ∫ x in -1..1, sqrt (1 - x^2) = π / 2 := sorry
  have sine_integral_res : ∫ x in -1..1, sin x = 0 := sorry
  simp [semicircle_integral, sine_integral, semi_circle_area, sine_integral_res]
  linarith

end integral_semicircle_minus_sin_l768_768461


namespace greatest_price_per_shirt_l768_768775

theorem greatest_price_per_shirt (budget : ℝ) (entrance_fee : ℝ) (tax_rate : ℝ) (num_shirts : ℕ) :
  budget = 200 → entrance_fee = 5 → tax_rate = 0.07 → num_shirts = 20 →
  let remaining_budget := budget - entrance_fee,
      total_amount_for_shirts := remaining_budget / (1 + tax_rate),
      price_per_shirt := total_amount_for_shirts / num_shirts in
  ⌊price_per_shirt⌋ = 9 :=
by
  intros h_budget h_fee h_tax h_num_shirts
  have remaining_budget := budget - entrance_fee,
  have total_amount_for_shirts := remaining_budget / (1 + tax_rate),
  have price_per_shirt := total_amount_for_shirts / num_shirts,
  have price_per_shirt_approx : floor price_per_shirt = 9,
  sorry

end greatest_price_per_shirt_l768_768775


namespace overall_percentage_change_l768_768771

def fall_increase (m : ℝ) : ℝ := m * 1.05
def winter_increase (m : ℝ) : ℝ := m * 1.14
def spring_decrease (m : ℝ) : ℝ := m * 0.81
def summer_decrease (m : ℝ) : ℝ := m * 0.93

theorem overall_percentage_change :
  let m0 := 100.0
  let m1 := fall_increase m0
  let m2 := winter_increase m1
  let m3 := spring_decrease m2
  let m4 := summer_decrease m3
  ((m4 - m0) / m0 * 100) ≈ -9.83 :=
by sorry

end overall_percentage_change_l768_768771


namespace curve_intersection_l768_768198

noncomputable def C1 (t : ℝ) (a : ℝ) : ℝ × ℝ :=
  (2 * t + 2 * a, -t)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.sin θ, 1 + 2 * Real.cos θ)

theorem curve_intersection (a : ℝ) :
  (∃ t θ : ℝ, C1 t a = C2 θ) ↔ 1 - Real.sqrt 5 ≤ a ∧ a ≤ 1 + Real.sqrt 5 :=
sorry

end curve_intersection_l768_768198


namespace total_leftover_value_l768_768766

theorem total_leftover_value : 
  let roll_of_quarters := 50
  let roll_of_dimes := 60
  let alice_quarters := 95
  let alice_dimes := 184
  let bob_quarters := 145
  let bob_dimes := 312
  let total_quarters := alice_quarters + bob_quarters
  let total_dimes := alice_dimes + bob_dimes
  let leftover_quarters := total_quarters % roll_of_quarters
  let leftover_dimes := total_dimes % roll_of_dimes
  let value_leftover_quarters := (leftover_quarters // 1) * (25 // 100)
  let value_leftover_dimes := (leftover_dimes // 1) * (10 // 100)
in 
  value_leftover_quarters + value_leftover_dimes = 11.60
:=
begin
  sorry
end

end total_leftover_value_l768_768766


namespace abs_eq_zero_implies_values_l768_768520

theorem abs_eq_zero_implies_values (x y : ℝ) (h : |x - 3| + |y + 2| = 0) : 
  (y - x = -5) ∧ (x * y = -6) :=
begin
  -- solution goes here
  sorry
end

end abs_eq_zero_implies_values_l768_768520


namespace point_inside_circle_l768_768688

theorem point_inside_circle : 
  ∀ θ : ℝ, ∃ x y : ℝ, (x = 1 + 2 * Real.cos θ) ∧ 
                      (y = 2 * Real.sin θ) → 
                      ((2 - 1) ^ 2 + (-1) ^ 2 < 4) :=
by
  intro θ
  dsimp
  sorry

end point_inside_circle_l768_768688


namespace number_of_special_four_digit_integers_l768_768127

theorem number_of_special_four_digit_integers : 
  let digits := [3, 6]
  let choices_per_digit := 2
  num_digits = 4
  ∑ i in range(num_digits), (choices_per_digit) = 2^4 :=
by
  sorry

end number_of_special_four_digit_integers_l768_768127


namespace smallest_y_76545_l768_768396

theorem smallest_y_76545 (y : ℕ) (h1 : ∀ z : ℕ, 0 < z → (76545 * z = k ^ 2 → (3 ∣ z ∨ 5 ∣ z) → z = y)) : y = 7 :=
sorry

end smallest_y_76545_l768_768396


namespace simplify_and_evaluate_expression_l768_768269

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 + 3) :
  ( (x^2 - 1) / (x^2 - 6 * x + 9) * (1 - x / (x - 1)) / ((x + 1) / (x - 3)) ) = - (Real.sqrt 2 / 2) :=
  sorry

end simplify_and_evaluate_expression_l768_768269


namespace emilia_grandchildren_is_8_l768_768477

noncomputable def num_emilia_grandchildren : ℕ :=
  let leonie_truths : list ℕ := [7, 8, some_other_number] in
  let gabrielle_truths : list ℕ := [some_number, some_number, 10] in
  -- Placeholder functions assuming the statements from Leonie and Gabrielle
  let leonie_statement (n : ℕ) : bool := n ∈ leonie_truths in
  let gabrielle_statement (n : ℕ) : bool := n ∈ gabrielle_truths in
  
  if leonie_statement 8 && gabrielle_statement 8 then 8 else sorry

theorem emilia_grandchildren_is_8 : num_emilia_grandchildren = 8 :=
begin
  unfold num_emilia_grandchildren,
  rw if_pos,
  -- Additional properties and logic checks can be added here as needed
  sorry,
end

end emilia_grandchildren_is_8_l768_768477


namespace harry_cookies_batch_l768_768921

theorem harry_cookies_batch
  (total_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips = 81)
  (batches = 3)
  (chips_per_cookie = 9) :
  (total_chips / batches) / chips_per_cookie = 3 := by
  sorry

end harry_cookies_batch_l768_768921


namespace forest_track_fraction_l768_768738

theorem forest_track_fraction (n : ℕ) : ∃ f : ℝ, 
    ( ∀ (A B C : ℂ), equilateral (triangle A B C) ∧ stations_on_track A B C ∧ 
    same_time_entry A B ∧ driver_in_forest Roma = (* some formula involving ℚ *): 
    (if n % 3 = 1 then f = 2/3 else if n % 3 = 2 then f = 1/3 else f = 0)) :=
by 
  sorry

end forest_track_fraction_l768_768738


namespace total_paintable_area_correct_l768_768772

-- Define the conditions
def warehouse_width := 12
def warehouse_length := 15
def warehouse_height := 7

def window_count_per_longer_wall := 3
def window_width := 2
def window_height := 3

-- Define areas for walls, ceiling, and floor
def area_wall_1 := warehouse_width * warehouse_height
def area_wall_2 := warehouse_length * warehouse_height
def window_area := window_width * window_height
def window_total_area := window_count_per_longer_wall * window_area
def area_wall_2_paintable := 2 * (area_wall_2 - window_total_area) -- both inside and outside
def area_ceiling := warehouse_width * warehouse_length
def area_floor := warehouse_width * warehouse_length

-- Total paintable area calculation
def total_paintable_area := 2 * area_wall_1 + area_wall_2_paintable + area_ceiling + area_floor

-- Final proof statement
theorem total_paintable_area_correct : total_paintable_area = 876 := by
  sorry

end total_paintable_area_correct_l768_768772


namespace volume_to_surface_area_ratio_l768_768436

theorem volume_to_surface_area_ratio :
  ∀ (V S : ℕ), 
    (V = 9 ∧ S = 45) → (V / S = 1 / 5) := by
  intros V S h
  cases h with volume_eq surface_area_eq
  rw [volume_eq, surface_area_eq]
  norm_num
  sorry

end volume_to_surface_area_ratio_l768_768436


namespace quiz_probability_l768_768431

theorem quiz_probability :
  let probMCQ := 1/3
  let probTF1 := 1/2
  let probTF2 := 1/2
  probMCQ * probTF1 * probTF2 = 1/12 := by
  sorry

end quiz_probability_l768_768431


namespace third_root_of_polynomial_l768_768346

theorem third_root_of_polynomial (a b : ℚ) 
  (h₁ : a*(-1)^3 + (a + 3*b)*(-1)^2 + (2*b - 4*a)*(-1) + (10 - a) = 0)
  (h₂ : a*(4)^3 + (a + 3*b)*(4)^2 + (2*b - 4*a)*(4) + (10 - a) = 0) :
  ∃ (r : ℚ), r = -24 / 19 :=
by
  sorry

end third_root_of_polynomial_l768_768346


namespace find_a_no_extremum_at_1_compare_size_l768_768642

-- Condition: Definition of f(x)
def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

-- Condition: Tangent line at x = e intersects y-axis at point (0, 2-e)
def tangent_condition (a : ℝ) : Prop :=
  ((f Real.exp a - (2 - Real.exp)) / (Real.exp - 0)) = 
  ((Real.exp + 1) * Real.log Real.exp + 1 - a)

-- Prove that a = 2 given the conditions
theorem find_a : ∃ a, tangent_condition a ∧ a = 2 := sorry

-- Prove that f(x) cannot have an extremum at x = 1
theorem no_extremum_at_1 (a : ℝ) : ¬ (∃ x : ℝ, x = 1 ∧ f x a = 0 ∧ ∃ p, ∀ y, x < y →  f y a ≤ f x a ) := sorry

-- Prove that when 1 < x < 2, (2 / (x - 1)) > (1 / (Real.log x)) - (1 / (Real.log (2 - x)))
theorem compare_size (x : ℝ) (h : 1 < x ∧ x < 2) : (2 / (x - 1)) > (1 / (Real.log x)) - (1 / (Real.log (2 - x))) := sorry

end find_a_no_extremum_at_1_compare_size_l768_768642


namespace plywood_cut_perimeter_difference_l768_768743

theorem plywood_cut_perimeter_difference :
  ∀ (length width : ℕ), length = 6 ∧ width = 9 → 
  ∃ p1 p2 : ℕ, 
    (∃ (config1 : length ≠ 0 ∧ width ≠ 0), p1 = 2 * (3 + width)) ∧
    (∃ (config2 : length ≠ 0 ∧ width ≠ 0), p2 = 2 * (6 + 3)) ∧
    (∀ n : ℕ, n = 3 → ∃ cut : length * width = 3 * (length * width / 3))),
  abs (p1 - p2) = 6 := 
by
  intro length width h
  obtain ⟨h1, h2⟩ := h
  have config1 := 2 * (3 + 9)
  have config2 := 2 * (6 + 3)
  have h3 := 6 * 9 = 3 * (6 * 9 / 3)
  use config1, config2
  split
  . use (6 ≠ 0 ∧ 9 ≠ 0)
    exact rfl
  . split
    . use (6 ≠ 0 ∧ 9 ≠ 0)
      exact rfl
    . intro n hn
      use h3
  rw [abs_eq_nat]
  rw [config1, config2]
  exact rfl

end plywood_cut_perimeter_difference_l768_768743


namespace divisible_by_lcm_of_4_5_6_l768_768929

theorem divisible_by_lcm_of_4_5_6 (n : ℕ) : (∃ k, 0 < k ∧ k < 300 ∧ k % 60 = 0) ↔ (∃! k, k = 4) :=
by
  let lcm_4_5_6 := Nat.lcm (Nat.lcm 4 5) 6
  have : lcm_4_5_6 = 60 := sorry
  have : ∀ k, (0 < k ∧ k < 300 ∧ k % lcm_4_5_6 = 0) ↔ (k = 60 ∨ k = 120 ∨ k = 180 ∨ k = 240) := sorry
  have : ∃ k, 0 < k ∧ k < 300 ∧ k % lcm_4_5_6 = 0 :=
    ⟨60, by norm_num, mk 120, by norm_num, mk 180, by norm_num, mk 240, by norm_num⟩
  have : ∃! k, (k = 60 ∨ k = 120 ∨ k = 180 ∨ k = 240) := sorry
  show _ ↔ (∃! k, k = 4) from sorry

end divisible_by_lcm_of_4_5_6_l768_768929


namespace simplify_radical_l768_768264

theorem simplify_radical (x : ℝ) :
  2 * Real.sqrt (50 * x^3) * Real.sqrt (45 * x^5) * Real.sqrt (98 * x^7) = 420 * x^7 * Real.sqrt (5 * x) :=
by
  sorry

end simplify_radical_l768_768264


namespace jarA_percentage_more_than_jarB_l768_768206

-- Definitions of the problem conditions.
def jarA_has_P_more_than_jarB (A B P : ℝ) : Prop :=
  A = (1 + P / 100) * B

def equalize_after_transfer (A B : ℝ) : Prop :=
  let k := 0.10317460317460316 in
  A - k * A = B + k * A

-- The target is to prove the percentage P is approximately 23.015873015873%
theorem jarA_percentage_more_than_jarB (A B : ℝ) (h1 : equalize_after_transfer A B) : 
  ∃ P, jarA_has_P_more_than_jarB A B P ∧ P ≈ 23.015873015873 :=
by
  sorry

end jarA_percentage_more_than_jarB_l768_768206


namespace percentage_passed_in_both_l768_768397

def percentage_of_students_failing_hindi : ℝ := 30
def percentage_of_students_failing_english : ℝ := 42
def percentage_of_students_failing_both : ℝ := 28

theorem percentage_passed_in_both (P_H_E: percentage_of_students_failing_hindi + percentage_of_students_failing_english - percentage_of_students_failing_both = 44) : 
  100 - (percentage_of_students_failing_hindi + percentage_of_students_failing_english - percentage_of_students_failing_both) = 56 := by
  sorry

end percentage_passed_in_both_l768_768397


namespace train_speed_second_half_today_l768_768691

noncomputable def initial_speed := 80
noncomputable def total_distance := 120
noncomputable def stop_duration_today := 3 / 20
noncomputable def scheduled_travel_time := 1.5

theorem train_speed_second_half_today : 
    ∃ v, 
    (60 / initial_speed) + stop_duration_today + (60 / v) = scheduled_travel_time ∧ v = 100 :=
by
    sorry

end train_speed_second_half_today_l768_768691


namespace question1_question2_l768_768894

open Real

def ellipse (x y : ℝ) := (x^2) / 5 + (y^2) / 4 = 1
def right_focus := (1 : ℝ, 0)
def line_l (k : ℝ) (x y : ℝ) := y = k * (x - 1)

theorem question1 ( M N : ℝ × ℝ ) (h : ∀ M N, (M, N ∈ { p : ℝ × ℝ | ellipse p.fst p.snd ∧ p ≠ right_focus }) → 
  ( ((M.1 * N.1) + (M.2 * N.2) = -3) → ∃ k, (line_l (√2) M.1 M.2 ∧ line_l (-√2) N.1 N.2) ∨ 
  (line_l (-√2) M.1 M.2 ∧ line_l (√2) N.1 N.2)) ) : sorry

theorem question2 ( M N : ℝ × ℝ ) (a : ℝ) (h : ∃ k, line_l k M.1 M.2 ∧ line_l k N.1 N.2 ∧ M ≠ N) :
  ( ∃ P : ℝ × ℝ, P = (a, 0) ∧ 0 ≤ a ∧ a < 1/5 ∧ (dist P M = dist P N) ) : sorry

end question1_question2_l768_768894


namespace sum_of_elements_in_A_otimes_B_l768_768512

-- Define the set A and B
def A : set ℕ := {x | x % 2 = 0 ∧ x ≤ 18}
def B : set ℕ := {x | x ∈ {98, 99, 100}}

-- Define the operation ⊗
def otimes (A B : set ℕ) : set ℕ :=
  {x | ∃ (a ∈ A) (b ∈ B), x = a * b + a + b}

-- Calculate the sum of all elements in A ⊗ B
noncomputable def sum_otimes (A B : set ℕ) : ℕ :=
  ∑ x in (otimes A B).to_finset, x

-- State the theorem
theorem sum_of_elements_in_A_otimes_B : sum_otimes A B = 29970 :=
sorry

end sum_of_elements_in_A_otimes_B_l768_768512


namespace max_distinct_numbers_in_table_l768_768186

theorem max_distinct_numbers_in_table 
  (table : ℕ → ℕ → ℕ)
  (h_dim : ∀ i j, i < 75 → j < 75)
  (h_row : ∀ i, i < 75 → (finset.image (table i) (finset.range 75)).card ≥ 15)
  (h_three_rows : ∀ i, i < 73 → 
    (finset.bUnion (finset.range 3)
      (λ k, finset.image (table (i + k)) (finset.range 75))).card ≤ 25) :
  ∃ distinct_numbers, distinct_numbers = 385 ∧ 
  ∀ (i j : ℕ), i < 75 → j < 75 → table i j ∈ distinct_numbers :=
sorry

end max_distinct_numbers_in_table_l768_768186


namespace find_positive_integer_divisible_by_18_and_cuberoot_between_8_and_8_1_l768_768492

noncomputable def n : ℕ := 522

theorem find_positive_integer_divisible_by_18_and_cuberoot_between_8_and_8_1 :
  (∃ k : ℕ, n = 18 * k) ∧ (512 < n ∧ n < 531.441) ∧ (8 < n ^ (1 / 3) ∧ n ^ (1 / 3) < 8.1) :=
by
  use 29
  have h1 : n = 18 * 29 := by rfl
  have h2 : 512 < n ∧ n < 531.441 := by
    simp [n]
    exact ⟨by norm_num, by norm_num⟩
  have h3 : 8 < n ^ (1 / 3) ∧ n ^ (1 / 3) < 8.1 := by
    have h : (522 : ℝ) = 522 := by norm_num
    rw h
    norm_num
  exact ⟨⟨29, h1⟩, h2, h3⟩

end find_positive_integer_divisible_by_18_and_cuberoot_between_8_and_8_1_l768_768492


namespace sum_of_consecutive_integers_product_336_l768_768304

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l768_768304


namespace num_students_with_B_in_cecilia_l768_768578

-- Conditions
def prop_jacob : ℝ := 3 / 5
def num_absent_cecilia : ℕ := 6
def num_total_cecilia : ℕ := 30
def num_present_cecilia := num_total_cecilia - num_absent_cecilia

-- Conclusion to prove
theorem num_students_with_B_in_cecilia :
  ∃ n : ℤ, n = (real.toInt (real.ofInt 24 * prop_jacob)) ∧ n = 14 :=
by
  sorry

end num_students_with_B_in_cecilia_l768_768578


namespace fish_remaining_correct_l768_768211

def remaining_fish (jordan_caught : ℕ) (total_catch_lost_fraction : ℚ) : ℕ :=
  let perry_caught := 2 * jordan_caught
  let total_catch := jordan_caught + perry_caught
  let lost_catch := total_catch * total_catch_lost_fraction
  let remaining := total_catch - lost_catch
  remaining.nat_abs

theorem fish_remaining_correct : (remaining_fish 4 (1/4)) = 9 :=
by 
  sorry

end fish_remaining_correct_l768_768211


namespace nine_chapters_coins_l768_768736

theorem nine_chapters_coins (a d : ℚ)
  (h1 : (a - 2 * d) + (a - d) = a + (a + d) + (a + 2 * d))
  (h2 : (a - 2 * d) + (a - d) + a + (a + d) + (a + 2 * d) = 5) :
  a - d = 7 / 6 :=
by 
  sorry

end nine_chapters_coins_l768_768736


namespace unit_prices_max_bundles_type_A_l768_768171

theorem unit_prices (x y : ℕ) : 
  (30 * x + 10 * y = 380) ∧ (50 * x + 30 * y = 740) → (x = 10) ∧ (y = 8) := by
  sorry

theorem max_bundles_type_A (m : ℕ) :
  ∀ (A_price B_price : ℕ), A_price = 10 ∧ B_price = 8 →
  (∀ n, ∃ m, m + n = 100) →
  (9 * m + 7.2 * (100 - m) ≤ 828) → 
  m ≤ 60 := by
  sorry

end unit_prices_max_bundles_type_A_l768_768171


namespace inequality_one_solution_inequality_system_solution_l768_768667

theorem inequality_one_solution :
  {x : ℕ | (5 * (x - 1) / 6 : ℚ) - 1 < (x + 2) / 3} = {1, 2, 3, 4} :=
by
  sorry

theorem inequality_system_solution :
  {x : ℝ | 3 * x - 2 ≤ x + 6 ∧ (5 * x + 3) / 2 > x} = Ioo (-1 : ℝ) 4 :=
by
  sorry

end inequality_one_solution_inequality_system_solution_l768_768667


namespace num_four_digit_36_combinations_l768_768099

theorem num_four_digit_36_combinations : 
  (∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
  (∀ d ∈ [digit n 1000, digit n 100, digit n 10, digit n 1], d = 3 ∨ d = 6)) → 
  16 :=
sorry

end num_four_digit_36_combinations_l768_768099


namespace tickets_difference_l768_768392

-- Define the conditions
def orchestra_ticket_price : ℕ := 12
def balcony_ticket_price : ℕ := 8
def total_tickets_sold : ℕ := 370
def total_cost : ℕ := 3320

-- Define the variables representing the number of tickets sold for orchestra and balcony
variables (O B : ℕ)

-- Asserting the conditions in Lean
def conditions : Prop :=
  (O + B = total_tickets_sold) ∧
  (orchestra_ticket_price * O + balcony_ticket_price * B = total_cost)

-- Statement of the proof problem: Proving the difference in ticket numbers
theorem tickets_difference (h : conditions) : B - O = 190 :=
sorry

end tickets_difference_l768_768392


namespace solve_for_x_l768_768843

theorem solve_for_x (x : ℝ) (h : sqrt (x + 3) = 7) : x = 46 :=
by {
  -- proof will be here
  sorry
}

end solve_for_x_l768_768843


namespace find_a_for_extremum_at_1_l768_768902

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x^2

-- Provided conditions
def has_extremum_at_1 (a : ℝ) : Prop :=
  ∀ h : Deriv (λ x, f a x) 1 = 0, h

theorem find_a_for_extremum_at_1 :
  ∃ a : ℝ, has_extremum_at_1 a ∧ a = -2 := 
begin
  use [-2],
  split,
  {
    intro h,
    dsimp [Deriv] at *,
    sorry,
  },
  {
    refl,
  }
end

end find_a_for_extremum_at_1_l768_768902


namespace opposite_sides_of_line_l768_768155

theorem opposite_sides_of_line (m : ℝ) (h : (2 * (-2 : ℝ) + m - 2) * (2 * m + 4 - 2) < 0) : -1 < m ∧ m < 6 :=
sorry

end opposite_sides_of_line_l768_768155


namespace three_numbers_sum_gt_100_l768_768521

theorem three_numbers_sum_gt_100
  (x : Fin 100 → ℝ)
  (h1 : ∑ i, x i^2 > 10000)
  (h2 : ∑ i, x i < 300)
  (h3 : ∀ i, x i > 0) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ x i + x j + x k > 100 :=
by
  sorry

end three_numbers_sum_gt_100_l768_768521


namespace sin_exp_eq_solutions_l768_768029

open Real

theorem sin_exp_eq_solutions : 
  ∀ f g : ℝ → ℝ, (∀ x, f x = sin x) → 
  (∀ x, g x = (1/3) ^ x) → 
  (Icc 0 (50 * π)).count (λ x, f x = g x) = 50 :=
by
  intro f g hf hg
  sorry

end sin_exp_eq_solutions_l768_768029


namespace number_of_special_four_digit_integers_l768_768126

theorem number_of_special_four_digit_integers : 
  let digits := [3, 6]
  let choices_per_digit := 2
  num_digits = 4
  ∑ i in range(num_digits), (choices_per_digit) = 2^4 :=
by
  sorry

end number_of_special_four_digit_integers_l768_768126


namespace broken_line_length_greater_than_1248_l768_768748

theorem broken_line_length_greater_than_1248 (A : ℕ → ℝ × ℝ) (n : ℕ) 
  (h1 : ∀ k < n - 1, dist (A k) (A (k + 1)) < 1)
  (h2 : ∀ p : ℝ × ℝ, p.1 >= 0 → p.1 <= 50 → p.2 >= 0 → p.2 <= 50 → 
    ∃ k < n - 1, dist p (A k) < 1 ∨ dist p (A (k + 1)) < 1) :
  let L := ∑ i in finset.range (n-1), dist (A i) (A (i+1)) in
  L > 1248 :=
by
  sorry

end broken_line_length_greater_than_1248_l768_768748


namespace min_value_ineq_l768_768228

open Real

theorem min_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 2) * (y^2 + 5 * y + 2) * (z^2 + 5 * z + 2) / (x * y * z) ≥ 512 :=
by sorry

noncomputable def optimal_min_value : ℝ := 512

end min_value_ineq_l768_768228


namespace initial_cd_count_l768_768373

variable (X : ℕ)

theorem initial_cd_count (h1 : (2 / 3 : ℝ) * X + 8 = 22) : X = 21 :=
by
  sorry

end initial_cd_count_l768_768373


namespace polynomial_coefficients_l768_768858

theorem polynomial_coefficients (a : Fin 10 → ℤ) :
  (1 - X) ^ 9 = ∑ i in Finset.range 10, (a i) * X ^ i →
  a 0 = 1 ∧
  a 1 + a 3 + a 5 + a 7 + a 9 = -256 ∧
  (2 : ℤ) * a 1 + (2 : ℤ)^2 * a 2 + (2 : ℤ)^3 * a 3 + (2 : ℤ)^4 * a 4 + (2 : ℤ)^5 * a 5 + 
  (2 : ℤ)^6 * a 6 + (2 : ℤ)^7 * a 7 + (2 : ℤ)^8 * a 8 + (2 : ℤ)^9 * a 9 = -2 := by
  sorry

end polynomial_coefficients_l768_768858


namespace percentage_of_number_l768_768718

-- Given definition: converting percentage to a decimal
def percentage_to_decimal (p : ℝ) : ℝ := p / 100

-- The problem statement could be translated to the following theorem in Lean:
theorem percentage_of_number (p n : ℝ) : 
  percentage_to_decimal p * n = 80 :=
by
  -- percentage_to_decimal 16.666666666666668 * 480 should be 80
  let p := 16.666666666666668
  let n := 480
  show percentage_to_decimal p * n = 80
  sorry

end percentage_of_number_l768_768718


namespace apples_taken_from_each_basket_l768_768205

theorem apples_taken_from_each_basket (total_apples : ℕ) (baskets : ℕ) (remaining_apples_per_basket : ℕ) 
(h1 : total_apples = 64) (h2 : baskets = 4) (h3 : remaining_apples_per_basket = 13) : 
(total_apples - (remaining_apples_per_basket * baskets)) / baskets = 3 :=
sorry

end apples_taken_from_each_basket_l768_768205


namespace repeating_decimal_as_fraction_l768_768487

-- Define the repeating decimal
def repeating_decimal_2_35 := 2 + (35 / 99 : ℚ)

-- Define the fraction form
def fraction_form := (233 / 99 : ℚ)

-- Theorem statement asserting the equivalence
theorem repeating_decimal_as_fraction : repeating_decimal_2_35 = fraction_form :=
by 
  -- Skipped proof
  sorry

end repeating_decimal_as_fraction_l768_768487


namespace valid_rearrangements_count_l768_768478

open Classical

-- Define the context of the problem
variables {n : ℕ} (table : Fin n → Fin n) (eight_friends : n = 8)

-- Conditions:
def no_one_in_same_seat : Prop :=
  ∀ i, table i ≠ i

def no_one_in_adjacent_seat : Prop :=
  ∀ i, table i ≠ (i + 1) % n ∧ table i ≠ (i - 1) % n

def no_one_in_opposite_seat : Prop :=
  ∀ i, table i ≠ (i + n/2) % n

-- The theorem we wish to prove
theorem valid_rearrangements_count (eight_friends' : eight_friends = 8):
  let rearrangements := {table : Fin n → Fin n // no_one_in_same_seat table ∧ no_one_in_adjacent_seat table ∧ no_one_in_opposite_seat table} in
  rearrangements.fintype.card = 14 :=
sorry

end valid_rearrangements_count_l768_768478


namespace solution_part_1_solution_part_2_l768_768895

theorem solution_part_1 (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : b = 1) (h4 : (sqrt (a^2 - b^2)) / a = sqrt 6 / 3) :
  (a = sqrt 3) ∧ b = 1 ∧ (∀ x y : ℝ, (x^2 / 3 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
by {
  sorry
}

theorem solution_part_2 (a b : ℝ) (k m : ℝ) (h1 : a = sqrt 3) (h2 : b = 1) (h3 : m^2 = 3/4 * (k^2 + 1)) :
  (let d := sqrt 3 / 2 in
   ∀ x1 y1 x2 y2 : ℝ, (dist (0,0) (line_through (x1,y1) (x2,y2)) = d) → |x1-x2| = 2 → 
   (maximum_area_triangle (x1, y1) (x2, y2) (0,0) = sqrt 3 / 2)) :=
by {
  sorry
}

end solution_part_1_solution_part_2_l768_768895


namespace arithmetic_sequence_solution_l768_768609

def a_n (n : ℕ) := 3 * n - 2
def b_n (n : ℕ) := 2 * a_n n

noncomputable def S (b : ℕ → ℤ) (n : ℕ) := 
  ∑ i in finset.range n, b i

theorem arithmetic_sequence_solution :
  let S_n := 6 * (1 / 2 : ℚ) * (a_n 1 + a_n 6) in
  S_n = 51 ∧ a_n 5 = 13 →
  (∀ n, a_n n = 3 * n - 2) ∧ (∀ n, S (λ n, b_n n) n = 27 * (8^n - 1)) :=
by {
  intro h,
  obtain ⟨h1, h2⟩ := h,
  sorry
}

end arithmetic_sequence_solution_l768_768609


namespace countMultiplesOf30Between900And27000_l768_768725

noncomputable def smallestPerfectSquareDivisibleBy30 : ℕ :=
  900

noncomputable def smallestPerfectCubeDivisibleBy30 : ℕ :=
  27000

theorem countMultiplesOf30Between900And27000 :
  let lower_bound := smallestPerfectSquareDivisibleBy30 / 30;
  let upper_bound := smallestPerfectCubeDivisibleBy30 / 30;
  upper_bound - lower_bound + 1 = 871 :=
  by
  let lower_bound := smallestPerfectSquareDivisibleBy30 / 30;
  let upper_bound := smallestPerfectCubeDivisibleBy30 / 30;
  show upper_bound - lower_bound + 1 = 871;
  sorry

end countMultiplesOf30Between900And27000_l768_768725


namespace highest_degree_p_has_horizontal_asymptote_l768_768816

-- Definition of the degree function for polynomials
def degree (p : Polynomial ℝ) : Nat := Polynomial.degree p

-- Definition of our specific polynomial g(x)
noncomputable def g : Polynomial ℝ := 3 * Polynomial.X^6 - 2 * Polynomial.X^3 + 5 * Polynomial.X - 9

-- Main statement
theorem highest_degree_p_has_horizontal_asymptote :
  ∃ (p : Polynomial ℝ), degree p ≤ degree g ∧ (degree p = 6) :=
sorry

end highest_degree_p_has_horizontal_asymptote_l768_768816


namespace plane_region_split_l768_768811

-- Define the condition that one coordinate is three times the other.
def condition1 (x y : ℝ) : Prop := y = 3 * x

-- Define the condition that the other coordinate is less than half times the first.
def condition2 (x y : ℝ) : Prop := x = (1 / 3) * y

-- Define the theorem that answers the question based on the conditions.
theorem plane_region_split : ∀ (x y : ℝ), condition1 x y ∨ condition2 x y → 1 := by
  intro x y h
  sorry

end plane_region_split_l768_768811


namespace trippy_div_by_11_l768_768760

def is_trippy (n : ℕ) : Prop := 
  ∀ i : ℕ, i < 3 → (n / 10^i) % 10 ≠ (n / 10^(i+1)) % 10 ∧ 
            (n / 10^i) % 10 ≠ (n / 10^(i+2)) % 10

def alternating_sum (n : ℕ) : ℕ := 
  let digits := List.map (λ i, (n / 10^i) % 10) [0, 1, 2, 3, 4, 5]
  List.sum (List.enumerate digits |>.map (λ (i, d), if i % 2 = 0 then d else -d) )

theorem trippy_div_by_11 (n : ℕ) : 
  is_trippy n ∧ alternating_sum n % 11 = 0 → (count six_digit_trippy_numbers n = 6) := 
sorry

end trippy_div_by_11_l768_768760


namespace decreasing_interval_of_sqrt_sinx_cosx_l768_768027

theorem decreasing_interval_of_sqrt_sinx_cosx (k : ℤ) :
  ∃ (k : ℤ), ∀ x ∈ set.Icc (k * π + π / 4) (k * π + π / 2), 
    derivative (λ x, sqrt (sin x * cos x)) x < 0 ∧ sqrt (sin x * cos x) ≥ 0 :=
by sorry

end decreasing_interval_of_sqrt_sinx_cosx_l768_768027


namespace terminal_side_of_neg_angle_400_l768_768602

def angle_in_quadrant (θ : ℝ) : string :=
  if 0 ≤ θ ∧ θ < 90 then "First quadrant"
  else if 90 ≤ θ ∧ θ < 180 then "Second quadrant"
  else if 180 ≤ θ ∧ θ < 270 then "Third quadrant"
  else if 270 ≤ θ ∧ θ < 360 then "Fourth quadrant"
  else "Out of range"

theorem terminal_side_of_neg_angle_400 : angle_in_quadrant (320 : ℝ) = "Fourth quadrant" :=
  by sorry

end terminal_side_of_neg_angle_400_l768_768602


namespace unique_root_in_interval_l768_768502

theorem unique_root_in_interval :
  ∀ (a : ℤ), (∃! x ∈ (Set.Icc (1 : ℝ) 8), (x - a - 4) ^ 2 + 2 * x - 2 * a - 16 = 0) ↔ 
  (a ∈ Set.Icc (-5 : ℤ) 0 ∨ a ∈ Set.Icc (3 : ℤ) 8) :=
begin
  sorry
end

end unique_root_in_interval_l768_768502


namespace common_tangent_l768_768939

theorem common_tangent (a : ℝ) :
  (∃ t : ℝ, t > 0 ∧ a * Real.log t = Real.sqrt t ∧ a / t = 1 / (2 * Real.sqrt t)) → a = Real.exp(1) / 2 :=
by
  intro h
  cases h with t ht
  have h1 : t = Real.exp(2),
  {
    sorry
  }
  have h2 : a = Real.sqrt t / 2,
  {
    sorry
  }
  rw [h1] at h2,
  rw [Real.sqrt_exp, Real.sqrt_sq, Real.exp_one_mul] at h2,
  exact h2

end common_tangent_l768_768939


namespace age_contradiction_problem_l768_768236

variables {A B C D : ℕ}

theorem age_contradiction_problem 
  (h1 : A + B = B + C + 11)
  (h2 : A + B + D = B + C + D + 8)
  (h3 : A + C = 2 * D) : False :=
by
  have h4 : A = C + 11 := by linarith
  have h5 : A = C + 8 := by linarith
  have h6 : C + 11 = C + 8 := by rw [h4, h5]
  have h7 : 11 = 8 := by linarith
  sorry

end age_contradiction_problem_l768_768236


namespace similarity_of_triangles_intersection_of_bisectors_l768_768423

-- Given definitions
variables (A B C A1 B1 C1 : Type) 
variables [triangle ABC] [triangle AB1C1] [triangle A1BC1] [triangle A1B1C]
variables (O Oa Ob Oc : Type) -- circumcenters
variables (H Ha Hb Hc : Type) -- orthocenters

-- Proof that △OaObOc ~ △ABC
theorem similarity_of_triangles :
  ∃ (P : Type), 
  is_miquel_point P A B C A1 B1 C1 →
  is_homothety P ABC OaObOc →
  triangle.similar ABC OaObOc := 
sorry

-- Proof that the perpendicular bisectors of OH, OaHa, ObHb, OcHc intersect
theorem intersection_of_bisectors :
  ∃ (M : Type), 
  is_common_point M OH OaHa ObHb OcHc :=
sorry

end similarity_of_triangles_intersection_of_bisectors_l768_768423


namespace find_Gary_gold_l768_768857

variable (G : ℕ) -- G represents the number of grams of gold Gary has.
variable (cost_Gary_gold_per_gram : ℕ) -- The cost per gram of Gary's gold.
variable (grams_Anna_gold : ℕ) -- The number of grams of gold Anna has.
variable (cost_Anna_gold_per_gram : ℕ) -- The cost per gram of Anna's gold.
variable (combined_cost : ℕ) -- The combined cost of both Gary's and Anna's gold.

theorem find_Gary_gold (h1 : cost_Gary_gold_per_gram = 15)
                       (h2 : grams_Anna_gold = 50)
                       (h3 : cost_Anna_gold_per_gram = 20)
                       (h4 : combined_cost = 1450)
                       (h5 : combined_cost = cost_Gary_gold_per_gram * G + grams_Anna_gold * cost_Anna_gold_per_gram) :
  G = 30 :=
by 
  sorry

end find_Gary_gold_l768_768857


namespace mod_pow_10_l768_768255

theorem mod_pow_10 (a : Fin 100 → ℕ) (ha : ∀ i, 0 < a i) :
  (9 ^ ∏ i in Finset.range 99, (a i + a (i + 1))) % 10 = 1 :=
sorry

end mod_pow_10_l768_768255


namespace fish_remaining_correct_l768_768210

def remaining_fish (jordan_caught : ℕ) (total_catch_lost_fraction : ℚ) : ℕ :=
  let perry_caught := 2 * jordan_caught
  let total_catch := jordan_caught + perry_caught
  let lost_catch := total_catch * total_catch_lost_fraction
  let remaining := total_catch - lost_catch
  remaining.nat_abs

theorem fish_remaining_correct : (remaining_fish 4 (1/4)) = 9 :=
by 
  sorry

end fish_remaining_correct_l768_768210


namespace number_of_prime_divisors_of_50_fac_l768_768137

-- Define the finite set of prime numbers up to 50
def primes_up_to_50 : finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

-- Main theorem statement
theorem number_of_prime_divisors_of_50_fac :
  (primes_up_to_50.filter prime).card = 15 := 
sorry

end number_of_prime_divisors_of_50_fac_l768_768137


namespace sin_half_alpha_l768_768514

theorem sin_half_alpha (α : ℝ) (h1 : cos α = 3 / 5) (h2 : α ∈ Ioo (3 / 2 * Real.pi) (2 * Real.pi)) :
  sin (α / 2) = sqrt(5) / 5 :=
by
  sorry

end sin_half_alpha_l768_768514


namespace remainder_product_mod_5_l768_768318

theorem remainder_product_mod_5 : (1657 * 2024 * 1953 * 1865) % 5 = 0 := by
  sorry

end remainder_product_mod_5_l768_768318


namespace sum_of_three_consecutive_integers_product_336_l768_768314

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l768_768314


namespace Isabel_earning_l768_768202

-- Define the number of bead necklaces sold
def bead_necklaces : ℕ := 3

-- Define the number of gem stone necklaces sold
def gemstone_necklaces : ℕ := 3

-- Define the cost of each necklace
def cost_per_necklace : ℕ := 6

-- Calculate the total number of necklaces sold
def total_necklaces : ℕ := bead_necklaces + gemstone_necklaces

-- Calculate the total earnings
def total_earnings : ℕ := total_necklaces * cost_per_necklace

-- Prove that the total earnings is 36 dollars
theorem Isabel_earning : total_earnings = 36 := by
  sorry

end Isabel_earning_l768_768202


namespace scaling_circle_not_hyperbola_l768_768345

-- Define a transformation function and what constitutes a valid circle and hyperbola
def is_circle (s : Set ℝ) : Prop :=
  ∃ c r, r > 0 ∧ s = { p : ℝ × ℝ | (p.1 - c.1) ^ 2 + (p.2 - c.2) ^ 2 = r ^ 2 }

def is_hyperbola (s : Set ℝ) : Prop :=
  ∃ a b c, a > 0 ∧ b > 0 ∧ s = { p : ℝ × ℝ | (p.1 / a) ^ 2 - (p.2 / b) ^ 2 = 1 }

def scale_transform (s : Set ℝ) (k : ℝ) : Set ℝ :=
  { p | ∃ q ∈ s, p.1 = k * q.1 ∧ p.2 = k * q.2 }

-- The Lean theorem statement
theorem scaling_circle_not_hyperbola (s : Set ℝ) (k : ℝ) (hk : k ≠ 0) :
  is_circle s → ¬ is_hyperbola (scale_transform s k) :=
by
  sorry

end scaling_circle_not_hyperbola_l768_768345


namespace regular_pyramid_theorem_l768_768252

-- Define the vertex, lateral edges, and base of the pyramid
variables {S O : Type} {A : ℕ → Type} [∀ n, metric_space (A n)]  
variables [metric_space S] [metric_space O]
variables {SA : ∀ n, metric_space (S × A n)}

-- Equally-lateral edges condition
def equal_lateral_edges (SA : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, i < n ∧ j < n → SA i = SA j

-- Equally-dihedral angles condition
def equal_dihedral_angles (angle : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, i < n ∧ j < n → angle i = angle j

-- Regular pyramid definition
def regular_pyramid (pyramid : Type) [htt : has_tilted_truncated_trapezoid pyramid] : Prop :=
  has_regular_base pyramid ∧ has_regular_lateral_faces pyramid

-- Statement of the problem in Lean 4
theorem regular_pyramid_theorem {n : ℕ} {pyramid : Type} [htt : has_tilted_truncated_trapezoid pyramid] 
  (SA : ℕ → ℝ) {angle : ℕ → ℝ} :
  equal_lateral_edges SA n → 
  equal_dihedral_angles angle n → 
  regular_pyramid pyramid :=
begin
  sorry -- Proof not required
end

end regular_pyramid_theorem_l768_768252


namespace min_area_of_triangle_l768_768737

-- Define the points A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (24, 10)

-- Define the area function using Shoelace formula
def area (p q : ℤ) : ℝ := (1 / 2) * |24 * q - 10 * p|

-- Prove that the minimum area is 1 given the integer coordinates condition
theorem min_area_of_triangle : (∃ p q : ℤ, area p q = 1) := 
sorry

end min_area_of_triangle_l768_768737


namespace arithmetic_mean_of_integers_from_neg6_to_7_l768_768705

theorem arithmetic_mean_of_integers_from_neg6_to_7 :
  let range := List.range' (-6) 14 in
  let mean := (range.sum : ℚ) / (range.length : ℚ) in
  mean = 0.5 := by
  -- Let's define the range of integers from -6 to 7 inclusive
  let range := List.range' (-6) 14
  -- Let the sum of this range be S
  let S : ℚ := (range.sum)
  -- Let the number of elements in this range be N
  let N : ℚ := (range.length)
  -- The mean of this range is S/N
  let mean := S / N
  -- We assert that this mean is equal to 0.5
  have h_correct_mean : S / N = 0.5 := sorry
  exact h_correct_mean

end arithmetic_mean_of_integers_from_neg6_to_7_l768_768705


namespace find_rs_value_l768_768161

theorem find_rs_value :
  let l := 1
  let w := 2
  let r := (1 : ℝ) + (1 / 2 : ℝ)
  let s := (1 / 1 : ℝ) * (1 / 2 : ℝ)
  rs = r * s :=
  by
  have roots_eq : (x : ℝ) -> (Polynomial).roots (x^2 - 3 * x + 2 : ℝ) = [l, w] := sorry
  have reciprocal_eq: (x : ℝ) -> (Polynomial).roots (x^2 - r * x + s : ℝ) = [1/l, 1/w] := sorry
  have rs_value : rs = 0.75 := by
    rw [r, s]
    norm_num
  exact rs_value

end find_rs_value_l768_768161


namespace find_a_values_l768_768909

variable (U : Set ℕ) (M : Set ℕ)
variable (a : ℕ)

axiom universe_def : U = {1, 3, 5, 7}
axiom set_def : M = {1, abs (a - 5)}
axiom complement_def : U \ M = {5, 7}

noncomputable def solve_a : Set ℕ :=
  { a : ℕ | {1, abs (a - 5)} = {1, 3} }

theorem find_a_values : solve_a = {2, 8} := sorry

end find_a_values_l768_768909


namespace five_S_n_minus_four_pow_n_a_n_eq_n_l768_768872

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom a1 : a 1 = 1
axiom recurrence_relation : ∀ n : ℕ, 0 < n → a n + a (n + 1) = (1 / 4)^n
axiom S_definition : ∀ n : ℕ, S n = ∑ i in Finset.range n, a (i + 1) * 4^i

theorem five_S_n_minus_four_pow_n_a_n_eq_n : ∀ n : ℕ, 0 < n → 5 * S n - 4^n * a n = n :=
by
  sorry

end five_S_n_minus_four_pow_n_a_n_eq_n_l768_768872


namespace tan_sin_sum_eq_sqrt3_l768_768324

theorem tan_sin_sum_eq_sqrt3 (tan20 sin20 : ℝ) (h1 : tan 20 = sin 20 / cos 20) (h2 : sin20 = sin 20) :
  tan20 + 4 * sin20 = sqrt 3 := by
  sorry

end tan_sin_sum_eq_sqrt3_l768_768324


namespace probability_endpoints_of_edge_l768_768366

noncomputable def num_vertices : ℕ := 12
noncomputable def edges_per_vertex : ℕ := 3

theorem probability_endpoints_of_edge :
  let total_ways := Nat.choose num_vertices 2,
      total_edges := (num_vertices * edges_per_vertex) / 2,
      probability := total_edges / total_ways in
  probability = 3 / 11 := by
  sorry

end probability_endpoints_of_edge_l768_768366


namespace sum_of_three_consecutive_integers_product_336_l768_768312

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l768_768312


namespace big_lots_sale_total_cost_l768_768792

variable (n : ℕ) (price : ℕ) (discount : ℕ) (additional_discount : ℕ)

def total_cost_of_chairs (num_chairs : ℕ) : ℕ :=
  let discounted_price := 20 - (20 * 25 / 100)
  let additional_discounted_price := discounted_price - (discounted_price * 1 / 3)
  if num_chairs ≤ 5 then
    num_chairs * discounted_price
  else
    (5 * discounted_price) + ((num_chairs - 5) * additional_discounted_price)

theorem big_lots_sale_total_cost (num_chairs : ℕ)
  (h1 : price = 20) (h2 : discount = price * 25 / 100) (h3 : additional_discount = (price - discount) * 1 / 3)
  (h4 : num_chairs = 8) :
  total_cost_of_chairs num_chairs = 105 :=
by
  rw [h4]
  simp [total_cost_of_chairs, h1, h2, h3]
  sorry

end big_lots_sale_total_cost_l768_768792


namespace average_tv_production_factory_tv_average_production_l768_768175

theorem average_tv_production (average_25_days : ℕ) (average_5_days : ℕ) (days_25 : ℕ) (days_5 : ℕ) : ℝ :=
  if (average_25_days = 50) ∧ (average_5_days = 20) ∧ (days_25 = 25) ∧ (days_5 = 5) then
    ((average_25_days * days_25 + average_5_days * days_5) / (days_25 + days_5) : ℝ)
  else
    0

theorem factory_tv_average_production 
  (average_25_days : ℕ) (average_5_days : ℕ) (days_25 : ℕ) (days_5 : ℕ) (total_days: ℕ) :
  average_tv_production average_25_days average_5_days days_25 days_5 = 45 :=
by 
  have h1 : average_25_days = 50 := rfl
  have h2 : average_5_days = 20 := rfl
  have h3 : days_25 = 25 := rfl
  have h4 : days_5 = 5 := rfl
  have h5 : total_days = 30 := rfl
  rw [h1, h2, h3, h4, h5]
  sorry

end average_tv_production_factory_tv_average_production_l768_768175


namespace no_rational_roots_l768_768494

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 2 * x^3 - 10 * x^2 + 4 * x + 1

theorem no_rational_roots : ∀ r : ℚ, polynomial r ≠ 0 :=
by
  intro r
  cases classical.em (polynomial r = 0) with prt h
  case intro:
    sorry

end no_rational_roots_l768_768494


namespace land_to_cabin_cost_ratio_l768_768425

theorem land_to_cabin_cost_ratio (total_cost cabin_cost land_cost : ℝ):
  total_cost = 30000 ∧ cabin_cost = 6000 ∧ land_cost = total_cost - cabin_cost -> 
  land_cost / cabin_cost = 4 :=
by
  intro h
  cases h with h_total h_rest
  cases h_rest with h_cabin h_land
  rw [h_total, h_cabin] at h_land
  rw [h_land, h_cabin]
  exact sorry

end land_to_cabin_cost_ratio_l768_768425


namespace area_of_eqn_l768_768379

theorem area_of_eqn (x y : ℝ) : (x^2 + y^2 = |x| + 2 * |y|) → 
  (enclosed_area x y = 5 * π / 4) :=
sorry

end area_of_eqn_l768_768379


namespace rectangle_probability_11_l768_768433

theorem rectangle_probability_11 (m n : ℕ) (h_rel_prime : Nat.coprime m n) 
(h_prob : (m : ℚ) / n = 5 / 6) : 
  m + n = 11 := by
sorry

end rectangle_probability_11_l768_768433


namespace cricketer_hit_two_sixes_l768_768417

/- Define the given conditions -/
def total_score : ℕ := 142
def boundaries : ℕ := 12
def running_percentage : ℝ := 57.74647887323944 / 100

/- Define the runs provided by boundaries -/
def runs_from_boundaries : ℕ := boundaries * 4

/- Define the runs made by running between the wickets (rounded to the nearest whole number) -/
def runs_from_running : ℕ := (total_score : ℝ * running_percentage).to_int

/- Define the runs made from sixes -/
def runs_from_sixes : ℕ := total_score - (runs_from_boundaries + runs_from_running)

/- Define the number of sixes -/
def number_of_sixes : ℕ := runs_from_sixes / 6

/- The theorem to prove -/
theorem cricketer_hit_two_sixes : number_of_sixes = 2 := by
  sorry

end cricketer_hit_two_sixes_l768_768417


namespace locus_of_p_ratio_distances_l768_768963

theorem locus_of_p_ratio_distances :
  (∀ (P : ℝ × ℝ), (dist P (1, 0) = (1 / 3) * abs (P.1 - 9)) →
  (P.1^2 / 9 + P.2^2 / 8 = 1)) :=
by
  sorry

end locus_of_p_ratio_distances_l768_768963


namespace proof_f_3_lt_e3_f_0_l768_768892

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := λ x, sorry

axiom differentiable_f : ∀ x, differentiable ℝ f

axiom condition1 : ∀ x, (x - 1) * (f'(x) - f(x)) > 0

axiom condition2 : ∀ x, f(2 - x) = f(x) * real.exp (2 - 2 * x)

theorem proof_f_3_lt_e3_f_0 : f(3) < real.exp 3 * f(0) :=
sorry

end proof_f_3_lt_e3_f_0_l768_768892


namespace proposition_p_and_not_q_l768_768570

theorem proposition_p_and_not_q (P Q : Prop) 
  (h1 : P ∨ Q) 
  (h2 : ¬ (P ∧ Q)) : (P ↔ ¬ Q) :=
sorry

end proposition_p_and_not_q_l768_768570


namespace num_valid_telephone_numbers_l768_768444

-- Define the set of available digits
def available_digits : finset ℕ := {3, 4, 5, 6, 7, 8, 9}

-- Define the distict and non-consecutive properties as predicates
def distinct_digits (l : list ℕ) : Prop := l.nodup

def non_consecutive (l : list ℕ) : Prop := ∀ (a b c : ℕ), (a ∈ l) → (b ∈ l) → (c ∈ l) → (abs (a - b) > 1) → (abs (b - c) > 1)

-- Define the valid telephone number formation
def valid_telephone_number (l : list ℕ) : Prop :=
  length l = 7 ∧ distinct_digits l ∧ non_consecutive l ∧ ∀ (d ∈ l), d ∈ available_digits

-- The main theorem to prove the number of such valid sequences is 16
theorem num_valid_telephone_numbers : finset.card (finset.filter valid_telephone_number (finset.powerset available_digits)) = 16 :=
  sorry

end num_valid_telephone_numbers_l768_768444


namespace player_A_wins_4_points_game_game_ends_after_5_points_l768_768274

def prob_A_winning_when_serving : ℚ := 2 / 3
def prob_A_winning_when_B_serving : ℚ := 1 / 4
def prob_A_winning_in_4_points : ℚ := 1 / 12
def prob_game_ending_after_5_points : ℚ := 19 / 216

theorem player_A_wins_4_points_game :
  (prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) = prob_A_winning_in_4_points := 
  sorry

theorem game_ends_after_5_points : 
  ((1 - prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) * 
  (1 - prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) * 
  (prob_A_winning_when_serving)) + 
  ((prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (1 - prob_A_winning_when_serving)) = 
  prob_game_ending_after_5_points :=
  sorry

end player_A_wins_4_points_game_game_ends_after_5_points_l768_768274


namespace cos_double_angle_l768_768042

theorem cos_double_angle (α : ℝ) : 
  sin ((π / 6) - α) = cos ((π / 6) + α) → cos (2 * α) = 0 :=
by
  sorry

end cos_double_angle_l768_768042


namespace slope_of_line_l768_768840

/-- 
Given points M(1, 2) and N(3, 4), prove that the slope of the line passing through these points is 1.
-/
theorem slope_of_line (x1 y1 x2 y2 : ℝ) (hM : x1 = 1 ∧ y1 = 2) (hN : x2 = 3 ∧ y2 = 4) : 
  (y2 - y1) / (x2 - x1) = 1 :=
by
  -- The proof is omitted here because only the statement is required.
  sorry

end slope_of_line_l768_768840


namespace solve_problem_l768_768075

noncomputable def problem (ω φ : ℝ) (k: ℤ) : Prop :=
  (ω > 0) ∧ (0 < φ ∧ φ ≤ π / 2) ∧
  (∀ x, (ω * x + φ) = k * π ↔ x = π / 6) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 6 → (f' (x) < 0))

theorem solve_problem (ω : ℝ) (φ : ℝ) (k: ℤ) [decidable (solve_problem ω φ k ) ] :
  problem ω φ k → ω = 3 :=
by
  sorry

end solve_problem_l768_768075


namespace part1_part2_l768_768886

-- Statement for Part (1)
theorem part1 (x : ℝ) (h : |2*x - 1| + |x - 2| > 2) : 
  x ∈ Set.Ioo ⊥ (1 / 3 : ℝ) ∪ Set.Ioo 1 ⊤ :=
sorry

-- Statement for Part (2)
theorem part2 (x m n : ℝ) (hm : m ≠ 0) (h : |m + n| + |m - n| ≥ |m| * (|2*x - 1| + |x - 2|)) : 
  x ∈ Set.Icc (1 / 3 : ℝ) 1 :=
sorry

end part1_part2_l768_768886


namespace multiply_metapolynomial_l768_768698

-- Definition of a metapolynomial given in the problem
def is_metapolynomial {k : Type} [CommRing k] (f : k → ℝ) : Prop :=
  ∃ (m n : ℕ) (P : ℕ → ℕ → (k → ℝ)), ∀ x, f x = max {i | 0 ≤ i ∧ i < m} (λ i, min {j | 0 ≤ j ∧ j < n} (λ j, P i j x))

-- Theorem statement that needs to be proven
theorem multiply_metapolynomial {k : Type} [CommRing k] {f g : k → ℝ} :
  is_metapolynomial f → is_metapolynomial g → is_metapolynomial (λ x, f x * g x) :=
begin
  sorry
end

end multiply_metapolynomial_l768_768698


namespace smallest_d0_l768_768033

theorem smallest_d0 (r : ℕ) (hr : r ≥ 3) : ∃ d₀, d₀ = 2^(r - 2) ∧ (7^d₀ ≡ 1 [MOD 2^r]) :=
by
  sorry

end smallest_d0_l768_768033


namespace num_four_digit_pos_integers_l768_768128

theorem num_four_digit_pos_integers : 
  ∃ n : ℕ, (n = 16) ∧ ∀ k : ℕ, (1000 ≤ k ∧ k < 10000 ∧ 
  ∀ d ∈ [k.digits 10], d = 3 ∨ d = 6) := sorry

end num_four_digit_pos_integers_l768_768128


namespace quadrilateral_area_5s_l768_768651

-- Definition of a convex quadrilateral and its extension properties
structure Quadrilateral :=
(ABCD : Bool) -- Placeholder for actual geometric interpretation
(area : ℝ)

-- Given conditions: convex quadrilateral with specified area, and the extension properties
variables {q : Quadrilateral} (s : ℝ) (A B C D A1 B1 C1 D1 : ℝ)

-- Given the extend segments are equal to respective sides
def extend_segments (q : Quadrilateral) : Prop :=
q.area = s ∧
BB1 = segment_length B B1 ∧
CC1 = segment_length C C1 ∧
DD1 = segment_length D D1 ∧
AA1 = segment_length A A1

-- Prove that the area of the new quadrilateral is 5s
theorem quadrilateral_area_5s (h : extend_segments q) : 
  area (Quadrilateral A1 B1 C1 D1) = 5 * s :=
sorry

end quadrilateral_area_5s_l768_768651


namespace repeating_decimal_as_fraction_l768_768486

-- Define the repeating decimal
def repeating_decimal_2_35 := 2 + (35 / 99 : ℚ)

-- Define the fraction form
def fraction_form := (233 / 99 : ℚ)

-- Theorem statement asserting the equivalence
theorem repeating_decimal_as_fraction : repeating_decimal_2_35 = fraction_form :=
by 
  -- Skipped proof
  sorry

end repeating_decimal_as_fraction_l768_768486


namespace find_BC_length_l768_768938

noncomputable def area_of_triangle (A B C : ℝ) (sin_angle : ℝ) : ℝ :=
  (1 / 2) * A * B * sin_angle

theorem find_BC_length {BC : ℝ} :
  (∃ sinA : ℝ, 
    area_of_triangle 5 8 sinA = 10 * Real.sqrt 3 ∧ 
    BC = Real.sqrt (5^2 + 8^2 - 2 * 5 * 8 * (Real.cos (Real.arcsin sinA)))) → 
  BC = 7 :=
by
  sorry

end find_BC_length_l768_768938


namespace ferry_problem_l768_768419

-- Define the given conditions
variables (speed_still current_speed distance : Real)

-- The ferry travels at 12 km/h in still water (speed_still = 12)
-- It takes 10 hours against the current (distance / (speed_still - current_speed) = 10)
-- It takes 6 hours with the current (distance / (speed_still + current_speed) = 6)
axiom conditions : speed_still = 12 ∧ distance / (speed_still - current_speed) = 10 ∧ distance / (speed_still + current_speed) = 6

-- Prove that the speed of the water current is 3 km/h and the distance between the two docks is 90 km
theorem ferry_problem : conditions → current_speed = 3 ∧ distance = 90 :=
by
  -- this is a placeholder for the actual proof
  sorry

end ferry_problem_l768_768419


namespace factorize_expression_l768_768828

theorem factorize_expression (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := 
by 
  sorry

end factorize_expression_l768_768828


namespace num_four_digit_36_combinations_l768_768101

theorem num_four_digit_36_combinations : 
  (∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
  (∀ d ∈ [digit n 1000, digit n 100, digit n 10, digit n 1], d = 3 ∨ d = 6)) → 
  16 :=
sorry

end num_four_digit_36_combinations_l768_768101


namespace problem_statement_l768_768935

theorem problem_statement (x : ℤ) (h : 81^6 = 27^(x+2)) : 3^(-x) = 1/729 := 
by {
  sorry
}

end problem_statement_l768_768935


namespace math_problem_l768_768595

-- Define the polar curve C
def curve_C (rho theta : ℝ) : Prop :=
  rho = 2 * real.cos theta

-- Define the parametric line l
def line_l (x y t : ℝ) : Prop :=
  (x = real.sqrt 3 * t) ∧ (y = -1 + t)

-- Define the point P in polar coordinates and its conversion to rectangular coordinates
def point_P_polar : ℝ × ℝ :=
  (1, 3 * real.pi / 2)

def point_P_rectangular (P : ℝ × ℝ) : Prop :=
  let x := P.1 * real.cos P.2 in
  let y := P.1 * real.sin P.2 in
  (x = 0) ∧ (y = -1)

-- Define the rectangular equation of curve C
def curve_C_rect (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * x

-- Define the general equation of line l
def line_l_general (x y : ℝ) : Prop :=
  x - real.sqrt 3 * y - real.sqrt 3 = 0

-- Define the distance calculation |PA| and |PB|
def distance_PA (P A : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

def distance_PB (P B : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)

-- Define the value calculation
def value (PA PB : ℝ) : ℝ :=
  (PA + 1) * (PB + 1)

-- Main theorem
theorem math_problem :
  ∀ t P A B : ℝ × ℝ,
  point_P_rectangular P →
  curve_C_rect A.1 A.2 ∧ line_l_general A.1 A.2 →
  curve_C_rect B.1 B.2 ∧ line_l_general B.1 B.2 →
  value (distance_PA P A) (distance_PB P B) = 3 + real.sqrt 3 :=
by
  sorry

end math_problem_l768_768595


namespace extra_men_needed_l768_768393

theorem extra_men_needed (total_length : ℝ) (total_days : ℕ) (initial_men : ℕ) (completed_length : ℝ) (days_passed : ℕ) 
  (remaining_length := total_length - completed_length)
  (remaining_days := total_days - days_passed)
  (current_rate := completed_length / days_passed)
  (required_rate := remaining_length / remaining_days)
  (rate_increase := required_rate / current_rate)
  (total_men_needed := initial_men * rate_increase)
  (extra_men_needed := ⌈total_men_needed⌉ - initial_men) :
  total_length = 15 → 
  total_days = 300 → 
  initial_men = 35 → 
  completed_length = 2.5 → 
  days_passed = 100 → 
  extra_men_needed = 53 :=
by
-- Prove that given the conditions, the number of extra men needed is 53
sorry

end extra_men_needed_l768_768393


namespace sum_of_three_consecutive_integers_product_336_l768_768313

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l768_768313


namespace card_at_73rd_is_8_l768_768003

-- Definition of the repeating pattern sequence
def card_sequence : List String := 
  ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", 
   "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]

-- Function to get the nth card in the repeating sequence
def nth_card (n : Nat) : String :=
  card_sequence[(n % 26)]

theorem card_at_73rd_is_8 : nth_card 73 = "8" := 
  by sorry  -- Proof omitted

end card_at_73rd_is_8_l768_768003


namespace terminal_side_of_neg_angle_in_second_quadrant_l768_768889

theorem terminal_side_of_neg_angle_in_second_quadrant (α : ℝ) (k : ℤ) :
  k * 360 + 180 < α ∧ α < k * 360 + 270 → -(k * 360 + 270) < -α ∧ -α < -(k * 360 + 180) :=
by
  intro h
  have hk1 : -(k * 360 + 270) = -k * 360 - 270, by linarith
  have hk2 : -(k * 360 + 180) = -k * 360 - 180, by linarith
  rw [hk1, hk2]
  exact h

end terminal_side_of_neg_angle_in_second_quadrant_l768_768889


namespace simplify_fraction_l768_768798

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) : (3 * m^3) / (6 * m^2) = m / 2 :=
by
  sorry

end simplify_fraction_l768_768798


namespace compare_segments_l768_768248

variables {A B C M N K : Point}
variables {AB AC AN AM MK MB : Length}

-- Definitions based on conditions
def is_isosceles_triangle (A B C : Point) : Prop :=
  length A B = length A C

def points_on_sides (M N : Point) (A B C : Point) : Prop :=
  on_line_segment M A B ∧ on_line_segment N A C

def an_greater_than_am (AN AM : Length) : Prop :=
  AN > AM

def lines_intersect_at_K (MN BC : Line) (K : Point) : Prop :=
  intersect MN BC K

-- Problem statement
theorem compare_segments (is_ABC_isosceles : is_isosceles_triangle A B C) 
    (MN_on_sides : points_on_sides M N A B C) 
    (AN_gt_AM : an_greater_than_am (length A N) (length A M))
    (MN_BC_intersect : lines_intersect_at_K (line_through M N) (line_through B C) K) :
    length M K > length M B :=
sorry

end compare_segments_l768_768248


namespace consecutive_integers_sum_l768_768295

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l768_768295


namespace sum_of_solutions_eq_sqrt_3_l768_768219

theorem sum_of_solutions_eq_sqrt_3 :
  let T := ∑ x : ℝ in { x | 0 < x ∧ x ^ (3 ^ real.sqrt 3) = (real.sqrt 3) ^ (3 ^ x) }, x
  in T = real.sqrt 3 :=
by
  let T := ∑ x : ℝ in { x | 0 < x ∧ x ^ (3 ^ real.sqrt 3) = (real.sqrt 3) ^ (3 ^ x) }, x
  show T = real.sqrt 3
  sorry

end sum_of_solutions_eq_sqrt_3_l768_768219


namespace find_c_and_d_l768_768838

theorem find_c_and_d : 
  ∃ (c d : ℝ), 
  (∀ p q : ℝ, (p ≠ q) → 
   (∀ x : ℝ, (x = p ∨ x = q) → 
    (x^3 + c * x^2 + 7 * x + 4 = 0 ∧ x^3 + d * x^2 + 10 * x + 6 = 0)) ∧ 
   (c - d = 1) ∧ 
   (3 * c - 2 * d = -3)) ∧
  (c = -5) ∧ (d = -6) := 
by
  exists (-5) (-6)
  intros p q hpq hx
  sorry

end find_c_and_d_l768_768838


namespace extremum_value_of_a_range_of_b_for_distinct_roots_l768_768079

theorem extremum_value_of_a (a : ℝ) :
  (∀ x < 0, deriv (λ x, Real.log (-x) + a * x - 1 / x) x = 0 → x = -1) → a = 0 :=
sorry

theorem range_of_b_for_distinct_roots (b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (λ x, Real.log x + 2 * x + 1 / x - b) x₁ = 0 ∧ (λ x, Real.log x + 2 * x + 1 / x - b) x₂ = 0) → b ∈ set.Ioi (3 - Real.log 2) :=
sorry

end extremum_value_of_a_range_of_b_for_distinct_roots_l768_768079


namespace probability_two_vertices_endpoints_l768_768355

theorem probability_two_vertices_endpoints (V E : Type) [Fintype V] [DecidableEq V] 
  (dodecahedron : Graph V E) (h1 : Fintype.card V = 20)
  (h2 : ∀ v : V, Fintype.card (dodecahedron.neighbors v) = 3)
  (h3 : Fintype.card E = 30) :
  (∃ A B : V, A ≠ B ∧ (A, B) ∈ dodecahedron.edgeSet) → 
  (∃ p : ℚ, p = 3/19) := 
sorry

end probability_two_vertices_endpoints_l768_768355


namespace roger_has_more_candy_l768_768261

-- Definitions from the conditions
def sandra_bags := 2
def pieces_per_sandra_bag := 6
def roger_bag_1 := 11
def roger_bag_2 := 3

-- Total pieces of candy Sandra had
def total_sandra := sandra_bags * pieces_per_sandra_bag

-- Total pieces of candy Roger had
def total_roger := roger_bag_1 + roger_bag_2

-- Statement to prove: How much more candy did Roger have than Sandra?
theorem roger_has_more_candy : total_roger - total_sandra = 2 :=
by
  have h1 : total_sandra = sandra_bags * pieces_per_sandra_bag, from rfl
  have h2 : total_roger = roger_bag_1 + roger_bag_2, from rfl
  rw [h1, h2]
  sorry

end roger_has_more_candy_l768_768261


namespace mary_total_payment_l768_768648

def fixed_fee : ℕ := 17
def hourly_charge : ℕ := 7
def rental_duration : ℕ := 9
def total_payment (f : ℕ) (h : ℕ) (r : ℕ) : ℕ := f + (h * r)

theorem mary_total_payment:
  total_payment fixed_fee hourly_charge rental_duration = 80 :=
by
  sorry

end mary_total_payment_l768_768648


namespace brooke_homework_time_l768_768510

/-- Define the problem parameters -/
def math_problem_count := 15
def ss_problem_count := 6
def science_problem_count := 10
def math_time_per_problem := 2
def ss_time_per_problem := 0.5
def science_time_per_problem := 1.5
def break_after_math := 5
def break_after_ss := 10
def break_after_science := 15

/-- Define the total time spent -/
def total_problem_solving_time :=
  (math_problem_count * math_time_per_problem) +
  (ss_problem_count * ss_time_per_problem) +
  (science_problem_count * science_time_per_problem)

def total_break_time :=
  break_after_math + break_after_ss + break_after_science

def total_time_spent :=
  total_problem_solving_time + total_break_time

/-- The theorem stating the total time spent equals 78 minutes -/
theorem brooke_homework_time : total_time_spent = 78 := by
  sorry

end brooke_homework_time_l768_768510


namespace prime_quadruples_l768_768468

theorem prime_quadruples (p q r : ℕ) (n : ℕ) 
  (hp : p.prime) (hq : q.prime) (hr : r.prime) (hn : 0 < n) :
  p^2 = q^2 + r^n ↔ (p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4) := by
  sorry

end prime_quadruples_l768_768468


namespace slope_line_divides_equal_areas_l768_768270

theorem slope_line_divides_equal_areas (c : ℝ) : 
  (Σ (c : ℝ), 6 = 2 * (4 - c) → c = 1 → 3 = 3) :=
by
  use 1,
  intros,
  sorry

end slope_line_divides_equal_areas_l768_768270


namespace f_1999_eq_l768_768734

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

noncomputable def f_n : ℤ → (ℝ → ℝ)
| 0       := id
| (n + 1) := f ∘ f_n n

theorem f_1999_eq : ∀ x : ℝ, f_n 1999 x = (x - 1) / (x + 1) :=
by sorry

end f_1999_eq_l768_768734


namespace num_prime_divisors_of_50_fac_l768_768140

-- Define the set of all prime numbers less than or equal to 50.
def primes_le_50 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

-- Define the factorial function.
noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the number of prime divisors of n.
noncomputable def num_prime_divisors (n : ℕ) : ℕ :=
(set.count (λ p, p ∣ n) primes_le_50)

-- The theorem statement.
theorem num_prime_divisors_of_50_fac : num_prime_divisors (factorial 50) = 15 :=
by
  sorry

end num_prime_divisors_of_50_fac_l768_768140


namespace problem1_cossine_problem2_sides_l768_768946

variables {A B C a b c S : ℝ}

def conditions (A B C a b c : ℝ) :=
  ∃ (A B C a b c : ℝ),
    ∀ (m n : ℝ × ℝ),
      m = (Real.cos B, Real.cos C) ∧
      n = (4 * a - b, c) ∧
      (∃ k : ℝ, m = (k * (4 * a - b), k * c)) ∧
      c = Real.sqrt 3 ∧
      S = Real.sqrt 15 / 4

theorem problem1_cossine {A B C a b c S : ℝ} (h : conditions A B C a b c) : 
  Real.cos C = 1 / 4 := sorry

theorem problem2_sides {A B C a b c S : ℝ} (h : conditions A B C a b c) 
  (h_cosC : Real.cos C = 1 / 4) :
  a = Real.sqrt 2 ∧ b = Real.sqrt 2 := sorry

end problem1_cossine_problem2_sides_l768_768946


namespace measure_angle_44_45_43_l768_768272

noncomputable def isosceles_right_triangle (A1 A2 A3 : Point) : Prop :=
  angle A1 A2 A3 = 45 ∧ angle A1 A3 A2 = 45

noncomputable def midpoint (A B C : Point) : Prop :=
  C = mid_point A B

noncomputable def rotated_triangle (A B C : Point) (n : ℕ) : Prop :=
  ∀ n : ℕ, ∃ A' B' C' : Point, 
    rotated45_clockwise A B C A' B' C' ∧ midpoint A B C ∧ midpoint B C A' ∧ midpoint C A B'

theorem measure_angle_44_45_43 
  (A : Point) 
  (isosceles : isosceles_right_triangle A1 A2 A3) 
  (midpoint_def : ∀ n, midpoint (A n) (A (n + 1)) (A (n + 3))) 
  (rotation_def : rotated_triangle A1 A2 A3 44) : 
  measure_angle (A 44) (A 45) (A 43) = 45 := 
sorry

end measure_angle_44_45_43_l768_768272


namespace sum_of_three_consecutive_integers_product_336_l768_768309

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l768_768309


namespace minimum_value_expression_l768_768227

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x * y * z) ≥ 343 :=
sorry

end minimum_value_expression_l768_768227


namespace max_pairing_distance_l768_768402

-- Given 1996 points on a line, half colored red and half colored blue,
-- prove the maximum sum of distances in an optimal red-blue pairing is 999004.
theorem max_pairing_distance :
  ∀ (points : Finset ℕ) (red_points blue_points : Finset ℕ),
    points.card = 1996 →
    red_points.card = 998 →
    blue_points.card = 998 →
    red_points ∪ blue_points = points →
    red_points ∩ blue_points = ∅ →
    ∃ (pairs : Finset (ℕ × ℕ)), 
      (∀ p ∈ pairs, p.1 ∈ red_points ∧ p.2 ∈ blue_points) ∧
      (∀ q ∈ points, ∃ p ∈ pairs, q = p.1 ∨ q = p.2) ∧
      (∑ p in pairs, abs (p.1 - p.2) = 999004) :=
by
  sorry

end max_pairing_distance_l768_768402


namespace number_of_relatives_Shiela_paintings_l768_768449

section
variable (total_paintings : ℕ)
variable (paintings_per_relative : ℕ)
variable (num_relatives : ℕ)

-- Given conditions
def condition1 : total_paintings = 18 := sorry
def condition2 : paintings_per_relative = 9 := sorry

-- Proof statement
theorem number_of_relatives : num_relatives = total_paintings / paintings_per_relative :=
sorry

-- Checking the specific instance
theorem Shiela_paintings :
  num_relatives = 2 :=
by
  have h1 : total_paintings = 18 := condition1
  have h2 : paintings_per_relative = 9 := condition2
  have h3 : 18 / 9 = 2 := by norm_num
  exact h3
end

end number_of_relatives_Shiela_paintings_l768_768449


namespace F2E1_base16_to_base10_l768_768465

theorem F2E1_base16_to_base10 :
  let F := 15,
      E := 14,
      two := 2,
      one := 1
  in 15 * 16^3 + 2 * 16^2 + 14 * 16^1 + 1 * 16^0 = 62177 :=
by
  have hF : F = 15 := rfl
  have hE : E = 14 := rfl
  have h2 : two = 2 := rfl
  have h1 : one = 1 := rfl
  calc
    15 * 16^3 + 2 * 16^2 + 14 * 16^1 + 1 * 16^0
        = 61440 + 2 * 256 + 14 * 16 + 1 * 1 : by rw [pow_succ, pow_succ, pow_succ, pow_zero, pow_add, pow_add, pow_add, hF, h2, hE, h1]
    ... = 61440 + 512 + 224 + 1 : by norm_num
    ... = 62177 : by norm_num

end F2E1_base16_to_base10_l768_768465


namespace sum_of_consecutive_integers_product_336_l768_768303

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l768_768303


namespace quadratic_y_real_iff_x_bounds_l768_768470

theorem quadratic_y_real_iff_x_bounds (x y : ℝ) (h : 3 * y^2 + 5 * x * y - 2 * x + 8 = 0) : 
  y ∈ ℝ ↔ x ≤ -2.4 ∨ x ≥ 1.6 :=
sorry

end quadratic_y_real_iff_x_bounds_l768_768470


namespace boy_sleep_duration_l768_768409

def sleep_duration (bedtime alarmtime : Nat) : Nat :=
  if alarmtime < bedtime then
    12 - bedtime + alarmtime
  else
    alarmtime - bedtime

theorem boy_sleep_duration : sleep_duration 7 9 = 2 :=
by
  unfold sleep_duration
  simp
  sorry

end boy_sleep_duration_l768_768409


namespace p_congruent_1_mod_5_infinitely_many_primes_of_form_5n_plus_1_l768_768627

-- Given condition: p is a prime number, p > 5
variable (p : ℕ)
variable (hp_prime : Nat.Prime p)
variable (hp_gt5 : p > 5)

-- Assume that x^4 + x^3 + x^2 + x + 1 ≡ 0 (mod p) is solvable
variable (x : ℕ)
variable (hx_solution : (x^4 + x^3 + x^2 + x + 1) % p = 0)

-- Prove that p ≡ 1 (mod 5)
theorem p_congruent_1_mod_5 : p % 5 = 1 := by
  sorry

-- Infer that there are infinitely many primes of the form 5n + 1
theorem infinitely_many_primes_of_form_5n_plus_1 :
  ∃ f : ℕ → ℕ, (∀ n, Nat.Prime (f n) ∧ f n = 5 * n + 1) :=
  by
  sorry

end p_congruent_1_mod_5_infinitely_many_primes_of_form_5n_plus_1_l768_768627


namespace min_value_y_l768_768986

theorem min_value_y (x y : ℝ) (h : x^2 + y^2 = 14 * x + 48 * y) : y = -1 := 
sorry

end min_value_y_l768_768986


namespace num_four_digit_36_combinations_l768_768100

theorem num_four_digit_36_combinations : 
  (∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
  (∀ d ∈ [digit n 1000, digit n 100, digit n 10, digit n 1], d = 3 ∨ d = 6)) → 
  16 :=
sorry

end num_four_digit_36_combinations_l768_768100


namespace relationship_between_a_and_b_l768_768992

theorem relationship_between_a_and_b 
  (x a b : ℝ)
  (hx : 0 < x)
  (ha : 0 < a)
  (hb : 0 < b)
  (hax : a^x < b^x) 
  (hbx : b^x < 1) : 
  a < b ∧ b < 1 := 
sorry

end relationship_between_a_and_b_l768_768992


namespace find_ordered_pair_exists_l768_768837

theorem find_ordered_pair_exists : 
  ∃ (a b : ℤ), 
    (a : ℝ) + (b : ℝ) * real.csc (real.pi / 3) = real.sqrt (16 - 12 * real.sin (real.pi / 3)) :=
by
  use [4, -√3]
  sorry

end find_ordered_pair_exists_l768_768837


namespace true_propositions_l768_768914

theorem true_propositions :
  (∀ a, (∀ x, x^2 + |x - a| = (−x)^2 + |−x - a| → a = 0)) → 
  (∀ m, m ∈ (set.Ioi 0) → (¬ ∃ x, m * x^2 - 2 * x + 1 = 0)) → 
  (∀ p q: Prop, 
    (p ∨ q) → ¬(p ∧ q) → ¬ (¬p ∧ q) → (¬p ∨ ¬q) →
    ((p ∨ q) ∧ (¬ (p ∧ q)) ∧ (¬ (¬p ∧ q)) ∧ (¬p ∨ ¬q))) :=
by 
  intros hp hq p q hpq hnppq hnnpq hnnpnq
  exact ⟨hpq, hnppq, hnnpq, hnnpnq⟩

end true_propositions_l768_768914


namespace find_radius_of_circle_l768_768695

theorem find_radius_of_circle {
    (a : ℝ) (R : ℝ) :
    a * a = 256 ∧  -- Condition 1: Area of the square
    (∃ O : point, ∀ A B : point, A ∈ circle O R ∧ B ∈ circle O R) ∧ -- Condition 2: Vertices on circle
    (∃ E F : point, tangent_to_circle E F O R) -- Condition 3: Vertices on tangent
    ↔ R = 10 :=
begin
  sorry
end

end find_radius_of_circle_l768_768695


namespace sum_floor_inequality_l768_768231

open BigOperators

noncomputable def ceil (x : ℝ) := x.toFloor
notation "⌊" x "⌋" => ceil x

theorem sum_floor_inequality (n : ℕ) (x : ℝ) (h1 : 0 < x) (h2 : 0 < n) 
  : ∑ k in Finset.range (n + 1), (x * ⌊k / x⌋ - (x + 1) * ⌊k / (x + 1)⌋) ≤ n :=
by
  sorry

end sum_floor_inequality_l768_768231


namespace find_a_l768_768639

-- Definitions of the complex numbers and the condition
def z1 : ℂ := 2 - complex.I
def z2 (a : ℝ) : ℂ := a + 2 * complex.I

-- Statement of the problem translated to Lean
theorem find_a (a : ℝ) (h : (z1 * z2 a).im = 0) : a = 4 :=
by sorry

end find_a_l768_768639


namespace simplify_expression_l768_768665

theorem simplify_expression (x : ℝ) :
  x * (4 * x^3 - 3 * x + 2) - 6 * (2 * x^3 + x^2 - 3 * x + 4) = 4 * x^4 - 12 * x^3 - 9 * x^2 + 20 * x - 24 :=
by sorry

end simplify_expression_l768_768665


namespace sum_of_integer_solutions_l768_768715

theorem sum_of_integer_solutions :
  let S := {n : ℤ | abs n > abs (n - 3) ∧ abs (n - 3) < 9}
  ∑ n in S, n = 65 :=
by
  sorry

end sum_of_integer_solutions_l768_768715


namespace minimum_value_at_three_l768_768819

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimum_value_at_three :
  ∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≥ f x := 
begin
  use 3,
  unfold f,
  split,
  {
    norm_num,
  },
  {
    intros y,
    unfold f,
    by_contradiction,
    sorry
  }
end

end minimum_value_at_three_l768_768819


namespace limit_sequence_equals_7_div_2_l768_768795

noncomputable def sequence_limit : Real :=
  \lim (n : ℕ) (1 / Real.to_nnreal (n)) * (Real.sqrt (n + 2) * (Real.sqrt (n + 3) - Real.sqrt (n - 4)))

theorem limit_sequence_equals_7_div_2 :
  sequence_limit = 7 / 2 := 
by
  sorry

end limit_sequence_equals_7_div_2_l768_768795


namespace sum_binomial_coefficient_identity_l768_768658

theorem sum_binomial_coefficient_identity
  (m n : ℕ) (h : m ≥ 2 * n) :
  (∑ k in range (n + 1), (nat.choose (m + 1) (2 * k + 1)) * (nat.choose (m - 2 * k) (n - k)) * 2 ^ (2 * k + 1))
  = (nat.choose (2 * m + 2) (2 * n + 1)) :=
sorry

end sum_binomial_coefficient_identity_l768_768658


namespace derivative_of_f_l768_768023

-- Define the function f(x)
def f (x : ℝ) : ℝ := (Real.sin (1/x)) ^ 3

-- State the theorem about the derivative of f(x)
theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : deriv f x = - (3 / x^2) * (Real.sin (1/x))^2 * Real.cos (1/x) :=
by
  -- The proof will be filled in by the user
  sorry

end derivative_of_f_l768_768023


namespace num_four_digit_36_combinations_l768_768103

theorem num_four_digit_36_combinations : 
  (∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
  (∀ d ∈ [digit n 1000, digit n 100, digit n 10, digit n 1], d = 3 ∨ d = 6)) → 
  16 :=
sorry

end num_four_digit_36_combinations_l768_768103


namespace distance_between_lines_l768_768281

noncomputable def distance_between_parallel_lines
  (a b m n : ℝ) : ℝ :=
  |m - n| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines
  (a b m n : ℝ) :
  distance_between_parallel_lines a b m n = 
  |m - n| / Real.sqrt (a^2 + b^2) :=
by
  sorry

end distance_between_lines_l768_768281


namespace number_of_special_four_digit_integers_l768_768123

theorem number_of_special_four_digit_integers : 
  let digits := [3, 6]
  let choices_per_digit := 2
  num_digits = 4
  ∑ i in range(num_digits), (choices_per_digit) = 2^4 :=
by
  sorry

end number_of_special_four_digit_integers_l768_768123


namespace trig_expr_eq_l768_768456

theorem trig_expr_eq : 
  sin (4/3 * π) * cos (5/6 * π) * tan (-4/3 * π) = - (3 * sqrt 3) / 4 :=
by sorry

end trig_expr_eq_l768_768456


namespace no_extremum_in_interval_l768_768940

theorem no_extremum_in_interval (a : ℝ) : 
  (∀ x ∈ Ioo (0:ℝ) 1, (3 * x^2 - 2 * a) ≠ 0) →
  a ∈ Iic 0 ∨ a ∈ Ici (3 / 2) :=
by sorry

end no_extremum_in_interval_l768_768940


namespace problem_solution_l768_768955

axiom {a_n : ℕ → ℝ}
axiom {b_n : ℕ → ℝ}
axiom {S_n : ℕ → ℝ}
axiom {d : ℝ}
axiom {a_1 : ℝ}

noncomputable def problem_conditions : Prop :=
  a_1 = -3 ∧
  (∀ n, a_n = a_1 + (n - 1) * d) ∧
  (∀ a_3 a_4 a_7, a_4 = (a_3 * a_7) ^ (1/2)) ∧
  (∀ n, S_n = 2 * b_n - 1)

theorem problem_solution :
  problem_conditions →
  (∀ n, a_n = 2 * n - 5) ∧
  (∀ n, b_n = 2 ^ (n - 1)) ∧
  ((Σ_ n' in finset.range n, a_n / b_n) = -2 - (2 * n - 1) * (1 / 2) ^ (n - 1)) :=
sorry

end problem_solution_l768_768955


namespace problem_a_problem_b_l768_768650

noncomputable def A : Type := sorry
def A1 : A := sorry
def A2 : A := sorry
def A3 : A := sorry
def A4 : A := sorry
def S : Set A := sorry
def H1 : A := sorry
def H2 : A := sorry
def H3 : A := sorry
def H4 : A := sorry

theorem problem_a (h_conditions : ∀ P ∈ {A1, A2, A3, A4}, A ∈ S) :
  ∃ H : A, ∀ P ∈ {H1, H2, H3, H4}, symmetric H P (A1, A2, A3, A4) :=
sorry

theorem problem_b (h_conditions : ∀ P ∈ {A1, A2, A3, A4}, A ∈ S) :
  ∀ (set : set (Set A)) (elem ∈ { {A1, A2, H3, H4}, {A1, A3, H2, H4}, {A1, A4, H2, H3},
    {A2, A3, H1, H4}, {A2, A4, H1, H3}, {A3, A4, H1, H2}, {H1, H2, H3, H4} }), 
  ∃ (S' : Set A), set elem ⊆ S' ∧ ∀ elem1 ∈ S, elem2 ∈ S', S = S' :=
sorry

end problem_a_problem_b_l768_768650


namespace repeating_decimal_as_fraction_l768_768483

noncomputable def repeating_decimal_value : ℚ :=
  let x := 2.35 + 35 / 99 in
  x

theorem repeating_decimal_as_fraction :
  repeating_decimal_value = 233 / 99 := by
  sorry

end repeating_decimal_as_fraction_l768_768483


namespace print_shop_cost_difference_l768_768849

theorem print_shop_cost_difference :
  let cost_per_copy_X := 1.25
  let cost_per_copy_Y := 2.75
  let num_copies := 40
  let total_cost_X := cost_per_copy_X * num_copies
  let total_cost_Y := cost_per_copy_Y * num_copies
  total_cost_Y - total_cost_X = 60 :=
by 
  dsimp only []
  sorry

end print_shop_cost_difference_l768_768849


namespace consecutive_integers_sum_l768_768294

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l768_768294


namespace four_digit_3_or_6_l768_768117

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end four_digit_3_or_6_l768_768117


namespace problem1_problem2_l768_768457

-- Proof Problem 1:

theorem problem1 : (5 / 3) ^ 2004 * (3 / 5) ^ 2003 = 5 / 3 := by
  sorry

-- Proof Problem 2:

theorem problem2 (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end problem1_problem2_l768_768457


namespace ladder_width_l768_768584

-- Define the variables and constants
variables (x y w : ℝ)

-- Define the conditions based on the problem statement
def condition_60_deg : Prop := w = x / (Real.sqrt 3)
def condition_45_deg : Prop := w = y

-- State the theorem to prove
theorem ladder_width (hx : condition_60_deg x y w) (hy : condition_45_deg x y w) : w = y :=
by sorry

end ladder_width_l768_768584


namespace sum_of_values_divisible_by_11_l768_768146

theorem sum_of_values_divisible_by_11 :
  (∑ (x : Finset ℕ) in (Finset.filter (λ x : ℕ, (∃ A B C : ℕ, 
                                                  0 ≤ A ∧ A ≤ 9 ∧
                                                  0 ≤ B ∧ B ≤ 9 ∧
                                                  0 ≤ C ∧ C ≤ 9 ∧
                                                  A + 5 + B + 7 + 9 + C = x ∧
                                                  (A + B - C - 3) % 11 = 0)), 
                             (Finset.range 30)), id) = 29 := 
by
  sorry

end sum_of_values_divisible_by_11_l768_768146


namespace big_lots_sale_total_cost_l768_768791

variable (n : ℕ) (price : ℕ) (discount : ℕ) (additional_discount : ℕ)

def total_cost_of_chairs (num_chairs : ℕ) : ℕ :=
  let discounted_price := 20 - (20 * 25 / 100)
  let additional_discounted_price := discounted_price - (discounted_price * 1 / 3)
  if num_chairs ≤ 5 then
    num_chairs * discounted_price
  else
    (5 * discounted_price) + ((num_chairs - 5) * additional_discounted_price)

theorem big_lots_sale_total_cost (num_chairs : ℕ)
  (h1 : price = 20) (h2 : discount = price * 25 / 100) (h3 : additional_discount = (price - discount) * 1 / 3)
  (h4 : num_chairs = 8) :
  total_cost_of_chairs num_chairs = 105 :=
by
  rw [h4]
  simp [total_cost_of_chairs, h1, h2, h3]
  sorry

end big_lots_sale_total_cost_l768_768791


namespace smallest_distance_parabola_l768_768446

theorem smallest_distance_parabola :
  let P1 := 1 / 2,
      P2 := 1,
      P3 := 1 / 4,
      P4 := 2
  in P3 = min (min P1 P2) (min P3 P4) := 
sorry

end smallest_distance_parabola_l768_768446


namespace part1_part2_l768_768054

variable (a : ℝ)
variable (x y : ℝ)
variable (P Q : ℝ × ℝ)

-- Part (1)
theorem part1 (hP : P = (2 * a - 2, a + 5)) (h_y : y = 0) : P = (-12, 0) :=
sorry

-- Part (2)
theorem part2 (hP : P = (2 * a - 2, a + 5)) (hQ : Q = (4, 5)) 
    (h_parallel : 2 * a - 2 = 4) : P = (4, 8) ∧ quadrant = "first" :=
sorry

end part1_part2_l768_768054


namespace hyperbola_equation_l768_768069

-- Definitions for the conditions and the conclusion
def parabola := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }

def hyperbola (a b : ℝ) := { p : ℝ × ℝ // (p.0^2 / a^2) - (p.1^2 / b^2) = 1 }

def focus_of_parabola := (1, 0 : ℝ)

def is_asymptote (slope : ℝ) (p : ℝ × ℝ) := p.1 = slope * p.0 ∨ p.1 = -slope * p.0

theorem hyperbola_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  ((∀ p ∈ hyperbola a b, (parabola.focus_of_parabola = (a,0))) ∧
  (∀ p, is_asymptote (Real.sqrt 3) p → p ∈ hyperbola a b) →
  (∀ p ∈ hyperbola 1 (Real.sqrt 3), p.0^2 - p.1^2 / 3 = 1)) :=
begin
  use [1, Real.sqrt 3],
  split,
  { linarith, },
  split,
  { apply Real.sqrt_pos.mpr, linarith, },
  intros h1 h2,
  sorry
end

end hyperbola_equation_l768_768069


namespace prime_divisors_of_50_fact_eq_15_l768_768134

theorem prime_divisors_of_50_fact_eq_15 :
  ∃ P : Finset Nat, (∀ p ∈ P, Prime p ∧ p ∣ (Nat.factorial 50)) ∧ P.card = 15 := by
  sorry

end prime_divisors_of_50_fact_eq_15_l768_768134


namespace sin_value_l768_768537

theorem sin_value (x : ℝ) (h : Real.sec x + Real.tan x = 5/4) : Real.sin x = 9/41 :=
sorry

end sin_value_l768_768537


namespace consecutive_integers_product_l768_768300

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l768_768300


namespace numberOfRottweilers_l768_768981

-- Define the grooming times in minutes for each type of dog
def groomingTimeRottweiler := 20
def groomingTimeCollie := 10
def groomingTimeChihuahua := 45

-- Define the number of each type of dog groomed
def numberOfCollies := 9
def numberOfChihuahuas := 1

-- Define the total grooming time in minutes
def totalGroomingTime := 255

-- Compute the time spent on grooming Collies
def timeSpentOnCollies := numberOfCollies * groomingTimeCollie

-- Compute the time spent on grooming Chihuahuas
def timeSpentOnChihuahuas := numberOfChihuahuas * groomingTimeChihuahua

-- Compute the time spent on grooming Rottweilers
def timeSpentOnRottweilers := totalGroomingTime - timeSpentOnCollies - timeSpentOnChihuahuas

-- The main theorem statement
theorem numberOfRottweilers :
  timeSpentOnRottweilers / groomingTimeRottweiler = 6 :=
by
  -- Proof placeholder
  sorry

end numberOfRottweilers_l768_768981


namespace derivative_x_pow_x_derivative_cos_alpha_pow_sin_2alpha_derivative_2t_sqrt_1_minus_t2_derivative_x_minus_1_cbrt_x_plus_1_2_x_minus_2_l768_768024

theorem derivative_x_pow_x (x : ℝ) : deriv (λ x, x^x) x = x^x * (Real.log x + 1) := sorry

theorem derivative_cos_alpha_pow_sin_2alpha (α : ℝ) : 
  deriv (λ α, (Real.cos α)^(Real.sin (2 * α))) α = 
  2 * ((Real.cos (2 * α) * Real.log (Real.cos α)) - (Real.sin (α)^2)) * ((Real.cos α)^(Real.sin (2 * α))) := sorry

theorem derivative_2t_sqrt_1_minus_t2 (t : ℝ) : 
  deriv (λ t, (2 * t) / Real.sqrt (1 - t ^ 2)) t = 
  2 / (Real.sqrt ((1 - t ^ 2) ^ 3)) := sorry

theorem derivative_x_minus_1_cbrt_x_plus_1_2_x_minus_2 (x : ℝ) : 
  deriv (λ x, (x - 1) * Real.cbrt ((x + 1)^2 * (x - 2))) x = 
  (2 * x ^ 2 - 3 * x - 1) / (Real.cbrt ((x + 1) * (x - 2) ^ 2)) := sorry

end derivative_x_pow_x_derivative_cos_alpha_pow_sin_2alpha_derivative_2t_sqrt_1_minus_t2_derivative_x_minus_1_cbrt_x_plus_1_2_x_minus_2_l768_768024


namespace trucks_have_160_containers_per_truck_l768_768333

noncomputable def containers_per_truck: ℕ :=
  let boxes1 := 7 * 20
  let boxes2 := 5 * 12
  let total_boxes := boxes1 + boxes2
  let total_containers := total_boxes * 8
  let trucks := 10
  total_containers / trucks

theorem trucks_have_160_containers_per_truck:
  containers_per_truck = 160 :=
by
  sorry

end trucks_have_160_containers_per_truck_l768_768333


namespace two_distinct_solutions_diff_l768_768634

theorem two_distinct_solutions_diff (a b : ℝ) (h1 : a ≠ b) (h2 : a > b)
  (h3 : ∀ x, (x = a ∨ x = b) ↔ (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3) :
  a - b = 3 :=
by
  -- Proof will be provided here.
  sorry

end two_distinct_solutions_diff_l768_768634


namespace problem_statement_l768_768053

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, 8 < x → f (x) > f (x + 1))
variable (h2 : ∀ x, f (x + 8) = f (-x + 8))

theorem problem_statement : f 7 > f 10 := by
  sorry

end problem_statement_l768_768053


namespace question1_question2_l768_768898

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem question1 (m : ℝ) (h1 : m > 0) 
(h2 : ∀ (x : ℝ), f (x + 1/2) ≤ 2 * m + 1 ↔ x ∈ [-2, 2]) : m = 3 / 2 := 
sorry

theorem question2 (x y : ℝ) : f x ≤ 2^y + 4 / 2^y + |2 * x + 3| := 
sorry

end question1_question2_l768_768898


namespace quadrant_of_complex_l768_768193

def complex_quad (z : ℂ) : string :=
  if z.re > 0 ∧ z.im > 0 then "first quadrant"
  else if z.re < 0 ∧ z.im > 0 then "second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "fourth quadrant"
  else "on an axis"

theorem quadrant_of_complex :
  complex_quad ((1 + complex.i) / complex.i) = "fourth quadrant" :=
by 
  sorry

end quadrant_of_complex_l768_768193


namespace rectangle_fold_angle_theorem_l768_768188

noncomputable def rectangle_fold_angle
  (a b : ℝ) (h : b > a) : Prop :=
  let M := (AB_bounds a b) in
  let N := (CD_bounds a b) in
  let A' := reflect(M, A) in
  let C' := reflect(N, C) in
  ∠ A'C MN = 90

theorem rectangle_fold_angle_theorem (a b : ℝ) (h : b > a) :
  rectangle_fold_angle a b h := sorry

end rectangle_fold_angle_theorem_l768_768188


namespace winning_probability_is_approx_0_103_l768_768475

/-- Definition of the total number of ways to choose 5 balls out of 10 -/
def total_outcomes : ℕ := nat.choose 10 5

/-- Number of ways to draw 4 red balls and 1 white ball -/
def draw_4_red_1_white : ℕ := nat.choose 5 4 * nat.choose 5 1

/-- Number of ways to draw all 5 red balls -/
def draw_5_red : ℕ := nat.choose 5 5

/-- Total number of winning outcomes -/
def winning_outcomes : ℕ := draw_4_red_1_white + draw_5_red

/-- Definition of the probability of winning as a rational number -/
def winning_probability : ℚ := winning_outcomes / total_outcomes

/-- Theorem: The probability of winning is approximately 0.103, rounded to three decimal places -/
theorem winning_probability_is_approx_0_103 : winning_probability ≈ 0.103 := by
  sorry

end winning_probability_is_approx_0_103_l768_768475


namespace arithmetic_sequence_ratio_l768_768557

-- Definitions based on conditions
variables {a_n b_n : ℕ → ℕ} -- Arithmetic sequences
variables {A_n B_n : ℕ → ℕ} -- Sums of the first n terms

-- Given condition
axiom sums_of_arithmetic_sequences (n : ℕ) : A_n n / B_n n = (7 * n + 1) / (4 * n + 27)

-- Theorem to prove
theorem arithmetic_sequence_ratio :
  ∀ (a_n b_n : ℕ → ℕ) (A_n B_n : ℕ → ℕ), 
    (∀ n, A_n n / B_n n = (7 * n + 1) / (4 * n + 27)) → 
    a_6 / b_6 = 78 / 71 := 
by {
  sorry
}

end arithmetic_sequence_ratio_l768_768557


namespace unique_point_in_equilateral_triangle_l768_768890

theorem unique_point_in_equilateral_triangle (A B C P : Type) 
  (h_eq_triangle_area : area (triangle A B C) = 1)
  (h_equal_areas : area (triangle P A B) = area (triangle P B C) ∧ area (triangle P B C) = area (triangle P C A)) :
  (∃! P, ∀ (P : Type), area (triangle P A B) = area (triangle P B C) ∧ area (triangle P B C) = area (triangle P C A)) ∧
  (area (triangle P A B) = 1 / 3) :=
by
  sorry

end unique_point_in_equilateral_triangle_l768_768890


namespace arithmetic_mean_of_integers_from_neg6_to7_l768_768700

noncomputable def arithmetic_mean : ℝ :=
  let integers := list.range' (-6) 14 -- list of integers from -6 to 7
  let sum := integers.sum
  let count := list.length integers
  (sum : ℝ) / count

theorem arithmetic_mean_of_integers_from_neg6_to7 : arithmetic_mean = 0.5 :=
by
  sorry

end arithmetic_mean_of_integers_from_neg6_to7_l768_768700


namespace evaluate_expression_l768_768479

theorem evaluate_expression :
  ⌈4 * (8 - 1 / 3)⌉ = 31 :=
by
  sorry

end evaluate_expression_l768_768479


namespace roots_irrational_l768_768508

theorem roots_irrational {k : ℝ} :
  (∃ k : ℝ, (∃ α β : ℝ, α * β = 2 * k^2 - 1 ∧ 2 * k^2 - 1 = 7 ∧ (x^2 - 3 * k * x + 2 * k^2 - 1 = 0) ∧ α ≠ β ∧ (α * β = 2 * k^2 - 1) ∧ (k = 2 ∨ k = -2)) (α + β) ∧ discriminant (x^2 - 6 * x + 7 = 0) ≠ 0 ∧ discriminant (x^2 + 6 * x + 7 = 0) ≠ 0) :=
by
  sorry

end roots_irrational_l768_768508


namespace average_greater_than_median_by_20_l768_768915

theorem average_greater_than_median_by_20 :
  let weights := [106, 5, 5, 6, 8] in
  let median := 6 in
  let average := (106 + 5 + 5 + 6 + 8) / 5 in
  average - median = 20 :=
by
  let weights := [106, 5, 5, 6, 8];
  let median := 6;
  let average := (106 + 5 + 5 + 6 + 8) / 5;
  have : average = 26 := by rfl;
  have : average - median = 20 := by rfl;
  exact this

end average_greater_than_median_by_20_l768_768915


namespace ants_in_third_anthill_l768_768951

-- Define the number of ants in the first anthill
def ants_first : ℕ := 100

-- Define the percentage reduction for each subsequent anthill
def percentage_reduction : ℕ := 20

-- Calculate the number of ants in the second anthill
def ants_second : ℕ := ants_first - (percentage_reduction * ants_first / 100)

-- Calculate the number of ants in the third anthill
def ants_third : ℕ := ants_second - (percentage_reduction * ants_second / 100)

-- Main theorem to prove that the number of ants in the third anthill is 64
theorem ants_in_third_anthill : ants_third = 64 := sorry

end ants_in_third_anthill_l768_768951


namespace number_of_prime_divisors_of_50_fac_l768_768139

-- Define the finite set of prime numbers up to 50
def primes_up_to_50 : finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

-- Main theorem statement
theorem number_of_prime_divisors_of_50_fac :
  (primes_up_to_50.filter prime).card = 15 := 
sorry

end number_of_prime_divisors_of_50_fac_l768_768139


namespace radius_of_sphere_in_tetrahedron_l768_768437

theorem radius_of_sphere_in_tetrahedron
    (a b c : ℝ)
    (h1 : a = b)
    (h2 : b = c)
    (R : ℝ)
    (h3 : R = 3 * sqrt 3) :
    ∃ r : ℝ, r = 6 * (sqrt 2 - 1) :=
by
  sorry

end radius_of_sphere_in_tetrahedron_l768_768437


namespace ratio_RB_BW_eq_one_third_l768_768400

open_locale big_operators

-- Definitions as per the problem statement
variables (ω1 ω2 : circle) (A B : point) (P Q R S W X : point)
variables [is_intersection ω1 ω2 A B]
variables [is_tangent P Q ω1] [is_tangent R S ω2]
variables [is_parallel RB PQ]
variables [is_ray_interaction RB ω2 W]

-- The proof statement
theorem ratio_RB_BW_eq_one_third :
  (RB / BW = 1 / 3) :=
sorry

end ratio_RB_BW_eq_one_third_l768_768400


namespace ratio_area_eq_9_l768_768617

theorem ratio_area_eq_9 (P Q R P' Q' R' : Type)
  (h1 : ∀ (s : ℝ), PQR s ∧
       (dist P Q = s) ∧ (dist Q R = s) ∧ (dist R P = s)) 
  (h2 : dist Q Q' = 2 * dist P Q ) 
  (h3 : dist R R' = 2 * dist Q R ) 
  (h4 : dist P P' = 2 * dist R P ) 
  (area_PQR : ℝ) 
  (area_P'Q'R' : ℝ) : 
  (area_P'Q'R' / area_PQR) = 9 :=
sorry

end ratio_area_eq_9_l768_768617


namespace travel_time_second_bus_l768_768280

def distance_AB : ℝ := 100 -- kilometers
def passengers_first : ℕ := 20
def speed_first : ℝ := 60 -- kilometers per hour
def breakdown_time : ℝ := 0.5 -- hours
def passengers_second_initial : ℕ := 22
def speed_second_initial : ℝ := 50 -- kilometers per hour
def additional_passengers_speed_decrease : ℝ := 1 -- speed decrease for every additional 2 passengers
def passenger_factor : ℝ := 2
def additional_passengers : ℕ := 20
def total_time_second_bus : ℝ := 2.35 -- hours

theorem travel_time_second_bus :
  let distance_first_half := (breakdown_time * speed_first)
  let remaining_distance := distance_AB - distance_first_half
  let time_to_reach_breakdown := distance_first_half / speed_second_initial
  let new_speed_second_bus := speed_second_initial - (additional_passengers / passenger_factor) * additional_passengers_speed_decrease
  let time_from_breakdown_to_B := remaining_distance / new_speed_second_bus
  total_time_second_bus = time_to_reach_breakdown + time_from_breakdown_to_B := 
sorry

end travel_time_second_bus_l768_768280


namespace centroid_circle_area_l768_768580

theorem centroid_circle_area {D E F : Point} (d_e_diameter : is_diameter D E)
  (d_e_length : distance D E = 30) (F_on_circumference : OnCircumference F) 
  (F_not_D : F ≠ D) (F_not_E : F ≠ E) : 
  area (circle (centroid (triangle D E F)) 5) = 25 * π :=
by sorry

end centroid_circle_area_l768_768580


namespace dodecahedron_edge_probability_l768_768358

theorem dodecahedron_edge_probability :
  ∀ (V E : ℕ), 
  V = 20 → 
  ((∀ v ∈ finset.range V, 3 = 3) → -- condition representing each of the 20 vertices is connected to 3 other vertices
  ∃ (p : ℚ), p = 3 / 19) :=
begin
  intros,
  use 3 / 19,
  split,
  sorry
end

end dodecahedron_edge_probability_l768_768358


namespace arithmetic_geometric_range_l768_768065

theorem arithmetic_geometric_range (a1 x y a2 b1 b2 : ℝ)
  (h_arith : a1 + a2 = x + y)
  (h_geom : b1 * b2 = x * y) :
  (∀ z, z = (a1 + a2)^2 / (b1 * b2) - 2 → z ∈ (-∞, -2] ∪ [2, +∞)) :=
begin
  sorry
end

end arithmetic_geometric_range_l768_768065


namespace inverse_of_A_is_zeroMatrix_l768_768026

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 8], ![-2, -4]]
def zeroMatrix : Matrix (Fin 2) (Fin 2) ℤ := ![![0, 0], ![0, 0]]

theorem inverse_of_A_is_zeroMatrix (h : A.det = 0) : A⁻¹ = zeroMatrix := by
  sorry

end inverse_of_A_is_zeroMatrix_l768_768026


namespace square_of_length_t_equals_306_l768_768670

noncomputable def p (x : ℝ) : ℝ := 2 * x + 1
noncomputable def q (x : ℝ) : ℝ := -x + 4
noncomputable def r (x : ℝ) : ℝ := x + 3
noncomputable def t (x : ℝ) : ℝ := min (min (p x) (q x)) (r x)

theorem square_of_length_t_equals_306 : 
  (let length_square := ∑ I in [-4, -1/2, 1, 4].zip [-1/2, 1, 4, 0], 
     (let (a, b) := I in dist (a, t a) (b, t b)))
  in length_square ^ 2 = 306 := 
by
  sorry

end square_of_length_t_equals_306_l768_768670


namespace resistor_value_l768_768694

/-- Two resistors with resistance R are connected in series to a DC voltage source U.
    An ideal voltmeter connected in parallel to one resistor shows a reading of 10V.
    The voltmeter is then replaced by an ideal ammeter, which shows a reading of 10A.
    Prove that the resistance R of each resistor is 2Ω. -/
theorem resistor_value (R U U_v I_A : ℝ)
  (hU_v : U_v = 10)
  (hI_A : I_A = 10)
  (hU : U = 2 * U_v)
  (hU_total : U = R * I_A) : R = 2 :=
by
  sorry

end resistor_value_l768_768694


namespace binomial_square_solution_l768_768018

variable (t u b : ℝ)

theorem binomial_square_solution (h1 : 2 * t * u = 12) (h2 : u^2 = 9) : b = t^2 → b = 4 :=
by
  sorry

end binomial_square_solution_l768_768018


namespace triangle_side_length_c_l768_768170

theorem triangle_side_length_c
  (a b A B C : ℝ)
  (ha : a = Real.sqrt 3)
  (hb : b = 1)
  (hA : A = 2 * B)
  (hAngleSum : A + B + C = Real.pi) :
  ∃ c : ℝ, c = 2 := 
by
  sorry

end triangle_side_length_c_l768_768170


namespace f_monotonically_increasing_f_geq_mx2_l768_768080

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan x - 2 * x

theorem f_monotonically_increasing : ∀ x ∈ Ioo (-Real.pi / 2) (Real.pi / 2), 
  has_deriv_at f (Real.cos x + (1 / Real.cos x ^ 2) - 2 : ℝ) x ∧ (Real.cos x + (1 / Real.cos x ^ 2) - 2 : ℝ) ≥ 0 := sorry

theorem f_geq_mx2 (m : ℝ) : (∀ x ∈ Ioo 0 (Real.pi / 2), f x ≥ m * x^2) ↔ m ≤ 0 := sorry

end f_monotonically_increasing_f_geq_mx2_l768_768080


namespace num_four_digit_36_combinations_l768_768102

theorem num_four_digit_36_combinations : 
  (∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
  (∀ d ∈ [digit n 1000, digit n 100, digit n 10, digit n 1], d = 3 ∨ d = 6)) → 
  16 :=
sorry

end num_four_digit_36_combinations_l768_768102


namespace conditions_for_k_b_l768_768064

theorem conditions_for_k_b (k b : ℝ) :
  (∀ x : ℝ, (x - (kx + b) + 2) * (2) > 0) →
  (k = 1) ∧ (b < 2) :=
by
  intros h
  sorry

end conditions_for_k_b_l768_768064


namespace transformation_equivalence_l768_768338

-- Define the transformation functions
def transformation_1 (x : ℝ) := 2 * (x + π / 4)
def transformation_2 (x : ℝ) := 2 * x + π / 8

-- Define the sine function and the target transformation
def sine_function (x : ℝ) := Real.sin x
def target_transformation (x : ℝ) := Real.sin (2 * x + π / 4)

-- Prove that transformations 1 and 2 result in the target transformation
theorem transformation_equivalence :
  (∀ x, sine_function (transformation_1 x) = target_transformation x) ∧ 
  (∀ x, sine_function (transformation_2 x) = target_transformation x) :=
  by
    sorry

end transformation_equivalence_l768_768338


namespace leila_marathon_yards_l768_768424

/-
Problem statement:
A marathon is 26 miles and 385 yards. One mile equals 1760 yards.
Leila has run five marathons in her life. 
If the total distance Leila covered in these marathons is m miles and y yards, where 0 ≤ y < 1760,
prove that y = 165.
-/

theorem leila_marathon_yards
  (n_marathons : ℕ)
  (marathon_miles : ℕ)
  (marathon_yards : ℕ)
  (yard_per_mile : ℕ)
  (total_miles : ℕ)
  (remainder_yards : ℕ)
  (h1 : n_marathons = 5)
  (h2 : marathon_miles = 26)
  (h3 : marathon_yards = 385)
  (h4 : yard_per_mile = 1760)
  (h5 : total_miles + remainder_yards / yard_per_mile = marathon_miles * n_marathons + marathon_yards * n_marathons / yard_per_mile)
  (h6 : remainder_yards % yard_per_mile = 165)
  (h7 : 0 ≤ remainder_yards % yard_per_mile ∧ remainder_yards % yard_per_mile < yard_per_mile) :
  remainder_yards % yard_per_mile = 165 := by
sory

end leila_marathon_yards_l768_768424


namespace price_of_pants_l768_768240

theorem price_of_pants
  (P S H : ℝ)
  (h1 : P + S + H = 340)
  (h2 : S = (3 / 4) * P)
  (h3 : H = P + 10) :
  P = 120 :=
by
  sorry

end price_of_pants_l768_768240


namespace fish_remaining_correct_l768_768212

def remaining_fish (jordan_caught : ℕ) (total_catch_lost_fraction : ℚ) : ℕ :=
  let perry_caught := 2 * jordan_caught
  let total_catch := jordan_caught + perry_caught
  let lost_catch := total_catch * total_catch_lost_fraction
  let remaining := total_catch - lost_catch
  remaining.nat_abs

theorem fish_remaining_correct : (remaining_fish 4 (1/4)) = 9 :=
by 
  sorry

end fish_remaining_correct_l768_768212


namespace tangent_through_circumcenter_l768_768576

-- Given points in a triangle and some constructions
variables {A B C X Y P Q A' : Type}
-- given common conditions:
variables [InTriangle A B C]
variables [OnSide X Y B C]
variables [XY_midpoint X Y B C]
variables [TwoXY_eq_BC X Y B C]
variables [Diameter_AA A A' X Y]
variables [Perpendicular_AX_P B BC A X P]
variables [Perpendicular_AY_Q C BC A Y Q]
variables [Circumcircle A X Y A']

-- Statement of the problem
theorem tangent_through_circumcenter :
  tangent_A'_circumcircle_AXY_eq_circumcenter_APQ A' A P Q :=
sorry

end tangent_through_circumcenter_l768_768576


namespace dodecahedron_edge_probability_l768_768351

def dodecahedron_vertices : ℕ := 20

def vertex_connections : ℕ := 3

theorem dodecahedron_edge_probability :
  (∃ (u v : fin dodecahedron_vertices), u ≠ v ∧ u.1 < vertex_connections → 
    (Pr (u, v) = 3 / (dodecahedron_vertices - 1))) :=
sorry

end dodecahedron_edge_probability_l768_768351


namespace incorrect_conclusion_b_l768_768511

noncomputable def given_function (x : ℝ) : ℝ := -2 * x - 4

theorem incorrect_conclusion_b :
  ¬ (∃ x, given_function x = 0 ∧ x = -2) :=
by
  intro h
  cases h with x hx
  cases hx with hx1 hx2
  rw hx2 at hx1
  dsimp [given_function] at hx1
  linarith

end incorrect_conclusion_b_l768_768511


namespace horizontal_asymptote_at_3_l768_768464

noncomputable def rational_function (x : ℝ) : ℝ :=
  (15 * x^4 + 2 * x^3 + 11 * x^2 + 6 * x + 4) / (5 * x^4 + x^3 + 10 * x^2 + 4 * x + 2)

theorem horizontal_asymptote_at_3 : 
  (∀ ε > 0, ∃ N > 0, ∀ x > N, |rational_function x - 3| < ε) := 
by
  sorry

end horizontal_asymptote_at_3_l768_768464


namespace no_such_function_exists_l768_768256

theorem no_such_function_exists : 
  ¬ ∃ (f : ℝ≥0 → ℝ), ∀ (x y : ℝ≥0), f (x + y^2) ≥ f x + y :=
by
  sorry

end no_such_function_exists_l768_768256


namespace exists_odd_sum_of_two_composites_l768_768657

theorem exists_odd_sum_of_two_composites :
  ∃ k : ℕ, (∀ n : ℕ, n ≥ k → odd n → (∃ a b : ℕ, composite a ∧ composite b ∧ a + b = n)) ∧ k = 13 :=
by
  let smallest_even_composite := 4
  let smallest_odd_composite := 9
  have k := 13
  have h1: smallest_even_composite = 4 := sorry
  have h2: smallest_odd_composite = 9 := sorry
  have h3: ∀ n : ℕ, n ≥ k → odd n → (∃ a b : ℕ, composite a ∧ composite b ∧ a + b = n) := sorry
  use k
  exact ⟨h3, by exact rfl⟩


-- Definitions related to composite numbers and odd numbers
def composite (n : ℕ) : Prop :=
∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

def odd (n : ℕ) : Prop :=
n % 2 = 1

end exists_odd_sum_of_two_composites_l768_768657


namespace probability_two_vertices_endpoints_l768_768353

theorem probability_two_vertices_endpoints (V E : Type) [Fintype V] [DecidableEq V] 
  (dodecahedron : Graph V E) (h1 : Fintype.card V = 20)
  (h2 : ∀ v : V, Fintype.card (dodecahedron.neighbors v) = 3)
  (h3 : Fintype.card E = 30) :
  (∃ A B : V, A ≠ B ∧ (A, B) ∈ dodecahedron.edgeSet) → 
  (∃ p : ℚ, p = 3/19) := 
sorry

end probability_two_vertices_endpoints_l768_768353


namespace tan_20_plus_4_sin_20_eq_sqrt_3_l768_768323

theorem tan_20_plus_4_sin_20_eq_sqrt_3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end tan_20_plus_4_sin_20_eq_sqrt_3_l768_768323


namespace num_four_digit_pos_integers_l768_768129

theorem num_four_digit_pos_integers : 
  ∃ n : ℕ, (n = 16) ∧ ∀ k : ℕ, (1000 ≤ k ∧ k < 10000 ∧ 
  ∀ d ∈ [k.digits 10], d = 3 ∨ d = 6) := sorry

end num_four_digit_pos_integers_l768_768129


namespace centroid_trajectory_eq_l768_768558

theorem centroid_trajectory_eq (x y xC yC : ℝ)
  (h1 : C ∈ set_of (λ p : ℝ × ℝ, p.fst^2 / 16 - p.snd^2 / 9 = 1)): 
  (∃ (x y : ℝ), xC = 3 * x - 6 ∧ yC = 3 * y ∧ (h2 : (xC^2 / 16 - yC^2 / 9 = 1))) →
  (9 * (x - 2)^2 / 16 - y^2 = 1) :=
by
  sorry

end centroid_trajectory_eq_l768_768558


namespace area_of_ADBC_l768_768812

noncomputable def quadrilateral_area : ℝ :=
  let A := (0, 0) in
  let B := (2, 0) in
  let C := (3/2, (real.sqrt 3) / 2) in
  let D := (0, -3 - real.sqrt 3) in
  let area_triangle_ABC := 1/2 * 2 * real.sqrt 3 * 1/2 in
  let area_triangle_ACD := 1/2 * abs (3/2 * (-3 - real.sqrt 3)) in
  (area_triangle_ABC + area_triangle_ACD)

theorem area_of_ADBC : quadrilateral_area = (5 * real.sqrt 3 + 9) / 4 :=
by
  sorry

end area_of_ADBC_l768_768812


namespace euler_product_l768_768220

noncomputable def zeta (s : ℝ) : ℝ := ∑' n : ℕ, n ^ (-s)
def prime_stream : Stream ℕ := Stream.map Nat.primes Stream.Nat

theorem euler_product (s : ℝ) (h1 : 1 < s) :
  zeta s = ∏' n : ℕ, (1 - (1 / (prime_stream n) ^ s))⁻¹ :=
sorry

end euler_product_l768_768220


namespace probability_even_sum_from_primes_l768_768168

theorem probability_even_sum_from_primes : 
  let primes := [2, 3, 5, 7, 11, 13, 17, 19] in
  let pairs := [(a, b) | a <- primes, b <- primes, a ≠ b] in
  let even_sum_pairs := [(a, b) | (a, b) <- pairs, (a + b) % 2 = 0] in
  (pairs.length > 0) → 
  ((even_sum_pairs.length : ℚ) / (pairs.length : ℚ) = 1) :=
by
  sorry

end probability_even_sum_from_primes_l768_768168


namespace harry_cookies_batch_l768_768920

theorem harry_cookies_batch
  (total_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips = 81)
  (batches = 3)
  (chips_per_cookie = 9) :
  (total_chips / batches) / chips_per_cookie = 3 := by
  sorry

end harry_cookies_batch_l768_768920


namespace given_eq_then_squares_l768_768517

theorem given_eq_then_squares (x : ℝ) (hx : x + x⁻¹ = 5) : x^2 + x⁻² = 23 :=
sorry

end given_eq_then_squares_l768_768517


namespace trigonometric_identity_proof_l768_768462

theorem trigonometric_identity_proof :
  (cos 0)^4 + (cos 90)^4 + (sin 45)^4 + (sin 135)^4 = 2 := by
  have h1 : (cos 0)^4 = 1 := by
    rw [cos_zero]
    simp
  have h2 : (cos 90)^4 = 0 := by
    rw [cos_pi_div_two]
    simp
  have h3 : (sin 45)^4 = (sqrt 2 / 2)^4 := by
    rw [sin_pi_div_four]
    simp
  have h4 : (sin 135)^4 = (sqrt 2 / 2)^4 := by
    rw [sin_pi_div_two_sub_alignment]
    simp
  calc
    (cos 0)^4 + (cos 90)^4 + (sin 45)^4 + (sin 135)^4
        = h1 + h2 + h3 + h4 := by rw [h1, h2, h3, h4]
    ... = 1 + 0 + (sqrt 2 / 2)^4 + (sqrt 2 / 2)^4 := by simp
    ... = 1 + 0 + 1/2 + 1/2 := by norm_num
    ... = 2 := by norm_num

end trigonometric_identity_proof_l768_768462


namespace initial_necklaces_count_l768_768788

theorem initial_necklaces_count (N : ℕ) 
  (h1 : N - 13 = 37) : 
  N = 50 := 
by
  sorry

end initial_necklaces_count_l768_768788


namespace rational_numbers_opposites_l768_768943

theorem rational_numbers_opposites (a b : ℚ) (h : (a + b) / (a * b) = 0) : a = -b ∧ a ≠ 0 ∧ b ≠ 0 :=
by
  sorry

end rational_numbers_opposites_l768_768943


namespace part_1_part_2_l768_768549

-- Part 1: Finding the value of n
theorem part_1 (n : ℕ) (h1 : n ≥ 2) (h2 : ∀ a b c : ℕ,
  a = (nat.choose n 0) ∧ b = (nat.choose n 1 ) / 2 ∧ 
  c = (nat.choose n 2) / 4 → 2 * b = a + c) : n = 8 :=
  sorry

-- Part 2: Finding the coefficient of x^4
theorem part_2 (n : ℕ) (h1 : n = 8) : ∀ r : ℕ,
  (8 - 2 * r = 4) → (nat.choose 8 r) / (2^r) = 7 :=
  sorry

end part_1_part_2_l768_768549


namespace sequence_sum_l768_768797

/--
  The sum of the sequence 1 - 2 - 3 + 4 + 5 - 6 - 7 + 8 + 9 - ... + 1992 + 1993 - 1994 - 1995 + 1996 + 1997 - 1998
  is equal to 2665.
-/
theorem sequence_sum : 
  (1 - 2 - 3) + (4 + 5 - 6) + (7 + 8 - 9) + ... + (1995 + 1996 - 1997) + 1998 = 2665 :=
  sorry

end sequence_sum_l768_768797


namespace four_digit_3_or_6_l768_768119

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end four_digit_3_or_6_l768_768119


namespace part1_part2_part3_l768_768057

-- Definitions for the vector set and Sn
def vector_set : Type := list (ℝ × ℝ)
def S_n (v: vector_set) : ℝ × ℝ := list.foldr (λ (a b : ℝ × ℝ), (a.1 + b.1, a.2 + b.2)) (0, 0) v

-- Definition of h vector
def is_h_vector (a : ℝ × ℝ) (v : vector_set) : Prop :=
  ∃ S, S = S_n v ∧ |a| ≥ |(S.1 - a.1, S.2 - a.2)|

-- Part 1: Range of values for x
theorem part1 (x : ℝ) (a_1 a_2 a_3 : ℝ × ℝ)
  (h1 : a_1 = (1, 4))
  (h2 : a_2 = (2, 4))
  (h3 : a_3 = (3, x + 3))
  (h_vector : is_h_vector a_3 [a_1, a_2, a_3]) :
  -2 ≤ x ∧ x ≤ 0 :=
sorry

-- Part 2: Existence of h vector
theorem part2 (a_n : ℕ → ℝ × ℝ)
  (h1 : ∀ n, a_n n = (1/3)^(n-1) * (-1)^(n : ℤ))
  (h_vector_exists : ∃ (p : ℕ), is_h_vector (a_n p) (list.of_fn a_n)) :
  ∃ p, p = 1 :=
sorry

-- Part 3: Minimum value of |Q_2013Q_2014|
theorem part3 (x : ℝ) (a_1 a_2 a_3 : ℝ × ℝ)
  (h1 : a_1 = (Real.sin x, Real.cos x))
  (h2 : a_2 = (2 * Real.cos x, 2 * Real.sin x))
  (h3 : is_h_vector a_1 [a_1, a_2, a_3])
  (h4 : is_h_vector a_2 [a_1, a_2, a_3])
  (h5 : is_h_vector a_3 [a_1, a_2, a_3]) :
  ∃ min_val, min_val = 4024 :=
sorry

end part1_part2_part3_l768_768057


namespace prime_divisors_of_50_fact_eq_15_l768_768136

theorem prime_divisors_of_50_fact_eq_15 :
  ∃ P : Finset Nat, (∀ p ∈ P, Prime p ∧ p ∣ (Nat.factorial 50)) ∧ P.card = 15 := by
  sorry

end prime_divisors_of_50_fact_eq_15_l768_768136


namespace polynomial_solution_l768_768019

theorem polynomial_solution (P : ℝ → ℝ) (h : ∀ x : ℝ, (x - 1) * P(x + 1) - (x + 2) * P(x) = 0) :
  ∃ a : ℝ, P = λ x, a * (x^3 - x) :=
by
  sorry

end polynomial_solution_l768_768019


namespace john_pays_correct_amount_l768_768209

noncomputable def candy_bar_price : ℝ := 1

def total_candy_bars : ℕ := 20

def dave_contribution : ℕ := 6

def discount_rate_10_or_more : ℝ := 0.20

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price - price * discount

def final_discounted_price : ℝ :=
  if total_candy_bars >= 10 then discounted_price candy_bar_price discount_rate_10_or_more
  else if total_candy_bars >= 5 then discounted_price candy_bar_price (discount_rate_10_or_more / 2)
  else candy_bar_price

def total_cost : ℝ := total_candy_bars * final_discounted_price

def dave_cost : ℝ := dave_contribution * final_discounted_price

def john_cost : ℝ := total_cost - dave_cost

theorem john_pays_correct_amount : john_cost = 20 * 0.80 - 6 * 0.80 := by
  unfold john_cost total_cost dave_cost final_discounted_price
  rw [if_pos (by decide : 20 >= 10)]
  simp [candy_bar_price, discount_rate_10_or_more, discounted_price]
  norm_num
  rw [←mul_sub]
  norm_num
  sorry

end john_pays_correct_amount_l768_768209


namespace cookies_per_batch_l768_768923

theorem cookies_per_batch
  (bag_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips : ℕ)
  (h1 : bag_chips = total_chips)
  (h2 : batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips = 81) :
  (bag_chips / batches) / chips_per_cookie = 3 := 
by
  sorry

end cookies_per_batch_l768_768923


namespace farmer_red_monthly_milk_l768_768490

def bess_daily := 2
def brownie_daily := 3 * bess_daily
def daisy_daily := bess_daily + 1
def ella_daily := 1.5 * daisy_daily
def flossie_daily := (bess_daily + brownie_daily) / 2
def month_days := 30
def total_daily := bess_daily + brownie_daily + daisy_daily + ella_daily + flossie_daily
def monthly_production := total_daily * month_days

theorem farmer_red_monthly_milk : monthly_production = 585 := by
  unfold monthly_production
  unfold total_daily
  unfold bess_daily
  unfold brownie_daily
  unfold daisy_daily
  unfold ella_daily
  unfold flossie_daily
  unfold month_days
  norm_num
  rfl

end farmer_red_monthly_milk_l768_768490


namespace triangle_covers_fraction_of_grid_l768_768770

noncomputable def triangleCoverageFraction : ℚ :=
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (6, 3)
  let C : ℝ × ℝ := (3, 6)
  let area_triangle := (1/2 : ℝ) * abs (fst A * (snd B - snd C) + fst B * (snd C - snd A) + fst C * (snd A - snd B))
  let area_grid := 7 * 6
  (area_triangle / area_grid).toRat

theorem triangle_covers_fraction_of_grid (h: triangleCoverageFraction = 5 / 28) : 
  true :=
by
  sorry

end triangle_covers_fraction_of_grid_l768_768770


namespace cyclic_points_l768_768968

/-- The incircle of triangle ABC touches sides AB and AC at points C1 and B1.
    Point M divides segment C1B1 in a 3:1 ratio from C1.
    Point N is the midpoint of AC.
    Given AC = 3 * (BC - AB), prove that the points I, M, B1, and N lie on a circle. -/

variables (A B C C1 B1 M N I : Type)
variables (h1 : touches_incircle_triangle A B C C1 B1) -- The incircle touches AB at C1 and AC at B1
variables (h2 : divides_segment_31_ratio C1 B1 M) -- M divides C1B1 in 3:1 ratio
variables (h3 : midpoint AC N) -- N is the midpoint of AC
variables (h4 : AC = 3 * (BC - AB)) -- Given condition AC = 3 * (BC - AB)

theorem cyclic_points (A B C C1 B1 M N I : Type)
  (h1 : touches_incircle_triangle A B C C1 B1)
  (h2 : divides_segment_31_ratio C1 B1 M)
  (h3 : midpoint AC N)
  (h4 : AC = 3 * (BC - AB)) :
  cyclic I M B1 N :=
sorry

end cyclic_points_l768_768968


namespace min_segments_needed_l768_768012

theorem min_segments_needed (n : ℕ) (colors : Fin 10) :
  (∀ (segments : Finset (Fin 67 × Fin 67)), 
  segments.card = n ∧ 
  (∀ (v : Fin 67), ∃ (c : Fin 10), ∃ (s : Finset (Fin 67 × Fin 67)), s.card = 7 ∧ ∀ (e : Fin 67 × Fin 67) ∈ s, e.fst = v ∨ e.snd = v ∧ colored e = c)
  ) → n = 2011 :=
  sorry

end min_segments_needed_l768_768012


namespace bugs_meet_again_at_point_P_in_six_minutes_l768_768693

-- Definitions based on the conditions
def radius_large : ℝ := 6
def radius_small : ℝ := 3
def speed_large : ℝ := 4 * Real.pi
def speed_small : ℝ := 3 * Real.pi

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

def time_to_revolution (circumference speed : ℝ) : ℝ := circumference / speed

-- Theorem stating the solution
theorem bugs_meet_again_at_point_P_in_six_minutes : 
  time_to_revolution (circumference radius_large) speed_large = 3 ∧
  time_to_revolution (circumference radius_small) speed_small = 2 ∧
  Nat.lcm (Nat.ofReal (time_to_revolution (circumference radius_large) speed_large)) (Nat.ofReal (time_to_revolution (circumference radius_small) speed_small)) = 6 := 
by 
  sorry

end bugs_meet_again_at_point_P_in_six_minutes_l768_768693


namespace find_f_of_2016_l768_768546

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^3 - 1
else if (x >= -1 ∧ x <= 1) then
  if x ≥ 0 then sorry else -f (-x)
else if x > 1/2 then f (x - 1/2)
else sorry

theorem find_f_of_2016 : f 2016 = 2 := 
sorry

end find_f_of_2016_l768_768546


namespace chicken_price_l768_768661

-- Definition of the conditions
def chickens : Nat := 4
def weekly_feed_cost : Int := 1
def eggs_per_chicken : Nat := 3
def weeks : Nat := 81
def dozen_egg_price : Int := 2
def eggs_per_dozen : Nat := 12
def weekly_eggs_needed : Nat := 1

theorem chicken_price (
    P : Real,
    total_chickens := chickens * P,
    total_feed_cost := weeks * weekly_feed_cost,
    total_egg_cost := weeks * dozen_egg_price,
    chicken_egg_production := chickens * eggs_per_chicken,
    is_cheaper_than_buying_eggs := total_chickens + total_feed_cost < total_egg_cost,
    chicken_egg_production = weekly_eggs_needed * eggs_per_dozen
  ) : P = 20.25 :=
by
  sorry

end chicken_price_l768_768661


namespace logical_relationship_l768_768912

variable (x : ℝ)

def statement_A := x^2 - 5 * x < 0
def statement_B := abs(x - 2) < 3

theorem logical_relationship : (∀ x : ℝ, statement_A x → statement_B x) ∧ ¬(∀ x : ℝ, statement_B x → statement_A x) :=
by { sorry }

end logical_relationship_l768_768912


namespace no_nat_num_divisible_l768_768604

open Nat

theorem no_nat_num_divisible : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 := sorry

end no_nat_num_divisible_l768_768604


namespace problem_l768_768518

theorem problem (x : ℝ) (h : x + x⁻¹ = 5) : x^2 + x⁻² = 23 :=
sorry

end problem_l768_768518


namespace tangent_lines_eq_l768_768025

noncomputable def tangent_line_through_point (p : ℝ × ℝ) := 
  {l | (∃ x : ℝ, l = (x, x^3) ∧ 3*x^2*(l.1 - x) = (l.2 - x^3))}

theorem tangent_lines_eq (L : set (ℝ × ℝ)) :
  (L = {(x, y) | y = 0} ∨ 
    L = {(x, y) | 27 * x - y - 54 = 0}) ↔
    ∃ p : ℝ × ℝ, p = (2, 0) ∧
    ∃ t : ℝ, 
    (∃ l : ((ℝ × ℝ) → ℝ),
     (l p = 0) ∧ 
     (∀ q : ℝ × ℝ, q ∈ tangent_line_through_point p → l q = 0)) :=
sorry

end tangent_lines_eq_l768_768025


namespace consecutive_integers_product_l768_768301

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l768_768301


namespace EG_length_in_quadrilateral_EFGH_l768_768187

theorem EG_length_in_quadrilateral_EFGH 
  (EF FG GH HE : ℝ) (EG : ℝ)
  (right_angle : ∠EFG = (90: ℝ)) :
  EF = 6 → FG = 18 → GH = 6 → HE = 10 → EG = 6 * Real.sqrt 10 := by
  intros h1 h2 h3 h4
  have h : EF^2 + FG^2 = (6 * Real.sqrt 10)^2, from sorry, -- Use Pythagorean theorem
  exact sorry -- Prove the lengths are correct

end EG_length_in_quadrilateral_EFGH_l768_768187


namespace angle_between_MN_and_PQ_in_tetrahedron_l768_768586

variable {A B C D M N P Q : ℝ^3}

-- Define a regular tetrahedron
def is_regular_tetrahedron (A B C D : ℝ^3) : Prop :=
  dist A B = dist A C ∧ dist A B = dist A D ∧ dist A B = dist B C ∧
  dist A C = dist A D ∧ dist A C = dist B D ∧ dist B C = dist B D ∧
  dist C D = dist A B

-- Define the midpoint M of AD
def midpoint_AD (A D M : ℝ^3) : Prop :=
  M = (A + D) / 2

-- Define the midpoint P of CD
def midpoint_CD (C D P : ℝ^3) : Prop :=
  P = (C + D) / 2

-- Define the centroid N of triangle BCD
def centroid_BCD (B C D N : ℝ^3) : Prop :=
  N = (B + C + D) / 3

-- Define the centroid Q of triangle ABC
def centroid_ABC (A B C Q : ℝ^3) : Prop :=
  Q = (A + B + C) / 3

-- Define the dot product for vectors to calculate the cosine of the angle
def dot_product (v1 v2 : ℝ^3) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Define the norm of a vector
def norm (v : ℝ^3) : ℝ :=
  real.sqrt (v.x^2 + v.y^2 + v.z^2)

-- Define the cosine of the angle between vectors v1 and v2
def cosine_angle (v1 v2 : ℝ^3) : ℝ :=
  (dot_product v1 v2) / (norm v1 * norm v2)

-- The angle between MN and PQ
def angle_between_lines (M N P Q : ℝ^3) : ℝ :=
  arccos (cosine_angle (N - M) (Q - P))

theorem angle_between_MN_and_PQ_in_tetrahedron (A B C D M N P Q : ℝ^3)
  (h_tetra : is_regular_tetrahedron A B C D)
  (h_M : midpoint_AD A D M) (h_N : centroid_BCD B C D N)
  (h_P : midpoint_CD C D P) (h_Q : centroid_ABC A B C Q) :
  angle_between_lines M N P Q = arccos (1 / 18) :=
sorry

end angle_between_MN_and_PQ_in_tetrahedron_l768_768586


namespace committee_selection_l768_768587

theorem committee_selection :
  let total_ways := Nat.choose 18 6 in
  let new_member_ways := Nat.choose 13 6 in
  total_ways - new_member_ways = 16848 :=
by
  let total_ways := Nat.choose 18 6
  let new_member_ways := Nat.choose 13 6
  have total_ways_eq : total_ways = 18564 := by sorry
  have new_member_ways_eq : new_member_ways = 1716 := by sorry
  have result_eq : total_ways - new_member_ways = 16848 := by
    rw [total_ways_eq, new_member_ways_eq]
    norm_num
  exact result_eq

end committee_selection_l768_768587


namespace count_four_digit_integers_with_3_and_6_l768_768113

theorem count_four_digit_integers_with_3_and_6 : 
  (∃ (count : ℕ), count = 16 ∧ 
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → 
  (∀ i : ℕ, i < 4 → (n / (10 ^ i)) % 10 = 3 ∨ (n / (10 ^ i)) % 10 = 6) ↔ n ∈ {3333, 3366, 3633, 3666, 6333, 6366, 6633, 6666}) :=
by
  have h : 2 ^ 4 = 16 := by norm_num
  use 16
  split
  · exact h
  · sorry

end count_four_digit_integers_with_3_and_6_l768_768113


namespace angle_measure_l768_768321

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * x + 7) : x = 34.6 :=
begin
  sorry
end

end angle_measure_l768_768321


namespace positive_integers_divisible_by_4_5_and_6_less_than_300_l768_768931

open Nat

theorem positive_integers_divisible_by_4_5_and_6_less_than_300 : 
    ∃ n : ℕ, n = 5 ∧ ∀ m, m < 300 → (m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0) → (m % 60 = 0) :=
by
  sorry

end positive_integers_divisible_by_4_5_and_6_less_than_300_l768_768931


namespace solution_set_bf_x2_solution_set_g_l768_768623

def f (x : ℝ) := x^2 - 5 * x + 6

theorem solution_set_bf_x2 (x : ℝ) : (2 < x ∧ x < 3) ↔ f x < 0 := sorry

noncomputable def g (x : ℝ) := 6 * x^2 - 5 * x + 1

theorem solution_set_g (x : ℝ) : (1 / 3 < x ∧ x < 1 / 2) ↔ g x < 0 := sorry

end solution_set_bf_x2_solution_set_g_l768_768623


namespace bun_cost_calculation_l768_768856

theorem bun_cost_calculation :
  (∃ (bun_cost : ℝ), 
    ∀ (num_buns : ℕ) (milk_price : ℝ) (milk_bottles : ℕ) (eggs_multiplier : ℝ) 
      (total_cost : ℝ),
      num_buns = 10 →
      milk_price = 2 →
      milk_bottles = 2 →
      eggs_multiplier = 3 →
      total_cost = 11 →
      0 < bun_cost ∧ 
      (total_cost - (milk_price * milk_bottles + milk_price * eggs_multiplier)) / num_buns = bun_cost) :=
begin
  use 0.1,
  intros num_buns milk_price milk_bottles eggs_multiplier total_cost,
  intros hbuns hmilk hmilk_bottles heggs htotal,
  split,
  { norm_num },  -- Prove that 0 < 0.1
  { rw [hbuns, hmilk, hmilk_bottles, heggs, htotal],
    norm_num }
end

end bun_cost_calculation_l768_768856


namespace eccentricity_of_ellipse_l768_768530

variables {a b : ℝ} (h : a > b) (ha : a > 0) (hb : b > 0)

def ellipse_eccentricity (a b : ℝ) (h : a > b) (ha : a > 0) (hb : b > 0) : ℝ :=
  have h_ab : a² - b² > 0 := by sorry
  let c := sqrt (a² - b²) / 2 in
  c / a

theorem eccentricity_of_ellipse (a b : ℝ) (h : a > b) (ha : a > 0) (hb : b > 0) :
   ellipse_eccentricity a b h ha hb = 1 / 4 :=
by
  sorry

end eccentricity_of_ellipse_l768_768530


namespace sum_of_coefficients_l768_768903

theorem sum_of_coefficients (a b : ℝ) (h1 : a = 1 * 5) (h2 : -b = 1 + 5) : a + b = -1 :=
by
  sorry

end sum_of_coefficients_l768_768903


namespace trig_identity_proof_l768_768824

theorem trig_identity_proof : (sin 35 - sin 25) / (cos 35 - cos 25) = - Real.sqrt 3 :=
by
  sorry

end trig_identity_proof_l768_768824


namespace distance_origin_complex_l768_768590

noncomputable def complex_point := 2 / (1 + (complex.i : ℂ))

theorem distance_origin_complex :
  complex.abs complex_point = real.sqrt 2 :=
by
  sorry

end distance_origin_complex_l768_768590


namespace no_nat_number_satisfies_l768_768606

theorem no_nat_number_satisfies (n : ℕ) : ¬ ((n^2 + 6 * n + 2019) % 100 = 0) :=
sorry

end no_nat_number_satisfies_l768_768606


namespace not_suitable_for_census_l768_768196

namespace CensusProblem

inductive Option : Type
| A
| B
| C
| D

open Option

def suitable_for_census (o : Option) : Prop :=
  match o with
  | A => true -- Security check for passengers before boarding a plane
  | B => true -- School recruiting teachers for interviews with applicants
  | C => true -- Understanding the extracurricular reading time of a class of students
  | D => false -- Understanding the service life of a batch of light bulbs

theorem not_suitable_for_census : ∃ (o : Option), ¬suitable_for_census o ∧ o = D :=
by 
  use D
  split
  . exact dec_trivial
  . refl

end CensusProblem

end not_suitable_for_census_l768_768196


namespace paula_bought_two_shirts_l768_768249

-- Define the conditions
def total_money : Int := 109
def shirt_cost : Int := 11
def pants_cost : Int := 13
def remaining_money : Int := 74

-- Calculate the expenditure on shirts and pants
def expenditure : Int := total_money - remaining_money

-- Define the number of shirts bought
def number_of_shirts (S : Int) : Prop := expenditure = shirt_cost * S + pants_cost

-- The theorem stating that Paula bought 2 shirts
theorem paula_bought_two_shirts : number_of_shirts 2 :=
by
  -- The proof is omitted as per instructions
  sorry

end paula_bought_two_shirts_l768_768249


namespace triangle_mn_length_l768_768966

open_locale classical

variables {A B C L K M N : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace L] [MetricSpace K] [MetricSpace M] [MetricSpace N]

def length (x y : Type) [MetricSpace x] [MetricSpace y] : Real := sorry

noncomputable def angle_bisector_intersection (x y z : Type) : Type := sorry

noncomputable def perpendicular_foot (x y z : Type) : Type := sorry

theorem triangle_mn_length (A B C : Type)
  (AB AC BC : Real) (AB_eq : AB = 125) (AC_eq : AC = 117) (BC_eq : BC = 120)
  (L : Type) (K : Type)
  (angle_bisector_A : L = angle_bisector_intersection A B C)
  (angle_bisector_B : K = angle_bisector_intersection B A C)
  (M : Type) (N : Type)
  (foot_from_C_to_BK : M = perpendicular_foot C B K)
  (foot_from_C_to_AL : N = perpendicular_foot C A L) :
  length M N = 56 :=
sorry

end triangle_mn_length_l768_768966


namespace problem_l768_768565

variable {a b c d : ℝ}

theorem problem (h1 : a > b) (h2 : c > d) : a - d > b - c := sorry

end problem_l768_768565


namespace atleast_three_red_and_blue_marked_l768_768947

noncomputable def grid_size := (10, 20)
def total_numbers := 200
def rows := 10
def columns := 20
def red_marked (grid : Matrix rows columns ℕ) : list ℕ :=
  (List.range rows).bind (λ r, (grid r).maxN 2)
def blue_marked (grid : Matrix rows columns ℕ) : list ℕ :=
  (List.range columns).bind (λ c, (Matrix.column grid c).maxN 2)

theorem atleast_three_red_and_blue_marked (grid : Matrix rows columns ℕ) (h_unique : ∀ i j, grid i j ≠ grid (i + 1) (j + 1)) :
  (red_marked grid).countp (λ n, n ∈ blue_marked grid) ≥ 3 :=
sorry

end atleast_three_red_and_blue_marked_l768_768947


namespace sum_area_le_two_l768_768217

theorem sum_area_le_two {n : ℕ} (b : Fin n → ℕ) (a : Fin (n + 1) → ℚ)
    (h_b_sum : (Finset.univ.sum (λ i, b i)) = 2)
    (h_a_0 : a 0 = 0)
    (h_a_n : a n = 0)
    (h_abs_diff_le : ∀ i, |a (i + 1) - a i| ≤ b i):
    (Finset.univ.sum (λ i, (a (i + 1) + a i) * b i) ≤ 2) :=
  sorry

end sum_area_le_two_l768_768217


namespace probability_of_D_or_E_in_a_box_l768_768337

-- Lean statement of the problem given conditions
theorem probability_of_D_or_E_in_a_box
  (balls : Finset ℕ)
  (boxes : Finset ℕ)
  (h_balls : Finset.card balls = 5)
  (h_boxes : Finset.card boxes = 3)
  (labels : balls = {1, 2, 3, 4, 5})
  : 
  (∃ (select : Finset ℕ), select ⊆ balls ∧ Finset.card select = 3 ∧
    ((4 ∈ select) ∨ (5 ∈ select)) → 
    (1 - ((1 * 6) / ((5.choose 3) * 3!)) = 9 / 10)) := sorry

end probability_of_D_or_E_in_a_box_l768_768337


namespace range_of_x_max_min_f_l768_768548

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop :=
  3^(2 * x - 4) - (10 / 3) * 3^(x - 1) + 9 ≤ 0

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  log 2 (x / 2) * log 2 (sqrt x / 2)

-- Prove the range of x
theorem range_of_x (x : ℝ) (h : inequality_condition x) : 2 ≤ x ∧ x ≤ 4 :=
sorry

-- Prove the maximum and minimum values of f(x)
theorem max_min_f (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4) :
  (∀ x, (inequality_condition x) → 0 ≤ f(x)) ∧
  (∀ x, (inequality_condition x) → (x = 2 ∨ x = 4)) ∧
  (f(2 * sqrt 2) = -1 / 8) :=
sorry

end range_of_x_max_min_f_l768_768548


namespace plywood_cut_perimeter_difference_l768_768742

theorem plywood_cut_perimeter_difference :
  ∀ (length width : ℕ), length = 6 ∧ width = 9 → 
  ∃ p1 p2 : ℕ, 
    (∃ (config1 : length ≠ 0 ∧ width ≠ 0), p1 = 2 * (3 + width)) ∧
    (∃ (config2 : length ≠ 0 ∧ width ≠ 0), p2 = 2 * (6 + 3)) ∧
    (∀ n : ℕ, n = 3 → ∃ cut : length * width = 3 * (length * width / 3))),
  abs (p1 - p2) = 6 := 
by
  intro length width h
  obtain ⟨h1, h2⟩ := h
  have config1 := 2 * (3 + 9)
  have config2 := 2 * (6 + 3)
  have h3 := 6 * 9 = 3 * (6 * 9 / 3)
  use config1, config2
  split
  . use (6 ≠ 0 ∧ 9 ≠ 0)
    exact rfl
  . split
    . use (6 ≠ 0 ∧ 9 ≠ 0)
      exact rfl
    . intro n hn
      use h3
  rw [abs_eq_nat]
  rw [config1, config2]
  exact rfl

end plywood_cut_perimeter_difference_l768_768742


namespace z_pow_12_eq_neg_64_i_l768_768150

noncomputable def z : ℂ :=
  2 * real.cos (real.pi / 8) * (real.sin (3 * real.pi / 4) + complex.i * real.cos (3 * real.pi / 4) + complex.i)

theorem z_pow_12_eq_neg_64_i : z^12 = -64 * complex.i :=
sorry

end z_pow_12_eq_neg_64_i_l768_768150


namespace part1_part2_l768_768989

def f (x : ℝ) (t : ℝ) : ℝ := x^2 + 2 * t * x + t - 1

theorem part1 (hf : ∀ x ∈ Set.Icc (-(3 : ℝ)) (1 : ℝ), f x 2 ≤ 6 ∧ f x 2 ≥ -3) : 
  ∀ x ∈ Set.Icc (-(3 : ℝ)) (1 : ℝ), f x 2 ≤ 6 ∧ f x 2 ≥ -3 :=
by 
  sorry
  
theorem part2 (ht : ∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f x t > 0) : 
  t ∈ Set.Ioi (0 : ℝ) :=
by 
  sorry

end part1_part2_l768_768989


namespace find_xy_l768_768088

theorem find_xy (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : (x - 10)^2 + (y - 10)^2 = 18) : 
  x * y = 91 := 
by {
  sorry
}

end find_xy_l768_768088


namespace find_m_geq_9_l768_768405

-- Define the real numbers
variables {x m : ℝ}

-- Define the conditions
def p (x : ℝ) := x ≤ 2
def q (x m : ℝ) := x^2 - 2*x + 1 - m^2 ≤ 0

-- Main theorem statement based on the given problem
theorem find_m_geq_9 (m : ℝ) (hm : m > 0) :
  (¬ p x → ¬ q x m) → (p x → q x m) → m ≥ 9 :=
  sorry

end find_m_geq_9_l768_768405


namespace range_of_function_l768_768593

theorem range_of_function : 
  (∀ x : ℝ, (x + 3 ≥ 0) → (x ≠ 0) → (x ≥ -3 ∧ x ≠ 0)) :=
by
  intros x hx hnx
  split
  { exact hx }
  { exact hnx }

end range_of_function_l768_768593


namespace sum_of_solutions_l768_768733

theorem sum_of_solutions :
  (∃ x1, 2 * x1 + 2^x1 = 5) → 
  (∃ x2, 2 * x2 + 2 * Real.log (x2 - 1) / Real.log 2 = 5) → 
  ∃ x1 x2, (2 * x1 + 2^x1 = 5 ∧ 2 * x2 + 2 * Real.log (x2 - 1) / Real.log 2 = 5) ∧ (x1 + x2 = 7 / 2) :=
begin
  intro h1,
  intro h2,
  cases h1 with x1 h1,
  cases h2 with x2 h2,
  use [x1, x2],
  split,
  { split; assumption },
  { sorry }
end

end sum_of_solutions_l768_768733


namespace Jordan_total_listens_by_end_of_year_l768_768977

theorem Jordan_total_listens_by_end_of_year :
  ∀ (initial_listens m : ℕ), initial_listens = 60000 → m = 3 →
  (∀ i : ℕ, i < m → ∀ current_listens : ℕ, current_listens = (2^i) * initial_listens → 
  ∑ i in finset.range(m), (2^i * initial_listens)) + initial_listens = 900000 :=
by sorry

end Jordan_total_listens_by_end_of_year_l768_768977


namespace range_of_a_l768_768522

def f (x a : ℝ) : ℝ := x^2 - a * x + (a / 2)

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ Ioo (-1 : ℝ) 1 → f x a > 0) ↔ (0 < a ∧ a ≤ 2) := 
sorry

end range_of_a_l768_768522


namespace part1_part2_l768_768515

-- Define the binomial expansion condition
def binomial_expansion (m : ℤ) (x : ℤ) : ℤ :=
  (1 + m * x)^7

-- Define the expressions a_3 and solution to part (1)
def a_3 (m : ℤ) : ℤ := 35 * m^3

-- Define the condition a3 = -280
def condition_a3 (m : ℤ) : Prop := a_3(m) = -280

-- Define the target value for m from part (1)
def target_m_part1 : ℤ := -2

theorem part1 (m : ℤ) (h : condition_a3 m) : m = target_m_part1 :=
by
  sorry

-- Define the target value for a_1 + a_3 + a_5 + a_7 from part (2)
def target_a1_a3_a5_a7_part2 : ℤ := -1094

-- Define the sum of a_1, a_3, a_5, and a_7
def sum_a1_a3_a5_a7 (m : ℤ) : ℤ :=
  let expansion := binomial_expansion m in
  expansion 1 - expansion 0

theorem part2 (m : ℤ) (h : condition_a3 m) (hm : m = target_m_part1) : sum_a1_a3_a5_a7 m = target_a1_a3_a5_a7_part2 :=
by
  sorry

end part1_part2_l768_768515


namespace characteristic_inequalities_l768_768735

noncomputable theory

open Complex

/-- The characteristic function of a random variable X is defined as φ(t) = ℰ[e^(itX)] -/
def characteristic_function (φ : ℝ → ℂ) (X : ℝ → ℝ → ℝ) (t : ℝ) := 
  φ t = ∫ x in Set.univ, Complex.exp (Complex.I * t * X x) -- Simplified for generality

theorem characteristic_inequalities (φ : ℝ → ℂ) (X : ℝ → ℝ) (t s : ℝ) (ht: t ∈ Set.univ) (hs: s > 0) :
  (φ = λ t, ∫ x in Set.univ, Complex.exp (Complex.I * t * X x)) →
  (|Complex.imag (φ t)|^2 ≤ (1 - Complex.real (φ (2 * t)) / 2)) ∧
  (|Complex.real (φ t)|^2 ≤ (1 + Complex.real (φ (2 * t)) / 2)) ∧
  (|φ t - φ s|^2 ≤ 2 * (1 - Complex.real (φ (t - s)))) ∧
  (|((1 / (2 * s)) * (∫ u in Ioc t (t + s), φ u))| ≤ (1 + Complex.real (φ s)) ^ (1 / 2)) :=
sorry

end characteristic_inequalities_l768_768735


namespace output_S_final_value_l768_768774

theorem output_S_final_value : 
  (let S := 0; i := 1;
   let S := S + (3*i - 1); i := i+1;
   let S := S + (3*i - 1); i := i+1;
   let S := S + (3*i - 1); i := i+1;
   let S := S + (3*i - 1); i := i+1;
   let S := S + (3*i - 1); i := i+1;
   S) = 40 :=
sorry

end output_S_final_value_l768_768774


namespace threeDigitNumbersWithDigit6or8_l768_768563

noncomputable def totalThreeDigitNumbers : ℕ := 999 - 100 + 1
def validHundredsDigits : finset ℕ := {1, 2, 3, 4, 5, 7, 9}
def validDigits : finset ℕ := {0, 1, 2, 3, 4, 5, 7, 9}

def countNumbersWithout6or8 : ℕ :=
  validHundredsDigits.card * validDigits.card * validDigits.card

theorem threeDigitNumbersWithDigit6or8 : 
  totalThreeDigitNumbers - countNumbersWithout6or8 = 452 := by
  sorry

end threeDigitNumbersWithDigit6or8_l768_768563


namespace probability_same_gender_l768_768014

theorem probability_same_gender :
  let males := 3
  let females := 2
  let total := males + females
  let total_ways := Nat.choose total 2
  let male_ways := Nat.choose males 2
  let female_ways := Nat.choose females 2
  let same_gender_ways := male_ways + female_ways
  let probability := (same_gender_ways : ℚ) / total_ways
  probability = 2 / 5 :=
by
  sorry

end probability_same_gender_l768_768014


namespace school_distance_is_seven_l768_768803

-- Definitions based on conditions
def distance_to_school (x : ℝ) : Prop :=
  let monday_to_thursday_distance := 8 * x
  let friday_distance := 2 * x + 4
  let total_distance := monday_to_thursday_distance + friday_distance
  total_distance = 74

-- The problem statement to prove
theorem school_distance_is_seven : ∃ (x : ℝ), distance_to_school x ∧ x = 7 := 
by {
  sorry
}

end school_distance_is_seven_l768_768803


namespace cookies_per_batch_l768_768922

theorem cookies_per_batch
  (bag_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips : ℕ)
  (h1 : bag_chips = total_chips)
  (h2 : batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips = 81) :
  (bag_chips / batches) / chips_per_cookie = 3 := 
by
  sorry

end cookies_per_batch_l768_768922


namespace lily_pad_growth_rate_is_one_l768_768952

def days_to_full_coverage : ℕ := 34
def days_to_half_coverage : ℕ := 33
def growth_rate (r : ℝ) : Prop :=
  let S := 1 / 2 -- assuming half coverage is the base size on day 33
  let full_on_34 := 1 -- full coverage on day 34
  full_on_34 = S * (1 + r)^(days_to_full_coverage - days_to_half_coverage)

theorem lily_pad_growth_rate_is_one : ∃ r : ℝ, growth_rate r ∧ r = 1 :=
begin
  -- We will later prove that if the conditions hold, then r = 1.
  sorry
end

end lily_pad_growth_rate_is_one_l768_768952


namespace james_trees_per_day_l768_768204

noncomputable def trees_cut_per_day (x : ℝ) : Prop :=
  let james_trees := 2 * x
  let brothers_trees := 3 * (2 * 0.8 * x)
  let total_trees := james_trees + brothers_trees
  total_trees = 196

theorem james_trees_per_day (x : ℝ) : trees_cut_per_day x → x = 196 / 6.8 :=
begin
  intro h,
  dsimp [trees_cut_per_day] at h,
  rw [←h],
  linarith,
end

end james_trees_per_day_l768_768204


namespace matt_new_average_commission_l768_768646

noncomputable def new_average_commission (x : ℝ) : ℝ :=
  (5 * x + 1000) / 6

theorem matt_new_average_commission
  (x : ℝ)
  (h1 : (5 * x + 1000) / 6 = x + 150)
  (h2 : x = 100) :
  new_average_commission x = 250 :=
by
  sorry

end matt_new_average_commission_l768_768646


namespace parabola_proof_l768_768687

-- Definition of the problem conditions
def parabola (p : ℝ) : (ℝ × ℝ) → Prop :=
λ point, point.2^2 = 2 * p * point.1

-- Definition stating that p is greater than 0
def p_positive (p : ℝ) : Prop :=
p > 0

-- The given area condition
def area_condition (p : ℝ) : Prop :=
∃ (A B A' B' : ℝ × ℝ), 
  parabola p A ∧ 
  parabola p B ∧
  (A.2 = B.2) ∧
  (abs (A.1 - B.1) = 8 * p) ∧
  let height := p in
  (1/2) * (A.2 + B.2) * height = 48

-- The target equation of the parabola
def target_equation (p : ℝ) : (ℝ × ℝ) → Prop :=
λ point, point.2^2 = 4 * (2:ℝ).sqrt * point.1

-- Final Lean statement to prove the target equation given the conditions
theorem parabola_proof (p : ℝ) :
  p_positive p →
  area_condition p →
  ∀ (point : ℝ × ℝ), parabola p point ↔ target_equation (2:ℝ).sqrt point :=
by sorry

end parabola_proof_l768_768687


namespace broken_line_property_l768_768603

theorem broken_line_property:
  ∀ (s : set (ℝ × ℝ)) (L : set (ℝ × ℝ)),
    (∀ x ∈ s, ∃ y ∈ L, dist x y ≤ 0.5) → 
    (∃ F1 F2 ∈ L, dist F1 F2 ≤ 1 ∧ path_length L F1 F2 ≥ 198) :=
begin
  sorry
end

-- Definitions that might be needed
def path_length (L : set (ℝ × ℝ)) (F1 F2 : ℝ × ℝ) : ℝ :=
  sorry -- function to compute the length of the path along L from F1 to F2

end broken_line_property_l768_768603


namespace optimal_strategy_l768_768659

theorem optimal_strategy (p : ℝ) (h : 0 < p ∧ p < 1) : 
  (if 0 < p ∧ p < 1/3 then "send_four_couriers" else "send_two_couriers") :=
begin
  sorry
end

end optimal_strategy_l768_768659


namespace number_of_trailing_zeros_remainder_l768_768796

theorem number_of_trailing_zeros_remainder : 
  let num_factors_5 := (Nat.floor (50 / 5) + Nat.floor (50 / 25))
  num_factors_5 % 500 = 12 := 
by
  let num_factors_5 := Nat.floor (50 / 5) + Nat.floor (50 / 25)
  have h1 : num_factors_5 = 12 := by sorry
  show num_factors_5 % 500 = 12 from by
    rw [h1]
    exact Nat.mod_self 12 500 sorry
    sorry

end number_of_trailing_zeros_remainder_l768_768796


namespace tetrahedron_if_all_vertices_connected_l768_768253

def is_convex_polyhedron (P : Type) [Fintype P] := sorry
def connected_vertices (P : Type) [Fintype P] := ∀ (v : P), ∀ (u : P), v ≠ u → (∃ (e : (v, u) : P), true)

theorem tetrahedron_if_all_vertices_connected (P : Type) [Fintype P] 
  (h1 : is_convex_polyhedron P) (h2 : connected_vertices P) : 
  Fintype.card P = 4 :=
sorry

end tetrahedron_if_all_vertices_connected_l768_768253


namespace no_matrix_M_doubles_first_and_triples_second_column_l768_768503

open Matrix

theorem no_matrix_M_doubles_first_and_triples_second_column :
  ¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ), ∀ (a b c d : ℝ),
    M.mul (λ (i j : Fin 2), ![![a, b], ![c, d]].get i j) =
    (λ (i j : Fin 2), ![![2 * a, 3 * b], ![2 * c, 3 * d]].get i j) := 
by 
  sorry

end no_matrix_M_doubles_first_and_triples_second_column_l768_768503


namespace min_value_ineq_l768_768229

open Real

theorem min_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 2) * (y^2 + 5 * y + 2) * (z^2 + 5 * z + 2) / (x * y * z) ≥ 512 :=
by sorry

noncomputable def optimal_min_value : ℝ := 512

end min_value_ineq_l768_768229


namespace problem_statement_l768_768233

theorem problem_statement
  (a b c : ℝ) (x₁ x₂ : ℂ)
  (h_root1 : a * x₁^2 + b * x₁ + c = 0)
  (h_root2 : a * x₂^2 + b * x₂ + c = 0)
  (h_imaginary : x₁.im ≠ 0)
  (h_real_frac : (x₁^2 / x₂).re = (x₁^2 / x₂) ∧ (x₁^2 / x₂).im = 0) :
  1 + (x₁ / x₂) + (x₁ / x₂)^2 + (x₁ / x₂)^4 + ... + (x₁ / x₂)^1999 = -1 :=
by 
  sorry

end problem_statement_l768_768233


namespace total_cost_of_chairs_l768_768790

theorem total_cost_of_chairs (original_price : ℝ) (discount_rate : ℝ) (extra_discount_rate : ℝ) (threshold : ℕ) (num_chairs : ℕ) (total_cost : ℝ) :
  original_price = 20 → 
  discount_rate = 0.25 → 
  extra_discount_rate = 1/3 → 
  threshold = 5 → 
  num_chairs = 8 →
  total_cost = 105 :=
by 
  intros h1 h2 h3 h4 h5 
  have discounted_price : ℝ := original_price * (1 - discount_rate),
  have initial_total : ℝ := discounted_price * num_chairs,
  have extra_discount : ℝ := if num_chairs > threshold then (num_chairs - threshold) * (discounted_price * extra_discount_rate) else 0,
  have final_total : ℝ := initial_total - extra_discount,
  rw [h1, h2, h4, h5] at *,
  rw final_total,
  have base_discount_price : ℝ := 20 * 0.75,
  have base_total : ℝ := base_discount_price * 8,
  have extra_price : ℝ := if 8 > 5 then (8 - 5) * base_discount_price / 3 else 0,
  have total : ℝ := base_total - extra_price,
  simp, norm_num at total,
  exact total

end total_cost_of_chairs_l768_768790


namespace geometric_series_sum_l768_768000

theorem geometric_series_sum :
  let a := 2
  let r := -2
  let n := 10
  let Sn := (a : ℚ) * (r^n - 1) / (r - 1)
  Sn = 2050 / 3 :=
by
  sorry

end geometric_series_sum_l768_768000


namespace odd_array_parity_l768_768528

theorem odd_array_parity (n : ℕ) (A : fin n → fin n → ℤ)
  (h_odd_n : n % 2 = 1) 
  (h_values : ∀ i j, A i j = 1 ∨ A i j = -1) :
  ¬ (∃ (r_odd_count c_odd_count : ℕ),
        (∀ i, r_odd_count = finset.card { j | A i j = -1 } ∧ r_odd_count % 2 = 1) ∧
        (∀ j, c_odd_count = finset.card { i | A i j = -1 } ∧ c_odd_count % 2 = 1) ∧
        (r_odd_count + c_odd_count = n)) :=
sorry

end odd_array_parity_l768_768528


namespace one_over_m_add_one_over_n_l768_768320

theorem one_over_m_add_one_over_n (m n : ℕ) (h_sum : m + n = 80) (h_hcf : Nat.gcd m n = 6) (h_lcm : Nat.lcm m n = 210) : 
  1 / (m:ℚ) + 1 / (n:ℚ) = 1 / 15.75 :=
by
  sorry

end one_over_m_add_one_over_n_l768_768320


namespace mixed_number_subtraction_l768_768384

theorem mixed_number_subtraction :
  2 + 5 / 6 - (1 + 1 / 3) = 3 / 2 := by
sorry

end mixed_number_subtraction_l768_768384


namespace sum_of_n_is_36_mod_100_l768_768618

theorem sum_of_n_is_36_mod_100:
  let T := ∑ n in { n : ℕ | ∃ k : ℤ, n^2 + 16 * n - 1733 = k^2 }, n
  T % 100 = 36 := 
by
  sorry

end sum_of_n_is_36_mod_100_l768_768618


namespace probability_even_sum_from_primes_l768_768167

theorem probability_even_sum_from_primes : 
  let primes := [2, 3, 5, 7, 11, 13, 17, 19] in
  let pairs := [(a, b) | a <- primes, b <- primes, a ≠ b] in
  let even_sum_pairs := [(a, b) | (a, b) <- pairs, (a + b) % 2 = 0] in
  (pairs.length > 0) → 
  ((even_sum_pairs.length : ℚ) / (pairs.length : ℚ) = 1) :=
by
  sorry

end probability_even_sum_from_primes_l768_768167


namespace minimum_value_expression_l768_768226

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x * y * z) ≥ 343 :=
sorry

end minimum_value_expression_l768_768226


namespace num_true_propositions_l768_768878

variables {ℝ : Type*} [real ℝ]

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry
def g' : ℝ → ℝ := sorry

-- Conditions
axiom h_deriv : ∀ x : ℝ, has_deriv_at g (g' x) x
axiom h_even : ∀ x : ℝ, g(x) = g(-x)
axiom h_eq1 : ∀ x : ℝ, f(x) + g'(x) = 10
axiom h_eq2 : ∀ x : ℝ, f(x) - g'(4 - x) = 10

-- Propositions
def p1 := f(1) + f(3) = 20
def p2 := f(4) = 10
def p3 := f(-1) = f(-3)
def p4 := f(2022) = 10

-- The number of true propositions
theorem num_true_propositions : (p1 ∧ p2 ∧ p4) ∧ ¬ p3 := sorry

end num_true_propositions_l768_768878


namespace simplify_complex_expression_l768_768265

theorem simplify_complex_expression : 
  let i : ℂ := complex.I in -- Imaginary unit
  (3 * (4 - 2 * i) + 2 * i * (3 + i)) = 10 := 
by
  have h1: i^2 = -1 := by sorry -- known property of the imaginary unit
  sorry

end simplify_complex_expression_l768_768265


namespace probability_even_sum_in_rows_and_columns_l768_768673

theorem probability_even_sum_in_rows_and_columns : 
  let nums := (Finset.range 16).map (λ x, x + 1)
  let perms := Equiv.Perm (Fin 16)
  let board_matrix := Matrix (Fin 4) (Fin 4) ℕ in
  ∃ M : board_matrix, 
    (∀ i : Fin 4, (∑ j : Fin 4, M i j) % 2 = 0) ∧ 
    (∀ j : Fin 4, (∑ i : Fin 4, M i j) % 2 = 0) →
  (nums.card = 16 ∧ 
    let successful_arrangements := 6 * 34 + 21 * 2 in
    let total_arrangements := Finset.card perms in
    probability_even_sum_in_rows_and_columns = (successful_arrangements * (8.factorial)) / (16.factorial) := 41 / 2145 := sorry

end probability_even_sum_in_rows_and_columns_l768_768673


namespace minimum_police_officers_needed_l768_768676

def grid := (5, 8)
def total_intersections : ℕ := 54
def max_distance_to_police := 2

theorem minimum_police_officers_needed (min_police_needed : ℕ) :
  (min_police_needed = 6) := sorry

end minimum_police_officers_needed_l768_768676


namespace S_2017_eq_18134_l768_768218

def S (n : ℕ) : ℕ :=
  (List.range' 1 n).sum (λ x => (Real.log2 x).floor)

theorem S_2017_eq_18134 : S 2017 = 18134 :=
  sorry

end S_2017_eq_18134_l768_768218


namespace sum_f_2001_l768_768066

noncomputable def f : ℝ → ℝ := sorry

axiom h_even : ∀ x : ℝ, f (-x) = f x
axiom h_shift_odd : ∀ x : ℝ, f (x + 1) = -f (x - 1)
axiom h_f_2 : f 2 = -2

theorem sum_f_2001 : (∑ i in finset.range 2001, f (i + 1)) = 0 :=
by
  sorry

end sum_f_2001_l768_768066


namespace sergey_number_l768_768263

def is_five_digit_number (x : ℕ) : Prop := 
  10000 ≤ x ∧ x < 100000

def reversed (x : ℕ) : ℕ :=
let a := x / 10000
let b := (x / 1000) % 10
let c := (x / 100) % 10
let d := (x / 10) % 10
let e := x % 10
in 10000 * e + 1000 * d + 100 * c + 10 * b + a

theorem sergey_number : 
  ∃ x : ℕ, is_five_digit_number x ∧ 9 * x = reversed x ∧ x = 10989 :=
by {
  sorry
}

end sergey_number_l768_768263


namespace union_A_B_l768_768235

def A : Set ℝ := { y | ∃ x : ℝ, y = 3^x }
def B : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem union_A_B : A ∪ B = { x | -1 < x ∨ (∃ y, y = 3^x ∧ y > 0) } := by
  sorry

end union_A_B_l768_768235


namespace rose_rice_amount_l768_768662

noncomputable def kg_to_g (kg: ℝ) : ℝ := kg * 1000
noncomputable def lb_to_g (lb: ℝ) : ℝ := lb * 453.592

theorem rose_rice_amount 
    (initial_kg: ℝ) 
    (cooked_morning_kg: ℝ) 
    (received_lb: ℝ) 
    (cooked_afternoon_fraction: ℝ) 
    (cooked_evening_fraction: ℝ) 
    (given_neighbor_lb: ℝ) : 
    initial_kg = 30 ∧ cooked_morning_kg = 4/5 ∧ received_lb = 5 ∧ 
    cooked_afternoon_fraction = 2/3 ∧ cooked_evening_fraction = 5/6 ∧ 
    given_neighbor_lb = 3 →
    let initial_g := kg_to_g initial_kg in
    let cooked_morning_g := kg_to_g cooked_morning_kg in
    let received_g := lb_to_g received_lb in
    let after_morning_g := initial_g - cooked_morning_g in
    let after_received_g := after_morning_g + received_g in
    let cooked_afternoon_g := after_received_g * cooked_afternoon_fraction in
    let after_afternoon_g := after_received_g - cooked_afternoon_g in
    let cooked_evening_g := after_afternoon_g * cooked_evening_fraction in
    let after_evening_g := after_afternoon_g - cooked_evening_g in
    let given_neighbor_g := lb_to_g given_neighbor_lb in
    let final_g := after_evening_g - given_neighbor_g in
    final_g = 387.444 :=
by sorry

end rose_rice_amount_l768_768662


namespace containers_per_truck_l768_768334

theorem containers_per_truck (trucks1 boxes1 trucks2 boxes2 boxes_to_containers total_trucks : ℕ)
  (h1 : trucks1 = 7) 
  (h2 : boxes1 = 20) 
  (h3 : trucks2 = 5) 
  (h4 : boxes2 = 12) 
  (h5 : boxes_to_containers = 8) 
  (h6 : total_trucks = 10) :
  (((trucks1 * boxes1) + (trucks2 * boxes2)) * boxes_to_containers) / total_trucks = 160 := 
sorry

end containers_per_truck_l768_768334


namespace plate_acceleration_l768_768369

noncomputable def cylindrical_roller_acceleration
  (R : ℝ) (r : ℝ) (m : ℝ) (alpha : ℝ) (g : ℝ) : Prop :=
  let a := g * Real.sqrt((1 - Real.cos(alpha)) / 2) in
  ∃ (accel : ℝ) (dir : ℝ), accel = a ∧ dir = Real.arcsin(0.2)

theorem plate_acceleration:
  cylindrical_roller_acceleration
    1.0  -- R = 1 m
    0.5  -- r = 0.5 m
    75.0 -- m = 75 kg
    (Real.arccos(0.82))  -- α = arccos(0.82)
    10.0  -- g = 10 m/s^2
:=
sorry

end plate_acceleration_l768_768369


namespace number_of_special_four_digit_integers_l768_768122

theorem number_of_special_four_digit_integers : 
  let digits := [3, 6]
  let choices_per_digit := 2
  num_digits = 4
  ∑ i in range(num_digits), (choices_per_digit) = 2^4 :=
by
  sorry

end number_of_special_four_digit_integers_l768_768122


namespace problem_part1_problem_part2_problem_part3_l768_768896

noncomputable def f (x : ℝ) : ℝ := 2 * cos x ^ 2 + sin (2 * x - π / 6)

theorem problem_part1 (k : ℤ) :
  ∀ x, x ∈ set.Icc (k * π - π / 3) (k * π + π / 6) → 
  f x ≥ 1 ∧ (f x ≤ 2) :=
sorry

theorem problem_part2 (k : ℤ) :
  f (k * π + π / 6) = 2 :=
sorry

variable {α : Type*} [linear_ordered_field α]

theorem problem_part3 {A b c : α} (h1 : 2 * cos A ^ 2 + sin (2 * A - (π/6)) = 3/2) (h2 : b + c = 2) :
  ∃ a : α, 1 ≤ a ∧ a < 2 :=
sorry

end problem_part1_problem_part2_problem_part3_l768_768896


namespace AH_perpendicular_BC_l768_768862

variable {A B C E D G F P T H : Point}
variable {triangle : Triangle}
variable (B_excircle : Circle)
variable (C_excircle : Circle)
variable (incircle : Circle)

-- Given conditions as Lean definitions
def triangle_ABC : Triangle := triangle
def B_excircle_touches_BC_at_E : touches B_excircle (side BC) E := sorry
def C_excircle_touches_BC_at_D : touches C_excircle (side BC) D := sorry
def incircle_touches_AC_at_G : touches incircle (side AC) G := sorry
def incircle_touches_AB_at_F : touches incircle (side AB) F := sorry
def extension_DF_and_EG_intersect_at_P : intersects (extension DF) (extension EG) P := sorry
def AP_extends_to_intersect_BC_at_T : intersects (extension AP) (side BC) T := sorry
def point_H_on_BC_such_that_BT_eq_CH : on_line_segment (side BC) H ∧ BT = CH := sorry

-- Proof statement
theorem AH_perpendicular_BC 
    (h_triangleABC: triangle = triangle_ABC) 
    (h_B_excircle_touches_E: B_excircle_touches_BC_at_E)
    (h_C_excircle_touches_D: C_excircle_touches_BC_at_D)
    (h_incircle_touches_G: incircle_touches_AC_at_G)
    (h_incircle_touches_F: incircle_touches_AB_at_F)
    (h_extensions_intersect_P: extension_DF_and_EG_intersect_at_P)
    (h_AP_intersect_T: AP_extends_to_intersect_BC_at_T)
    (h_H_on_BC_BT_eq_CH: point_H_on_BC_such_that_BT_eq_CH) : 
    perpendicular AH BC := 
by 
  sorry

end AH_perpendicular_BC_l768_768862


namespace work_days_for_A_and_B_l768_768421

theorem work_days_for_A_and_B (W_A W_B : ℝ) (h1 : W_A = (1/2) * W_B) (h2 : W_B = 1/21) : 
  1 / (W_A + W_B) = 14 := by 
  sorry

end work_days_for_A_and_B_l768_768421


namespace prime_pair_probability_even_sum_l768_768165

open Finset

-- Conditions given in the problem
def firstEightPrimes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Main statement of the problem to prove
theorem prime_pair_probability_even_sum : 
  let total_pairs := (firstEightPrimes.choose 2).card in
  let odd_pairs := (firstEightPrimes.filter (λ x, x ≠ 2)).card in
  total_pairs > 0 → 
  (total_pairs - odd_pairs) / total_pairs = 3 / 4 :=
by
  intros total_pairs odd_pairs h
  sorry

end prime_pair_probability_even_sum_l768_768165


namespace continuous_sqrt_power_l768_768663

theorem continuous_sqrt_power (x a : ℝ) (n k : ℝ) (hx : x > 0) (ha : a > 0) (hn : n > 0) :
  tendsto (λ x, x^(k/n)) (𝓝 a) (𝓝 (a^(k/n))) :=
begin
  sorry
end

end continuous_sqrt_power_l768_768663


namespace sequence_S_decreasing_l768_768223

open Real

def S (n : ℕ) : ℝ := (e - 1) ^ 2 / (2 * e ^ (n + 2))

theorem sequence_S_decreasing (n : ℕ) : n > 0 → S n > S (n + 1) := by
  intro h
  let term_1 := (e - 1) ^ 2 / (2 * e ^ (n + 2))
  let term_2 := (e - 1) ^ 2 / (2 * e ^ (n + 3))
  suffices : 2 * e ^ (n + 2) < 2 * e ^ (n + 3)
  { -- Completes the proof due to the property of fractions
    rw [S, S]
    exact this }
  sorry

end sequence_S_decreasing_l768_768223


namespace solve_cubic_l768_768841

noncomputable def solution_set : Set ℂ :=
  {-2, 1 + complex.I * real.sqrt 3, 1 - complex.I * real.sqrt 3}

theorem solve_cubic : 
  {z : ℂ | z^3 = -8} = solution_set := sorry

end solve_cubic_l768_768841


namespace solve_for_m_l768_768156

def is_purely_imaginary (c : ℂ) : Prop :=
  c.re = 0 ∧ c.im ≠ 0

def imaginary_condition (m : ℂ) : Prop :=
  let z := complex.log (m^2 - 2*m - 2) + ((m^2 + 3*m + 2) * complex.I) in
  is_purely_imaginary z

theorem solve_for_m (m : ℂ) : imaginary_condition m → m = 3 :=
begin
  intros h,
  sorry
end

end solve_for_m_l768_768156


namespace olives_per_jar_correct_l768_768804

-- Define the conditions as constants
constant total_money : ℝ := 10
constant change : ℝ := 4
constant total_olives_needed : ℕ := 80
constant cost_per_jar : ℝ := 1.5

-- Define the number of jars he can buy
noncomputable def jars_bought : ℕ := (total_money - change) / cost_per_jar

-- The function to calculate the number of olives per jar
noncomputable def olives_per_jar : ℕ := total_olives_needed / jars_bought

-- The theorem to prove
theorem olives_per_jar_correct :
  olives_per_jar = 20 :=
sorry

end olives_per_jar_correct_l768_768804


namespace ellipse_1_standard_equation_ellipse_2_standard_equation_l768_768740

noncomputable def ellipse_equation_1 : Prop :=
  let f1 := (-1 : ℝ, 0 : ℝ);
  let f2 := (1 : ℝ, 0 : ℝ);
  let p := (1 / 2 : ℝ, sqrt 14 / 4 : ℝ) in
  ∃ (a b : ℝ), a = sqrt 2 ∧ b = 1 ∧ ((x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1))

theorem ellipse_1_standard_equation : ellipse_equation_1 :=
sorry

noncomputable def ellipse_equation_2 : Prop :=
  let p1 := (sqrt 2 : ℝ, -1 : ℝ);
  let p2 := (-1 : ℝ, sqrt 6 / 2 : ℝ) in
  ∃ (m n : ℝ), m = 1 / 4 ∧ n = 1 / 2 ∧ ((x y : ℝ), (x^2 / m + y^2 / n = 1))

theorem ellipse_2_standard_equation : ellipse_equation_2 :=
sorry

end ellipse_1_standard_equation_ellipse_2_standard_equation_l768_768740


namespace number_of_factors_of_2310_for_which_product_of_factors_is_perfect_square_is_27_l768_768847

namespace Proof

-- Definitions
def num_factors_2310 : ℕ := 2310

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

def product_of_factors_is_perfect_square (n : ℕ) : Prop :=
  is_perfect_square (n ^ (Nat.div (Nat.factors n).length 2))

def count_factors_which_product_is_perfect_square (d : ℕ) (p : ℕ) : ℕ :=
  Nat.card {n : ℕ | n ∣ d ∧ product_of_factors_is_perfect_square n}

-- Theorem statement
theorem number_of_factors_of_2310_for_which_product_of_factors_is_perfect_square_is_27 :
  count_factors_which_product_is_perfect_square num_factors_2310 2310 = 27 := sorry

end Proof

end number_of_factors_of_2310_for_which_product_of_factors_is_perfect_square_is_27_l768_768847


namespace compare_a_b_c_l768_768048

noncomputable def a : ℝ := Real.log (Real.sqrt 2)
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := 1 / Real.exp 1

theorem compare_a_b_c : a < b ∧ b < c := by
  -- Proof will be done here
  sorry

end compare_a_b_c_l768_768048


namespace value_of_x_plus_y_l768_768942

theorem value_of_x_plus_y 
  (x y : ℝ) 
  (h1 : -x = 3) 
  (h2 : |y| = 5) : 
  x + y = 2 ∨ x + y = -8 := 
  sorry

end value_of_x_plus_y_l768_768942


namespace min_modulus_complex_l768_768052

theorem min_modulus_complex (z : ℂ) (h : complex.abs (z - (1 + complex.I)) = 1) : complex.abs z ≥ real.sqrt 2 - 1 :=
sorry

end min_modulus_complex_l768_768052


namespace consecutive_integers_product_l768_768297

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l768_768297


namespace bishops_non_threaten_count_l768_768588

theorem bishops_non_threaten_count : 
  let dark_square_diagonals := 7
  let light_square_diagonals := 7
  let placements_per_color := 16
  placements_per_color * placements_per_color = 256
:= 
by
  let dark_square_diagonals := 7
  let light_square_diagonals := 7
  let placements_per_color := 16
  show placements_per_color * placements_per_color = 256 by 
    sorry

end bishops_non_threaten_count_l768_768588


namespace num_four_digit_pos_integers_l768_768132

theorem num_four_digit_pos_integers : 
  ∃ n : ℕ, (n = 16) ∧ ∀ k : ℕ, (1000 ≤ k ∧ k < 10000 ∧ 
  ∀ d ∈ [k.digits 10], d = 3 ∨ d = 6) := sorry

end num_four_digit_pos_integers_l768_768132


namespace consecutive_integers_sum_l768_768296

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l768_768296


namespace solve_equation_l768_768005

theorem solve_equation : ∀ x : ℝ, x ≠ 2 → x - 8 / (x - 2) = 5 + 8 / (x - 2) ↔ (x = 9 ∨ x = -2) :=
by {
  intro x,
  intro h,
  split,
  {
    intro eq,
    -- steps to simplify and solve would go here, skipped for brevity.
    sorry
  },
  {
    intro hx,
    cases hx,
    {
      rw hx,
      -- check for x = 9
      sorry
    },
    {
      rw hx,
      -- check for x = -2
      sorry
    }
  }
}

end solve_equation_l768_768005


namespace max_tan_theta_of_parabola_l768_768086

noncomputable theory
open Real

-- Definition of points A and B on the given parabola y^2 = 2px
def parabola_points (p : ℝ) (t1 t2 : ℝ) : Prop :=
  p > 0 ∧
  t1 ≠ 0 ∧
  t2 ≠ 0 ∧
  t1 * t2 ≠ -1 ∧
  (∀ t, y = 2 * p * t)

-- Definition of area of triangle formed by points O, A, and B
def area_triangle (OA OB : ℝ) (θ : ℝ) : ℝ :=
  1 / 2 * OA * OB * sin θ

-- Given that the area of ΔAOB is m * tan θ
def area_eq_m_tan (S : ℝ) (m θ : ℝ) : Prop :=
  S = m * tan θ

-- Lean statement for the given problem
theorem max_tan_theta_of_parabola (p m θ : ℝ) (A B : parabola_points p) :
  (∀ t, y = 2 * p * t) →
  (t1 ≠ 0 ∧ t2 ≠ 0 ∧ t1 * t2 ≠ -1) →
  area_eq_m_tan (area_triangle (dist (0, 0) (2 * p * t1 , 2 * p * t1^2))
                               (dist (0, 0) (2 * p * t2 , 2 * p * t2^2)) θ) m →
  (∀ t, y = 2 * p * t) →
  ∃ θ : ℝ, θ = -2 * sqrt 2 :=
  sorry

end max_tan_theta_of_parabola_l768_768086


namespace travel_between_cities_maintained_l768_768339

open Finset Function

/-- Given a graph representing cities and airlines from different companies,
assure that connectivity is maintained even after canceling N-1 airlines.--/
theorem travel_between_cities_maintained
  (V : Type) [Fintype V]
  (E : set (V × V))
  (N : ℕ)
  (company_edges : Fin N → set (V × V))
  (h1 : ∀ i, is_perfect_matching (company_edges i) V) -- each company connects cities perfectly
  (h2 : connected (induce E)) -- initial graph is connected
  (h3 : E = ⋃ i, company_edges i) -- E is union of edges of all companies
  (h4 : ∃ (S : Finset (Fin N)), S.card = N - 1) -- select N-1 edges to cancel
  (h5 : let T := E ∖ ⋃ i ∈ S, company_edges i in ¬connected (induce T)) -- assume resulting graph disconnected
  : false :=
by sorry

end travel_between_cities_maintained_l768_768339


namespace intersection_midpoint_ratio_3_l768_768523

def hyperbola (a b x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

def asymptotes (a b : ℝ) (x y : ℝ) := y = (b / a) * x ∨ y = -(b / a) * x

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem intersection_midpoint_ratio_3 
  (a b m x y x0 y0 : ℝ) 
  (h_asymptotes : asymptotes a b x y) 
  (h_vertex: a = 1) 
  (h_intersection : ∃ A B : ℝ × ℝ, A ≠ B ∧ hyperbola a b A.1 A.2 ∧ hyperbola a b B.1 B.2 ∧ 
                     midpoint A B = (x0, y0) ∧ x0 = m / 2 ∧ y0 = (3 * m) / 2) :
  (x0 ≠ 0) → (y0 / x0 = 3) := 
by
  sorry

end intersection_midpoint_ratio_3_l768_768523


namespace arithmetic_sequence_sum_l768_768589

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 8 = 8)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) (h3 : a 1 + a 15 = 2 * a 8) :
  S 15 = 120 := sorry

end arithmetic_sequence_sum_l768_768589


namespace proof_problem_l768_768547

def param_eqns_line_l (x y t : ℝ) : Prop := (x = 1 + 1/2 * t) ∧ (y = sqrt 3 + (sqrt 3)/2 * t)

def polar_eqn_curve_C (ρ θ : ℝ) : Prop := ρ = 4 * sin θ

def rect_eqn_curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0

def polar_eqn_line_l (θ : ℝ) : Prop := θ = π/3

noncomputable def intersection_points {t1 t2 : ℝ} 
    (h1 : t1 + t2 = 2 * sqrt 3 - 4) 
    (h2 : t1 * t2 = -4 * sqrt 3 + 4) : 
    (|t1|/|t2| + |t2|/|t1| = (2 * sqrt 3 - 4)^2 - 2*(-4 * sqrt 3 + 4)) := sorry

theorem proof_problem : 
    (∀ ρ θ, polar_eqn_curve_C ρ θ → ∀ x y, rect_eqn_curve_C x y) ∧
    (∀ x y t, param_eqns_line_l x y t → ∀ θ, polar_eqn_line_l θ) ∧
    (∀ t1 t2, intersection_points t1 t2 = (3 * sqrt 3 - 1)/2) := sorry

end proof_problem_l768_768547


namespace parabola_equation_l768_768834

theorem parabola_equation (h k : ℝ) (p : ℝ × ℝ) (a b c : ℝ) :
  h = 3 ∧ k = -2 ∧ p = (4, -5) ∧
  (∀ x y : ℝ, y = a * (x - h) ^ 2 + k → p.2 = a * (p.1 - h) ^ 2 + k) →
  -(3:ℝ) = a ∧ 18 = b ∧ -29 = c :=
by sorry

end parabola_equation_l768_768834


namespace number_of_k_l768_768028

theorem number_of_k :
  ∃ (N : ℕ), N = 4000 ∧ ∀ k : ℕ, k ≤ 291000 → (k^2 - 1) % 291 = 0 ↔ k ∈ (Set.range ((λ n : ℕ, 291 * n + 1) ∪ (λ n, 291 * n + 98) ∪ (λ n, 291 * n + 193) ∪ (λ n, 291 * n + 290))) :=
begin
  sorry
end

end number_of_k_l768_768028


namespace german_students_l768_768581

theorem german_students (total_students : ℕ) (french_students : ℕ) (both_students : ℕ) (neither_students : ℕ) :
  total_students = 79 →
  french_students = 41 →
  both_students = 9 →
  neither_students = 25 →
  ∃ G : ℕ, G = 22 :=
by 
  intros h1 h2 h3 h4
  use 22
  sorry

end german_students_l768_768581


namespace polynomial_expansion_properties_l768_768861

theorem polynomial_expansion_properties : 
  let a := (λ x, (1 - x)^9) in
  let coeffs := (λ x, coeffs_of_polynomial x) in
  let a_0 := coeffs a 0 in
  let a_1 := coeffs a 1 in
  let a_2 := coeffs a 2 in
  let a_3 := coeffs a 3 in
  let a_4 := coeffs a 4 in
  let a_5 := coeffs a 5 in
  let a_6 := coeffs a 6 in
  let a_7 := coeffs a 7 in
  let a_8 := coeffs a 8 in
  let a_9 := coeffs a 9 in
  (a_0 = 1) ∧ 
  (a_1 + a_3 + a_5 + a_7 + a_9 = -256) ∧ 
  (2 * a_1 + 2^2 * a_2 + 2^3 * a_3 + 2^4 * a_4 + 2^5 * a_5 + 2^6 * a_6 + 2^7 * a_7 + 2^8 * a_8 + 2^9 * a_9 = -2) :=
by sorry

end polynomial_expansion_properties_l768_768861


namespace reconstruct_triangle_l768_768967

-- Definitions and Conditions
variables {A B C I H I_C : Type} 

-- Assuming these points form a triangle and have the defined properties
axiom incenter : incenter C I
axiom altitude_foot : altitude_foot C H AB
axiom excenter : excenter_opposite_C C I_C

-- Target theorem/prove statement
theorem reconstruct_triangle :
  reconstruct_triangle I H I_C → (∃ A B C, is_triangle A B C ∧ incenter A B C I ∧ altitude_foot A B C H ∧ excenter_opposite A B C I_C) :=
sorry

end reconstruct_triangle_l768_768967


namespace each_parent_suitcases_l768_768644

namespace SuitcaseProblem

-- Definitions based on conditions
def siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def total_suitcases : Nat := 14

-- Theorem statement corresponding to the question and correct answer
theorem each_parent_suitcases (suitcases_per_parent : Nat) :
  (siblings * suitcases_per_sibling + 2 * suitcases_per_parent = total_suitcases) →
  suitcases_per_parent = 3 := by
  intro h
  sorry

end SuitcaseProblem

end each_parent_suitcases_l768_768644


namespace find_n_l768_768996

def reverse_digits (n : ℕ) : ℕ := 
  n.digits.reverse.foldl (λ acc d, acc * 10 + d) 0

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits.sum

def remove_zeros (n : ℕ) : ℕ := 
  n.digits.filter (≠ 0).foldr (λ d acc, acc * 10 + d) 0

def condition1 (n : ℕ) : Prop := 
  n = sum_of_digits n * reverse_digits (sum_of_digits n)

def condition2 (n : ℕ) : Prop := 
  let prime_factors := n.factors;
  let sum_of_squares := prime_factors.map (λ p, p * p).sum;
  remove_zeros (sum_of_squares / 2) = n

theorem find_n : ∃ n : ℕ, n = 1729 ∧ condition1 n ∧ condition2 n := 
  by
    use 1729
    -- Proof of conditions will be provided here
    sorry

end find_n_l768_768996


namespace find_number_l768_768395

theorem find_number (n : ℝ) (h : (1/2) * n + 5 = 11) : n = 12 :=
by
  sorry

end find_number_l768_768395


namespace dodecahedron_edge_probability_l768_768361

theorem dodecahedron_edge_probability :
  ∀ (V E : ℕ), 
  V = 20 → 
  ((∀ v ∈ finset.range V, 3 = 3) → -- condition representing each of the 20 vertices is connected to 3 other vertices
  ∃ (p : ℚ), p = 3 / 19) :=
begin
  intros,
  use 3 / 19,
  split,
  sorry
end

end dodecahedron_edge_probability_l768_768361


namespace minimum_sum_of_squares_l768_768636

theorem minimum_sum_of_squares (α p q : ℝ) 
  (h1: p + q = α - 2) (h2: p * q = - (α + 1)) :
  p^2 + q^2 ≥ 5 :=
by
  sorry

end minimum_sum_of_squares_l768_768636


namespace value_of_a_plus_b_l768_768164

theorem value_of_a_plus_b (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 3 ↔ ax^2 + bx + 3 < 0) :
  a + b = -3 :=
sorry

end value_of_a_plus_b_l768_768164


namespace arithmetic_progression_sum_l768_768507

def arithmetic_progression (p : ℕ) (n : ℕ) : ℕ := p + (n - 1) * 3 * p
def sum_first_n_terms (p : ℕ) (n : ℕ) : ℕ := n * (p + arithmetic_progression p n) / 2

def S_p (p : ℕ) : ℕ := sum_first_n_terms p 30

theorem arithmetic_progression_sum : (Finset.range 8).sum (λ p, S_p (p + 1)) = 48060 := by
  sorry

end arithmetic_progression_sum_l768_768507


namespace probability_endpoints_of_edge_l768_768364

noncomputable def num_vertices : ℕ := 12
noncomputable def edges_per_vertex : ℕ := 3

theorem probability_endpoints_of_edge :
  let total_ways := Nat.choose num_vertices 2,
      total_edges := (num_vertices * edges_per_vertex) / 2,
      probability := total_edges / total_ways in
  probability = 3 / 11 := by
  sorry

end probability_endpoints_of_edge_l768_768364


namespace jimmy_bag_weight_l768_768669

def totalWeightSuki (numBags : ℝ) (weightPerBag : ℝ) := numBags * weightPerBag
def totalWeightCombined (numContainers : ℝ) (weightPerContainer : ℝ) := numContainers * weightPerContainer
def totalWeightJimmy (totalWeightCombined : ℝ) (weightSuki : ℝ) := totalWeightCombined - weightSuki
def weightPerBagJimmy (totalWeightJimmy : ℝ) (numBagsJimmy : ℝ) := totalWeightJimmy / numBagsJimmy

theorem jimmy_bag_weight (numBagsSuki weightPerBagSuki numContainers weightPerContainer numBagsJimmy : ℝ) : 
  numBagsSuki = 6.5 →
  weightPerBagSuki = 22 →
  numContainers = 28 →
  weightPerContainer = 8 →
  numBagsJimmy = 4.5 →
  weightPerBagJimmy (totalWeightJimmy (totalWeightCombined numContainers weightPerContainer) (totalWeightSuki numBagsSuki weightPerBagSuki)) numBagsJimmy = 18 :=
by
  intros
  sorry

end jimmy_bag_weight_l768_768669


namespace angle_ABF_l768_768960

noncomputable theory

variables {a b c : ℝ}
variable [fact (0 < a)]
variable [fact (0 < b)]
variable [fact (b < a)]
variable [fact (c = a * (Real.sqrt 5 - 1) / 2)]

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem angle_ABF (h : ellipse a b)
(h_ecc : (Real.sqrt 5 - 1) / 2 = c / a)
: sorry := sorry

end angle_ABF_l768_768960


namespace tan_double_angle_l768_768047

theorem tan_double_angle (α : Real) (h1 : Real.sin α - Real.cos α = 4 / 3) (h2 : α ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4)) :
  Real.tan (2 * α) = (7 * Real.sqrt 2) / 8 :=
by
  sorry

end tan_double_angle_l768_768047


namespace eval_expression_l768_768081

def f (x : ℝ) := Real.sin x - Real.cos x

theorem eval_expression (x : ℝ)
  (h₁ : ∀ x, (D^2 f) x = 2 * f x)
  : (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin (2 * x)) = -19 / 5 := 
sorry

end eval_expression_l768_768081


namespace pair_count_3750_l768_768463

theorem pair_count_3750 :
  (∀ m n : ℕ, (2^2876 < 3^1250) ∧ (3^1250 < 2^2877) →
    (1 ≤ m ∧ m ≤ 2875) →
    (3^n < 2^m ∧ 2^m < 2^(m + 3) ∧ 2^(m + 3) < 3^(n + 1))) →
    (∃ S, S.card = 3750 ∧ ∀ (m, n) ∈ S, 1 ≤ m ∧ m ≤ 2875 ∧ 3^n < 2^m ∧ 2^m < 2^(m + 3) ∧ 2^(m + 3) < 3^(n + 1)) :=
sorry

end pair_count_3750_l768_768463


namespace john_made_47000_from_car_l768_768973

def cost_to_fix_before_discount := 20000
def discount := 0.20
def prize := 70000
def keep_percentage := 0.90

def cost_to_fix_after_discount := cost_to_fix_before_discount - (discount * cost_to_fix_before_discount)
def prize_kept := keep_percentage * prize
def money_made := prize_kept - cost_to_fix_after_discount

theorem john_made_47000_from_car : money_made = 47000 := by
  sorry

end john_made_47000_from_car_l768_768973


namespace find_angle_y_l768_768958

def lines_parallel (m n : Type) [linear_order m] [linear_order n] : Prop :=
∀ (A B : m) (C D : n), A ≤ B → C ≤ D → (B - A) = (D - C)

variables {m n : Type} [linear_order m] [linear_order n]

def angle_30 (a b : ℝ) : Prop := b - a = 30
def angle_40 (a b : ℝ) : Prop := b - a = 40
def angle_y_correct (y : ℝ) : Prop := y = 70

theorem find_angle_y (a b c d e : ℝ) (m n : Type) [linear_order m] [linear_order n]
  (h1 : lines_parallel m n)
  (h2 : angle_30 a b)
  (h3 : angle_40 c d) :
  ∃ y : ℝ, angle_y_correct y :=
sorry

end find_angle_y_l768_768958


namespace max_min_values_l768_768553

noncomputable def f (x k : ℝ) : ℝ := (1 - x) / x + k * Real.log x

theorem max_min_values (k : ℝ) (hk : k < 1 / Real.exp 1) :
  (∀ x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1, f x k ≤ e - k - 1) ∧
  (∀ x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1, f x k ≥ 1 / Real.exp 1 + k - 1) :=
by
  sorry

end max_min_values_l768_768553


namespace sum_of_three_consecutive_integers_product_336_l768_768310

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l768_768310


namespace quadratic_real_roots_range_l768_768158

-- Given conditions and definitions
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def equation_has_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c ≥ 0

-- Problem translated into a Lean statement
theorem quadratic_real_roots_range (m : ℝ) :
  equation_has_real_roots 1 (-2) (-m) ↔ m ≥ -1 :=
by
  sorry

end quadratic_real_roots_range_l768_768158


namespace arithmetic_sequence_common_difference_l768_768181

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (h₁ : a 3 = 0)
  (h₂ : a 1 = 4)
  (h₃ : ∀ n, a n = a 1 + (n - 1) * (-2)) :
  ∃ d, d = -2 :=
by {
  use -2,
  exact h₃ 3,
  sorry
}

end arithmetic_sequence_common_difference_l768_768181


namespace invertible_graphs_l768_768810

def is_invertible (f : ℝ → ℝ) : Prop := ∀ y1 y2, f y1 = f y2 → y1 = y2

def graph1 : ℝ → ℝ := λ x, x^2 - 1   -- Example: a parabolic curve opens upwards
def graph2 : ℝ → ℝ := λ x, x          -- Example: a straight line through the origin
def graph3 : ℝ → ℝ := λ x, 3          -- Example: a horizontal line at y = 3
def graph4 : ℝ → ℝ := λ x, real.sqrt(9 - x^2)  -- Example: the top half of a semicircle
def graph5 : ℝ → ℝ := λ x, x^3 - x    -- Example: a cubic function with one turning point

theorem invertible_graphs :
  is_invertible graph2 ∧ is_invertible graph4 ∧
  ¬ is_invertible graph1 ∧ ¬ is_invertible graph3 ∧ ¬ is_invertible graph5 :=
by
  sorry

end invertible_graphs_l768_768810


namespace simplify_complex_expression_l768_768266

theorem simplify_complex_expression : 
  let i : ℂ := complex.I in -- Imaginary unit
  (3 * (4 - 2 * i) + 2 * i * (3 + i)) = 10 := 
by
  have h1: i^2 = -1 := by sorry -- known property of the imaginary unit
  sorry

end simplify_complex_expression_l768_768266


namespace count_four_digit_integers_with_3_and_6_l768_768112

theorem count_four_digit_integers_with_3_and_6 : 
  (∃ (count : ℕ), count = 16 ∧ 
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → 
  (∀ i : ℕ, i < 4 → (n / (10 ^ i)) % 10 = 3 ∨ (n / (10 ^ i)) % 10 = 6) ↔ n ∈ {3333, 3366, 3633, 3666, 6333, 6366, 6633, 6666}) :=
by
  have h : 2 ^ 4 = 16 := by norm_num
  use 16
  split
  · exact h
  · sorry

end count_four_digit_integers_with_3_and_6_l768_768112


namespace no_silver_matrix_1997_exists_infinitely_many_silver_matrices_l768_768779

def is_silver_matrix (n : ℕ) (M : matrix (fin n) (fin n) ℕ) : Prop :=
  ∀ i : fin n, (finset.univ.image (λj, M i j)).union (finset.univ.image (λj, M j i)) = finset.range (2 * n - 1 + 1)

theorem no_silver_matrix_1997 : ¬ ∃ M : matrix (fin 1997) (fin 1997) ℕ, is_silver_matrix 1997 M := 
sorry

theorem exists_infinitely_many_silver_matrices : ∃ᶠ n in at_top, ∃ M : matrix (fin n) (fin n) ℕ, is_silver_matrix n M := 
sorry

end no_silver_matrix_1997_exists_infinitely_many_silver_matrices_l768_768779


namespace lcm_difference_l768_768498

def lcm3 (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem lcm_difference :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 10 ∧ 1 ≤ b ∧ b ≤ 10 ∧ 1 ≤ c ∧ c ≤ 10 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  504 - 4 = (let min_lcm := lcm3 a b c in
             let max_lcm := lcm3 (if h : a ≠ b then a else a + 1) (if h : b ≠ c then b else b + 1) (if h : a ≠ c then c else c + 1) in
             max_lcm - min_lcm) :=
sorry

end lcm_difference_l768_768498


namespace number_of_special_four_digit_integers_l768_768125

theorem number_of_special_four_digit_integers : 
  let digits := [3, 6]
  let choices_per_digit := 2
  num_digits = 4
  ∑ i in range(num_digits), (choices_per_digit) = 2^4 :=
by
  sorry

end number_of_special_four_digit_integers_l768_768125


namespace positive_t_satisfies_2sqrt17_l768_768513

theorem positive_t_satisfies_2sqrt17 (t : ℝ) (ht_pos : 0 < t) :
  abs (-3 + complex.I * t) = 2 * real.sqrt 17 ↔ t = real.sqrt 59 :=
by
  have : abs (-3 + complex.I * t) = real.sqrt (9 + t^2), from complex.abs_def (-3 + complex.I * t),
  have h_abs_equiv : abs (-3 + complex.I * t) = 2 * real.sqrt 17 ↔ real.sqrt (9 + t^2) = 2 * real.sqrt 17, from this,
  have : real.sqrt (9 + t^2) = 2 * real.sqrt 17 ↔ 9 + t^2 = (2 * real.sqrt 17)^2, from real.sqrt_eq real.sqrt_sq,
  have : (2 * real.sqrt 17)^2 = 4 * 17, by ring_exp,
  have h_t_eq : 9 + t^2 = 4 * 17 ↔ t^2 = 59, from by ring_exp,
  have h_final : t^2 = 59 ↔ t = real.sqrt 59, by real.sqrt_eq ht_pos,
  exact ⟨λ ht, h_final.mpr (h_t_eq.mp (this.mp (h_abs_equiv.mp ht))), λ ht, h_abs_equiv.mpr (this.mpr (h_t_eq.mpr (h_final.mpl ht)))⟩

end positive_t_satisfies_2sqrt17_l768_768513


namespace b4_lt_b7_l768_768441

noncomputable def b : ℕ → ℝ
| 1       := 1 + 1 / α 1
| (n + 1) := 1 + 1 / (α 1 + b n)

theorem b4_lt_b7 (α : ℕ → ℝ) (hα : ∀ k, α k > 0) : b α 4 < b α 7 :=
by { sorry }

end b4_lt_b7_l768_768441


namespace probability_two_vertices_endpoints_l768_768352

theorem probability_two_vertices_endpoints (V E : Type) [Fintype V] [DecidableEq V] 
  (dodecahedron : Graph V E) (h1 : Fintype.card V = 20)
  (h2 : ∀ v : V, Fintype.card (dodecahedron.neighbors v) = 3)
  (h3 : Fintype.card E = 30) :
  (∃ A B : V, A ≠ B ∧ (A, B) ∈ dodecahedron.edgeSet) → 
  (∃ p : ℚ, p = 3/19) := 
sorry

end probability_two_vertices_endpoints_l768_768352


namespace number_of_special_permutations_l768_768985

theorem number_of_special_permutations : 
  let b : Fin 20 → ℕ := sorry in
  (∀ i, 1 ≤ b i ∧ b i ≤ 20) ∧ 
  (∀ j k, j < k → b j ≠ b k) ∧ 
  b 0 > b 1 > b 2 > b 3 > b 4 > b 5 > b 6 > b 7 > b 8 ∧ 
  b 8 < b 9 < b 10 < b 11 < b 12 < b 13 < b 14 < b 15 < b 16 < b 17 < b 18 < b 19 →
  (Finset.card (Finset.univ.image (λ (x : (Fin 20 → ℕ)), x)) = 75582) :=
sorry

end number_of_special_permutations_l768_768985


namespace problem_statement_l768_768615

open Real

noncomputable def sum_sin_ge_nat_sin {n : ℕ} (a : ℝ) (x : Fin n → ℝ) : Prop :=
  (0 < a ∧ a < π / 2) ∧ (∑ i, sin (x i) ≥ n * sin a)

theorem problem_statement {n : ℕ} (a : ℝ) (x : Fin n → ℝ) 
  (h1 : 0 < a) (h2 : a < π / 2)
  (h3 : ∑ i, sin (x i) ≥ n * sin a) :
  ∑ i, sin (x i - a) ≥ 0 := sorry

end problem_statement_l768_768615


namespace fibonacci_sixth_is_eight_l768_768656

def fibonacci : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n+2) := fibonacci n + fibonacci (n+1)

theorem fibonacci_sixth_is_eight :
  fibonacci 5 = 8 :=
by {
  -- proof steps are not needed as mentioned.
  sorry
}

end fibonacci_sixth_is_eight_l768_768656


namespace find_m_collinear_l768_768879

structure Point :=
  (x : ℝ)
  (y : ℝ)

def isCollinear (A B C : Point) : Prop :=
  (B.y - A.y) * (C.x - A.x) = (C.y - A.y) * (B.x - A.x)

theorem find_m_collinear :
  ∀ (m : ℝ),
  let A := Point.mk (-2) 3
  let B := Point.mk 3 (-2)
  let C := Point.mk (1 / 2) m
  isCollinear A B C → m = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end find_m_collinear_l768_768879


namespace problem_statement_l768_768949

noncomputable def ps (r b bl : ℕ) : ℚ :=
  (r * (r - 1) / 2 + b * (b - 1) / 2 + bl * (bl - 1) / 2) / ((r + b + bl) * (r + b + bl - 1) / 2)

noncomputable def pd (r b bl : ℕ) : ℚ :=
  (r * b + r * bl + b * bl) / ((r + b + bl) * (r + b + bl - 1) / 2)

theorem problem_statement :
  let r := 501
      b := 1501
      bl := 1000
  in |ps r b bl - pd r b bl| = 2 / 9 := 
by
  sorry

end problem_statement_l768_768949


namespace max_product_sum_1988_l768_768826

theorem max_product_sum_1988 :
  ∃ (n : ℕ) (a : ℕ), n + a = 1988 ∧ a = 1 ∧ n = 662 ∧ (3^n * 2^a) = 2 * 3^662 :=
by
  sorry

end max_product_sum_1988_l768_768826


namespace solution_to_problem_l768_768990

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable h : f = f'

noncomputable def problem_statement : Prop :=
  (∀ x, f x + f' x > 1) →
  (f 0 = 2018) →
  (∀ x, f x > (2017 / Real.exp x) + 1 ↔ x ∈ Set.Ioi 0)

theorem solution_to_problem : problem_statement f f' h :=
by
  intros h₁ h₂
  sorry

end solution_to_problem_l768_768990


namespace polynomial_inequality_l768_768809

variable {a b c d x1 x2 x3 : ℝ}
variable (phi : ℝ → ℝ)

noncomputable def hasPositiveRealRootsAndNegativeAtZero (phi : ℝ → ℝ) (a b c d x1 x2 x3: ℝ) : Prop :=
  (phi 0 < 0) ∧ (a > 0) ∧ (phi = λ x, a * x^3 + b * x^2 + c * x + d) ∧ 
  (b = -a * (x1 + x2 + x3)) ∧ (c = a * (x1 * x2 + x2 * x3 + x3 * x1)) ∧ (d = -a * (x1 * x2 * x3)) ∧ 
  (x1 > 0) ∧ (x2 > 0) ∧ (x3 > 0)

theorem polynomial_inequality
  (h : hasPositiveRealRootsAndNegativeAtZero phi a b c d x1 x2 x3) :
  2 * b^3 + 9 * a^2 * d - 7 * a * b * c ≤ 0 :=
sorry

end polynomial_inequality_l768_768809


namespace cubic_roots_nature_l768_768004

-- Define the cubic polynomial function
def cubic_poly (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x - 4

-- Define the statement about the roots of the polynomial
theorem cubic_roots_nature :
  ∃ a b c : ℝ, cubic_poly a = 0 ∧ cubic_poly b = 0 ∧ cubic_poly c = 0 
  ∧ 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end cubic_roots_nature_l768_768004


namespace quadratic_roots_k_eq_4_l768_768035

theorem quadratic_roots_k_eq_4 :
  ∀ k : ℝ, (∀ x : ℝ, (2*x^2 - 7*x + k = 0) → (x = (7 + sqrt 17) / 4) ∨ (x = (7 - sqrt 17) / 4)) → k = 4 :=
by
  sorry

end quadratic_roots_k_eq_4_l768_768035


namespace yoongi_age_l768_768169

theorem yoongi_age (Y H : ℕ) (h1 : Y + H = 16) (h2 : Y = H + 2) : Y = 9 :=
by
  sorry

end yoongi_age_l768_768169


namespace repeating_decimal_as_fraction_l768_768484

noncomputable def repeating_decimal_value : ℚ :=
  let x := 2.35 + 35 / 99 in
  x

theorem repeating_decimal_as_fraction :
  repeating_decimal_value = 233 / 99 := by
  sorry

end repeating_decimal_as_fraction_l768_768484


namespace vanessas_mother_picked_14_carrots_l768_768696

-- Define the problem parameters
variable (V : Nat := 17)  -- Vanessa picked 17 carrots
variable (G : Nat := 24)  -- Total good carrots
variable (B : Nat := 7)   -- Total bad carrots

-- Define the proof goal: Vanessa's mother picked 14 carrots
theorem vanessas_mother_picked_14_carrots : (G + B) - V = 14 := by
  sorry

end vanessas_mother_picked_14_carrots_l768_768696


namespace repeating_decimal_eq_fraction_l768_768482

noncomputable def repeating_decimal_to_fraction (x : ℝ) (h : x = 2.353535...) : ℝ :=
  233 / 99

theorem repeating_decimal_eq_fraction :
  (∃ x : ℝ, x = 2.353535... ∧ x = repeating_decimal_to_fraction x (by sorry)) :=
begin
  use 2.353535...,
  split,
  { exact rfl },
  { have h : 2.353535... = 233 / 99, by sorry,
    exact h, }
end

end repeating_decimal_eq_fraction_l768_768482


namespace plywood_perimeter_difference_l768_768745

theorem plywood_perimeter_difference :
  ∃ (p_max p_min : ℕ),
  let plywood_width := 6,
      plywood_height := 9,
      n := 3,
      possible_dimensions := [(6, 3), (2, 9)],
      perimeters := possible_dimensions.map (λ dim, dim.1 + dim.1 + dim.2 + dim.2) in
  p_max = perimeters.maximum ∧
  p_min = perimeters.minimum ∧
  p_max - p_min = 4 :=
sorry

end plywood_perimeter_difference_l768_768745


namespace factorial_expression_l768_768794

theorem factorial_expression :
  (factorial 13 - factorial 12) / factorial 10 = 1584 :=
by
  sorry

end factorial_expression_l768_768794


namespace evaluate_expression_l768_768621

noncomputable def complex_numbers_condition (a b : ℂ) := a ≠ 0 ∧ b ≠ 0 ∧ (a^2 + a * b + b^2 = 0)

theorem evaluate_expression (a b : ℂ) (h : complex_numbers_condition a b) : 
  (a^5 + b^5) / (a + b)^5 = -2 := 
by
  sorry

end evaluate_expression_l768_768621


namespace area_of_rhombus_roots_l768_768022

noncomputable def rhombus_area_of_roots : ℂ :=
  let coeffs : ℕ → ℂ :=
    λ n, match n with
    | 0 => 8 - 2 * complex.I
    | 1 => -(6 + 4 * complex.I)
    | 2 => 1 - 3 * complex.I
    | 3 => 2 + 2 * complex.I
    | 4 => 1
    | _ => 0
  in
  let roots := polynomial.roots (polynomial.of_fn @[coeffs 0, coeffs 1, coeffs 2, coeffs 3, coeffs 4]) in
  let center := -(2 + 2 * complex.I) / 4 in
  let shifted_roots := (λ z, z + center) <$> roots.to_list in
  let product_of_distances := list.prod (shifted_roots.map complex.abs) in
  let pq := real.sqrt ( (25 / 8) ^ 2 + (21 / 8) ^ 2) in
  pq / 2

theorem area_of_rhombus_roots :
  rhombus_area_of_roots = (5 * real.sqrt 218) / 8 :=
sorry

end area_of_rhombus_roots_l768_768022


namespace nail_polish_total_l768_768214

theorem nail_polish_total :
  let kim := 12
  let heidi := kim + 5
  let karen := kim - 4
  heidi + karen = 25 :=
by
  let kim := 12
  let heidi := kim + 5
  let karen := kim - 4
  show heidi + karen = 25 from sorry

end nail_polish_total_l768_768214


namespace power_function_value_l768_768681

theorem power_function_value (a : ℝ) (f g : ℝ → ℝ) (A : ℝ × ℝ)
  (h₁ : f = λ x, log a (x - 1) + 8)
  (h₂ : A = (2, 8))
  (h₃ : g = λ x, x ^ 3)
  (h₄ : g 2 = 8) :
  g 3 = 27 :=
sorry

end power_function_value_l768_768681


namespace distinct_digits_999_l768_768143

open Finset

theorem distinct_digits_999 (S : Finset ℕ) (H₁ : S = {6, 7, 8, 9}) (H₂ : ∀ n ∈ S, n > 5) :
  (∃ (l : List ℕ) (h : l.nodup ∧ ∀ x ∈ l, x ∈ S), l.sum % 9 = 0 ∧ l.length = 3 ∧ l.permutations.length = 6) :=
  sorry

end distinct_digits_999_l768_768143


namespace hyperbola_asymptotes_l768_768085

open Real

noncomputable def hyperbola (x y m : ℝ) : Prop := (x^2 / 9) - (y^2 / m) = 1

noncomputable def on_line (x y : ℝ) : Prop := x + y = 5

theorem hyperbola_asymptotes (m : ℝ) (hm : 9 + m = 25) :
    (∃ x y : ℝ, hyperbola x y m ∧ on_line x y) →
    (∀ x : ℝ, on_line x ((4 / 3) * x) ∧ on_line x (-(4 / 3) * x)) :=
by
  sorry

end hyperbola_asymptotes_l768_768085


namespace part_I_solution_set_part_II_t_range_l768_768641

def f (x : ℝ) : ℝ := |x - 2| - |2 * x + 1|

-- (I) Solve the inequality f(x) ≤ x for its solution set
theorem part_I_solution_set :
  {x : ℝ | f x ≤ x} = set.Ici (1 / 4) :=
sorry

-- (II) If the inequality f(x) ≥ t^2 - t holds for all x ∈ [-2, -1], find the range of the real number t
theorem part_II_t_range (t : ℝ) :
  (∀ x ∈ set.Icc (-2 : ℝ) (-1), f x ≥ t^2 - t) →
  (real.sqrt 5 - 1) / 2 ≤ t ∧ t ≤ (1 + real.sqrt 5) / 2 :=
sorry

end part_I_solution_set_part_II_t_range_l768_768641


namespace smallest_period_of_f_l768_768818

def f (x : ℝ) : ℝ := Real.cos (x / 2) * (Real.sin (x / 2) - Real.sqrt 3 * Real.cos (x / 2))

theorem smallest_period_of_f : (∀ x : ℝ, f (x + 2 * Real.pi) = f x) ∧ (∀ p > 0, (∀ x : ℝ, f (x + p) = f x) → p ≥ 2 * Real.pi) :=
by
  sorry

end smallest_period_of_f_l768_768818


namespace arithmetic_mean_of_integers_from_neg6_to_7_l768_768706

theorem arithmetic_mean_of_integers_from_neg6_to_7 : 
  (list.sum (list.range (7 + 1 + 6) ∘ λ n => n - 6) : ℚ) / (7 - (-6) + 1 : ℚ) = 0.5 := 
by
  sorry

end arithmetic_mean_of_integers_from_neg6_to_7_l768_768706


namespace max_min_values_of_f_l768_768504

noncomputable def f : ℝ → ℝ := λ x, -x^3 + 3 * x - 1

theorem max_min_values_of_f :
  (∃ x : ℝ, f x = 1) ∧ (∀ y : ℝ, f y ≤ 1) ∧
  (∃ x : ℝ, f x = -3) ∧ (∀ y : ℝ, f y ≥ -3) :=
by
  sorry

end max_min_values_of_f_l768_768504


namespace number_of_special_four_digit_integers_l768_768124

theorem number_of_special_four_digit_integers : 
  let digits := [3, 6]
  let choices_per_digit := 2
  num_digits = 4
  ∑ i in range(num_digits), (choices_per_digit) = 2^4 :=
by
  sorry

end number_of_special_four_digit_integers_l768_768124


namespace num_four_digit_integers_with_3_and_6_l768_768107

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end num_four_digit_integers_with_3_and_6_l768_768107


namespace find_lambda_l768_768544

-- Define the vectors AB and AC with given magnitudes and angle between them
structure Vector (α : Type) :=
  (x : α)
  (y : α)

variables {α : Type} [InnerProductSpace ℝ α]

-- Given conditions
variables (A B C P : Vector α)
  (AB AC : Vector α)
  (λ : ℝ)
  (h_angle : ∡ AB AC = 120)
  (h_AB_mag : ‖AB‖ = 2)
  (h_AC_mag : ‖AC‖ = 3)
  (h_perpendicular : ⟪(λ • AB + AC), (AC - AB)⟫ = 0)

-- Define AP vector
noncomputable def AP : Vector α := λ • AB + AC

-- Proof statement
theorem find_lambda : λ = 12 / 7 :=
by
  sorry

end find_lambda_l768_768544


namespace minimal_tetrahedron_volume_when_centroid_l768_768059

-- Define the problem setup
variables (P : Type) [EuclideanSpace P] 
variables (α β γ : Ray P) -- The trihedral angle formed by rays α, β, and γ
variables (M : P) -- The point inside the trihedral angle
variables (π : Plane) -- The plane passing through point M

-- Assume that the plane π intersects the rays forming the trihedral angle at points A, B, C forming a triangle ΔABC
variables (A B C : P)
-- Plane π intersects the rays α, β, γ respectively
variable (hA : α.intersects_plane_at_point A π)
variable (hB : β.intersects_plane_at_point B π)
variable (hC : γ.intersects_plane_at_point C π)

-- Assume the point M is inside the triangle ΔABC
variables (hM : M ∈ triangle_interior (triangle A B C))

-- Define the volume function for the tetrahedron
def tetrahedron_volume (A B C M : P) : ℝ := sorry -- Definition of the volume function goes here

-- Define the centroid of triangle ΔABC
def centroid (A B C : P) : P := sorry -- Definition of the centroid calculation goes here

-- The theorem statement
theorem minimal_tetrahedron_volume_when_centroid (h : ∀ (A B C : P), tetrahedron_volume A B C (centroid A B C) ≤ tetrahedron_volume A B C M) :
  tetrahedron_volume A B C (centroid A B C) = tetrahedron_volume A B C M := 
    sorry

end minimal_tetrahedron_volume_when_centroid_l768_768059


namespace four_digit_3_or_6_l768_768118

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end four_digit_3_or_6_l768_768118


namespace problem_neg_two_in_S_l768_768984

theorem problem_neg_two_in_S
  (S : Set ℤ)
  (h₁ : 0 ∈ S)
  (h₂ : 1996 ∈ S)
  (h₃ : ∀ (P : ℤ[X]), (∀ (c : ℤ), c ∈ S → P.coeff c = 0 → P ≠ 0) → (P ≠ 0) → ∀ root, IsRoot P root → root ∈ S) :
  -2 ∈ S := sorry

end problem_neg_two_in_S_l768_768984


namespace find_angle_A_max_altitude_AD_l768_768199

-- Assuming the existence of triangle ABC with given conditions
variable {a b c : ℝ} {A B C : ℝ}
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) -- the sides are positive
variable (h_eq1 : sqrt 2 * b * sin C + a * sin A = b * sin B + c * sin C) 
variable (h_law_sines : a / sin A = b / sin B ∧ b / sin B = c / sin C)

theorem find_angle_A : A = π / 4 :=
sorry

variable (h_a : a = sqrt 2) -- additional condition for part 2

theorem max_altitude_AD : ∃ AD, AD = 1 + sqrt 2 / 2 :=
sorry

end find_angle_A_max_altitude_AD_l768_768199


namespace similarity_of_midpoints_l768_768994

-- Define notation and basic settings regarding similar triangles
open_locale euclidean_geometry

variables {A B C A' B' C' A'' B'' C'' : Point}

-- Since we are working in Euclidean Geometry, Euclidean space is assumed

-- Similarity condition denoted by is_similar
-- midpoint condition denoted by midpoint P Q R where R is the midpoint of segment [P Q]

theorem similarity_of_midpoints (h1 : is_similar A B C A' B' C')
  (h2 : midpoint A A' A'') (h3 : midpoint B B' B'') (h4 : midpoint C C' C'') :
  is_similar A B C A'' B'' C'' :=
sorry

end similarity_of_midpoints_l768_768994


namespace consecutive_integers_sum_l768_768293

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l768_768293


namespace math_problem_l768_768036

noncomputable def num_divisors (n : ℕ) : ℕ := 
if h : n = 0 then 0 else (nat.divisors n).card

noncomputable def f1 (n : ℕ) : ℕ := 
3 * num_divisors n

noncomputable def f (j : ℕ) (n : ℕ) : ℕ :=
if j = 1 then f1 n else f1 (f (j - 1) n)

def count_values (N : ℕ) (target : ℕ) : ℕ := 
nat.card {n : ℕ | n ≤ N ∧ f 50 n = target}

theorem math_problem (N : ℕ) (target : ℕ) : count_values 100 16 = 1 := 
sorry

end math_problem_l768_768036


namespace plate_acceleration_magnitude_plate_acceleration_direction_l768_768372

variable (R r : ℝ) (m g : ℝ) (alpha : ℝ)

def plate_conditions := (R = 1) ∧ (r = 0.5) ∧ (m = 75) ∧ (alpha = Real.arccos 0.82) ∧ (g = 10)

theorem plate_acceleration_magnitude (h : plate_conditions R r m g alpha) : 
  let a := g * Real.sqrt ((1 - Real.cos alpha) / 2) 
  in a = 3 := by
  sorry

theorem plate_acceleration_direction (h : plate_conditions R r m g alpha) : 
  let direction := Real.arcsin 0.2 
  in direction = (alpha / 2) := by
  sorry

end plate_acceleration_magnitude_plate_acceleration_direction_l768_768372


namespace coefficient_x3_expansion_l768_768678

theorem coefficient_x3_expansion : 
  (polynomial.coeff (((1 - polynomial.X)^5 * (1 + polynomial.X)^3) : polynomial ℚ) 3) = 6 := 
by sorry

end coefficient_x3_expansion_l768_768678


namespace repeating_decimal_as_fraction_l768_768485

noncomputable def repeating_decimal_value : ℚ :=
  let x := 2.35 + 35 / 99 in
  x

theorem repeating_decimal_as_fraction :
  repeating_decimal_value = 233 / 99 := by
  sorry

end repeating_decimal_as_fraction_l768_768485


namespace contradiction_even_numbers_l768_768344

theorem contradiction_even_numbers (a b c : ℕ) :
  ¬(∃ (is_even : ℕ → Prop),
    (is_even a ∨ is_even b ∨ is_even c) ∧
    (is_even a → is_even b → is_even c → False) ∧
    ((is_even a ∧ ¬is_even b ∧ ¬is_even c) ∨
    (¬is_even a ∧ is_even b ∧ ¬is_even c) ∨
    (¬is_even a ∧ ¬is_even b ∧ is_even c)
    )) →
  (∀ (is_odd : ℕ → Prop), 
    (is_odd a ∧ is_odd b ∧ is_odd c) ∨
    (∃ (x y : ℕ), ((x = a ∨ x = b ∨ x = c) ∧
      (y = a ∨ y = b ∨ y = c) ∧
      (x ≠ y) ∧
      (is_even x ∧ is_even y))) :=
begin
  sorry
end

end contradiction_even_numbers_l768_768344


namespace quadratic_real_roots_range_l768_768160

theorem quadratic_real_roots_range (m : ℝ) : (∃ x : ℝ, x^2 - 2 * x - m = 0) → -1 ≤ m := 
sorry

end quadratic_real_roots_range_l768_768160


namespace factorize_expression_l768_768829

theorem factorize_expression (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := 
by 
  sorry

end factorize_expression_l768_768829


namespace find_first_factor_of_lcm_l768_768287

theorem find_first_factor_of_lcm (hcf : ℕ) (A : ℕ) (X : ℕ) (B : ℕ) (lcm_val : ℕ) 
  (h_hcf : hcf = 59)
  (h_A : A = 944)
  (h_lcm_val : lcm_val = 59 * X * 16)
  (h_A_lcm : A = lcm_val) :
  X = 1 := 
by
  sorry

end find_first_factor_of_lcm_l768_768287


namespace find_lambda_l768_768045

variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions as definitions
variables (a b : V) (λ : ℝ)
hypothesis1 : a ⬝ b = 0
hypothesis2 : ‖a‖ = 2
hypothesis3 : ‖b‖ = 3
hypothesis4 : (3 • a + 2 • b) ⬝ (λ • a - b) = 0

-- Statement to prove λ = 3 / 2
theorem find_lambda (a b : V) (λ : ℝ) (hypothesis1 : a ⬝ b = 0) (hypothesis2 : ‖a‖ = 2) 
(hypothesis3 : ‖b‖ = 3) (hypothesis4 : (3 • a + 2 • b) ⬝ (λ • a - b) = 0) : λ = 3 / 2 :=
by
  sorry

end find_lambda_l768_768045


namespace math_scores_individuals_l768_768340

theorem math_scores_individuals :
  ∀ (population : ℕ) (sample_size : ℕ) (math_scores : Vector ℝ sample_size),
  population > 100000 ∧ sample_size = 1000 → 
  (∀ (i : Fin sample_size), math_scores[i] ∈ math_scores) :=
by
  sorry

end math_scores_individuals_l768_768340


namespace apple_cost_l768_768782

theorem apple_cost (cost_per_pound : ℚ) (weight : ℚ) (total_cost : ℚ) : cost_per_pound = 1 ∧ weight = 18 → total_cost = 18 :=
by
  sorry

end apple_cost_l768_768782


namespace orangeade_price_l768_768398

theorem orangeade_price (O W : ℝ) (h1 : O = W) (price_day1 : ℝ) (price_day2 : ℝ) 
    (volume_day1 : ℝ) (volume_day2 : ℝ) (revenue_day1 : ℝ) (revenue_day2 : ℝ) : 
    volume_day1 = 2 * O ∧ volume_day2 = 3 * O ∧ revenue_day1 = revenue_day2 ∧ price_day1 = 0.82 
    → price_day2 = 0.55 :=
by
    intros
    sorry

end orangeade_price_l768_768398


namespace bob_plate_price_correct_l768_768037

-- Assuming units and specific values for the problem
def anne_plate_area : ℕ := 20 -- in square units
def bob_clay_usage : ℕ := 600 -- total clay used by Bob in square units
def bob_number_of_plates : ℕ := 15
def anne_plate_price : ℕ := 50 -- in cents
def anne_number_of_plates : ℕ := 30
def total_anne_earnings : ℕ := anne_number_of_plates * anne_plate_price

-- Condition
def bob_plate_area : ℕ := bob_clay_usage / bob_number_of_plates

-- Prove the price of one of Bob's plates
theorem bob_plate_price_correct : bob_number_of_plates * bob_plate_area = bob_clay_usage →
                                  bob_number_of_plates * 100 = total_anne_earnings :=
by
  intros 
  sorry

end bob_plate_price_correct_l768_768037


namespace ellipse_properties_l768_768090

-- Given conditions
def F1 : ℝ × ℝ := (-sqrt 3, 0)
def F2 : ℝ × ℝ := (sqrt 3, 0)
def a : ℝ := 2
def b : ℝ := 1

-- The ellipse equation
def ellipse_eq (x y : ℝ) := (x^2 / (a^2)) + (y^2 / (b^2)) = 1

-- Prove that the above ellipse_eq is valid and there exists a point E such that the dot product PE·QE is constant
theorem ellipse_properties :
  ellipse_eq x y ∧ 
  ∃ (E : ℝ × ℝ), E = (17 / 8, 0) ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P ≠ Q) → 
    (P = (p1, p2)) → 
    (Q = (q1, q2)) → 
    (l_pass (1, 0) (P Q)) → 
    (intersects P Q (ellipse_eq)) →
    (let PE := (E.1 - P.1, -P.2),
         QE := (E.1 - Q.1, -Q.2) in
     PE.1 * QE.1 + PE.2 * QE.2 = 33 / 64) := 
sorry

noncomputable def l_pass : ℝ × ℝ → ℝ × ℝ → Prop := sorry
noncomputable def intersects : ℝ × ℝ → ℝ × ℝ → (ℝ → ℝ → Prop) → Prop := sorry

end ellipse_properties_l768_768090


namespace parabola_equation_correct_l768_768761

noncomputable def parabola_equation : Prop :=
  ∃ (a b c d e f : ℤ), 
    (a = 4 ∧ b = -20 ∧ c = 25 ∧ d = -40 ∧ e = -16 ∧ f = -509) ∧
    (a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0) ∧
    (a > 0) ∧
    (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.natAbs a) 
                                                  (Int.natAbs b)) 
                                            (Int.natAbs c)) 
                                  (Int.natAbs d)) 
                        (Int.natAbs e)) 
              (Int.natAbs f) = 1)

theorem parabola_equation_correct (x y : ℝ) : parabola_equation := sorry

end parabola_equation_correct_l768_768761


namespace no_distinct_triple_exists_for_any_quadratic_trinomial_l768_768509

theorem no_distinct_triple_exists_for_any_quadratic_trinomial (f : ℝ → ℝ) 
    (hf : ∃ a b c : ℝ, ∀ x, f x = a*x^2 + b*x + c) :
    ¬ ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = b ∧ f b = c ∧ f c = a := 
by 
  sorry

end no_distinct_triple_exists_for_any_quadratic_trinomial_l768_768509


namespace Maria_bought_7_roses_l768_768783

theorem Maria_bought_7_roses
  (R : ℕ)
  (h1 : ∀ f : ℕ, 6 * f = 6 * f)
  (h2 : ∀ r : ℕ, ∃ d : ℕ, r = R ∧ d = 3)
  (h3 : 6 * R + 18 = 60) : R = 7 := by
  sorry

end Maria_bought_7_roses_l768_768783


namespace swimming_class_attendance_l768_768327

theorem swimming_class_attendance (total_students : ℕ) (chess_percentage : ℝ) (swimming_percentage : ℝ) 
  (H1 : total_students = 1000) 
  (H2 : chess_percentage = 0.20) 
  (H3 : swimming_percentage = 0.10) : 
  200 * 0.10 = 20 := 
by sorry

end swimming_class_attendance_l768_768327


namespace right_triangle_from_lengths_l768_768174

variables {A B C D P Q R : Type} [euclidean_geometry A B C D P Q R]

noncomputable def isRightTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_triangle_from_lengths (ABCD : ConvexQuadrilateral) (H1 : ∠ADB + ∠ACB = 30°)
  (H2 : ∠CAB + ∠DBA = 30°) (H3 : AD = BC) :
  ∃ (DB CA DC : ℝ), isRightTriangle DB CA DC :=
sorry

end right_triangle_from_lengths_l768_768174


namespace telescoping_sum_eq_l768_768635

noncomputable def telescoping_sum (x : ℝ) (hx : x > 1) : ℝ :=
  ∑' n : ℕ, 1 / (x^(3^n) - x^(-3^n))

theorem telescoping_sum_eq (x : ℝ) (hx : x > 1) :
  telescoping_sum x hx = 1 / (x - 1) :=
sorry

end telescoping_sum_eq_l768_768635


namespace minimum_length_CD_l768_768367

theorem minimum_length_CD {α β : Plane} (AB CD : ℝ) (angle : ℝ) 
  (h1: parallel α β) (h2: AB = 6) (h3: 0 ≤ angle ∧ angle = 60) (h4: CD > 0) (h5: AB ⊥ CD): 
  CD = 3 * sqrt 3 := 
sorry

end minimum_length_CD_l768_768367


namespace cuboid_volume_l768_768757

theorem cuboid_volume (base_area height : ℝ) (h_base_area : base_area = 18) (h_height : height = 8) : 
  base_area * height = 144 :=
by
  rw [h_base_area, h_height]
  norm_num

end cuboid_volume_l768_768757


namespace marble_difference_l768_768970

variable (Jason_blue_marbles : Nat) (Tom_blue_marbles : Nat)
variable h1 : Jason_blue_marbles = 44
variable h2 : Tom_blue_marbles = 24
variable difference : Nat := Jason_blue_marbles - Tom_blue_marbles

theorem marble_difference : difference = 20 := by
  rw [h1, h2]
  rfl

end marble_difference_l768_768970


namespace problem1_proof_problem2_proof_l768_768799

section Problems

variable {x a : ℝ}

-- Problem 1
theorem problem1_proof : 3 * x^2 * x^4 - (-x^3)^2 = 2 * x^6 := by
  sorry

-- Problem 2
theorem problem2_proof : a^3 * a + (-a^2)^3 / a^2 = 0 := by
  sorry

end Problems

end problem1_proof_problem2_proof_l768_768799


namespace solve_equation_l768_768666

theorem solve_equation:
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ 
    (x - y - x / y - (x^3 / y^3) + (x^4 / y^4) = 2017) ∧ 
    ((x = 2949 ∧ y = 983) ∨ (x = 4022 ∧ y = 2011)) :=
sorry

end solve_equation_l768_768666


namespace trajectory_of_point_G_l768_768533

open Real EuclideanGeometry

noncomputable def is_on_circle (P : ℝ × ℝ) (h : ℝ) (k : ℝ) (r : ℝ) : Prop :=
  (P.fst - h) ^ 2 + (P.snd - k) ^ 2 = r ^ 2

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

noncomputable def perpendicular_bisector (A B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  (P.fst - midpoint A B.fst) * (B.snd - A.snd) +
  (P.snd - midpoint A B.snd) * (A.fst - B.fst) = 0

theorem trajectory_of_point_G :
  ∀ (M N P Q G : ℝ × ℝ),
    is_on_circle P (-sqrt 5) 0 6 →
    N = (sqrt 5, 0) →
    Q.fst = N.fst + 2 * (P.fst - N.fst) / 2 ∧ Q.snd = N.snd + 2 * (P.snd - N.snd) / 2 →
    perpendicular_bisector N P G →
    let e := ellipse M N 3 2 in
    is_on_ellipse G e :=
sorry

end trajectory_of_point_G_l768_768533


namespace defective_items_count_l768_768246

variables 
  (total_items : ℕ)
  (total_video_games : ℕ)
  (total_DVDs : ℕ)
  (total_books : ℕ)
  (working_video_games : ℕ)
  (working_DVDs : ℕ)

theorem defective_items_count
  (h1 : total_items = 56)
  (h2 : total_video_games = 30)
  (h3 : total_DVDs = 15)
  (h4 : total_books = total_items - total_video_games - total_DVDs)
  (h5 : working_video_games = 20)
  (h6 : working_DVDs = 10)
  : (total_video_games - working_video_games) + (total_DVDs - working_DVDs) = 15 :=
sorry

end defective_items_count_l768_768246


namespace find_x_value_l768_768194

theorem find_x_value (x y : ℝ) :
  (∀ x y : ℝ, (y = (1/5 : ℝ) * x ∧ (x, y) ≠ (0, 0)) → 
    (y = 1 ↔ x = 5)) :=
by
  intros
  constructor
  { intro h
    have : 1 = (1/5)*x := h
    have h_eq : x = 5 
    from (eq_div_iff (by norm_num)).1 (by norm_num [*]),
    exact h_eq
  }
  sorry

end find_x_value_l768_768194


namespace luncheon_cost_l768_768276

theorem luncheon_cost
  (s c p : ℝ)
  (h1 : 5 * s + 8 * c + p = 5.25)
  (h2 : 7 * s + 12 * c + p = 7.35) :
  s + c + p = 1.05 :=
begin
  sorry
end

end luncheon_cost_l768_768276


namespace sequence_term_l768_768871

theorem sequence_term :
  (∀ n : ℕ, (∑ k in Finset.range (n + 1), a k) = 2^n + n - 1) →
  a 6 = 33 := by
  intro h
  sorry

end sequence_term_l768_768871


namespace min_value_sin_cos_l768_768505

theorem min_value_sin_cos : ∀ x : ℝ, sin x ^ 6 + 2 * cos x ^ 6 ≥ 1 / 3 := 
by
  sorry

end min_value_sin_cos_l768_768505


namespace rhombus_perimeter_l768_768278

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  let side := real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in 
  4 * side = 16 * real.sqrt 13 := 
by {
  sorry
}

end rhombus_perimeter_l768_768278


namespace tan_20_plus_4_sin_20_eq_sqrt_3_l768_768322

theorem tan_20_plus_4_sin_20_eq_sqrt_3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end tan_20_plus_4_sin_20_eq_sqrt_3_l768_768322


namespace average_salary_is_8900_for_feb_mar_apr_may_l768_768275

theorem average_salary_is_8900_for_feb_mar_apr_may
  (avg_jan_feb_mar_apr : ℝ)
  (sal_jan : ℝ)
  (sal_may : ℝ)
  (avg_some_months : ℝ)
  (total_salary_jan_feb_mar_apr : ℝ)
  (total_salary_feb_mar_apr : ℝ)
  (total_salary_feb_mar_apr_may : ℝ)
  (num_months : ℝ) :
  avg_jan_feb_mar_apr = 8000 →
  sal_jan = 2900 →
  sal_may = 6500 →
  avg_some_months = 8900 →
  total_salary_jan_feb_mar_apr = 32000 →
  total_salary_feb_mar_apr = 29100 →
  total_salary_feb_mar_apr_may = 35600 →
  num_months = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  have h8 : total_salary_feb_mar_apr_may / avg_some_months = 4,
    from calc
      total_salary_feb_mar_apr_may : 35600
      ... / avg_some_months : 8900
      ... = 4
  exact h8

end average_salary_is_8900_for_feb_mar_apr_may_l768_768275


namespace closest_point_on_line_l768_768030

def line (s : ℝ) := (1, 4, 2) + s • (3, -1, 5)

def point : (ℝ × ℝ × ℝ) := (3, 2, 1)

theorem closest_point_on_line : 
  ∃ s : ℝ, 
    let p := line s in
    p = (44 / 35, 137 / 35, 85 / 35) ∧
    ∀ t : ℝ, (let q := line t in 
    (p - point) • (q - p) = 0) :=
sorry

end closest_point_on_line_l768_768030


namespace num_four_digit_36_combinations_l768_768098

theorem num_four_digit_36_combinations : 
  (∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
  (∀ d ∈ [digit n 1000, digit n 100, digit n 10, digit n 1], d = 3 ∨ d = 6)) → 
  16 :=
sorry

end num_four_digit_36_combinations_l768_768098


namespace find_missing_figure_l768_768394

theorem find_missing_figure :
  ∃ x : ℝ, (1.2 / 100) * x = 0.6 ∧ x = 50 :=
begin
  use 50,
  split,
  { simp, linarith },
  { refl }
end

end find_missing_figure_l768_768394


namespace find_a_if_lines_perpendicular_l768_768070

-- Define the lines and the statement about their perpendicularity
theorem find_a_if_lines_perpendicular 
    (a : ℝ)
    (h_perpendicular : (2 * a) / (3 * (a - 1)) = 1) :
    a = 3 :=
by
  sorry

end find_a_if_lines_perpendicular_l768_768070


namespace arithmetic_operators_correct_l768_768831

theorem arithmetic_operators_correct :
  let op1 := (132: ℝ) - (7: ℝ) * (6: ℝ)
  let op2 := (12: ℝ) + (3: ℝ)
  (op1 / op2) = (6: ℝ) := by 
  sorry

end arithmetic_operators_correct_l768_768831


namespace distinguishable_arrangements_l768_768926

theorem distinguishable_arrangements :
  let brown := 1
  let purple := 1
  let green := 3
  let yellow := 3
  let blue := 2
  let total := brown + purple + green + yellow + blue
  (Nat.factorial total) / (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow * Nat.factorial blue) = 50400 := 
by
  let brown := 1
  let purple := 1
  let green := 3
  let yellow := 3
  let blue := 2
  let total := brown + purple + green + yellow + blue
  sorry

end distinguishable_arrangements_l768_768926


namespace smallest_period_of_cosine_half_l768_768472

def f (x : ℝ) : ℝ := cos (x / 2)

theorem smallest_period_of_cosine_half (T : ℝ) : T > 0 ∧ (∀ x : ℝ, f(x + T) = f(x)) ∧ 
  (∀ ε > 0, ∃ δ > 0, δ < ε ∧ δ ∈ {t : ℝ | t > 0 ∧ ∀ x : ℝ, f(x + t) = f(x)}) → T = 4 * π :=
sorry

end smallest_period_of_cosine_half_l768_768472


namespace problem_part_a_problem_part_b_l768_768728

noncomputable def circular_permutations (n : ℕ) : ℕ :=
  (Fintype.card (Equiv.Perm (Fin n))) / n

theorem problem_part_a : circular_permutations 7 = 720 := by
  sorry

noncomputable def necklace_count (n : ℕ) : ℕ :=
  circular_permutations n / 2

theorem problem_part_b : necklace_count 7 = 360 := by
  sorry

end problem_part_a_problem_part_b_l768_768728


namespace chipmunk_families_left_l768_768825

theorem chipmunk_families_left (orig : ℕ) (left : ℕ) (h1 : orig = 86) (h2 : left = 65) : orig - left = 21 := by
  sorry

end chipmunk_families_left_l768_768825


namespace question_1_question_2_l768_768897

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.exp x + a * real.exp (-x)

theorem question_1 (a : ℝ) : (∀ x : ℝ, 0 ≤ x → f x a - (f 0 a) ≥ 0) → a ≤ 1 := sorry

theorem question_2 (m : ℝ) (x : ℝ) : 
  (∀ x : ℝ, m * (f (2*x) 1 + 2) ≥ f x 1 + 1) → m ≥ 3 / 4 := sorry

end question_1_question_2_l768_768897


namespace minimize_expr_l768_768622

-- Define the problem conditions
variables (a b c : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variables (h4 : a * b * c = 8)

-- Define the target expression and the proof goal
def expr := (3 * a + b) * (a + 3 * c) * (2 * b * c + 4)

-- Prove the main statement
theorem minimize_expr : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * b * c = 8) ∧ expr a b c = 384 :=
sorry

end minimize_expr_l768_768622


namespace quadratic_function_properties_maximum_value_on_interval_l768_768525

noncomputable theory

def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x

theorem quadratic_function_properties :
  (f (-2) = -16) ∧ (f 4 = -16) ∧ (∀ x, f x ≤ 2) ∧ 
  (f (1) = 2) := sorry

theorem maximum_value_on_interval (t : ℝ) :
  let ft := x ↦ f x in
  ∃ (max_val : ℝ),
  (if t + 1 ≤ 1 then max_val = ft (t + 1) else if t ≥ 1 then max_val = ft t else max_val = ft 1) :=
sorry

end quadratic_function_properties_maximum_value_on_interval_l768_768525


namespace geometric_mean_7_3sqrt5_7_neg3sqrt5_eq_pm2_l768_768501

def geometric_mean (a b : ℝ) := Real.sqrt (a * b)

theorem geometric_mean_7_3sqrt5_7_neg3sqrt5_eq_pm2 :
  geometric_mean (7 + 3 * Real.sqrt 5) (7 - 3 * Real.sqrt 5) = ±2 :=
by
  let a := 7 + 3 * Real.sqrt 5
  let b := 7 - 3 * Real.sqrt 5
  have h : (a * b) = 4 := by sorry
  show geometric_mean a b = Real.sqrt 4 = ±2 from sorry

end geometric_mean_7_3sqrt5_7_neg3sqrt5_eq_pm2_l768_768501


namespace find_norm_a_l768_768545

variables {a b : ℝ^3}
variables (θ : ℝ) (dot_product : ℝ) (norm_b : ℝ)

def angle_between_vectors (a b : ℝ^3) (θ : ℝ) : Prop :=
cos θ = (a • b) / ((|a|) * (|b|))

def given_conditions : Prop :=
angle_between_vectors a b (2 * Real.pi / 3) ∧ (a • b = -3) ∧ (|b| = 2)

theorem find_norm_a (a b : ℝ^3) (h₁ : angle_between_vectors a b (2 * Real.pi / 3))
(h₂ : a • b = -3)
(h₃ : |b| = 2)
: |a| = 3 := 
sorry

end find_norm_a_l768_768545


namespace triangle_angle_equality_l768_768551

theorem triangle_angle_equality (A B C : ℝ) (h : ∃ (x : ℝ), x^2 - x * (Real.cos A * Real.cos B) - Real.cos (C / 2)^2 = 0 ∧ x = 1) : A = B :=
by {
  sorry
}

end triangle_angle_equality_l768_768551


namespace tangent_segment_equality_l768_768414

-- Definitions for the geometric setup
variable {α : Type*} [metric_space α] [inner_product_space ℝ α]

structure triangle := (A B C : α)

structure circle := (center : α) (radius : ℝ)

def incircle (T : triangle α) (A1 B1 C1 : α) : circle α :=
sorry

def circumcircle (T : triangle α) : circle α :=
sorry

def midpoint (P Q : α) : α :=
sorry

def tangent_segment_length (P : α) (C : circle α) : ℝ :=
sorry

-- Problem statement as a Lean 4 theorem
theorem tangent_segment_equality
  {α : Type*} [metric_space α] [inner_product_space ℝ α]
  (A B C A1 B1 C1 P M : α)
  (T : triangle α)
  (I C : circle α)
  (h_incircle : I = incircle T A1 B1 C1)
  (h_circumcircle : C = circumcircle T)
  (h_P : P ∈ line_through B1 C1)
  (h_P_bc : P ∈ line_through B C)
  (h_M_midpoint : M = midpoint P A1) :
  tangent_segment_length M I = tangent_segment_length M C :=
sorry

end tangent_segment_equality_l768_768414


namespace hypotenuse_of_30_60_90_triangle_l768_768653

theorem hypotenuse_of_30_60_90_triangle (l : ℝ) (θ : ℝ) 
  (leg_condition : l = 15) (angle_condition : θ = 30) : 
  ∃ h : ℝ, h = 30 :=
by
  use 2 * l
  rw [leg_condition, angle_condition]
  trivial

end hypotenuse_of_30_60_90_triangle_l768_768653


namespace sin_value_l768_768538

theorem sin_value (x : ℝ) (h : (sec x + tan x) = 5 / 4) : sin x = 9 / 41 :=
sorry

end sin_value_l768_768538


namespace area_of_semicircle_on_triangle_l768_768983

noncomputable def semicircle_area (r : ℝ) : ℝ := (1 / 2) * real.pi * r^2

theorem area_of_semicircle_on_triangle (AB AC BC : ℝ) (AB_pos : AB = 2) (AC_pos : AC = 3) (BC_pos : BC = 4) :
  ∃ r : ℝ, semicircle_area r = 27 * real.pi / 40 := by
  sorry

end area_of_semicircle_on_triangle_l768_768983


namespace geometric_sum_first_10_l768_768594

noncomputable def geometric_sequence := Σ (a : ℕ → ℝ) (r : ℝ), 
  (a 1 + a 2 = 20) ∧ (a 3 + a 4 = 80) ∧ ∀ n, a (n + 1) = (a n) * r

theorem geometric_sum_first_10 
  (a : ℕ → ℝ) (r : ℝ) 
  (h1 : a 1 + a 2 = 20) 
  (h2 : a 3 + a 4 = 80) 
  (h_geom : ∀ n, a (n + 1) = (a n) * r) : 
  (a 1 + a 2 * r + a 2 * r^2 + a 2 * r^3 + a 2 * r^4 
  + a 2 * r^5 + a 2 * r^6 + a 2 * r^7 + a 2 * r^8 + a 2 * r^9) = 6820 := 
sorry

end geometric_sum_first_10_l768_768594


namespace sheets_in_stack_l768_768428

theorem sheets_in_stack (thickness_per_500_sheets : ℝ) (stack_height : ℝ) (total_sheets : ℕ) :
  thickness_per_500_sheets = 4 → stack_height = 10 → total_sheets = 1250 :=
by
  intros h1 h2
  -- We will provide the mathematical proof steps here.
  sorry

end sheets_in_stack_l768_768428


namespace tan_of_angle_B_l768_768200

theorem tan_of_angle_B (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (hABC : ∃ (c : C), Triangle.hasSum ABC (B, c)) (angleC : ∠ C = 90) (AB : Real := 13) (AC : Real := 5) :
  ∃ (BC : Real), tan B = 12 / 5 :=
by
  assume A B C
  have h1: ∠ C = 90 := angleC,
  have h2: AB = 13 := AB,
  have h3: AC = 5 := AC,
  sorry

end tan_of_angle_B_l768_768200


namespace base_rate_second_telephone_company_l768_768375

theorem base_rate_second_telephone_company : 
  ∃ B : ℝ, (11 + 20 * 0.25 = B + 20 * 0.20) ∧ B = 12 := by
  sorry

end base_rate_second_telephone_company_l768_768375


namespace find_n_l768_768039

theorem find_n (n : ℕ) (h1 : n > 0)
  (h2 : ∃ (p : ℚ), p = 1 / 14 ∧ 
        ∃ (pairs : set (ℕ × ℕ)), pairs = {(1, 4), (2, 3)} ∧ 
        ∑ (x : ℕ × ℕ) in pairs, 1 = 2 ∧ 
        p = 2 / (n * (n - 1) / 2)) : n = 8 :=
by {
  sorry
}

end find_n_l768_768039


namespace containers_per_truck_l768_768336

theorem containers_per_truck (trucks1 boxes1 trucks2 boxes2 boxes_to_containers total_trucks : ℕ)
  (h1 : trucks1 = 7) 
  (h2 : boxes1 = 20) 
  (h3 : trucks2 = 5) 
  (h4 : boxes2 = 12) 
  (h5 : boxes_to_containers = 8) 
  (h6 : total_trucks = 10) :
  (((trucks1 * boxes1) + (trucks2 * boxes2)) * boxes_to_containers) / total_trucks = 160 := 
sorry

end containers_per_truck_l768_768336


namespace no_nat_num_divisible_l768_768605

open Nat

theorem no_nat_num_divisible : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 := sorry

end no_nat_num_divisible_l768_768605


namespace cone_base_diameter_l768_768416

theorem cone_base_diameter
  (h_cone : ℝ) (r_sphere : ℝ) (waste_percentage : ℝ) (d : ℝ) :
  h_cone = 9 → r_sphere = 9 → waste_percentage = 0.75 → 
  (V_cone = 1/3 * π * (d/2)^2 * h_cone) →
  (V_sphere = 4/3 * π * r_sphere^3) →
  (V_cone = (1 - waste_percentage) * V_sphere) →
  d = 9 :=
by
  intros h_cond r_cond waste_cond v_cone_eq v_sphere_eq v_cone_sphere_eq
  sorry

end cone_base_diameter_l768_768416


namespace relationship_between_3a_3b_4a_l768_768404

variable (a b : ℝ)
variable (h : a > b)
variable (hb : b > 0)

theorem relationship_between_3a_3b_4a (a b : ℝ) (h : a > b) (hb : b > 0) :
  3 * b < 3 * a ∧ 3 * a < 4 * a := 
by
  sorry

end relationship_between_3a_3b_4a_l768_768404


namespace sin_30_to_cos_60_and_function_value_l768_768933

theorem sin_30_to_cos_60_and_function_value :
  let f : ℝ → ℝ := λ x, cos (3 * real.arccos x) in
  f (real.sin (real.pi / 6)) = -1 :=
by
  -- Definitions
  def f : ℝ → ℝ := λ x, real.cos (3 * real.arccos x)
  have h1 : real.sin (real.pi / 6) = real.cos (real.pi / 3), from sorry
  calc
    f (real.sin (real.pi / 6)) = f (real.cos (real.pi / 3)) : by rw [h1]
                            ... = real.cos (3 * (real.pi / 3)) : by simp [f]
                            ... = real.cos real.pi : by congr
                            ... = -1 : by norm_num

end sin_30_to_cos_60_and_function_value_l768_768933


namespace reflection_of_vector_l768_768031

section ReflectionOverVector

open Real

noncomputable def reflection_over_vector 
(v u : ℝ × ℝ) : ℝ × ℝ :=
  let p := ((v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)) * u in
  (2 * p.1 - v.1, 2 * p.2 - v.2)

theorem reflection_of_vector :
  reflection_over_vector (1, 2) (2, 1) = (11 / 5, 2 / 5) :=
by {
  sorry
}

end ReflectionOverVector

end reflection_of_vector_l768_768031


namespace polynomial_coefficients_l768_768859

theorem polynomial_coefficients (a : Fin 10 → ℤ) :
  (1 - X) ^ 9 = ∑ i in Finset.range 10, (a i) * X ^ i →
  a 0 = 1 ∧
  a 1 + a 3 + a 5 + a 7 + a 9 = -256 ∧
  (2 : ℤ) * a 1 + (2 : ℤ)^2 * a 2 + (2 : ℤ)^3 * a 3 + (2 : ℤ)^4 * a 4 + (2 : ℤ)^5 * a 5 + 
  (2 : ℤ)^6 * a 6 + (2 : ℤ)^7 * a 7 + (2 : ℤ)^8 * a 8 + (2 : ℤ)^9 * a 9 = -2 := by
  sorry

end polynomial_coefficients_l768_768859


namespace union_sets_l768_768556

open Set

/-- Given sets A and B defined as follows:
    A = {x | -1 ≤ x ∧ x ≤ 2}
    B = {x | x ≤ 4}
    Prove that A ∪ B = {x | x ≤ 4}
--/
theorem union_sets  :
    let A := {x | -1 ≤ x ∧ x ≤ 2}
    let B := {x | x ≤ 4}
    A ∪ B = {x | x ≤ 4} :=
by
    intros A B
    have : A = {x | -1 ≤ x ∧ x ≤ 2} := rfl
    have : B = {x | x ≤ 4} := rfl
    sorry

end union_sets_l768_768556


namespace condition_iff_l768_768002

theorem condition_iff :
  ∀ (a : ℝ), (a > 1 ↔ a > real.sqrt a) :=
by
  intro a
  sorry

end condition_iff_l768_768002


namespace four_digit_3_or_6_l768_768121

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end four_digit_3_or_6_l768_768121


namespace impossible_sum_l768_768201

-- Define the variables for the number of edges, faces, and vertices
variables (e F V : ℕ)
-- Define the functions representing the number of faces with i sides and vertices with i edges
variables (ℓ : ℕ → ℕ) (c : ℕ → ℕ)
-- Define condition that each face is a polygon with an odd number of sides
axiom faces_are_odd : ∀ i, ℓ i > 0 → i % 2 = 1
-- Define condition that each vertex is where an odd number of edges meet
axiom vertices_are_odd : ∀ i, c i > 0 → i % 2 = 1
-- Euler's formula for a convex polyhedron
axiom eulers_formula : V - e + F = 2
-- The sum condition of the problem
axiom sum_condition : ℓ 3 + c 3 = 9

-- Given these conditions, we need to prove it is impossible
theorem impossible_sum : ¬ ∃ (ℓ c : ℕ → ℕ), 
  (∀ i, ℓ i > 0 → i % 2 = 1) ∧
  (∀ i, c i > 0 → i % 2 = 1) ∧
  (ℓ 3 + c 3 = 9) ∧
  (V - e + (ℓ 3 + ℓ 5 + ℓ 7 + ...) + (c 3 + c 5 + c ...
sorry

end impossible_sum_l768_768201


namespace sin_shift_equiv_l768_768343

theorem sin_shift_equiv :
  ∀ (x : ℝ), sin (2 * (x - (π / 12))) = sin (2 * x - π / 6) :=
by
  intro x
  sorry

end sin_shift_equiv_l768_768343


namespace tangent_line_on_x_axis_l768_768542

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 1/4

theorem tangent_line_on_x_axis (x0 a : ℝ) (h1: f x0 a = 0) (h2: (3 * x0^2 + a) = 0) : a = -3/4 :=
by sorry

end tangent_line_on_x_axis_l768_768542


namespace function_properties_l768_768554

def f (x : ℝ) : ℝ := x / (x + 1)

theorem function_properties :
  (∀ x, f x ≠ 1) ∧
  (∀ x, x ≠ -1 → monotone f) ∧
  (∀ x, f x + f (1 / x) = 1) →
  f 1 + ∑ i in (Finset.range 2023).map (Fin.val) (\λ i, f (i + 2) + f (1/(i + 2))) = 4045 / 2 :=
by
  sorry

end function_properties_l768_768554


namespace atomic_weight_of_nitrogen_l768_768836

noncomputable def compound : Type :=
{ atomic_weight : ℕ → ℝ -> ℝ // ∀ N H I : ℝ, H = 1.008 ∧ I = 126.9 ∧ 
                                             let molecular_weight := N + 4 * H + I in
                                             molecular_weight = 145 }

theorem atomic_weight_of_nitrogen : ∃ N : ℝ, (N + 4 * 1.008 + 126.9 = 145) ∧ N = 14.068 :=
by {
  sorry
}

end atomic_weight_of_nitrogen_l768_768836


namespace find_angle_A_l768_768230

-- Definitions of the geometry objects and conditions.
variables {A B C O I : Point}
variable (α β γ δ θ : ℝ)

-- Angles and calculations.
def angle_BIC : ℝ := 140
def angle_BOC (angle_BIC : ℝ) : ℝ := 2 * (angle_BIC - 90)
def angle_A (angle_BOC : ℝ) : set ℝ := {1/2 * angle_BOC, 180 - 1/2 * angle_BOC}
  
-- Given conditions and the statement we want to prove.
theorem find_angle_A (h₁ : circumcenter O A B C) (h₂ : incenter I O B C) (h₃ : angle_BIC = 140) :
  50 ∈ angle_A (angle_BOC angle_BIC) ∧ 130 ∈ angle_A (angle_BOC angle_BIC) :=
by
  sorry

end find_angle_A_l768_768230


namespace part1_part2_l768_768234

def p (m : ℝ) : Prop := ∀ x y : ℝ, (1 ≤ x ∧ x ≤ y) → f(x, m) ≤ f(y, m)
def q (m : ℝ) : Prop := ∀ x : ℝ, (x² - m*x + 1) > 0

-- First part, m = 2 implies p is false
theorem part1 : ¬ p 2 := sorry

-- Second part, if exactly one of p and q is true, provide range of m
theorem part2 : (p m ∧ ¬ q m) ∨ (¬ p m ∧ q m) → m ∈ set.Iic (-2) ∪ set.Ioo 1 2 := sorry

end part1_part2_l768_768234


namespace perpendicular_axis_of_symmetry_point_N_fixed_line_range_of_x_intercept_l768_768543

noncomputable def parabola (x y : ℝ) : Prop :=
  x * x = 4 * y

structure Point :=
  (x : ℝ)
  (y : ℝ)

def on_parabola (P : Point) : Prop :=
  parabola P.x P.y

def dot_product (M A B : Point) : ℝ :=
  (A.x - M.x) * (B.x - M.x) + (A.y - M.y) * (B.y - M.y)

def line_through_AB (A B M : Point) :=
  ∃ λ : ℝ, dot_product M A B = λ

theorem perpendicular_axis_of_symmetry (A B M : Point) (λ : ℝ)
    (hA_on_para : on_parabola A) (hB_on_para : on_parabola B)
    (hM_cond : M.x = 0 ∧ M.y = 4) (h_dot : dot_product M A B = λ) :
  -- Prove that line AB is perpendicular to the axis of symmetry of the parabola
  sorry

theorem point_N_fixed_line (A B M : Point) (λ : ℝ)
    (hA_on_para : on_parabola A) (hB_on_para : on_parabola B)
    (hM_cond : M.x = 0 ∧ M.y = 4) (h_dot : dot_product M A B = λ) :
  -- Prove that point N lies on the line y = -4
  sorry

theorem range_of_x_intercept (A B M : Point) (λ : ℝ)
    (hA_on_para : on_parabola A) (hB_on_para : on_parabola B)
    (hM_cond : M.x = 0 ∧ M.y = 4) (h_dot : 4 ≤ λ ∧ λ ≤ 9)
    (h_dot_prod : dot_product M A B = λ ) :
  -- Find the range of values for the x-intercept of line MN
  sorry

end perpendicular_axis_of_symmetry_point_N_fixed_line_range_of_x_intercept_l768_768543


namespace boys_from_other_communities_l768_768730

theorem boys_from_other_communities (total_boys : ℕ) (percent_muslims percent_hindus percent_sikhs : ℕ) 
    (h_total_boys : total_boys = 300)
    (h_percent_muslims : percent_muslims = 44)
    (h_percent_hindus : percent_hindus = 28)
    (h_percent_sikhs : percent_sikhs = 10) :
  ∃ (percent_others : ℕ), percent_others = 100 - (percent_muslims + percent_hindus + percent_sikhs) ∧ 
                             (percent_others * total_boys / 100) = 54 := 
by 
  sorry

end boys_from_other_communities_l768_768730


namespace train_crosses_pole_in_time_l768_768601

-- Define the conditions
def train_length : ℝ := 130 -- length of the train in meters
def train_speed_kmph : ℝ := 144 -- speed of the train in kilometers per hour

-- Convert speed from km/hr to m/s
def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

-- Theorem stating that the train crosses the electric pole in 3.25 seconds
theorem train_crosses_pole_in_time : train_length / train_speed_mps = 3.25 :=
by
  -- proof goes here
  sorry

end train_crosses_pole_in_time_l768_768601


namespace problem_statement_l768_768719

def fair_coin_twice := [("H", "H"), ("H", "T"), ("T", "H"), ("T", "T")]

def event_A := [("H", "H"), ("H", "T")]
def event_B := [("H", "H"), ("T", "H")]

def probability (e : List (String × String)) := e.length.toFloat / fair_coin_twice.length.toFloat

theorem problem_statement :
  probability (fair_coin_twice.filter (not ∘ event_A.contains)) = 0.5 ∧
  probability ((event_A ++ event_B).eraseDups) = 0.75 ∧
  ¬ (event_A.filter (event_B.contains) = []) ∧
  probability (event_A.filter (event_B.contains)) = (probability event_A) * (probability event_B) :=
by
  sorry

end problem_statement_l768_768719


namespace cookies_per_batch_l768_768924

theorem cookies_per_batch
  (bag_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips : ℕ)
  (h1 : bag_chips = total_chips)
  (h2 : batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips = 81) :
  (bag_chips / batches) / chips_per_cookie = 3 := 
by
  sorry

end cookies_per_batch_l768_768924


namespace count_distinct_sequences_l768_768559

theorem count_distinct_sequences : 
  ∀ (letters : List Char), letters = ['C', 'O', 'M', 'P', 'U', 'T', 'E', 'R'] →
  let sequences := { seq : List Char | seq.length = 4 ∧ seq.head! = 'T' ∧ seq.last! ≠ 'C' ∧ (∀ ch ∈ seq, ch ∈ letters) ∧ seq.nodup } in
  sequences.to_finset.card = 100 :=
by
  sorry

end count_distinct_sequences_l768_768559


namespace dogs_Carly_worked_on_l768_768801

-- Define the parameters for the problem
def total_nails := 164
def three_legged_dogs := 3
def three_nail_paw_dogs := 2
def extra_nail_paw_dog := 1
def regular_dog_nails := 16
def three_legged_nails := (regular_dog_nails - 4)
def three_nail_paw_nails := (regular_dog_nails - 1)
def extra_nail_paw_nails := (regular_dog_nails + 1)

-- Lean statement to prove the number of dogs Carly worked on today
theorem dogs_Carly_worked_on :
  (3 * three_legged_nails) + (2 * three_nail_paw_nails) + extra_nail_paw_nails 
  = 83 → ((total_nails - 83) / regular_dog_nails ≠ 0) → 5 + 3 + 2 + 1 = 11 :=
by sorry

end dogs_Carly_worked_on_l768_768801


namespace sum_of_consecutive_integers_product_336_l768_768307

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l768_768307


namespace smallest_N_with_divisor_digit_sums_1_to_9_is_288_l768_768427

-- Definition of sum of digits
def digit_sum (n : ℕ) : ℕ :=
  n.digits₁0.sum

-- Definition of the condition: divisors' digit sums cover 1 to 9
def covers_1_to_9 (N : ℕ) : Prop :=
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 9 → ∃ d : ℕ, d ∣ N ∧ digit_sum d = m

-- The theorem we need to prove
theorem smallest_N_with_divisor_digit_sums_1_to_9_is_288 :
  ∃ N : ℕ, covers_1_to_9 N ∧ ∀ M : ℕ, (covers_1_to_9 M → N ≤ M) :=
by
  use 288
  split
  · unfold covers_1_to_9
    intro m
    intro h
    sorry -- proof of the sum of digits of divisors of 288 covering 1 to 9

  · intros M H
    sorry -- proof that 288 is the smallest such number

end smallest_N_with_divisor_digit_sums_1_to_9_is_288_l768_768427


namespace lucy_should_give_five_dollars_l768_768568

theorem lucy_should_give_five_dollars : 
    ∀ X : ℕ, 
    (Lucy_initial : ℕ := 20) → 
    (Linda_initial : ℕ := 10) → 
    (Lucy_final := Lucy_initial - X) →
    (Linda_final := Linda_initial + X) →
    (Lucy_final = Linda_final) ↔ X = 5 := 
by
  intros
  unfold Lucy_final Linda_final
  have h : 20 - X = 10 + X ↔ X = 5
  {
    split
    {
      intros h
      linarith
    }
    {
      intros h
      rw h
      linarith
    }
  }
  exact h

end lucy_should_give_five_dollars_l768_768568


namespace number_of_integers_with_abs_le_4_l768_768686

theorem number_of_integers_with_abs_le_4 : 
  ∃ (S : Set Int), (∀ x ∈ S, |x| ≤ 4) ∧ S.card = 9 :=
by
  let S := {x : Int | |x| ≤ 4}
  use S
  have h1: ∀ x ∈ S, |x| ≤ 4 := by
    intros x hx
    exact hx
  have h2: S.card = 9 := sorry
  exact ⟨h1, h2⟩

end number_of_integers_with_abs_le_4_l768_768686


namespace plywood_perimeter_difference_l768_768746

theorem plywood_perimeter_difference :
  ∃ (p_max p_min : ℕ),
  let plywood_width := 6,
      plywood_height := 9,
      n := 3,
      possible_dimensions := [(6, 3), (2, 9)],
      perimeters := possible_dimensions.map (λ dim, dim.1 + dim.1 + dim.2 + dim.2) in
  p_max = perimeters.maximum ∧
  p_min = perimeters.minimum ∧
  p_max - p_min = 4 :=
sorry

end plywood_perimeter_difference_l768_768746


namespace diameter_other_endpoint_l768_768460

theorem diameter_other_endpoint (center : ℝ × ℝ) (endpoint1 : ℝ × ℝ) : (center = (0, 0)) → (endpoint1 = (3, 4)) → (let (x, y) := endpoint1 in (0 - x, 0 - y) = (-3, -4)) :=
by
  intros h1 h2
  sorry

end diameter_other_endpoint_l768_768460


namespace dodecahedron_edge_probability_l768_768357

theorem dodecahedron_edge_probability :
  ∀ (V E : ℕ), 
  V = 20 → 
  ((∀ v ∈ finset.range V, 3 = 3) → -- condition representing each of the 20 vertices is connected to 3 other vertices
  ∃ (p : ℚ), p = 3 / 19) :=
begin
  intros,
  use 3 / 19,
  split,
  sorry
end

end dodecahedron_edge_probability_l768_768357


namespace calculate_overall_duration_of_stoppages_per_hour_l768_768643

noncomputable def overall_duration_of_stoppages_per_hour
  (speed_excluding_stoppages : ℚ)
  (speed_including_stoppages : ℚ) : ℚ :=
  let distance := speed_excluding_stoppages - speed_including_stoppages
  let time := distance / (speed_excluding_stoppages / 60)
  in time

theorem calculate_overall_duration_of_stoppages_per_hour :
  overall_duration_of_stoppages_per_hour 42 27 = 21.43 :=
by
  -- Placeholder for actual proof
  sorry

end calculate_overall_duration_of_stoppages_per_hour_l768_768643


namespace circles_axis_of_symmetry_l768_768910

theorem circles_axis_of_symmetry (a b c d r : ℝ) (h_diff: a ≠ c ∨ b ≠ d) :
  (∃x y, (x = (a + c) / 2 ∧ y = (b + d) / 2) →
    ((∀ x y, ((x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) → ((x - c) ^ 2 + (y - d) ^ 2 = r ^ 2) → 
    (x = (a + c) / 2 ∧ y = (b + d) / 2))) := 
  sorry

end circles_axis_of_symmetry_l768_768910


namespace solvable_system_l768_768496

theorem solvable_system (a : ℝ) : 
  (∃ (b x y : ℝ), x = |y + a| + 4 / a ∧ x^2 + y^2 + 24 + b * (2 * y + b) = 10 * x) ↔ 
  (a ∈ set.Ici (2 / 3) ∨ a < 0) := 
by sorry

end solvable_system_l768_768496


namespace remainders_sum_l768_768720

theorem remainders_sum (n : ℤ) (h₁ : n % 18 = 11) : 
  (let r3 := n % 3 in let r6 := n % 6 in r3 + r6 = 7 ∧ r3 % 2 = 0 ∧ r6 % 2 ≠ 0) :=
by
  let r3 := n % 3
  let r6 := n % 6
  have h_r3 : r3 = 2 := by sorry
  have h_r6 : r6 = 5 := by sorry
  have h_sum : r3 + r6 = 7 := by
    rw [h_r3, h_r6]
    exact rfl
  have h_r3_even : r3 % 2 = 0 := by
    rw h_r3
    exact rfl
  have h_r6_odd : r6 % 2 ≠ 0 := by
    rw h_r6
    exact dec_trivial
  exact ⟨h_sum, h_r3_even, h_r6_odd⟩

end remainders_sum_l768_768720


namespace binary_representation_88_l768_768469

def binary_representation (n : Nat) : String := sorry

theorem binary_representation_88 : binary_representation 88 = "1011000" := sorry

end binary_representation_88_l768_768469


namespace corn_cob_count_l768_768750

theorem corn_cob_count (bushel_weight : ℕ) (ear_weight : ℝ) (bushels_picked : ℕ) :
  bushel_weight = 56 → ear_weight = 0.5 → bushels_picked = 2 → 
  (bushels_picked * bushel_weight) / ear_weight = 224 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end corn_cob_count_l768_768750


namespace gravitational_force_on_asteroid_l768_768683

theorem gravitational_force_on_asteroid :
  ∃ (k : ℝ), ∃ (f : ℝ), 
  (∀ (d : ℝ), f = k / d^2) ∧
  (d = 5000 → f = 700) →
  (∃ (f_asteroid : ℝ), f_asteroid = k / 300000^2 ∧ f_asteroid = 7 / 36) :=
sorry

end gravitational_force_on_asteroid_l768_768683


namespace padic_fibonacci_geometric_repr_l768_768620

-- Variables and conditions
variables (p : ℕ) [fact (p ≠ 2)] [fact (p ≠ 5)]

-- Definition of the roots of the characteristic polynomial
def alpha := (1 + real.sqrt 5) / 2
def beta := (1 - real.sqrt 5) / 2

-- Definition of Fibonacci sequence
def fibonacci_sequence (n : ℕ) : ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci_sequence n + fibonacci_sequence (n+1)

-- Theorem stating the problem
theorem padic_fibonacci_geometric_repr :
  ∃ (A B : ℝ), ∀ (n : ℕ), fibonacci_sequence n = A * alpha^n + B * beta^n :=
sorry

end padic_fibonacci_geometric_repr_l768_768620


namespace john_made_money_l768_768974

theorem john_made_money 
  (repair_cost : ℕ := 20000) 
  (discount_percentage : ℕ := 20) 
  (prize_money : ℕ := 70000) 
  (keep_percentage : ℕ := 90) : 
  (prize_money * keep_percentage / 100) - (repair_cost - (repair_cost * discount_percentage / 100)) = 47000 := 
by 
  sorry

end john_made_money_l768_768974


namespace parabola_chord_solution_l768_768677

noncomputable def parabola_chord : Prop :=
  ∃ x_A x_B : ℝ, (140 = 5 * x_B^2 + 2 * x_A^2) ∧ 
  ((x_A = -5 * Real.sqrt 2 ∧ x_B = 2 * Real.sqrt 2) ∨ 
   (x_A = 5 * Real.sqrt 2 ∧ x_B = -2 * Real.sqrt 2))

theorem parabola_chord_solution : parabola_chord := 
sorry

end parabola_chord_solution_l768_768677


namespace ratio_of_money_spent_on_ice_cream_l768_768259

variables (randy_money_initial randy_lunch_spent randy_money_final : ℕ)
variables (ice_cream_spent money_left_after_lunch gcd : ℕ)

-- Definitions based on the conditions
def initial_money := randy_money_initial = 30
def lunch_spent := randy_lunch_spent = 10
def money_after_lunch := randy_money_initial - randy_lunch_spent
def money_spent_on_ice_cream := randy_money_final = 15
def ice_cream_spent_money := ice_cream_spent = money_after_lunch - randy_money_final
def ratio_simplified := gcd = Nat.gcd ice_cream_spent money_after_lunch

-- The proof problem
theorem ratio_of_money_spent_on_ice_cream :
  initial_money → lunch_spent → money_spent_on_ice_cream → 
  gcd = Nat.gcd (money_after_lunch - randy_money_final) money_after_lunch →
  (ice_cream_spent / gcd) = 1 ∧ (money_after_lunch / gcd) = 4 :=
by
  sorry

end ratio_of_money_spent_on_ice_cream_l768_768259


namespace arithmetic_mean_divisors_is_integer_l768_768632

theorem arithmetic_mean_divisors_is_integer (p q : ℕ) (hp : p.prime) (hq : q.prime) (hpq : p ≠ q) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧
  let n := p^a * q^b in
  let σ := ∑ d in (nat.divisors n), d in
  let mean := σ / (nat.divisors n).card in
  mean = (σ / (nat.divisors n).card) :=
by
  sorry

end arithmetic_mean_divisors_is_integer_l768_768632


namespace number_of_prime_divisors_of_50_fac_l768_768138

-- Define the finite set of prime numbers up to 50
def primes_up_to_50 : finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

-- Main theorem statement
theorem number_of_prime_divisors_of_50_fac :
  (primes_up_to_50.filter prime).card = 15 := 
sorry

end number_of_prime_divisors_of_50_fac_l768_768138


namespace max_squares_on_checkerboard_l768_768711

theorem max_squares_on_checkerboard (n : ℕ) (h1 : n = 7) (h2 : ∀ s : ℕ, s = 2) : ∃ max_squares : ℕ, max_squares = 18 := sorry

end max_squares_on_checkerboard_l768_768711


namespace min_value_2x_plus_y_l768_768739

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 2 / (y + 1) = 2) :
  2 * x + y = 3 :=
sorry

end min_value_2x_plus_y_l768_768739


namespace f_even_and_monotonic_solve_inequality_l768_768082

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.abs x) - 1 / (x ^ 2 + 1)

theorem f_even_and_monotonic :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x y : ℝ, 0 ≤ x → x < y → f x < f y) ∧ (∀ x y : ℝ, x < 0 → x < y → f x > f y) :=
by
  sorry

theorem solve_inequality (x : ℝ) : f x > f (2 * x - 1) ↔ (1 / 3 < x ∧ x < 1) :=
by
  sorry

end f_even_and_monotonic_solve_inequality_l768_768082


namespace redistribute_oil_l768_768328

def total_boxes (trucks1 trucks2 boxes1 boxes2 : Nat) :=
  (trucks1 * boxes1) + (trucks2 * boxes2)

def total_containers (boxes containers_per_box : Nat) :=
  boxes * containers_per_box

def containers_per_truck (total_containers trucks : Nat) :=
  total_containers / trucks

theorem redistribute_oil :
  ∀ (trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks : Nat),
  trucks1 = 7 →
  trucks2 = 5 →
  boxes1 = 20 →
  boxes2 = 12 →
  containers_per_box = 8 →
  total_trucks = 10 →
  containers_per_truck (total_containers (total_boxes trucks1 trucks2 boxes1 boxes2) containers_per_box) total_trucks = 160 :=
by
  intros trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks
  intros h_trucks1 h_trucks2 h_boxes1 h_boxes2 h_containers_per_box h_total_trucks
  sorry

end redistribute_oil_l768_768328


namespace listens_by_end_of_year_l768_768978

theorem listens_by_end_of_year 
  (initial_listens : ℕ) 
  (months_remaining : ℕ) 
  (doubles_each_month : ℕ → ℕ) 
  (h_doubles : ∀ n, doubles_each_month n = 2 * n) 
  (h_initial : initial_listens = 60000) 
  (h_months : months_remaining = 3) 
  : ∑ i in finset.range (months_remaining + 1), (doubles_each_month (initial_listens * 2 ^ i)) = 900000 := 
sorry

end listens_by_end_of_year_l768_768978


namespace julio_orange_soda_bottles_l768_768612

def mateo_orange_bottles : ℕ := 1
def mateo_grape_bottles : ℕ := 3
def bottle_volume : ℕ := 2  -- in liters
def julios_grape_bottles : ℕ := 7
def extra_soda_julio : ℕ := 14  -- in liters

theorem julio_orange_soda_bottles :
  let mateo_total_soda := mateo_orange_bottles * bottle_volume + mateo_grape_bottles * bottle_volume in
  let julio_total_soda := mateo_total_soda + extra_soda_julio in
  let julio_grape_soda := julios_grape_bottles * bottle_volume in
  let julio_orange_soda := julio_total_soda - julio_grape_soda in
  let julio_orange_bottles := julio_orange_soda / bottle_volume in
  julio_orange_bottles = 4 :=
by
  sorry

end julio_orange_soda_bottles_l768_768612


namespace part1_part2_l768_768865

-- Define the function f(x)
def f (x a : ℝ) : ℝ := (4 * x + a) * real.log x / (3 * x + 1)

-- Condition: the slope of the tangent line at (1, f(1)) is perpendicular to x + y + 1 = 0
theorem part1 (a : ℝ) (h : (deriv (λ x => f x a) 1) = 1) : a = 0 :=
sorry

-- Condition: f(x) ≤ mx for any x in [1, e]
def f_m (x : ℝ) : ℝ := (4 * x * real.log x) / (3 * x + 1)

theorem part2 (m : ℝ) (h : ∀ x ∈ set.Icc 1 real.exp, f_m x ≤ m * x) : 
  m ∈ set.Ici (4 / (3 * real.exp + 1)) :=
sorry

end part1_part2_l768_768865


namespace length_of_leg_of_isosceles_right_triangle_l768_768289

def is_isosceles_right_triangle (a b h : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = h^2

def median_to_hypotenuse (m h : ℝ) : Prop :=
  m = h / 2

theorem length_of_leg_of_isosceles_right_triangle (m : ℝ) (h a : ℝ)
  (h1 : median_to_hypotenuse m h)
  (h2 : h = 2 * m)
  (h3 : is_isosceles_right_triangle a a h) :
  a = 15 * Real.sqrt 2 :=
by
  -- Skipping the proof
  sorry

end length_of_leg_of_isosceles_right_triangle_l768_768289


namespace positive_multiples_of_4_with_units_digit_4_l768_768562

theorem positive_multiples_of_4_with_units_digit_4 (n : ℕ) : 
  ∃ n ≤ 15, ∀ m, m = 4 + 10 * (n - 1) → m < 150 ∧ m % 10 = 4 :=
by {
  sorry
}

end positive_multiples_of_4_with_units_digit_4_l768_768562


namespace arithmetic_mean_of_integers_from_neg6_to7_l768_768701

noncomputable def arithmetic_mean : ℝ :=
  let integers := list.range' (-6) 14 -- list of integers from -6 to 7
  let sum := integers.sum
  let count := list.length integers
  (sum : ℝ) / count

theorem arithmetic_mean_of_integers_from_neg6_to7 : arithmetic_mean = 0.5 :=
by
  sorry

end arithmetic_mean_of_integers_from_neg6_to7_l768_768701


namespace collinear_vectors_y_value_l768_768564

theorem collinear_vectors_y_value :
  ∀ (y : ℝ), (∃ k : ℝ, (2, 3) = k • (-6, y)) → y = -9 :=
begin
  intro y,
  intro h,
  cases h with k hk,
  simp at hk,
  cases hk with hx hy,
  have k_val : k = -1 / 3,
  { linarith, },
  rw [k_val] at hy,
  field_simp at hy,
  linarith,
end

end collinear_vectors_y_value_l768_768564


namespace Q_inverse_zero_matrix_l768_768988

noncomputable def Q : Matrix (Fin 2) (Fin 2) ℚ :=
  let v := ![1/real.sqrt 17, 4/real.sqrt 17]
  v ⬝ v.transpose

theorem Q_inverse_zero_matrix : Q⁻¹ = (0 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end Q_inverse_zero_matrix_l768_768988


namespace min_value_g_l768_768162

variables {ℝ : Type*} [linear_ordered_field ℝ] {f g : ℝ → ℝ}

-- Definitions of odd and even functions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Given conditions
variables (h1 : is_odd f)
variables (h2 : is_even g)
variables (h3 : ∀ x, f x + g x = 2^x)

-- Prove statement
theorem min_value_g : (∀ x, f x + g x = 2^x) → (∀ x, is_odd f) → (∀ x, is_even g) → ∃ x, g x = 1 :=
by 
  sorry

end min_value_g_l768_768162


namespace shoe_count_l768_768408

theorem shoe_count 
  (pairs : ℕ)
  (total_shoes : ℕ)
  (prob : ℝ)
  (h_pairs : pairs = 12)
  (h_prob : prob = 0.043478260869565216)
  (h_total_shoes : total_shoes = pairs * 2) :
  total_shoes = 24 :=
by
  sorry

end shoe_count_l768_768408


namespace dodecahedron_edge_probability_l768_768360

theorem dodecahedron_edge_probability :
  ∀ (V E : ℕ), 
  V = 20 → 
  ((∀ v ∈ finset.range V, 3 = 3) → -- condition representing each of the 20 vertices is connected to 3 other vertices
  ∃ (p : ℚ), p = 3 / 19) :=
begin
  intros,
  use 3 / 19,
  split,
  sorry
end

end dodecahedron_edge_probability_l768_768360


namespace number_of_intersections_l768_768817

def ellipse (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1
def vertical_line (x : ℝ) : Prop := x = 4

theorem number_of_intersections : 
    (∃ y : ℝ, ellipse 4 y ∧ vertical_line 4) ∧ 
    ∀ y1 y2, (ellipse 4 y1 ∧ vertical_line 4) → (ellipse 4 y2 ∧ vertical_line 4) → y1 = y2 :=
by
  sorry

end number_of_intersections_l768_768817


namespace num_four_digit_integers_with_3_and_6_l768_768104

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end num_four_digit_integers_with_3_and_6_l768_768104


namespace arithmetic_progression_of_LCMs_impossible_l768_768697
  
theorem arithmetic_progression_of_LCMs_impossible (n : ℕ) 
  (h1 : 100 < n) 
  (numbers : set ℕ) 
  (h2 : set.finite numbers ∧ numbers.card = n) 
  (LCMs : set ℕ) 
  (h3 : LCMs = { lcm a b | a b : ℕ, a ∈ numbers, b ∈ numbers, a ≠ b }) 
  (h4 : ∀ a b ∈ LCMs, ∃ d : ℕ, b = a + d)
  : ¬∃ d : ℕ, d ≠ 0 ∧ LCMs = { k : ℕ | ∃ m : ℕ, k = m * d + (LCMs.min' sorry) } := 
sorry

end arithmetic_progression_of_LCMs_impossible_l768_768697


namespace triangle_area_double_sides_l768_768152

theorem triangle_area_double_sides (a b : ℝ) (θ : ℝ) (sin_θ_pos : sin θ > 0) :
    let A := (a * b * sin θ) / 2
    let A' := (2 * a * 2 * b * sin θ) / 2
  in A' = 4 * A :=
by
  let A := (a * b * sin θ) / 2
  let A' := (2 * a * 2 * b * sin θ) / 2
  sorry

end triangle_area_double_sides_l768_768152


namespace seventh_term_geometric_seq_l768_768283

theorem seventh_term_geometric_seq (a r : ℝ) (h_pos: 0 < r) (h_fifth: a * r^4 = 16) (h_ninth: a * r^8 = 4) : a * r^6 = 8 := by
  sorry

end seventh_term_geometric_seq_l768_768283


namespace area_decrease_1_percent_l768_768941

variable (l w : ℝ)

def percentChangeInArea (l w : ℝ) : ℝ := 
  let A := l * w
  let l' := 1.1 * l
  let w' := 0.9 * w
  let A' := l' * w'
  ((A' - A) / A) * 100

theorem area_decrease_1_percent :
  percentChangeInArea l w = -1 := by
  sorry

end area_decrease_1_percent_l768_768941


namespace rectangle_breadth_l768_768288

theorem rectangle_breadth (sq_area : ℝ) (rect_area : ℝ) (radius_rect_relation : ℝ → ℝ) 
  (rect_length_relation : ℝ → ℝ) (breadth_correct: ℝ) : 
  (sq_area = 3600) →
  (rect_area = 240) →
  (forall r, radius_rect_relation r = r) →
  (forall r, rect_length_relation r = (2/5) * r) →
  breadth_correct = 10 :=
by
  intros h_sq_area h_rect_area h_radius_rect h_rect_length
  sorry

end rectangle_breadth_l768_768288


namespace num_four_digit_integers_with_3_and_6_l768_768109

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end num_four_digit_integers_with_3_and_6_l768_768109


namespace least_cost_to_buy_rice_and_millet_l768_768685

-- Defining the conditions
def needed_rice : ℕ := 1000
def needed_millet : ℕ := 200
def price_per_jin_rice : ℕ := 1
def price_per_jin_millet : ℕ := 2

-- Promotions
def millet_per_10_jin_rice : ℕ := 1
def rice_per_5_jin_millet : ℕ := 2

-- Calculate the total cost needed
theorem least_cost_to_buy_rice_and_millet : 
  Σ (total_cost : ℕ), 
    (total_cost = needed_rice * price_per_jin_rice + (needed_millet - (needed_rice / 10) * millet_per_10_jin_rice) * price_per_jin_millet ∨
     total_cost = needed_millet * price_per_jin_millet + (needed_rice - (needed_millet / 5) * rice_per_5_jin_millet) * price_per_jin_rice) ∧
    (total_cost = 1200) :=
begin
  sorry
end

end least_cost_to_buy_rice_and_millet_l768_768685


namespace min_value_of_derivative_l768_768077

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2 * a * x^2 + (1 / a) * x

noncomputable def f' (a : ℝ) : ℝ := 3 * 2^2 + 4 * a * 2 + (1 / a)

theorem min_value_of_derivative (a : ℝ) (h : a > 0) : 
  f' a ≥ 12 + 8 * Real.sqrt 2 :=
sorry

end min_value_of_derivative_l768_768077


namespace proof_statement_l768_768382

noncomputable def proof_problem : Prop :=
  let a := -(0.175 * 4925)
  let b := (0.3067 * 960)
  let c := (0.7245 * 4500)
  let d := (0.87625 * 1203)
  (a + b + c - d = 1638.77)

theorem proof_statement : proof_problem :=
by
  rw proof_problem
  sorry

end proof_statement_l768_768382


namespace consecutive_integers_product_l768_768302

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l768_768302


namespace sum_of_coefficients_l768_768944

theorem sum_of_coefficients (a b : ℝ) (h : ∀ x : ℝ, (x > 1 ∧ x < 4) ↔ (ax^2 + bx - 2 > 0)) :
  a + b = 2 :=
by
  sorry

end sum_of_coefficients_l768_768944


namespace proportion_equality_l768_768773

variables {A B C R P Q : Type}
variables [Point : Type]
variables [IsTriangle : Triangle A B C]
variables [OnSegment : R ∈ Segment A B]
variables [IsIntersection1 : P = Intersection (LineThrough A (ParallelToLine (Line C R)) (Line B C))]
variables [IsIntersection2 : Q = Intersection (LineThrough B (ParallelToLine (Line C R)) (Line A C))]

theorem proportion_equality :
  ∀ (A B C R P Q : Point),
  IsTriangle A B C →
  R ∈ Segment A B →
  P = Intersection (LineThrough A (ParallelToLine (Line C R))) (Line B C) →
  Q = Intersection (LineThrough B (ParallelToLine (Line C R))) (Line A C) →
  (1 / Distance A P) + (1 / Distance B Q) = (1 / Distance C R) :=
begin
  sorry
end

end proportion_equality_l768_768773


namespace price_of_pants_l768_768239

theorem price_of_pants (P : ℝ) 
  (h1 : (3 / 4) * P + P + (P + 10) = 340)
  (h2 : ∃ P, (3 / 4) * P + P + (P + 10) = 340) : 
  P = 120 :=
sorry

end price_of_pants_l768_768239


namespace quarterback_sacks_for_loss_l768_768764

theorem quarterback_sacks_for_loss (throws : ℕ) (percent_not_thrown : ℕ) (half_of_not_thrown : ℕ) 
  (h1 : throws = 80) 
  (h2 : percent_not_thrown = 30) 
  (h3 : half_of_not_thrown = 2) : 
  let not_thrown := (percent_not_thrown * throws) / 100 in 
  let sacks := not_thrown / half_of_not_thrown in 
  sacks = 12 := 
by 
  sorry

end quarterback_sacks_for_loss_l768_768764


namespace foma_waiting_probability_l768_768015

noncomputable def prob_foma_waiting_max_4_mins : ℝ :=
  let total_area := (10 - 2) * (10 - 2) / 2 in
  let waiting_area := 
    let x₁ := 2 in
    let x₂ := 6 in
    let y₁ := 6 in
    let y₂ := 10 in
    ((x₂ - x₁) + (y₂ - y₁)) * (y₂ - x₁) / 2 in
  (waiting_area / total_area)

theorem foma_waiting_probability :
  prob_foma_waiting_max_4_mins = 3 / 4 := by
  sorry

end foma_waiting_probability_l768_768015


namespace twenty_four_times_ninety_nine_l768_768793

theorem twenty_four_times_ninety_nine : 24 * 99 = 2376 :=
by sorry

end twenty_four_times_ninety_nine_l768_768793


namespace repeating_decimal_eq_fraction_l768_768481

noncomputable def repeating_decimal_to_fraction (x : ℝ) (h : x = 2.353535...) : ℝ :=
  233 / 99

theorem repeating_decimal_eq_fraction :
  (∃ x : ℝ, x = 2.353535... ∧ x = repeating_decimal_to_fraction x (by sorry)) :=
begin
  use 2.353535...,
  split,
  { exact rfl },
  { have h : 2.353535... = 233 / 99, by sorry,
    exact h, }
end

end repeating_decimal_eq_fraction_l768_768481


namespace calculation_correct_l768_768454

theorem calculation_correct : 2 * (3 ^ 2) ^ 4 = 13122 := by
  sorry

end calculation_correct_l768_768454


namespace intersection_A_B_l768_768063

def A : Set ℝ := { x | x ≤ 1 }
def B : Set ℝ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_A_B_l768_768063


namespace find_ellipse_eq_find_chord_length_l768_768068

-- Definitions of conditions
def ellipse_eq (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def hyperbola_eq (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def focal_length (a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 + b^2)

def angle_with_x_axis (angle : ℝ) : Prop :=
  angle = 30 * (real.pi / 180)

-- Proof statements
theorem find_ellipse_eq (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b)
                        (asymptote_angle : angle_with_x_axis 30) (focal_len : focal_length a b = 4 * real.sqrt 2) :
  ellipse_eq 3 (√2) sorry :=  -- Here appropriate constraints on a, b should be assumed to solve the equations and find a, b
sorry

theorem find_chord_length (a b : ℝ)
                         (eq_ellipse : ellipse_eq a b 3 2)
                         (r_focus : 2)
                         (area_max : ∀ m : ℝ, ∃ S_max : ℝ, true) -- Representing area maximization condition
                         : ∃DE : ℝ, DE = 4 :=
sorry

end find_ellipse_eq_find_chord_length_l768_768068


namespace pure_imaginary_x_l768_768569

theorem pure_imaginary_x (x : ℝ) :
  (let z := ((x^2 + 2*x - 3) : ℂ) + (x + 3) * complex.I in
  z.im = 0 ∧ z.re = 0) → (x = 1) :=
by {
  intros h,
  sorry
}

end pure_imaginary_x_l768_768569


namespace find_parallel_line_through_P_l768_768835

noncomputable def line_parallel_passing_through (p : (ℝ × ℝ)) (line : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, _) := line
  let (x, y) := p
  (a, b, - (a * x + b * y))

theorem find_parallel_line_through_P :
  line_parallel_passing_through (4, -1) (3, -4, 6) = (3, -4, -16) :=
by 
  sorry

end find_parallel_line_through_P_l768_768835


namespace positive_difference_diagonals_zero_l768_768747

def original_matrix := [ [1, 2, 3, 4, 5], 
                         [11, 12, 13, 14, 15], 
                         [21, 22, 23, 24, 25], 
                         [31, 32, 33, 34, 35], 
                         [41, 42, 43, 44, 45] ]

def reversed_matrix := [ [1, 2, 3, 4, 5], 
                         [15, 14, 13, 12, 11], 
                         [25, 24, 23, 22, 21], 
                         [35, 34, 33, 32, 31], 
                         [41, 42, 43, 44, 45] ]

def diagonal_sum (matrix : List (List ℕ)) (main : Bool) : ℕ :=
  let idxs := List.range 5
  idxs.foldl (λ acc i => acc + (if main then matrix.get! i |>.get! i else matrix.get! i |>.get! (4 - i))) 0

theorem positive_difference_diagonals_zero :
  abs (diagonal_sum reversed_matrix true - diagonal_sum reversed_matrix false) = 0 :=
by
  sorry

end positive_difference_diagonals_zero_l768_768747


namespace radius_of_spherical_balloon_l768_768767

theorem radius_of_spherical_balloon (r : ℝ) (R : ℝ) (h_r : r = 5 * real.cbrt 2) (h_volume : (4 / 3) * real.pi * R^3 = (2 / 3) * real.pi * r^3) : 
  R = 5 :=
sorry

end radius_of_spherical_balloon_l768_768767


namespace min_value_of_sum_l768_768863

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (1 / (2 * a)) + (1 / b) = 1) :
  a + 2 * b = 9 / 2 :=
sorry

end min_value_of_sum_l768_768863


namespace divisible_by_lcm_of_4_5_6_l768_768928

theorem divisible_by_lcm_of_4_5_6 (n : ℕ) : (∃ k, 0 < k ∧ k < 300 ∧ k % 60 = 0) ↔ (∃! k, k = 4) :=
by
  let lcm_4_5_6 := Nat.lcm (Nat.lcm 4 5) 6
  have : lcm_4_5_6 = 60 := sorry
  have : ∀ k, (0 < k ∧ k < 300 ∧ k % lcm_4_5_6 = 0) ↔ (k = 60 ∨ k = 120 ∨ k = 180 ∨ k = 240) := sorry
  have : ∃ k, 0 < k ∧ k < 300 ∧ k % lcm_4_5_6 = 0 :=
    ⟨60, by norm_num, mk 120, by norm_num, mk 180, by norm_num, mk 240, by norm_num⟩
  have : ∃! k, (k = 60 ∨ k = 120 ∨ k = 180 ∨ k = 240) := sorry
  show _ ↔ (∃! k, k = 4) from sorry

end divisible_by_lcm_of_4_5_6_l768_768928


namespace find_a_l768_768153

-- Here we define the conditions
def z1 (a : ℝ) : ℂ := a - complex.I
def z2 : ℂ := 1 + complex.I
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Here we state the problem
theorem find_a (a : ℝ) (h : is_pure_imaginary (z1 a * z2)) : a = -1 := by
  sorry

end find_a_l768_768153


namespace increasing_interval_minimum_value_in_interval_l768_768096

noncomputable def f (x : ℝ) : ℝ :=
  1 * sin x + (-sqrt 3 * sin (x / 2)) * (2 * sin (x / 2)) + sqrt 3

theorem increasing_interval (k : ℤ) : 
  ∃ (a b : ℝ), [2 * Real.pi * k - (5 * Real.pi / 6), 2 * Real.pi * k + (Real.pi / 6)] = Icc a b := sorry

theorem minimum_value_in_interval :
  ∀ x ∈ Icc (0 : ℝ) (2 * Real.pi / 3), f x ≥ 0 := sorry

end increasing_interval_minimum_value_in_interval_l768_768096


namespace distance_between_fourth_and_work_l768_768237

theorem distance_between_fourth_and_work (x : ℝ) (h₁ : x > 0) :
  let total_distance := x + 0.5 * x + 2 * x
  let to_fourth := (1 / 3) * total_distance
  let total_to_fourth := total_distance + to_fourth
  3 * total_to_fourth = 14 * x :=
by
  sorry

end distance_between_fourth_and_work_l768_768237


namespace total_percent_decrease_l768_768727

theorem total_percent_decrease (initial_value : ℝ) (val1 val2 : ℝ) :
  initial_value > 0 →
  val1 = initial_value * (1 - 0.60) →
  val2 = val1 * (1 - 0.10) →
  (initial_value - val2) / initial_value * 100 = 64 :=
by
  intros h_initial h_val1 h_val2
  sorry

end total_percent_decrease_l768_768727


namespace plywood_cut_perimeter_difference_l768_768741

theorem plywood_cut_perimeter_difference :
  ∀ (length width : ℕ), length = 6 ∧ width = 9 → 
  ∃ p1 p2 : ℕ, 
    (∃ (config1 : length ≠ 0 ∧ width ≠ 0), p1 = 2 * (3 + width)) ∧
    (∃ (config2 : length ≠ 0 ∧ width ≠ 0), p2 = 2 * (6 + 3)) ∧
    (∀ n : ℕ, n = 3 → ∃ cut : length * width = 3 * (length * width / 3))),
  abs (p1 - p2) = 6 := 
by
  intro length width h
  obtain ⟨h1, h2⟩ := h
  have config1 := 2 * (3 + 9)
  have config2 := 2 * (6 + 3)
  have h3 := 6 * 9 = 3 * (6 * 9 / 3)
  use config1, config2
  split
  . use (6 ≠ 0 ∧ 9 ≠ 0)
    exact rfl
  . split
    . use (6 ≠ 0 ∧ 9 ≠ 0)
      exact rfl
    . intro n hn
      use h3
  rw [abs_eq_nat]
  rw [config1, config2]
  exact rfl

end plywood_cut_perimeter_difference_l768_768741


namespace derivative_of_log_base_3_derivative_of_exp_base_2_l768_768389

noncomputable def log_base_3_deriv (x : ℝ) : ℝ := (Real.log x / Real.log 3)
noncomputable def exp_base_2_deriv (x : ℝ) : ℝ := Real.exp (x * Real.log 2)

theorem derivative_of_log_base_3 (x : ℝ) (h : x > 0) :
  (log_base_3_deriv x) = (1 / (x * Real.log 3)) :=
by
  sorry

theorem derivative_of_exp_base_2 (x : ℝ) :
  (exp_base_2_deriv x) = (Real.exp (x * Real.log 2) * Real.log 2) :=
by
  sorry

end derivative_of_log_base_3_derivative_of_exp_base_2_l768_768389


namespace tan_sin_sum_eq_sqrt3_l768_768325

theorem tan_sin_sum_eq_sqrt3 (tan20 sin20 : ℝ) (h1 : tan 20 = sin 20 / cos 20) (h2 : sin20 = sin 20) :
  tan20 + 4 * sin20 = sqrt 3 := by
  sorry

end tan_sin_sum_eq_sqrt3_l768_768325


namespace m_not_in_P_l768_768906

def P : set ℝ := { x | 0 ≤ x ∧ x ≤ Real.sqrt 2 }
def m := Real.sqrt 3

theorem m_not_in_P : m ∉ P :=
by 
  have h : Real.sqrt 3 > Real.sqrt 2 := Real.sqrt_lt.mpr (by norm_num)
  simp [P, m]
  intro h_mem
  have := h_mem.2
  linarith

end m_not_in_P_l768_768906


namespace relay_race_distance_l768_768038

theorem relay_race_distance (x : ℝ) (h : 0 ≤ x) (h1 : x ≤ 100)
  (h2 : (∃ s1 s2 s3 : ℝ, 100 = x + s1 + s2 + s3 ∧ 
                           (∀ d ∈ {x, s1, s2, s3}.pairs, d.1 ≤ 3 * d.2 ∧ d.2 ≤ 3 * d.1))):
  10 ≤ x ∧ x ≤ 50 :=
by
  sorry

end relay_race_distance_l768_768038


namespace solve_inequality_l768_768833
open Set

theorem solve_inequality (x : ℝ) : 
  (2*x + 1) / (x + 2) - (1 / 3) * ((x - 3) / (x - 2)) ≤ 0 ↔ x ∈ Ioo (-2 : ℝ) 0 ∪ Ioo (2 : ℝ) (14 / 5) ∪ {(14 / 5) : ℝ} := 
sorry

end solve_inequality_l768_768833


namespace jake_has_3_peaches_l768_768610

-- Define the number of peaches Steven has.
def steven_peaches : ℕ := 13

-- Define the number of peaches Jake has based on the condition.
def jake_peaches (P_S : ℕ) : ℕ := P_S - 10

-- The theorem that states Jake has 3 peaches.
theorem jake_has_3_peaches : jake_peaches steven_peaches = 3 := sorry

end jake_has_3_peaches_l768_768610


namespace parallel_lines_necessary_not_sufficient_l768_768007

variables {a1 b1 a2 b2 c1 c2 : ℝ}

def determinant (a1 b1 a2 b2 : ℝ) : ℝ := a1 * b2 - a2 * b1

theorem parallel_lines_necessary_not_sufficient
  (h1 : a1^2 + b1^2 ≠ 0)
  (h2 : a2^2 + b2^2 ≠ 0)
  : (determinant a1 b1 a2 b2 = 0) → 
    (a1 * x + b1 * y + c1 = 0 ∧ a2 * x + b2 * y + c2 =0 → exists k : ℝ, (a1 = k ∧ b1 = k)) ∧ 
    (determinant a1 b1 a2 b2 = 0 → (a2 * x + b2 * y + c2 = a1 * x + b1 * y + c1 → false)) :=
sorry

end parallel_lines_necessary_not_sufficient_l768_768007


namespace find_defective_pens_l768_768579

noncomputable def defective_pens (total_pens defective_count : ℕ) : Prop :=
  ∃ (non_defective_count : ℕ),
    total_pens = defective_count + non_defective_count ∧
    (non_defective_count * (non_defective_count - 1)) / (total_pens * (total_pens - 1)) = 0.4242424242424242

theorem find_defective_pens :
  defective_pens 12 4 :=
sorry

end find_defective_pens_l768_768579


namespace average_rainfall_correct_l768_768172

/-- In July 1861, 366 inches of rain fell in Cherrapunji, India. -/
def total_rainfall : ℤ := 366

/-- July has 31 days. -/
def days_in_july : ℤ := 31

/-- Each day has 24 hours. -/
def hours_per_day : ℤ := 24

/-- The total number of hours in July -/
def total_hours_in_july : ℤ := days_in_july * hours_per_day

/-- The average rainfall in inches per hour during July 1861 in Cherrapunji, India -/
def average_rainfall_per_hour : ℤ := total_rainfall / total_hours_in_july

/-- Proof that the average rainfall in inches per hour is 366 / (31 * 24) -/
theorem average_rainfall_correct : average_rainfall_per_hour = 366 / (31 * 24) :=
by
  /- We skip the proof as it is not required. -/
  sorry

end average_rainfall_correct_l768_768172


namespace area_of_triangle_NF1F2_l768_768891

noncomputable def hyperbola_eqn : (ℝ × ℝ) → ℝ :=
  λ (p : ℝ × ℝ), p.1^2 - p.2^2

def passes_through (p : ℝ × ℝ) (λ : ℝ) : Prop :=
  hyperbola_eqn p = λ

def point_N_condition (N F1 F2 : ℝ × ℝ) : Prop :=
  let m := dist N F1 in
  let n := dist N F2 in
  abs (m - n) = 2 * real.sqrt 3 ∧ m^2 + n^2 = 24

def area_triangle (a b c : ℝ × ℝ) : ℝ :=
  let base = dist a b in
  let height = dist a c in
  1/2 * base * height

theorem area_of_triangle_NF1F2 :
  ∃ (F1 F2 N : ℝ × ℝ), passes_through (2, 1) 3 ∧
    point_N_condition N F1 F2 ∧
    area_triangle N F1 F2 = 3 :=
begin
  sorry
end

end area_of_triangle_NF1F2_l768_768891


namespace triangle_angle_A_l768_768873

theorem triangle_angle_A (AC BC : ℝ) (angle_B : ℝ) (h_AC : AC = Real.sqrt 2) (h_BC : BC = 1) (h_angle_B : angle_B = 45) :
  ∃ (angle_A : ℝ), angle_A = 30 :=
by
  sorry

end triangle_angle_A_l768_768873


namespace jennifer_green_sweets_l768_768207

theorem jennifer_green_sweets :
  ∀ (blue_sweets yellow_sweets : ℕ),
  blue_sweets = 310 →
  yellow_sweets = 502 →
  (∀ (num_people sweets_per_person total_sweets : ℕ),
  num_people = 4 →
  sweets_per_person = 256 →
  total_sweets = num_people * sweets_per_person →
  ∃ (green_sweets : ℕ), total_sweets = blue_sweets + yellow_sweets + green_sweets ∧ green_sweets = 212) :=
begin
  sorry
end

end jennifer_green_sweets_l768_768207


namespace coloring_satisfies_requirements_l768_768948

def is_lattice_point (p : ℤ × ℤ) : Prop := true

def color_lattice_point (p : ℤ × ℤ) : Color :=
if (p.1 + p.2) % 2 = 1 then Color.red 
else if (p.1 % 2 = 1 ∧ p.2 % 2 = 0) then Color.white 
else if (p.1 % 2 = 0 ∧ p.2 % 2 = 1) then Color.black 
else Color.undefined

theorem coloring_satisfies_requirements :
  (∀ y : ℤ, ∃ (infinitely many red x : ℤ), color_lattice_point (x, y) = Color.red) ∧
  (∀ y : ℤ, ∃ (infinitely many white x : ℤ), color_lattice_point (x, y) = Color.white) ∧
  (∀ y : ℤ, ∃ (infinitely many black x : ℤ), color_lattice_point (x, y) = Color.black) ∧
  (∀ (A B C : ℤ × ℤ), color_lattice_point A = Color.white → color_lattice_point B = Color.red → color_lattice_point C = Color.black →
    ∃ D : ℤ × ℤ, color_lattice_point D = Color.red ∧ 
    (D = (C.1 + (A.1 - B.1), C.2 + (A.2 - B.2)) ∧ (A.1 + C.1) = (B.1 + D.1) ∧ (A.2 + C.2) = (B.2 + D.2))) :=
by sorry

end coloring_satisfies_requirements_l768_768948


namespace trapezoid_midsegment_l768_768598

theorem trapezoid_midsegment (a b : ℝ)
  (AB CD E F: ℝ) -- we need to indicate that E and F are midpoints somehow
  (h1 : AB = a)
  (h2 : CD = b)
  (h3 : AB = CD) 
  (h4 : E = (AB + CD) / 2)
  (h5 : F = (CD + AB) / 2) : 
  EF = (1/2) * (a - b) := sorry

end trapezoid_midsegment_l768_768598


namespace volume_of_solid_l768_768260

/--
Given an isosceles right triangle with a hypotenuse of length 4,
the volume of the solid formed by rotating this triangle around the line containing its hypotenuse is $\frac{16\pi}{3}$.
-/
theorem volume_of_solid (hypotenuse : ℝ) (h : hypotenuse = 4) :
  let a := hypotenuse / Math.sqrt 2
  let b := 2 * (1 / 3 : ℝ) * Real.pi * a^2 * a
  b = 16 * Real.pi / 3 :=
by
  -- Using a hypothetical Lean function to calculate the radius from hypotenuse
  let radius := hypotenuse / Real.sqrt 2
  -- The height of each cone in the solid
  let height := radius
  -- The volume of each cone
  have volume_cone : ℝ := (1 / 3 : ℝ) * Real.pi * radius^2 * height
  -- The volume of the entire solid
  let volume_solid := 2 * volume_cone
  -- Prove the final volume
  calc
    b = 2 * volume_cone : by sorry
    ... = 2 * ((1 / 3 : ℝ) * Real.pi * radius^2 * radius) : by sorry
    ... = 16 * Real.pi / 3 : by sorry

end volume_of_solid_l768_768260


namespace repeating_decimal_eq_fraction_l768_768480

noncomputable def repeating_decimal_to_fraction (x : ℝ) (h : x = 2.353535...) : ℝ :=
  233 / 99

theorem repeating_decimal_eq_fraction :
  (∃ x : ℝ, x = 2.353535... ∧ x = repeating_decimal_to_fraction x (by sorry)) :=
begin
  use 2.353535...,
  split,
  { exact rfl },
  { have h : 2.353535... = 233 / 99, by sorry,
    exact h, }
end

end repeating_decimal_eq_fraction_l768_768480


namespace route_B_no_quicker_l768_768245

noncomputable def time_route_A (distance_A : ℕ) (speed_A : ℕ) : ℕ :=
(distance_A * 60) / speed_A

noncomputable def time_route_B (distance_B : ℕ) (speed_B1 : ℕ) (speed_B2 : ℕ) : ℕ :=
  let distance_B1 := distance_B - 1
  let distance_B2 := 1
  (distance_B1 * 60) / speed_B1 + (distance_B2 * 60) / speed_B2

theorem route_B_no_quicker : time_route_A 8 40 = time_route_B 6 50 10 :=
by
  sorry

end route_B_no_quicker_l768_768245


namespace part_I_part_II_l768_768078

variable {x a : ℝ}

-- (I)
theorem part_I : (a = 1) → (|x - a| + |2 * x - 1| ≥ 2 ↔ x ∈ set.Iic 0 ∪ set.Ici (4 / 3)) :=
sorry

-- (II)
theorem part_II : (|x - a| + |2 * x - 1| ≥ |a - 1 / 2|) :=
sorry

end part_I_part_II_l768_768078


namespace fourth_root_approx_62_l768_768806

theorem fourth_root_approx_62 : ∃ (x : ℝ), x^4 = 13824000 ∧ x ≈ 62 :=
sorry

end fourth_root_approx_62_l768_768806


namespace root_in_interval_l768_768597

noncomputable def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem root_in_interval : (∀ x ∈ set.Icc (1 : ℝ) (2 : ℝ), continuous_at f x) → 
                           (f 2 > 0) → 
                           (f 1.5 < 0) → 
                           (f 1.75 > 0) → 
                           ∃ x ∈ set.Ioo (1.5 : ℝ) (1.75 : ℝ), f x = 0 :=
by
  intros
  apply exists_of_continuous_on_ivt
  sorry

end root_in_interval_l768_768597


namespace polynomial_perfect_square_condition_l768_768387

theorem polynomial_perfect_square_condition (a b c d e f : ℝ) :
  (∃ x y z : ℝ, (a * x^2 + b * y^2 + c * z^2 + 2 * d * x * y + 2 * e * y * z + 2 * f * z * x = (a * x^2 + b * y^2 + c * z^2 + 2 * sqrt(a) * sqrt(b) * x * y + 2 * sqrt(b) * sqrt(c) * y * z + 2 * sqrt(c) * sqrt(a) * z * x)) ↔
  (a * b = d^2) ∧ (b * c = e^2) ∧ (c * a = f^2) ∧ (a * e = d * f) ∧ (b * f = d * e) ∧ (c * d = e * f)) :=
sorry

end polynomial_perfect_square_condition_l768_768387


namespace find_pairs_of_positive_numbers_l768_768832

theorem find_pairs_of_positive_numbers
  (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (exists_triangle : ∃ (C D E A B : ℝ), true)
  (points_on_hypotenuse : ∀ (C D E A B : ℝ), A ∈ [D, E] ∧ B ∈ [D, E]) 
  (equal_vectors : ∀ (D A B E : ℝ), (D - A) = (A - B) ∧ (A - B) = (B - E))
  (AC_eq_a : (C - A) = a)
  (BC_eq_b : (C - B) = b) :
  (1 / 2) < (a / b) ∧ (a / b) < 2 :=
by {
  sorry
}

end find_pairs_of_positive_numbers_l768_768832


namespace minimize_product_of_roots_l768_768467

noncomputable def polynomial_min_product (p q : ℝ) (q_poly : ℝ → ℝ) : ℝ :=
  q_poly 1

theorem minimize_product_of_roots (p q : ℝ) :
  let q_poly := λ x, x^2 - p * x + q in
  (∀ y, q_poly y = 0 → y ∈ set.univ → ∃ u v1 v2 v3 : ℝ, u ≠ v1 ∧ u ≠ v2 ∧ u ≠ v3 ∧ v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3 ∧ (q_poly u)*(q_poly v1)*(q_poly v2)*(q_poly v3) < (min_value (λ a b c d, (q_poly a)*(q_poly b)*(q_poly c)*(q_poly d))) ) →
  polynomial_min_product p q (λ x, x^2 - p * x + q) = <specific value> :=
sorry

end minimize_product_of_roots_l768_768467


namespace worth_of_cloth_sold_l768_768447

-- Define the given conditions
def commission_rate : ℝ := 2.5 / 100
def commission : ℝ := 21

-- Define the proof problem
theorem worth_of_cloth_sold :
  ∃ (total_sales : ℝ), total_sales = 840 ∧ commission = commission_rate * total_sales :=
by
  -- Placeholder for proof, specifying the theorem statement
  sorry

end worth_of_cloth_sold_l768_768447


namespace no_such_function_exists_l768_768820

theorem no_such_function_exists :
  ¬ (∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2) :=
sorry

end no_such_function_exists_l768_768820


namespace arithmetic_mean_of_integers_from_neg6_to7_l768_768702

noncomputable def arithmetic_mean : ℝ :=
  let integers := list.range' (-6) 14 -- list of integers from -6 to 7
  let sum := integers.sum
  let count := list.length integers
  (sum : ℝ) / count

theorem arithmetic_mean_of_integers_from_neg6_to7 : arithmetic_mean = 0.5 :=
by
  sorry

end arithmetic_mean_of_integers_from_neg6_to7_l768_768702


namespace problem_solution_l768_768191

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * cos theta, rho * sin theta)

def line_l (t : ℝ) : ℝ × ℝ :=
  (- (sqrt 3 / 2) * t, -5 + (1 / 2) * t)

def point_C : ℝ × ℝ :=
  (sqrt 3, 0)

theorem problem_solution : 
  (∃ rho theta, 2 * sqrt 3 * cos theta = rho) →
  (∃ x y, (x - sqrt 3) ^ 2 + y ^ 2 = 3) ∧ 
  (∃ t, (∃ (P : ℝ × ℝ), P = line_l t) ∧ 
        (∃ P_x P_y, P = (P_x, P_y) ∧ t = 1 ∧ 
          min (sqrt ((P_x - sqrt 3) ^ 2 + P_y ^ 2)) = sqrt 27) ∧ 
          line_l 1 = (- sqrt 3 / 2, - 9 / 2)) :=
by {
  sorry
}

end problem_solution_l768_768191


namespace lg_five_l768_768884

theorem lg_five (h : log 2 = 0.3010) : log 5 = 0.6990 :=
by sorry

end lg_five_l768_768884


namespace upper_bound_on_antichain_size_l768_768216

theorem upper_bound_on_antichain_size (S : Type *) (n : ℕ) (F : Finset (Finset S)) 
  (hS : F.card ≤ n)
  (hF : ∀ (A B : Finset S), A ∈ F → B ∈ F → A ⊆ B → A = B) : 
  F.card ≤ nat.choose n (n / 2) := sorry

end upper_bound_on_antichain_size_l768_768216


namespace four_digit_integer_l768_768279

theorem four_digit_integer (a b c d : ℕ) (h1 : a + b + c + d = 18)
  (h2 : b + c = 11) (h3 : a - d = 1) (h4 : 11 ∣ (1000 * a + 100 * b + 10 * c + d)) :
  1000 * a + 100 * b + 10 * c + d = 4653 :=
by sorry

end four_digit_integer_l768_768279


namespace sum_of_consecutive_integers_product_336_l768_768306

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l768_768306


namespace transversal_line_construct_l768_768093

theorem transversal_line_construct {P : Type} [MetricSpace P] {g l : AffineSubspace ℝ P} 
  (hg : line g) (hl : line l) 
  (h_intersect : g ∩ l ≠ ∅) 
  {α β : ℝ}
  (hα : 0 ≤ α ∧ α ≤ π)
  (hβ : 0 ≤ β ∧ β ≤ π) :
  ∃ t : AffineSubspace ℝ P, line t ∧
  (∃ e f : AffineSubspace ℝ P, line e ∧ line f ∧ 
  e ∥ g ∧ f ∥ l ∧ 
  angle t g = α ∧ angle t l = β) :=
sorry

end transversal_line_construct_l768_768093


namespace anna_scores_statistics_l768_768781

theorem anna_scores_statistics :
  let scores := [86, 90, 92, 98, 87, 95, 93] in
  ( (scores.sum.toFloat / scores.length.toFloat == 91.5714) ∧
    (scores.sorted.inth (scores.length / 2) == 92) ) :=
by
  let scores := [86, 90, 92, 98, 87, 95, 93]
  sorry

end anna_scores_statistics_l768_768781


namespace corn_cob_count_l768_768749

theorem corn_cob_count (bushel_weight : ℕ) (ear_weight : ℝ) (bushels_picked : ℕ) :
  bushel_weight = 56 → ear_weight = 0.5 → bushels_picked = 2 → 
  (bushels_picked * bushel_weight) / ear_weight = 224 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end corn_cob_count_l768_768749


namespace no_x_satisfies_inequalities_l768_768020

theorem no_x_satisfies_inequalities : ¬ ∃ x : ℝ, 4 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 9 * x - 5 :=
sorry

end no_x_satisfies_inequalities_l768_768020


namespace even_function_f_at_1_l768_768624

-- Define the function f(x) under the conditions specified
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2*x^2 - x else 2*(-x)^2 - (-x)

-- Define the property of an even function.
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Proof problem statement
theorem even_function_f_at_1 : is_even_function f → f 1 = 3 := by
  intro h
  sorry

end even_function_f_at_1_l768_768624


namespace goose_has_feathers_l768_768243

theorem goose_has_feathers :
  (∀ (pillow_feathers_weight : ℝ) (pound_feathers_count : ℝ) (total_pillows : ℝ),
    pillow_feathers_weight = 2 ∧
    pound_feathers_count = 300 ∧
    total_pillows = 6 →
    pillow_feathers_weight * total_pillows * pound_feathers_count = 3600) :=
begin
  sorry
end

end goose_has_feathers_l768_768243


namespace probability_two_vertices_endpoints_l768_768356

theorem probability_two_vertices_endpoints (V E : Type) [Fintype V] [DecidableEq V] 
  (dodecahedron : Graph V E) (h1 : Fintype.card V = 20)
  (h2 : ∀ v : V, Fintype.card (dodecahedron.neighbors v) = 3)
  (h3 : Fintype.card E = 30) :
  (∃ A B : V, A ≠ B ∧ (A, B) ∈ dodecahedron.edgeSet) → 
  (∃ p : ℚ, p = 3/19) := 
sorry

end probability_two_vertices_endpoints_l768_768356


namespace next_combined_activity_day_l768_768208

-- Define the intervals
def dance_interval : ℕ := 6
def karate_interval : ℕ := 12
def library_interval : ℕ := 18

-- Define the LCM computation
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The LCM of the intervals
def lcm_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem next_combined_activity_day : lcm_three dance_interval karate_interval library_interval = 36 := 
by 
  have h1 : lcm 6 12 = 12 := Nat.lcm_eq 6 12 rfl rfl
  have h2 : lcm 12 18 = 36 := Nat.lcm_eq 12 18 rfl rfl
  rw [lcm_three, h1, h2]
  exact rfl

end next_combined_activity_day_l768_768208


namespace cubic_function_value_l768_768073

noncomputable def g (x : ℝ) (p q r s : ℝ) : ℝ := p * x ^ 3 + q * x ^ 2 + r * x + s

theorem cubic_function_value (p q r s : ℝ) (h : g (-3) p q r s = -2) :
  12 * p - 6 * q + 3 * r - s = 2 :=
sorry

end cubic_function_value_l768_768073


namespace part1_part2_l768_768221

-- Definitions from condition part
def f (a x : ℝ) := a * x^2 + (1 + a) * x + a

-- Part (1) Statement
theorem part1 (a : ℝ) : 
  (a ≥ -1/3) → (∀ x : ℝ, f a x ≥ 0) :=
sorry

-- Part (2) Statement
theorem part2 (a : ℝ) : 
  (a > 0) → 
  (∀ x : ℝ, f a x < a - 1) → 
  ((0 < a ∧ a < 1) → (-1/a < x ∧ x < -1) ∨ 
   (a = 1) → False ∨
   (a > 1) → (-1 < x ∧ x < -1/a)) :=
sorry

end part1_part2_l768_768221


namespace find_DE_l768_768993

-- Definition of isosceles triangle ABC with AB = AC
variable {A B C D E H : Type*}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space H]

-- AB = AC
variable (h1 : dist A B = dist A C)

-- D is the foot of the perpendicular from B to AC
variable (h2 : ∃ D, ⟂ (B, D) (A, C))

-- E is the foot of the perpendicular from C to AB
variable (h3 : ∃ E, ⟂ (C, E) (A, B))

-- CE and BD intersect at H
variable (h4 : point H ∈ line (C, E) ∧ point H ∈ line (B, D))

-- EH = 1
variable (h5 : dist E H = 1)

-- AD = 4
variable (h6 : dist A D = 4)

-- Prove DE = 8 * sqrt 17 / 17
theorem find_DE (h1 h2 h3 h4 h5 h6 : Type*) : 
  dist D E = (8 * real.sqrt 17) / 17 :=
sorry

end find_DE_l768_768993


namespace temperatures_product_product_of_N_l768_768785

theorem temperatures_product (N: ℤ) (L M : ℤ) (M_6: ℤ) (L_6: ℤ):
  M = L + N →
  M_6 = M - 8 →
  L_6 = L + 6 →
  |M_6 - L_6| = 4 →
  N = 18 ∨ N = 10 :=
by {
  sorry
}

theorem product_of_N:
  (∀ (L M : ℤ) (M_6: ℤ) (L_6: ℤ),
  M = L + 18 →
  M_6 = M - 8 →
  L_6 = L + 6 →
  |M_6 - L_6| = 4) ∧
  (∀ (L M : ℤ) (M_6: ℤ) (L_6: ℤ),
  M = L + 10 →
  M_6 = M - 8 →
  L_6 = L + 6 →
  |M_6 - L_6| = 4) →
  ∃ N: ℤ,  N = 180 :=
by {
  use 180,
  sorry
}

end temperatures_product_product_of_N_l768_768785


namespace num_divisors_greater_than_9_factorial_l768_768561

theorem num_divisors_greater_than_9_factorial : 
  ∃ n, n = 9 ∧ ∀ d, d ∣ (nat.factorial 10) → d > (nat.factorial 9) → n = 9 :=
by
  sorry

end num_divisors_greater_than_9_factorial_l768_768561


namespace number_with_property_count_l768_768815

-- Define the predicate to check if a number has the desired property
def has_property (n : ℕ) : Prop :=
  n >= 3000 ∧ n < 4000 ∧
  let digit := fun n i => (n / 10^(i : ℕ)) % 10 in
  digit n 0 = (3 * digit n 1 * digit n 2) % 10

-- Main theorem statement
theorem number_with_property_count : 
  ∃ cnt : ℕ, cnt = 100 ∧ (∀ n, has_property n → n ∈ Finset.range 10000 ∧ cnt = Finset.filter has_property (Finset.Ico 3000 4000)).card sorry

end number_with_property_count_l768_768815


namespace ellipse_equation_line_existence_l768_768061

open Real

-- Define the ellipse and the conditions
def ellipse_C (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1
def circle := ∀ x y : ℝ, x^2 + y^2 = 4
def line_dist (x₀ y₀: ℝ) := abs(x₀ + y₀ - 2 * sqrt 3) / sqrt 2 = sqrt 6 / 2

-- Proof problem part 1: Proving the equation of the ellipse
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : ellipse_C 2 b)
                         (h3 : line_dist (sqrt 3) 1) : ellipse_C 2 1 :=
sorry

-- Definition for the line condition
def line (k m x : ℝ) := k * x + m
def quadratic := ∀ a b c x : ℝ, a * x^2 + b * x + c = 0
def discriminant_positive (a b c : ℝ) :=
  b^2 - 4 * a * c > 0

-- Proof problem part 2: Prove existence and range for m
theorem line_existence (k m : ℝ) (hx : ∀ x, quadratic 5 k x = 0)
                      (hΔ : discriminant_positive 4 k (m^2 - k^2)) :
                      m < -sqrt(3)/2 ∨ m > sqrt(3)/2 :=
sorry

end ellipse_equation_line_existence_l768_768061


namespace consecutive_integers_product_l768_768298

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end consecutive_integers_product_l768_768298


namespace sum_of_three_consecutive_integers_product_336_l768_768311

theorem sum_of_three_consecutive_integers_product_336 :
  ∃ (n : ℕ), (n - 1) * n * (n + 1) = 336 ∧ (n - 1) + n + (n + 1) = 21 :=
sorry

end sum_of_three_consecutive_integers_product_336_l768_768311


namespace s_g6_l768_768628

def s (x : ℝ) : ℝ := real.sqrt (4 * x + 2)
def g (x : ℝ) : ℝ := 7 - s x

theorem s_g6 : s (g 6) = real.sqrt (30 - 4 * real.sqrt 26) := by
  sorry

end s_g6_l768_768628


namespace negation_of_p_equiv_l768_768637

-- Define the initial proposition p
def p : Prop := ∃ x : ℝ, x^2 - 5*x - 6 < 0

-- State the theorem for the negation of p
theorem negation_of_p_equiv : ¬p ↔ ∀ x : ℝ, x^2 - 5*x - 6 ≥ 0 :=
by
  sorry

end negation_of_p_equiv_l768_768637


namespace equivalence_of_statements_l768_768723

-- Variables used in the statements
variable (P Q : Prop)

-- Proof problem statement
theorem equivalence_of_statements : (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by sorry

end equivalence_of_statements_l768_768723


namespace train_passes_jogger_in_31_secs_l768_768759

-- Define the constants and conditions given in the problem
def speed_jogger := 9 -- in km/hr
def speed_train := 45 -- in km/hr
def initial_distance := 190 -- in meters
def length_train := 120 -- in meters

-- Convert speeds to m/s
def speed_jogger_ms := speed_jogger * (5 / 18 : Real) -- in m/s
def speed_train_ms := speed_train * (5 / 18 : Real) -- in m/s

-- Relative speed of the train with respect to the jogger
def relative_speed := speed_train_ms - speed_jogger_ms

-- Total distance the train needs to cover to pass the jogger
def total_distance := initial_distance + length_train -- in meters

-- Correct answer
def correct_time := 31 -- in seconds

-- Lean statement to prove
theorem train_passes_jogger_in_31_secs :
  (total_distance / relative_speed) = correct_time := by
  sorry

end train_passes_jogger_in_31_secs_l768_768759


namespace range_of_a_l768_768885

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

theorem range_of_a (hf_even : is_even f) (hf_inc : is_increasing_on f (set.Ici (0 : ℝ))) :
  {a : ℝ | f 3 < f a} = {a : ℝ | a < -3} ∪ {a : ℝ | a > 3} :=
by {
  -- sorry is used to skip the proof
  sorry
}

end range_of_a_l768_768885


namespace first_divisor_of_1011_minus_3_l768_768506

theorem first_divisor_of_1011_minus_3 :
  ∃ d, (d = 16 ∨ d = 18 ∨ d = 21 ∨ d = 28) ∧ ∀ x, (x = 16 ∨ x = 18 ∨ x = 21 ∨ x = 28) → d ≤ x ∧ (1011 - 3) % d = 0 := 
begin
  sorry
end

end first_divisor_of_1011_minus_3_l768_768506


namespace intersecting_squares_black_greater_gray_l768_768591

theorem intersecting_squares_black_greater_gray :
  let A := 12^2
  let B := 9^2
  let C := 7^2
  let D := 3^2
  A - B + C - D = 103 :=
by
  let A := 12^2
  let B := 9^2
  let C := 7^2
  let D := 3^2
  calc
    A - B + C - D = 144 - 81 + 49 - 9 := by rfl
                ... = 103 := by rfl

end intersecting_squares_black_greater_gray_l768_768591


namespace find_some_number_l768_768574

-- Definitions of symbol replacements
def replacement_minus (a b : Nat) := a + b
def replacement_plus (a b : Nat) := a * b
def replacement_times (a b : Nat) := a / b
def replacement_div (a b : Nat) := a - b

-- The transformed equation using the replacements
def transformed_equation (some_number : Nat) :=
  replacement_minus
    some_number
    (replacement_div
      (replacement_plus 9 (replacement_times 8 3))
      25) = 5

theorem find_some_number : ∃ some_number : Nat, transformed_equation some_number ∧ some_number = 6 :=
by
  exists 6
  unfold transformed_equation
  unfold replacement_minus replacement_plus replacement_times replacement_div
  sorry

end find_some_number_l768_768574


namespace units_digit_of_expression_l768_768455

theorem units_digit_of_expression :
  (4 ^ 101 * 5 ^ 204 * 9 ^ 303 * 11 ^ 404) % 10 = 0 := 
sorry

end units_digit_of_expression_l768_768455


namespace positive_even_multiples_of_3_less_than_10000_l768_768927

theorem positive_even_multiples_of_3_less_than_10000 : set.count {n : ℕ | ∃ k : ℕ, 0 < k ∧ n = 36 * k^2 ∧ n < 10000} = 16 := 
sorry

end positive_even_multiples_of_3_less_than_10000_l768_768927


namespace sample_teachers_from_c_l768_768341

theorem sample_teachers_from_c : 
  (A B C : ℕ) (S : ℕ) 
  (hA : A = 180) (hB : B = 270) (hC : C = 90) (hS : S = 60) : (C * S) / (A + B + C) = 10 :=
by
  rw [hA, hB, hC, hS]
  rw [Nat.add_comm (Nat.add_comm 180 270)]
  norm_num
  sorry

end sample_teachers_from_c_l768_768341


namespace circle_eq_of_diameter_line_eq_of_chord_l768_768684

theorem circle_eq_of_diameter :
  (∃ A B : ℝ × ℝ, A = (0, -3) ∧ B = (-4, 0) ∧ (3 * (fst A) - 4 * (snd A) + 12 = 0) ∧ (3 * (fst B) - 4 * (snd B) + 12 = 0)) →
  ∃ C : ℝ × ℝ, (C = (-2, -3 / 2)) ∧ ∃ r : ℝ, r = 5 / 2 ∧
  (∀ x y : ℝ, ((x + 2)^2 + (y + 3 / 2)^2 = (5 / 2)^2)) :=
by
  sorry

theorem line_eq_of_chord :
  (∃ P : ℝ × ℝ, P = (1, 1 / 2)) →
  (∃ l : ℝ, l = sqrt 21) →
  (∃ C : ℝ × ℝ, C = (-2, -3 / 2) ∧ ∃ r : ℝ, r = 5 / 2) →
  ∃ k : ℝ, ∀ x y : ℝ, ((k = 0 ∧ y = 1 / 2) ∨ (k = 3 / 4 ∧ (3 * x + 4 * y - 5 = 0))) :=
by
  sorry

end circle_eq_of_diameter_line_eq_of_chord_l768_768684


namespace circle_radius_l768_768755

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 100 * π) :
  r = sqrt 101 - 1 :=
by
  sorry

end circle_radius_l768_768755


namespace find_f_l768_768049

def g (x : ℝ) := -x^2 - 3

noncomputable def quadratic (a b c x : ℝ) := a*x^2 + b*x + c

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

def minimum_in_interval (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) := ∀ x ∈ set.Icc a b, f x ≥ m ∧ (∃ y ∈ set.Icc a b, f y = m)

theorem find_f (f : ℝ → ℝ) (a b c : ℝ) :
    (∀ x : ℝ, g x + quadratic a b c x = f x)
    → is_odd_function (λ x, quadratic a b c x + g x)
    → minimum_in_interval (quadratic a b c) (-1) 2 1
    → (f = (λ x, x^2 + 3*x + 3) ∨ f = (λ x, x^2 - (2/2)*x + 3)) :=
sorry

end find_f_l768_768049


namespace number_of_students_in_line_l768_768271

-- Definitions for the conditions
def yoojung_last (n : ℕ) : Prop :=
  n = 14

def eunjung_position : ℕ := 5

def students_between (n : ℕ) : Prop :=
  n = 8

noncomputable def total_students : ℕ := 14

-- The theorem to be proven
theorem number_of_students_in_line 
  (last : yoojung_last total_students) 
  (eunjung_pos : eunjung_position = 5) 
  (between : students_between 8) :
  total_students = 14 := by
  sorry

end number_of_students_in_line_l768_768271


namespace inscribed_circle_center_medians_l768_768526

-- Given a triangle ABC
variables {A B C : Point}

-- Define the medians of triangle ABC
noncomputable def median_A : Line := sorry
noncomputable def median_B : Line := sorry
noncomputable def median_C : Line := sorry

-- Define the medial triangle DEF
noncomputable def D : Point := sorry
noncomputable def E : Point := sorry
noncomputable def F : Point := sorry

-- Statement: The center of the inscribed circle of the medial triangle DEF
theorem inscribed_circle_center_medians :
  ∃ (I : Point), is_incenter I D E F :=
sorry

end inscribed_circle_center_medians_l768_768526


namespace repeating_decimal_as_fraction_l768_768488

-- Define the repeating decimal
def repeating_decimal_2_35 := 2 + (35 / 99 : ℚ)

-- Define the fraction form
def fraction_form := (233 / 99 : ℚ)

-- Theorem statement asserting the equivalence
theorem repeating_decimal_as_fraction : repeating_decimal_2_35 = fraction_form :=
by 
  -- Skipped proof
  sorry

end repeating_decimal_as_fraction_l768_768488


namespace sum_f_ln_l768_768899

def f (x : ℝ) : ℝ := (Real.exp x) / (Real.exp x + 1)

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k, a (n + k) = a n * a k

axiom a : ℕ → ℝ
axiom a_positive : ∀ n, a n > 0
axiom a_geometric : geometric_sequence a
axiom a_1009 : a 1009 = 1

theorem sum_f_ln :
  (∑ n in Finset.range 2017, f (Real.log (a (n + 1)))) = 2017 / 2 :=
by
  sorry

end sum_f_ln_l768_768899


namespace laura_owes_correct_amount_l768_768613

def principal : ℝ := 35
def annual_rate : ℝ := 0.07
def time_years : ℝ := 1
def interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T
def total_amount_owed (P : ℝ) (I : ℝ) : ℝ := P + I

theorem laura_owes_correct_amount :
  total_amount_owed principal (interest principal annual_rate time_years) = 37.45 :=
sorry

end laura_owes_correct_amount_l768_768613


namespace prob_not_lose_money_proof_min_purchase_price_proof_l768_768830

noncomputable def prob_not_lose_money : ℚ :=
  let pr_normal_rain := (2 : ℚ) / 3
  let pr_less_rain := (1 : ℚ) / 3
  let pr_price_6_normal := (1 : ℚ) / 4
  let pr_price_6_less := (2 : ℚ) / 3
  pr_normal_rain * pr_price_6_normal + pr_less_rain * pr_price_6_less

theorem prob_not_lose_money_proof : prob_not_lose_money = 7 / 18 := sorry

noncomputable def min_purchase_price : ℚ :=
  let old_exp_income := 500
  let new_yield := 2500
  let cost_increase := 1000
  (7000 + 1500 + cost_increase) / new_yield
  
theorem min_purchase_price_proof : min_purchase_price = 3.4 := sorry

end prob_not_lose_money_proof_min_purchase_price_proof_l768_768830


namespace first_number_value_l768_768674

theorem first_number_value (A B LCM HCF : ℕ) (h_lcm : LCM = 2310) (h_hcf : HCF = 30) (h_b : B = 210) (h_mul : A * B = LCM * HCF) : A = 330 := 
by
  -- Use sorry to skip the proof
  sorry

end first_number_value_l768_768674


namespace quadratic_real_roots_range_l768_768159

theorem quadratic_real_roots_range (m : ℝ) : (∃ x : ℝ, x^2 - 2 * x - m = 0) → -1 ≤ m := 
sorry

end quadratic_real_roots_range_l768_768159


namespace checkerboard_diagonal_squares_l768_768412

theorem checkerboard_diagonal_squares (m n : ℕ) (h_m : m = 91) (h_n : n = 28) :
  let gcd_mn := Nat.gcd m n
  m + n - gcd_mn = 112 :=
by
  have h_gcd : gcd_mn = 7 := by
    rw [h_m, h_n]
    exact Nat.gcd_rec 91 28
  sorry

end checkerboard_diagonal_squares_l768_768412


namespace simplified_sum_l768_768713

def exp1 := -( -1 ^ 2006 )
def exp2 := -( -1 ^ 2007 )
def exp3 := -( 1 ^ 2008 )
def exp4 := -( -1 ^ 2009 )

theorem simplified_sum : 
  exp1 + exp2 + exp3 + exp4 = 0 := 
by 
  sorry

end simplified_sum_l768_768713


namespace tennis_balls_first_set_l768_768660

theorem tennis_balls_first_set (X : ℕ) 
    (total_balls : X + 75 = 175)
    (hit_first_set : float_of_int(2) * float_of_int(X) / 5)
    (hit_second_set : 75 / 3 = 25)
    (not_hit : float_of_int(3) * float_of_int(X) / 5 + 50 = 110) : 
    X = 100 := 
sorry

end tennis_balls_first_set_l768_768660


namespace minimum_red_points_l768_768668

def red_point := (ℕ × ℕ)

def is_valid_grid (n : ℕ) (red_points : List red_point) : Prop :=
  ∀ k m i j, 1 ≤ k ∧ k ≤ n ∧ 1 ≤ m ∧ m ≤ n →
  i + k ≤ n + 1 ∧ j + m ≤ n + 1 →
  (∃ (p ∈ red_points), (p.1 = i ∨ p.1 = i + k - 1) ∧ (p.2 ≥ j ∧ p.2 ≤ j + m - 1) ∨
                       (p.2 = j ∨ p.2 = j + m - 1) ∧ (p.1 ≥ i ∧ p.1 ≤ i + k - 1))

theorem minimum_red_points : ∀ red_points, is_valid_grid 6 red_points → red_points.length ≥ 16 :=
sorry

end minimum_red_points_l768_768668


namespace transformed_curve_eq_l768_768679

/-- Given the initial curve equation and the scaling transformation,
    prove that the resulting curve has the transformed equation. -/
theorem transformed_curve_eq 
  (x y x' y' : ℝ)
  (h_curve : x^2 + 9*y^2 = 9)
  (h_transform_x : x' = x)
  (h_transform_y : y' = 3*y) :
  (x')^2 + y'^2 = 9 := 
sorry

end transformed_curve_eq_l768_768679


namespace square_side_length_l768_768058

theorem square_side_length (d : ℝ) (h : d = 24) : ∃ s : ℝ, s = 12 * Real.sqrt 2 ∧ 2 * s^2 = d^2 :=
by
  use 12 * Real.sqrt 2
  split
  . sorry
  . sorry

end square_side_length_l768_768058


namespace given_eq_then_squares_l768_768516

theorem given_eq_then_squares (x : ℝ) (hx : x + x⁻¹ = 5) : x^2 + x⁻² = 23 :=
sorry

end given_eq_then_squares_l768_768516


namespace line_passing_A_parallel_BC_eq_l768_768907

-- Definitions and structures for points and slopes
structure Point where
  x : ℝ
  y : ℝ

def slope (P Q : Point) : ℝ :=
  if P.x = Q.x then 0 else (Q.y - P.y) / (Q.x - P.x)

-- Definitions of points A, B, and C
def A : Point := ⟨4, 0⟩
def B : Point := ⟨8, 10⟩
def C : Point := ⟨0, 6⟩

-- The slope of line BC
def slopeBC : ℝ := slope B C

-- The equation of the line passing through A and parallel to BC
def lineEquation (P : Point) (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y - P.y = k * (x - P.x)

theorem line_passing_A_parallel_BC_eq :
  ∀ (x y : ℝ), lineEquation A slopeBC x y ↔ x - 2*y - 4 = 0 :=
by
  sorry

end line_passing_A_parallel_BC_eq_l768_768907


namespace luke_fish_fillets_l768_768476

theorem luke_fish_fillets (daily_fish : ℕ) (days : ℕ) (fillets_per_fish : ℕ) 
  (h1 : daily_fish = 2) (h2 : days = 30) (h3 : fillets_per_fish = 2) : 
  daily_fish * days * fillets_per_fish = 120 := 
by 
  sorry

end luke_fish_fillets_l768_768476


namespace num_prime_divisors_of_50_fac_l768_768141

-- Define the set of all prime numbers less than or equal to 50.
def primes_le_50 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

-- Define the factorial function.
noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the number of prime divisors of n.
noncomputable def num_prime_divisors (n : ℕ) : ℕ :=
(set.count (λ p, p ∣ n) primes_le_50)

-- The theorem statement.
theorem num_prime_divisors_of_50_fac : num_prime_divisors (factorial 50) = 15 :=
by
  sorry

end num_prime_divisors_of_50_fac_l768_768141


namespace find_value_of_a_l768_768754

noncomputable def average (s : List ℚ) : ℚ :=
  s.sum / s.length

theorem find_value_of_a (a : ℚ)
    (h : ∀ x, 2.1 * x - 0.3 = y)
    (hx_values: [1, 2, 3, 4, 5])
    (hy_values: [2, 3, 7, 8, a]) :
  a = 10 := 
by
  -- Let's define the means of x and y
  let x_values := [1, 2, 3, 4, 5]
  let y_values := [2, 3, 7, 8, a]
  let x_mean := average x_values
  let y_mean := average y_values

  -- Express the condition on y mean as given by the regression line
  have :  y_mean = 2.1 * x_mean - 0.3, from h x_mean

  -- Calculate x_mean manually
  have : x_mean = 3, by sorry

  -- By substitution y_mean = 6
  have : y_mean = 6, by sorry

  -- Equate and solve for a
  have : y_mean = (20 + a) / 5, by sorry

  -- Complete the proof
  calc
    (20 + a) / 5 = 6 => from sorry
    20 + a = 30 => from sorry
    a = 10 => from sorry

end find_value_of_a_l768_768754


namespace same_marked_numbers_l768_768527

variable (n : ℕ)
variable (table : Fin n → Fin n → ℕ)
variable (distinct : ∀ i j i' j', (i ≠ i' ∨ j ≠ j') → table i j ≠ table i' j')

theorem same_marked_numbers :
  (∀ i, ∃ j, ∀ k, table i k ≥ table i j ∧ ∀ i', table i j ≠ table i' j) →
  (∀ j, ∃ i, ∀ k, table k j ≥ table i j ∧ ∀ j', table i j ≠ table i j') →
  (∀ i j, (∀ k : Fin n, table i k ≥ table i j = (table (some i')) 
  sorry

end same_marked_numbers_l768_768527


namespace pastries_solution_l768_768458

def pastries_problem : Prop :=
  ∃ (F Calvin Phoebe Grace : ℕ),
  (Calvin = F + 8) ∧
  (Phoebe = F + 8) ∧
  (Grace = 30) ∧
  (F + Calvin + Phoebe + Grace = 97) ∧
  (Grace - Calvin = 5) ∧
  (Grace - Phoebe = 5)

theorem pastries_solution : pastries_problem :=
by
  sorry

end pastries_solution_l768_768458


namespace clyde_picked_correct_number_of_cobs_l768_768752

-- Definitions based on conditions
def bushel_weight : ℝ := 56
def ear_weight : ℝ := 0.5
def bushels_picked : ℝ := 2

-- Calculation based on conditions
def cobs_per_bushel (bushel_weight ear_weight : ℝ) : ℝ := bushel_weight / ear_weight
def total_cobs (bushels_picked cobs_per_bushel : ℝ) : ℝ := bushels_picked * cobs_per_bushel

-- Theorem statement
theorem clyde_picked_correct_number_of_cobs :
  total_cobs bushels_picked (cobs_per_bushel bushel_weight ear_weight) = 224 := 
by
  sorry

end clyde_picked_correct_number_of_cobs_l768_768752


namespace pin_probability_l768_768383

theorem pin_probability :
  let total_pins := 9 * 10^5
  let valid_pins := 10^4
  ∃ p : ℚ, p = valid_pins / total_pins ∧ p = 1 / 90 := by
  sorry

end pin_probability_l768_768383


namespace john_made_money_l768_768975

theorem john_made_money 
  (repair_cost : ℕ := 20000) 
  (discount_percentage : ℕ := 20) 
  (prize_money : ℕ := 70000) 
  (keep_percentage : ℕ := 90) : 
  (prize_money * keep_percentage / 100) - (repair_cost - (repair_cost * discount_percentage / 100)) = 47000 := 
by 
  sorry

end john_made_money_l768_768975


namespace length_CD_is_7_div_3_l768_768247

noncomputable def length_CD (A B C D : ℝ × ℝ) : ℝ :=
  Real.sqrt ((fst D - fst A) ^ 2 + (snd D - snd A) ^ 2)

theorem length_CD_is_7_div_3 (A B C D : ℝ × ℝ) 
  (hAD : length_CD A D = 3)
  (hAB : length_CD A B = 1)
  (hBC : length_CD B C = 1)
  : length_CD C D = 7 / 3 :=
by
  sorry

end length_CD_is_7_div_3_l768_768247


namespace trucks_have_160_containers_per_truck_l768_768331

noncomputable def containers_per_truck: ℕ :=
  let boxes1 := 7 * 20
  let boxes2 := 5 * 12
  let total_boxes := boxes1 + boxes2
  let total_containers := total_boxes * 8
  let trucks := 10
  total_containers / trucks

theorem trucks_have_160_containers_per_truck:
  containers_per_truck = 160 :=
by
  sorry

end trucks_have_160_containers_per_truck_l768_768331


namespace prime_divisors_of_50_fact_eq_15_l768_768135

theorem prime_divisors_of_50_fact_eq_15 :
  ∃ P : Finset Nat, (∀ p ∈ P, Prime p ∧ p ∣ (Nat.factorial 50)) ∧ P.card = 15 := by
  sorry

end prime_divisors_of_50_fact_eq_15_l768_768135


namespace sufficient_not_necessary_l768_768148

theorem sufficient_not_necessary (a b : ℝ) (Ha : 0 < a) (Hb : 0 < b) :
  (a + b = 2 → a * b ≤ 1) ∧ ¬ (a * b ≤ 1 → a + b = 2) :=
by {
  sorry,
}

end sufficient_not_necessary_l768_768148


namespace min_area_QAMB_equation_of_MQ_l768_768874

-- Given conditions
def circle_M (x y : ℝ) := x^2 + (y - 2)^2 = 1
def Q_on_x_axis (Q : ℝ × ℝ) := Q.2 = 0 
def is_tangent (x y : ℝ) := x * y = 1

-- Statements to prove
theorem min_area_QAMB :
  ∀ (Q : ℝ × ℝ) (A B : ℝ × ℝ),
  Q_on_x_axis Q → 
  circle_M A.1 A.2 →
  circle_M B.1 B.2 →
  is_tangent Q.1 A.1 →
  is_tangent Q.1 B.1 →
  (min_area Q A B = sqrt 3) :=
sorry

theorem equation_of_MQ :
  ∀ (Q : ℝ × ℝ) (A B : ℝ × ℝ),
  |A.1 - B.1| = 4 * sqrt 2 / 3 →
  Q_on_x_axis Q →
  circle_M A.1 A.2 →
  circle_M B.1 B.2 →
  is_tangent Q.1 A.1 →
  is_tangent Q.1 B.1 →
  (line_eq Q A B = (2 * x + sqrt 5 * y - 2 * sqrt 5 = 0) ∨ (2 * x - sqrt 5 * y - 2 * sqrt 5 = 0)) :=
sorry

end min_area_QAMB_equation_of_MQ_l768_768874


namespace problem_l768_768625

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the conjugate of a complex number
def conjugate (z : ℂ) : ℂ := Complex.conj z

-- Given conditions
def c1 : conjugate z = (4 - 5 * i) / i := sorry

-- Proof goal
theorem problem : (3 - i) * z = -11 + 17 * i :=
by
  -- Use the given condition c1
  have h : conjugate z = (4 - 5 * i) / i := c1
  sorry

end problem_l768_768625


namespace probability_indep_seq_limit_one_l768_768640

noncomputable def i_o (A : ℕ → Set Ω) : Set Ω :=
  ⋂ k, ⋃ n (hn : k ≤ n), A n

theorem probability_indep_seq_limit_one {Ω : Type*} {A : ℕ → Set Ω} [MeasureSpace Ω] (P : Measure Ω)
  (indep : IndepEvents A P) (h : ∀ n, P (A n) < 1) :
  P (i_o A) = 1 ↔ P (⋃ n, A n) = 1 := 
by
  sorry

end probability_indep_seq_limit_one_l768_768640


namespace find_angle_A_find_sin_2C_add_A_find_area_triangle_l768_768954

section TriangleProblem

variables {A B C a b c : ℝ}

-- Conditions in the problem
axiom acute_triangle (h : 2 * sin A * (c * cos B + b * cos C) = sqrt 3 * a)
axiom side_ac_given_eq : c = 2 * sqrt 2
axiom side_a_given_eq : a = 3

-- Part (1): Find the value of angle A
theorem find_angle_A (h : 2 * sin A * (c * cos B + b * cos C) = sqrt 3 * a) : 
  A = π / 3 :=
sorry

-- Part (2)(i): Find sin(2C + A)
theorem find_sin_2C_add_A (h : 2 * sin A * (c * cos B + b * cos C) = sqrt 3 * a)
  (hc : c = 2 * sqrt 2) (ha : a = 3) (hA : A = π / 3) : 
  sin (2 * C + A) = (2 * sqrt 2 - sqrt 3) / 6 :=
sorry

-- Part (2)(ii): Find the area of triangle ABC
theorem find_area_triangle (h : 2 * sin A * (c * cos B + b * cos C) = sqrt 3 * a)
  (hc : c = 2 * sqrt 2) (ha : a = 3) (hA : A = π / 3) : 
  ∃ (b : ℝ), (c = 2 * sqrt 2 ∧ a = 3) ∧ (A = π / 3) ∧
  let area := (1 / 2) * b * c * sin A in 
  area = (3 * sqrt 2 + 2 * sqrt 3) / 2 :=
sorry

end TriangleProblem

end find_angle_A_find_sin_2C_add_A_find_area_triangle_l768_768954


namespace sum_of_prime_sequence_l768_768842

-- Definitions for necessary conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def sequence : List ℕ := [7, 17, 37, 47, 67]

-- The Lean statement for the proof problem
theorem sum_of_prime_sequence : 
  (∀ n ∈ sequence, is_prime n) ∧ 
  sequence.head = some 7 →
  sequence.sum = 175 := 
sorry

end sum_of_prime_sequence_l768_768842


namespace average_of_integers_between_ratios_l768_768381

theorem average_of_integers_between_ratios :
  let lower_bound := 32 + 1  -- 32 < N
  let upper_bound := 60 - 1  -- N < 60
  let integers_in_range := list.range' 33 (59 - 33 + 1)
  let sum := list.sum integers_in_range
  let count := list.length integers_in_range
  let average := sum / count
in
  average = 46 :=
by sorry

end average_of_integers_between_ratios_l768_768381


namespace height_of_poster_l768_768426

theorem height_of_poster (w A : ℕ) (h : ℕ) (hw : w = 4) (hA : A = 28) (h_area : A = w * h) : h = 7 :=
by
  rw [hw, hA] at h_area
  norm_num at h_area
  assumption 
  sorry -- Placeholder for required proof

end height_of_poster_l768_768426


namespace listens_by_end_of_year_l768_768979

theorem listens_by_end_of_year 
  (initial_listens : ℕ) 
  (months_remaining : ℕ) 
  (doubles_each_month : ℕ → ℕ) 
  (h_doubles : ∀ n, doubles_each_month n = 2 * n) 
  (h_initial : initial_listens = 60000) 
  (h_months : months_remaining = 3) 
  : ∑ i in finset.range (months_remaining + 1), (doubles_each_month (initial_listens * 2 ^ i)) = 900000 := 
sorry

end listens_by_end_of_year_l768_768979


namespace true_propositions_l768_768535

def proposition_p (x : ℝ) : Prop := (0 < x ∧ x < π / 2) → sin x > x
def proposition_q (x : ℝ) : Prop := (0 < x ∧ x < π / 2) → tan x > x

theorem true_propositions :
  (∃ x, proposition_p x ∨ proposition_q x)
  ∧ (∀ x, ¬proposition_p x ∨ proposition_q x) :=
sorry

end true_propositions_l768_768535


namespace smallest_multiple_of_36_with_digit_product_divisible_by_9_l768_768386

theorem smallest_multiple_of_36_with_digit_product_divisible_by_9 :
  ∃ n : ℕ, n > 0 ∧ n % 36 = 0 ∧ (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ (d1 * d2 * d3) % 9 = 0) ∧ n = 936 := 
by
  sorry

end smallest_multiple_of_36_with_digit_product_divisible_by_9_l768_768386


namespace incenter_vector_ratio_l768_768575

theorem incenter_vector_ratio (AB BC AC : ℝ) (O : Point)
  (hAB : AB = 6) (hBC : BC = 7) (hAC : AC = 4) (hIncenter : is_incenter O ABC)
  (p q : ℝ) (hAO : vector_of AO = p * vector_of AB + q * vector_of AC) :
  p / q = 2 / 3 :=
sorry

end incenter_vector_ratio_l768_768575


namespace unbroken_bottles_not_in_crates_l768_768980

theorem unbroken_bottles_not_in_crates :
  let total_bottles := 250
  let small_crates := 5
  let medium_crates := 5
  let large_crates := 5
  let small_crate_capacity := 8
  let medium_crate_capacity := 12
  let large_crate_capacity := 20
  let max_small_used := 3
  let max_medium_used := 4
  let max_large_used := 5
  let broken_bottles := 11
  let bottles_in_crates := 3 * small_crate_capacity +
                           4 * medium_crate_capacity +
                           5 * large_crate_capacity
  let unbroken_bottles := total_bottles - broken_bottles
  in unbroken_bottles - bottles_in_crates = 67 := by
  sorry

end unbroken_bottles_not_in_crates_l768_768980


namespace problem1_problem2_problem3_l768_768095

-- Define the vectors
def a : ℝ × ℝ × ℝ := (-2, -1,  2)
def b : ℝ × ℝ × ℝ := (-1,  1,  2)
def c (x : ℝ) : ℝ × ℝ × ℝ := (x, 2, 2)

-- Problem 1: Prove |a - 2b| = sqrt(13)
theorem problem1 : (real.sqrt (((0 - 2 * (-1))^2 + (1)^2 + (2 - 2 * 2)^2) = real.sqrt 13) 
 := sorry

-- Problem 2: prove values of x and k
theorem problem2 (k x : ℝ) (h1 : ∥c x∥ = 2 * real.sqrt 2) (h2 : (k * a + b) • c x = 0) : x = 0 ∧ k = -3
 := sorry

-- Problem 3: prove x from linear combination of a and b
theorem problem3 (x : ℝ) (λ μ : ℝ) (h3 : c x = λ • a + μ • b) : x = -1/2
 := sorry

end problem1_problem2_problem3_l768_768095


namespace sqrt_a_div_sqrt_b_l768_768473

theorem sqrt_a_div_sqrt_b (a b : ℝ) 
  (h : ( ((3 / 5 : ℝ)^2 + (2 / 7 : ℝ)^2) / ((2 / 9 : ℝ)^2 + (1 / 6 : ℝ)^2) = 28 * a / (45 * b) ) ) :
  real.sqrt a / real.sqrt b = 2 * real.sqrt 105 / 7 :=
by
  sorry

end sqrt_a_div_sqrt_b_l768_768473


namespace b_is_some_even_number_l768_768016

noncomputable def factorable_b (b : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    (m * p = 15 ∧ n * q = 15) ∧ 
    (b = m * q + n * p)

theorem b_is_some_even_number (b : ℤ) 
  (h : factorable_b b) : ∃ k : ℤ, b = 2 * k := 
by
  sorry

end b_is_some_even_number_l768_768016


namespace cookie_calories_l768_768645

theorem cookie_calories 
  (burger_calories : ℕ)
  (carrot_stick_calories : ℕ)
  (num_carrot_sticks : ℕ)
  (total_lunch_calories : ℕ) :
  burger_calories = 400 ∧ 
  carrot_stick_calories = 20 ∧ 
  num_carrot_sticks = 5 ∧ 
  total_lunch_calories = 750 →
  (total_lunch_calories - (burger_calories + num_carrot_sticks * carrot_stick_calories) = 250) :=
by sorry

end cookie_calories_l768_768645


namespace price_of_pants_l768_768238

theorem price_of_pants (P : ℝ) 
  (h1 : (3 / 4) * P + P + (P + 10) = 340)
  (h2 : ∃ P, (3 / 4) * P + P + (P + 10) = 340) : 
  P = 120 :=
sorry

end price_of_pants_l768_768238


namespace trivia_game_points_l768_768802

theorem trivia_game_points :
  let points_first_round := 17
  let points_second_round_raw := 6 * 2
  let points_after_two_rounds := points_first_round + points_second_round_raw
  let points_lost_last_round := 16 * 3
  points_after_two_rounds - points_lost_last_round = -19 :=
by
  let points_first_round := 17
  let points_second_round_raw := 6 * 2
  let points_after_two_rounds := points_first_round + points_second_round_raw
  let points_lost_last_round := 16 * 3
  show points_after_two_rounds - points_lost_last_round = -19 from sorry

end trivia_game_points_l768_768802


namespace chime_count_1000_occurs_on_feb_29_l768_768415

def starting_time := (10, 0) -- 10:00 AM
def start_date := (2000, 2, 20) -- February 20, 2000

/- A function that defines the chime behavior of the clock within a 24-hour period -/
def chimes_in_day : ℕ := 
  let chimes_at_quarters := 24 in
  let chimes_on_hours := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) in
  chimes_at_quarters + chimes_on_hours

/- A function that computes the number of chimes from 10:00 AM to midnight -/
def chimes_on_feb_20 : ℕ :=
  let morning_chimes := 1 + 11 + 1 + 12 in
  let afternoon_chimes := chimes_in_day - 78 + 12 in -- 78 chimes for (1 PM to 11 PM), 12 chimes at midnight
  morning_chimes + afternoon_chimes

/- Lean statement specifying the goal of proving the date of the 1000th chime -/
theorem chime_count_1000_occurs_on_feb_29 : 
  ∃ (date : ℕ × ℕ × ℕ), count_chimes_until date starting_time start_date 1000 → date = (2000, 2, 29) :=
sorry

end chime_count_1000_occurs_on_feb_29_l768_768415


namespace problem1_problem2_l768_768905

variable (m : ℝ)

-- Condition for p: m ∈ ℝ and m + 1 ≤ 0
def p : Prop := (m ∈ ℝ) ∧ (m + 1 ≤ 0)

-- Condition for q: ∀ x ∈ ℝ, x^2 + mx + 1 > 0
def q : Prop := ∀ (x : ℝ), x^2 + m * x + 1 > 0

-- Problem 1: Prove that if q is true, then -2 < m < 2
theorem problem1 (h_q : q) : -2 < m ∧ m < 2 :=
  sorry

-- Problem 2: Prove that if p ∧ q is false and p ∨ q is true, then m ≤ -2 or -1 < m < 2
theorem problem2 (h1 : ¬(p ∧ q)) (h2 : p ∨ q) : m ≤ -2 ∨ (-1 < m ∧ m < 2) :=
  sorry

end problem1_problem2_l768_768905


namespace largest_divisible_n_l768_768008

/-- Largest positive integer n for which n^3 + 10 is divisible by n + 1 --/
theorem largest_divisible_n (n : ℕ) :
  n = 0 ↔ ∀ m : ℕ, (m > n) → ¬ ((m^3 + 10) % (m + 1) = 0) :=
by
  sorry

end largest_divisible_n_l768_768008


namespace valid_permutations_count_l768_768995

noncomputable def number_of_valid_permutations (n : ℕ) : ℕ :=
  ∑ (p : Perm (Fin n)),
    if (∀ k ∈ {1, 2, 3},
      3 * k ∣ (∑ i in (Finset.range (3 * k)), p i.succ)) then 1 else 0

theorem valid_permutations_count : number_of_valid_permutations 10 = 45792 :=
  sorry

end valid_permutations_count_l768_768995


namespace initial_girls_count_l768_768758

variable (p : ℝ) (g : ℝ) (b : ℝ) (initial_girls : ℝ)

-- Conditions
def initial_percentage_of_girls (p g : ℝ) : Prop := g / p = 0.6
def final_percentage_of_girls (g : ℝ) (p : ℝ) : Prop := (g - 3) / p = 0.5

-- Statement only (no proof)
theorem initial_girls_count (p : ℝ) (h1 : initial_percentage_of_girls p (0.6 * p)) (h2 : final_percentage_of_girls (0.6 * p) p) :
  initial_girls = 18 :=
by
  sorry

end initial_girls_count_l768_768758


namespace prove_x3_y3_le_2_l768_768671

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

axiom positive_x : 0 < x
axiom positive_y : 0 < y
axiom condition : x^3 + y^4 ≤ x^2 + y^3

theorem prove_x3_y3_le_2 : x^3 + y^3 ≤ 2 := 
by
  sorry

end prove_x3_y3_le_2_l768_768671


namespace partition_into_groups_l768_768401

theorem partition_into_groups (n m k : ℕ) (h1 : m >= n) (h2 : (n * (n + 1) / 2) = m * k) :
  ∃ (partition : finset (finset ℕ)), (∀ x ∈ partition, x.sum = m) ∧ (partition.card = k) ∧ (⋃ x ∈ partition, x) = finset.range (n + 1) :=
sorry

end partition_into_groups_l768_768401


namespace trigonometric_transform_l768_768285

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := f (x - 3)
noncomputable def g (x : ℝ) : ℝ := 3 * h (x / 3)

theorem trigonometric_transform (x : ℝ) : g x = 3 * Real.sin (x / 3 - 3) := by
  sorry

end trigonometric_transform_l768_768285


namespace complex_simplify_l768_768267

theorem complex_simplify :
  3 * (4 - 2 * complex.i) + 2 * complex.i * (3 + complex.i) = 10 :=
by
  sorry

end complex_simplify_l768_768267


namespace quadratic_root_equality_l768_768319

theorem quadratic_root_equality : 
  ∀ (x : ℝ), - x^2 + 2 * x - 1 = 0 → discriminant (-1 : ℝ) 2 (-1) = 0 :=
by
  intro x h
  sorry

end quadratic_root_equality_l768_768319


namespace find_extreme_points_and_values_l768_768499

def f (x : ℝ) : ℝ := (x^2 - 1)^2 + 2

theorem find_extreme_points_and_values :
  (differentiable ℝ f) ∧ (∀ x : ℝ, deriv f x = 0 → x = 0 ∨ x = 1 ∨ x = -1) ∧ 
  (f 0 = 3) ∧ (f (-1) = 2) ∧ (f 1 = 2) :=
begin
  sorry
end

end find_extreme_points_and_values_l768_768499


namespace probability_endpoints_of_edge_l768_768363

noncomputable def num_vertices : ℕ := 12
noncomputable def edges_per_vertex : ℕ := 3

theorem probability_endpoints_of_edge :
  let total_ways := Nat.choose num_vertices 2,
      total_edges := (num_vertices * edges_per_vertex) / 2,
      probability := total_edges / total_ways in
  probability = 3 / 11 := by
  sorry

end probability_endpoints_of_edge_l768_768363


namespace Paul_work_time_l768_768040

def work_completed (rate: ℚ) (time: ℚ) : ℚ := rate * time

noncomputable def George_work_rate : ℚ := 3 / 5 / 9

noncomputable def combined_work_rate : ℚ := 2 / 5 / 4

noncomputable def Paul_work_rate : ℚ := combined_work_rate - George_work_rate

theorem Paul_work_time :
  (work_completed Paul_work_rate 30) = 1 :=
by
  have h_george_rate : George_work_rate = 1 / 15 :=
    by norm_num [George_work_rate]
  have h_combined_rate : combined_work_rate = 1 / 10 :=
    by norm_num [combined_work_rate]
  have h_paul_rate : Paul_work_rate = 1 / 30 :=
    by norm_num [Paul_work_rate, h_combined_rate, h_george_rate]
  sorry -- Complete proof statement here

end Paul_work_time_l768_768040


namespace sin_value_l768_768536

theorem sin_value (x : ℝ) (h : Real.sec x + Real.tan x = 5/4) : Real.sin x = 9/41 :=
sorry

end sin_value_l768_768536


namespace cake_area_l768_768712

theorem cake_area (n : ℕ) (a area_per_piece : ℕ) 
  (h1 : n = 25) 
  (h2 : a = 16) 
  (h3 : area_per_piece = 4 * 4) 
  (h4 : a = area_per_piece) : 
  n * a = 400 := 
by
  sorry

end cake_area_l768_768712


namespace point_A_not_on_transformed_plane_l768_768631

def point_A := (1, 2, -1)
def plane_a (x y z : ℝ) := 2 * x + 3 * y + z - 1 = 0
def k := 2
def plane_a_transformed (x y z : ℝ) := 2 * x + 3 * y + z - k = 0

theorem point_A_not_on_transformed_plane : ¬ plane_a_transformed point_A.1 point_A.2 point_A.3 :=
by {
  sorry 
}

end point_A_not_on_transformed_plane_l768_768631


namespace cost_of_filling_all_pots_l768_768377

def cost_palm_fern : ℝ := 15.00
def cost_creeping_jenny_per_plant : ℝ := 4.00
def num_creeping_jennies : ℝ := 4
def cost_geranium_per_plant : ℝ := 3.50
def num_geraniums : ℝ := 4
def cost_elephant_ear_per_plant : ℝ := 7.00
def num_elephant_ears : ℝ := 2
def cost_purple_fountain_grass_per_plant : ℝ := 6.00
def num_purple_fountain_grasses : ℝ := 3
def num_pots : ℝ := 4

def total_cost_per_pot : ℝ := 
  cost_palm_fern +
  (num_creeping_jennies * cost_creeping_jenny_per_plant) +
  (num_geraniums * cost_geranium_per_plant) +
  (num_elephant_ears * cost_elephant_ear_per_plant) +
  (num_purple_fountain_grasses * cost_purple_fountain_grass_per_plant)

def total_cost : ℝ := total_cost_per_pot * num_pots

theorem cost_of_filling_all_pots : total_cost = 308.00 := by
  sorry

end cost_of_filling_all_pots_l768_768377


namespace value_of_x_l768_768173

theorem value_of_x (x : ℝ) (combined_area : ℝ) : 
  (let area_3x_square := (3 * x) ^ 2,
       area_4x_square := (4 * x) ^ 2,
       area_triangle := (1 / 2) * (3 * x) * (4 * x) in 
       area_3x_square + area_4x_square + area_triangle = combined_area) → 
  combined_area = 962 →
  x = Real.sqrt 31 :=
sorry

end value_of_x_l768_768173


namespace number_of_elements_in_P_l768_768638

def P : set ℝ := {x : ℝ | x^2 + x - 6 = 0}

theorem number_of_elements_in_P : finset.card (P.to_finset) = 2 :=
sorry

end number_of_elements_in_P_l768_768638


namespace constant_term_bin_expansion_l768_768195

theorem constant_term_bin_expansion : 
  let a := 2
  let b := (1 : ℚ)
  let n := 4
  ∃ k : ℕ, (k ≤ n) ∧ 
  ((a*x + b/x)^n).coeff k = 24 := 
begin
  sorry
end

end constant_term_bin_expansion_l768_768195


namespace probability_shaded_region_l768_768420

-- Assume definitions
def game_board_equilateral_triangle : Prop := 
-- Placeholder for the definition of an equilateral triangle game board
sorry

def regions_divided_by_altitude_and_medians (board : Prop) : Nat := 6 
-- Placeholder for the count of regions, assuming a triangle divided by an altitude and two medians creates 6 regions
sorry

def shaded_regions (board : Prop) : Nat := 3
-- Number of shaded regions

-- State the theorem
theorem probability_shaded_region (board : Prop) 
  (h1 : game_board_equilateral_triangle board)
  (h2 : regions_divided_by_altitude_and_medians board = 6)
  (h3 : shaded_regions board = 3) : 
  3 / 6 = 1 / 2 :=
by
  rw [div_eq_iff_mul_eq]
  simp
  exact sorry

end probability_shaded_region_l768_768420


namespace ellipse_equation_l768_768531

theorem ellipse_equation 
  (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (eccentricity_is_one_third : (1 / 3) = sqrt (1 - (b / a)^2))
  (F1 F2 P : ℝ × ℝ) 
  (point_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (area_triangle : 4 = abs ((F1.1 * P.2 + P.1 * F2.2 + F2.1 * F1.2 - F2.1 * P.2 - F1.1 * F2.2 - P.1 * F1.2) / 2))
  (cosine_angle : cos (∠ F1 P F2) = 3 / 5) :
  a = 3 ∧ b = sqrt(8) := 
sorry

end ellipse_equation_l768_768531


namespace value_of_y_l768_768149

theorem value_of_y (y : ℕ) (h : 9 / (y^2) = y / 81) : y = 9 :=
by
-- Since we are only required to state the theorem, we leave the proof out for now.
sorry

end value_of_y_l768_768149


namespace cookies_per_batch_l768_768916

-- Define the necessary conditions
def total_chips : ℕ := 81
def batches : ℕ := 3
def chips_per_cookie : ℕ := 9

-- Theorem stating the number of cookies per batch
theorem cookies_per_batch : (total_chips / batches) / chips_per_cookie = 3 :=
by
  -- Here would be the proof, but we use sorry as placeholder
  sorry

end cookies_per_batch_l768_768916


namespace distance_midpoint_y_axis_l768_768904

noncomputable def C : parabola := sorry  -- Define the parabola y^2 = 4x.
noncomputable def F : point := sorry     -- Define the focus F.
noncomputable def M : point := sorry     -- Define one intersection point M on the parabola.
noncomputable def N : point := sorry     -- Define the other intersection point N on the parabola.
def MN_length := 10                      -- Given the length |MN| = 10.

theorem distance_midpoint_y_axis : 
  let x1 := M.x,
      x2 := N.x,
      x0 := (x1 + x2) / 2,               -- Midpoint's x-coordinate.
      F_x := 2                           -- Given focus' x-coordinate.
  in abs (x0 - 0) = 2 := sorry           -- Prove that the midpoint's x-coordinate is 2.

end distance_midpoint_y_axis_l768_768904


namespace sum_of_distances_eq_l768_768987

variable {α : Type*} [LinearOrder α] [Add α] [Neg α] [HasAdd α] [HasSub α] [Zero α]
open Finset

theorem sum_of_distances_eq (n : ℕ) (f : Fin (2 * n) → α) (h_sym : ∀ i, f 2 * n - 1 - i = -f i)
    (blue red : Fin n → Fin (2 * n)) (h_bij : ∀ i ∈ Ico 0 (2 * n), ∃ j, (f ∘ blue) j = f i)
    (h_disj : ∀ i j, blue i = red j → False) :
    (∑ i, f (blue i) + 1) = (∑ i, 1 - f (red i)) := sorry

end sum_of_distances_eq_l768_768987
