import Mathlib

namespace geometric_series_arithmetic_sequence_l61_61852

noncomputable def geometric_seq_ratio (a : ℕ → ℝ) (q : ℝ) : Prop := 
∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_series_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_seq_ratio a q)
  (h_pos : ∀ n, a n > 0)
  (h_arith : a 1 = (a 0 + 2 * a 1) / 2) :
  a 5 / a 3 = 3 + 2 * Real.sqrt 2 :=
sorry

end geometric_series_arithmetic_sequence_l61_61852


namespace apples_used_l61_61802

theorem apples_used (initial_apples remaining_apples : ℕ) (h_initial : initial_apples = 40) (h_remaining : remaining_apples = 39) : initial_apples - remaining_apples = 1 := 
by
  sorry

end apples_used_l61_61802


namespace salary_increase_l61_61974

theorem salary_increase (S0 S3 : ℕ) (r : ℕ) : 
  S0 = 3000 ∧ S3 = 8232 ∧ (S0 * (1 + r / 100)^3 = S3) → r = 40 :=
by
  sorry

end salary_increase_l61_61974


namespace sum_of_decimals_l61_61031

theorem sum_of_decimals : 5.46 + 2.793 + 3.1 = 11.353 := by
  sorry

end sum_of_decimals_l61_61031


namespace exists_integers_gcd_eq_one_addition_l61_61988

theorem exists_integers_gcd_eq_one_addition 
  (n k : ℕ) 
  (hnk_pos : n > 0 ∧ k > 0) 
  (hn_even_or_nk_even : (¬ n % 2 = 0) ∨ (n % 2 = 0 ∧ k % 2 = 0)) :
  ∃ a b : ℤ, Int.gcd a ↑n = 1 ∧ Int.gcd b ↑n = 1 ∧ k = a + b :=
by
  sorry

end exists_integers_gcd_eq_one_addition_l61_61988


namespace edge_length_of_cube_l61_61250

theorem edge_length_of_cube (V : ℝ) (e : ℝ) (h1 : V = 2744) (h2 : V = e^3) : e = 14 := 
by 
  sorry

end edge_length_of_cube_l61_61250


namespace cube_face_parallel_probability_l61_61276

theorem cube_face_parallel_probability :
  ∃ (n m : ℕ), (n = 15) ∧ (m = 3) ∧ (m / n = (1 / 5 : ℝ)) := 
sorry

end cube_face_parallel_probability_l61_61276


namespace time_for_A_to_complete_work_l61_61028

-- Defining the work rates and the condition
def workRateA (a : ℕ) : ℚ := 1 / a
def workRateB : ℚ := 1 / 12
def workRateC : ℚ := 1 / 24
def combinedWorkRate (a : ℕ) : ℚ := workRateA a + workRateB + workRateC
def togetherWorkRate : ℚ := 1 / 4

-- Stating the theorem
theorem time_for_A_to_complete_work : 
  ∃ (a : ℕ), combinedWorkRate a = togetherWorkRate ∧ a = 8 :=
by
  sorry

end time_for_A_to_complete_work_l61_61028


namespace min_airlines_needed_l61_61458

theorem min_airlines_needed 
  (towns : Finset ℕ) 
  (h_towns : towns.card = 21)
  (flights : Π (a : Finset ℕ), a.card = 5 → Finset (Finset ℕ))
  (h_flight : ∀ {a : Finset ℕ} (ha : a.card = 5), (flights a ha).card = 10):
  ∃ (n : ℕ), n = 21 :=
sorry

end min_airlines_needed_l61_61458


namespace max_product_price_l61_61617

/-- Conditions: 
1. Company C sells 50 products.
2. The average retail price of the products is $2,500.
3. No product sells for less than $800.
4. Exactly 20 products sell for less than $2,000.
Goal:
Prove that the greatest possible selling price of the most expensive product is $51,000.
-/
theorem max_product_price (n : ℕ) (avg_price : ℝ) (min_price : ℝ) (threshold_price : ℝ) (num_below_threshold : ℕ) :
  n = 50 → 
  avg_price = 2500 → 
  min_price = 800 → 
  threshold_price = 2000 → 
  num_below_threshold = 20 → 
  ∃ max_price : ℝ, max_price = 51000 :=
by 
  sorry

end max_product_price_l61_61617


namespace result_of_y_minus_3x_l61_61649

theorem result_of_y_minus_3x (x y : ℝ) (h1 : x + y = 8) (h2 : y - x = 7.5) : y - 3 * x = 7 :=
sorry

end result_of_y_minus_3x_l61_61649


namespace find_number_l61_61961

def hash (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_number :
  (∃ x : ℕ, hash 3 x = 63 ∧ x = 7) :=
sorry

end find_number_l61_61961


namespace find_offset_length_l61_61832

theorem find_offset_length 
  (diagonal_offset_7 : ℝ) 
  (area_of_quadrilateral : ℝ) 
  (diagonal_length : ℝ) 
  (result : ℝ) : 
  (diagonal_length = 10) 
  ∧ (diagonal_offset_7 = 7) 
  ∧ (area_of_quadrilateral = 50) 
  → (∃ x, x = result) :=
by
  sorry

end find_offset_length_l61_61832


namespace factorization_correctness_l61_61774

theorem factorization_correctness :
  (∀ x : ℝ, (x + 1) * (x - 1) = x^2 - 1 → false) ∧
  (∀ x : ℝ, x^2 - 4 * x + 4 = x * (x - 4) + 4 → false) ∧
  (∀ x : ℝ, (x + 3) * (x - 4) = x^2 - x - 12 → false) ∧
  (∀ x : ℝ, x^2 - 4 = (x + 2) * (x - 2)) :=
by
  sorry

end factorization_correctness_l61_61774


namespace geometric_probability_l61_61597

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l61_61597


namespace sum_of_squares_eq_two_l61_61200

theorem sum_of_squares_eq_two {a b : ℝ} (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 := sorry

end sum_of_squares_eq_two_l61_61200


namespace value_of_function_at_2_l61_61408

theorem value_of_function_at_2 (q : ℝ → ℝ) : q 2 = 5 :=
by
  -- Condition: The point (2, 5) lies on the graph of q
  have point_on_graph : q 2 = 5 := sorry
  exact point_on_graph

end value_of_function_at_2_l61_61408


namespace complex_magnitude_l61_61572

variable (a b : ℝ)

theorem complex_magnitude :
  ((1 + 2 * a * Complex.I) * Complex.I = 1 - b * Complex.I) →
  Complex.normSq (a + b * Complex.I) = 5/4 :=
by
  intro h
  -- Add missing logic to transform assumption to the norm result
  sorry

end complex_magnitude_l61_61572


namespace percentage_sum_of_v_and_w_l61_61352

variable {x y z v w : ℝ} 

theorem percentage_sum_of_v_and_w (h1 : 0.45 * z = 0.39 * y) (h2 : y = 0.75 * x) 
                                  (h3 : v = 0.80 * z) (h4 : w = 0.60 * y) :
                                  v + w = 0.97 * x :=
by 
  sorry

end percentage_sum_of_v_and_w_l61_61352


namespace exponent_proof_l61_61216

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end exponent_proof_l61_61216


namespace numberOfBags_l61_61137

-- Define the given conditions
def totalCookies : Nat := 33
def cookiesPerBag : Nat := 11

-- Define the statement to prove
theorem numberOfBags : totalCookies / cookiesPerBag = 3 := by
  sorry

end numberOfBags_l61_61137


namespace ec_value_l61_61578

theorem ec_value (AB AD : ℝ) (EFGH1 EFGH2 : ℝ) (x : ℝ)
  (h1 : AB = 2)
  (h2 : AD = 1)
  (h3 : EFGH1 = 1 / 2 * AB)
  (h4 : EFGH2 = 1 / 2 * AD)
  (h5 : 1 + 2 * x = 1)
  : x = 1 / 3 :=
by sorry

end ec_value_l61_61578


namespace extra_sweets_l61_61528

theorem extra_sweets (S : ℕ) (h1 : ∀ n: ℕ, S = 120 * 38) : 
    (38 - (S / 190) = 14) :=
by
  -- Here we will provide the proof 
  sorry

end extra_sweets_l61_61528


namespace geometric_sequence_sixth_term_l61_61192

theorem geometric_sequence_sixth_term (a : ℕ) (a2 : ℝ) (aₖ : ℕ → ℝ) (r : ℝ) (k : ℕ) (h1 : a = 3) (h2 : a2 = -1/6) (h3 : ∀ n, aₖ n = a * r^(n-1)) (h4 : r = a2 / a) (h5 : k = 6) :
  aₖ k = -1 / 629856 :=
by sorry

end geometric_sequence_sixth_term_l61_61192


namespace tetrahedron_volume_ratio_l61_61975

theorem tetrahedron_volume_ratio
  (a b : ℝ)
  (larger_tetrahedron : a = 6)
  (smaller_tetrahedron : b = a / 2) :
  (b^3 / a^3) = 1 / 8 := 
by 
  sorry

end tetrahedron_volume_ratio_l61_61975


namespace identify_translation_l61_61537

def phenomenon (x : String) : Prop :=
  x = "translational"

def option_A : Prop := phenomenon "rotational"
def option_B : Prop := phenomenon "rotational"
def option_C : Prop := phenomenon "translational"
def option_D : Prop := phenomenon "rotational"

theorem identify_translation :
  (¬ option_A) ∧ (¬ option_B) ∧ option_C ∧ (¬ option_D) :=
  by {
    sorry
  }

end identify_translation_l61_61537


namespace total_amount_shared_l61_61898

-- Define the initial conditions
def ratioJohn : ℕ := 2
def ratioJose : ℕ := 4
def ratioBinoy : ℕ := 6
def JohnShare : ℕ := 2000
def partValue : ℕ := JohnShare / ratioJohn

-- Define the shares based on the ratio and part value
def JoseShare := ratioJose * partValue
def BinoyShare := ratioBinoy * partValue

-- Prove the total amount shared is Rs. 12000
theorem total_amount_shared : (JohnShare + JoseShare + BinoyShare) = 12000 :=
  by
  sorry

end total_amount_shared_l61_61898


namespace distance_C_to_C_l61_61318

noncomputable def C : ℝ × ℝ := (-3, 2)
noncomputable def C' : ℝ × ℝ := (3, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_C_to_C' : distance C C' = 2 * Real.sqrt 13 := by
  sorry

end distance_C_to_C_l61_61318


namespace identity_map_a_plus_b_l61_61811

theorem identity_map_a_plus_b (a b : ℝ) (h : ∀ x ∈ ({-1, b / a, 1} : Set ℝ), x ∈ ({a, b, b - a} : Set ℝ)) : a + b = -1 ∨ a + b = 1 :=
by
  sorry

end identity_map_a_plus_b_l61_61811


namespace triangle_base_l61_61406

theorem triangle_base (sum_of_sides : ℕ) (left_side : ℕ) (right_side : ℕ) (base : ℕ)
  (h1 : sum_of_sides = 50)
  (h2 : right_side = left_side + 2)
  (h3 : left_side = 12) :
  base = 24 :=
by
  sorry

end triangle_base_l61_61406


namespace proof_l61_61272

noncomputable def question := ∀ x : ℝ, (0.12 * x = 36) → (0.5 * (0.4 * 0.3 * x) = 18) 

theorem proof : question :=
by
  intro x
  intro h
  sorry

end proof_l61_61272


namespace max_min_values_l61_61119

open Real

noncomputable def circle_condition (x y : ℝ) :=
  (x - 3) ^ 2 + (y - 3) ^ 2 = 6

theorem max_min_values (x y : ℝ) (hx : circle_condition x y) :
  ∃ k k' d d', 
    k = 3 + 2 * sqrt 2 ∧
    k' = 3 - 2 * sqrt 2 ∧
    k = y / x ∧
    k' = y / x ∧
    d = sqrt ((x - 2) ^ 2 + y ^ 2) ∧
    d' = sqrt ((x - 2) ^ 2 + y ^ 2) ∧
    d = sqrt (10) + sqrt (6) ∧
    d' = sqrt (10) - sqrt (6) :=
sorry

end max_min_values_l61_61119


namespace tangent_line_at_M_l61_61416

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 6)

theorem tangent_line_at_M :
  let M : ℝ × ℝ := (2, 0)
  ∃ (m n : ℝ), n = f m ∧ m = 4 ∧ n = -2 * Real.exp 4 ∧
    ∀ (x y : ℝ), y = -Real.exp 4 * (x - 2) →
    M.2 = y :=
by
  sorry

end tangent_line_at_M_l61_61416


namespace ratio_of_oranges_l61_61583

def num_good_oranges : ℕ := 24
def num_bad_oranges : ℕ := 8
def ratio_good_to_bad : ℕ := num_good_oranges / num_bad_oranges

theorem ratio_of_oranges : ratio_good_to_bad = 3 := by
  show 24 / 8 = 3
  sorry

end ratio_of_oranges_l61_61583


namespace maximum_area_of_flower_bed_l61_61069

-- Definitions based on conditions
def length_of_flower_bed : ℝ := 150
def total_fencing : ℝ := 450

-- Question reframed as a proof statement
theorem maximum_area_of_flower_bed :
  ∀ (w : ℝ), 2 * w + length_of_flower_bed = total_fencing → (length_of_flower_bed * w = 22500) :=
by
  intro w h
  sorry

end maximum_area_of_flower_bed_l61_61069


namespace find_sum_of_A_and_B_l61_61457

theorem find_sum_of_A_and_B :
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ B = A - 2 ∧ A = 5 + 3 ∧ A + B = 14 :=
by
  sorry

end find_sum_of_A_and_B_l61_61457


namespace sector_area_150_degrees_l61_61016

def sector_area (radius : ℝ) (central_angle : ℝ) : ℝ :=
  0.5 * radius^2 * central_angle

theorem sector_area_150_degrees (r : ℝ) (angle_rad : ℝ) (h1 : r = Real.sqrt 3) (h2 : angle_rad = (5 * Real.pi) / 6) : 
  sector_area r angle_rad = (5 * Real.pi) / 4 :=
by
  simp [sector_area, h1, h2]
  sorry

end sector_area_150_degrees_l61_61016


namespace inequality_proof_l61_61497

theorem inequality_proof (x1 x2 y1 y2 z1 z2 : ℝ) 
  (hx1 : x1 > 0) (hx2 : x2 > 0) (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hx1y1_pos : x1 * y1 - z1^2 > 0) (hx2y2_pos : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 
    1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
by
  sorry

end inequality_proof_l61_61497


namespace solve_system_of_equations_l61_61730

theorem solve_system_of_equations
  (x y : ℝ)
  (h1 : 1 / x + 1 / (2 * y) = (x^2 + 3 * y^2) * (3 * x^2 + y^2))
  (h2 : 1 / x - 1 / (2 * y) = 2 * (y^4 - x^4)) :
  x = (3 ^ (1 / 5) + 1) / 2 ∧ y = (3 ^ (1 / 5) - 1) / 2 :=
by
  sorry

end solve_system_of_equations_l61_61730


namespace isosceles_triangle_angles_l61_61914

theorem isosceles_triangle_angles (α β γ : ℝ) 
  (h1 : α = 50)
  (h2 : α + β + γ = 180)
  (isosceles : (α = β ∨ α = γ ∨ β = γ)) :
  (β = 50 ∧ γ = 80) ∨ (γ = 50 ∧ β = 80) :=
by
  sorry

end isosceles_triangle_angles_l61_61914


namespace subdivide_tetrahedron_l61_61664

/-- A regular tetrahedron with edge length 1 can be divided into smaller regular tetrahedrons and octahedrons,
    such that the edge lengths of the resulting tetrahedrons and octahedrons are less than 1 / 100 after a 
    finite number of subdivisions. -/
theorem subdivide_tetrahedron (edge_len : ℝ) (h : edge_len = 1) :
  ∃ (k : ℕ), (1 / (2^k : ℝ) < 1 / 100) :=
by sorry

end subdivide_tetrahedron_l61_61664


namespace evaluate_composite_function_l61_61735

def g (x : ℝ) : ℝ := 3 * x^2 - 4

def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem evaluate_composite_function : g (h 2) = 5288 := by
  sorry

end evaluate_composite_function_l61_61735


namespace initial_ratio_l61_61189

-- Define the initial number of horses and cows
def initial_horses (H : ℕ) : Prop := H = 120
def initial_cows (C : ℕ) : Prop := C = 20

-- Define the conditions of the problem
def condition1 (H C : ℕ) : Prop := H - 15 = 3 * (C + 15)
def condition2 (H C : ℕ) : Prop := H - 15 = C + 15 + 70

-- The statement that initial ratio is 6:1
theorem initial_ratio (H C : ℕ) (h1 : condition1 H C) (h2 : condition2 H C) : 
  H = 6 * C :=
by {
  sorry
}

end initial_ratio_l61_61189


namespace animals_on_farm_l61_61757

theorem animals_on_farm (cows : ℕ) (sheep : ℕ) (pigs : ℕ) 
  (h1 : cows = 12) 
  (h2 : sheep = 2 * cows) 
  (h3 : pigs = 3 * sheep) : 
  cows + sheep + pigs = 108 := 
by
  sorry

end animals_on_farm_l61_61757


namespace repeating_decimal_product_l61_61827

theorem repeating_decimal_product (x y : ℚ) (h₁ : x = 8 / 99) (h₂ : y = 1 / 3) :
    x * y = 8 / 297 := by
  sorry

end repeating_decimal_product_l61_61827


namespace expand_expression_l61_61510

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l61_61510


namespace concert_attendance_l61_61434

-- Define the given conditions
def buses : ℕ := 8
def students_per_bus : ℕ := 45

-- Statement of the problem
theorem concert_attendance :
  buses * students_per_bus = 360 :=
sorry

end concert_attendance_l61_61434


namespace neg_forall_sin_gt_zero_l61_61164

theorem neg_forall_sin_gt_zero :
  ¬ (∀ x : ℝ, Real.sin x > 0) ↔ ∃ x : ℝ, Real.sin x ≤ 0 := 
sorry

end neg_forall_sin_gt_zero_l61_61164


namespace last_digit_322_pow_369_l61_61704

theorem last_digit_322_pow_369 : (322^369) % 10 = 2 := by
  sorry

end last_digit_322_pow_369_l61_61704


namespace simplify_fraction_l61_61866

theorem simplify_fraction :
  (1 / (1 / (1 / 2) ^ 1 + 1 / (1 / 2) ^ 2 + 1 / (1 / 2) ^ 3)) = (1 / 14) :=
by 
  sorry

end simplify_fraction_l61_61866


namespace find_f_l61_61936

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f (x - 1/x) = x^2 + 1/x^2 - 4) :
  ∀ x : ℝ, f x = x^2 - 2 :=
by
  intros x
  sorry

end find_f_l61_61936


namespace single_elimination_games_l61_61658

theorem single_elimination_games (n : Nat) (h : n = 21) : games_needed = n - 1 :=
by
  sorry

end single_elimination_games_l61_61658


namespace min_value_nS_n_l61_61212

theorem min_value_nS_n (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) 
  (h2 : m ≥ 2)
  (h3 : S (m - 1) = -2)
  (h4 : S m = 0)
  (h5 : S (m + 1) = 3) :
  ∃ n : ℕ, n * S n = -9 :=
sorry

end min_value_nS_n_l61_61212


namespace slope_transformation_l61_61368

theorem slope_transformation :
  ∀ (b : ℝ), ∃ k : ℝ, 
  (∀ x : ℝ, k * x + b = k * (x + 4) + b + 1) → k = -1/4 :=
by
  intros b
  use -1/4
  intros h
  sorry

end slope_transformation_l61_61368


namespace pet_store_cages_l61_61331

def initial_puppies : ℕ := 18
def puppies_sold : ℕ := 3
def puppies_per_cage : ℕ := 5

def remaining_puppies : ℕ := initial_puppies - puppies_sold
def number_of_cages : ℕ := remaining_puppies / puppies_per_cage

theorem pet_store_cages : number_of_cages = 3 :=
by sorry

end pet_store_cages_l61_61331


namespace operation_result_l61_61928

-- Define the operation
def operation (a b : ℝ) : ℝ := (a - b) ^ 3

theorem operation_result (x y : ℝ) : operation ((x - y) ^ 3) ((y - x) ^ 3) = -8 * (y - x) ^ 9 := 
  sorry

end operation_result_l61_61928


namespace planting_area_l61_61014

variable (x : ℝ)

def garden_length := x + 2
def garden_width := 4
def path_width := 1

def effective_garden_length := garden_length x - 2 * path_width
def effective_garden_width := garden_width - 2 * path_width

theorem planting_area : effective_garden_length x * effective_garden_width = 2 * x := by
  simp [garden_length, garden_width, path_width, effective_garden_length, effective_garden_width]
  sorry

end planting_area_l61_61014


namespace arithmetic_sequence_a7_l61_61375

/--
In an arithmetic sequence {a_n}, it is known that a_1 = 2 and a_3 + a_5 = 10.
Then, we need to prove that a_7 = 8.
-/
theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 2) 
  (h2 : a 3 + a 5 = 10) 
  (h3 : ∀ n, a n = 2 + (n - 1) * d) : 
  a 7 = 8 := by
  sorry

end arithmetic_sequence_a7_l61_61375


namespace simplify_rationalize_expr_l61_61551

theorem simplify_rationalize_expr :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_rationalize_expr_l61_61551


namespace men_count_in_first_group_is_20_l61_61219

noncomputable def men_needed_to_build_fountain (work1 : ℝ) (days1 : ℕ) (length1 : ℝ) (workers2 : ℕ) (days2 : ℕ) (length2 : ℝ) (work_per_man_per_day2 : ℝ) : ℕ :=
  let work_per_day2 := length2 / days2
  let work_per_man_per_day2 := work_per_day2 / workers2
  let total_work1 := length1 / days1
  Nat.floor (total_work1 / work_per_man_per_day2)

theorem men_count_in_first_group_is_20 :
  men_needed_to_build_fountain 56 6 56 35 3 49 (49 / (35 * 3)) = 20 :=
by
  sorry

end men_count_in_first_group_is_20_l61_61219


namespace balls_into_boxes_l61_61756

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l61_61756


namespace problem_l61_61247

-- Definitions of the function g and its values at specific points
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

-- Conditions given in the problem
theorem problem (d e f : ℝ)
  (h0 : g d e f 0 = 8)
  (h1 : g d e f 1 = 5) :
  d + e + 2 * f = 13 :=
by
  sorry

end problem_l61_61247


namespace sweater_cost_l61_61655

theorem sweater_cost (S : ℚ) (M : ℚ) (C : ℚ) (h1 : S = 80) (h2 : M = 3 / 4 * 80) (h3 : C = S - M) : C = 20 := by
  sorry

end sweater_cost_l61_61655


namespace train_speed_is_45_km_per_hr_l61_61930

/-- 
  Given the length of the train (135 m), the time to cross a bridge (30 s),
  and the length of the bridge (240 m), we want to prove that the speed of the 
  train is 45 km/hr.
--/

def length_of_train : ℕ := 135
def time_to_cross_bridge : ℕ := 30
def length_of_bridge : ℕ := 240
def speed_of_train_in_km_per_hr (L_t t L_b : ℕ) : ℕ := 
  ((L_t + L_b) * 36 / 10) / t

theorem train_speed_is_45_km_per_hr : 
  speed_of_train_in_km_per_hr length_of_train time_to_cross_bridge length_of_bridge = 45 :=
by 
  -- Assuming the calculations are correct, the expected speed is provided here directly
  sorry

end train_speed_is_45_km_per_hr_l61_61930


namespace sum_of_squares_of_roots_eq_14_l61_61203

theorem sum_of_squares_of_roots_eq_14 {α β γ : ℝ}
  (h1: ∀ x: ℝ, (x^3 - 6*x^2 + 11*x - 6 = 0) → (x = α ∨ x = β ∨ x = γ)) :
  α^2 + β^2 + γ^2 = 14 :=
by
  sorry

end sum_of_squares_of_roots_eq_14_l61_61203


namespace sum_of_squares_l61_61282

theorem sum_of_squares (a b c : ℝ) (h_arith : a + b + c = 30) (h_geom : a * b * c = 216) 
(h_harm : 1/a + 1/b + 1/c = 3/4) : a^2 + b^2 + c^2 = 576 := 
by 
  sorry

end sum_of_squares_l61_61282


namespace arithmetic_sequence_problem_l61_61875

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ) 
  (a1 : a 1 = 3) 
  (d : ℕ := 2) 
  (h : ∀ n, a n = a 1 + (n - 1) * d) 
  (h_25 : a n = 25) : 
  n = 12 := 
by
  sorry

end arithmetic_sequence_problem_l61_61875


namespace mary_marbles_l61_61824

theorem mary_marbles (total_marbles joan_marbles mary_marbles : ℕ) 
  (h1 : total_marbles = 12) 
  (h2 : joan_marbles = 3) 
  (h3 : total_marbles = joan_marbles + mary_marbles) : 
  mary_marbles = 9 := 
by
  rw [h1, h2, add_comm] at h3
  linarith

end mary_marbles_l61_61824


namespace asian_population_percentage_in_west_is_57_l61_61582

variable (NE MW South West : ℕ)

def total_asian_population (NE MW South West : ℕ) : ℕ :=
  NE + MW + South + West

def west_asian_population_percentage
  (NE MW South West : ℕ) (total_asian_population : ℕ) : ℚ :=
  (West : ℚ) / (total_asian_population : ℚ) * 100

theorem asian_population_percentage_in_west_is_57 :
  total_asian_population 2 3 4 12 = 21 →
  west_asian_population_percentage 2 3 4 12 21 = 57 :=
by
  intros
  sorry

end asian_population_percentage_in_west_is_57_l61_61582


namespace number_of_blue_crayons_given_to_Becky_l61_61238

-- Definitions based on the conditions
def initial_green_crayons : ℕ := 5
def initial_blue_crayons : ℕ := 8
def given_out_green_crayons : ℕ := 3
def total_crayons_left : ℕ := 9

-- Statement of the problem and expected proof
theorem number_of_blue_crayons_given_to_Becky (initial_green_crayons initial_blue_crayons given_out_green_crayons total_crayons_left : ℕ) : 
  initial_green_crayons = 5 →
  initial_blue_crayons = 8 →
  given_out_green_crayons = 3 →
  total_crayons_left = 9 →
  ∃ num_blue_crayons_given_to_Becky, num_blue_crayons_given_to_Becky = 1 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_blue_crayons_given_to_Becky_l61_61238


namespace find_xyz_l61_61834

theorem find_xyz : ∃ (x y z : ℕ), x + y + z = 12 ∧ 7 * x + 5 * y + 8 * z = 79 ∧ x = 5 ∧ y = 4 ∧ z = 3 :=
by
  sorry

end find_xyz_l61_61834


namespace Vishal_investment_percentage_more_than_Trishul_l61_61447

-- Definitions from the conditions
def R : ℚ := 2400
def T : ℚ := 0.90 * R
def total_investments : ℚ := 6936

-- Mathematically equivalent statement to prove
theorem Vishal_investment_percentage_more_than_Trishul :
  ∃ V : ℚ, V + T + R = total_investments ∧ (V - T) / T * 100 = 10 := 
by
  sorry

end Vishal_investment_percentage_more_than_Trishul_l61_61447


namespace justin_run_time_l61_61323

theorem justin_run_time : 
  let flat_ground_rate := 2 / 2 -- Justin runs 2 blocks in 2 minutes on flat ground
  let uphill_rate := 2 / 3 -- Justin runs 2 blocks in 3 minutes uphill
  let total_blocks := 10 -- Justin is 10 blocks from home
  let uphill_blocks := 6 -- 6 of those blocks are uphill
  let flat_ground_blocks := total_blocks - uphill_blocks -- Remainder are flat ground
  let flat_ground_time := flat_ground_blocks * flat_ground_rate
  let uphill_time := uphill_blocks * uphill_rate
  let total_time := flat_ground_time + uphill_time
  total_time = 13 := 
by 
  sorry

end justin_run_time_l61_61323


namespace fx_properties_l61_61940

-- Definition of the function
def f (x : ℝ) : ℝ := x * |x|

-- Lean statement for the proof problem
theorem fx_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) :=
by
  -- Definition used directly from the conditions
  sorry

end fx_properties_l61_61940


namespace find_total_shaded_area_l61_61102

/-- Definition of the rectangles' dimensions and overlap conditions -/
def rect1_length : ℕ := 4
def rect1_width : ℕ := 15
def rect2_length : ℕ := 5
def rect2_width : ℕ := 10
def rect3_length : ℕ := 3
def rect3_width : ℕ := 18
def shared_side_length : ℕ := 4
def trip_overlap_width : ℕ := 3

/-- Calculation of the rectangular overlap using given conditions -/
theorem find_total_shaded_area : (rect1_length * rect1_width + rect2_length * rect2_width + rect3_length * rect3_width - shared_side_length * shared_side_length - trip_overlap_width * shared_side_length) = 136 :=
    by sorry

end find_total_shaded_area_l61_61102


namespace bags_filled_on_saturday_l61_61387

-- Definitions of the conditions
def bags_sat (S : ℕ) := S
def bags_sun := 4
def cans_per_bag := 9
def total_cans := 63

-- The statement to prove
theorem bags_filled_on_saturday (S : ℕ) 
  (h : total_cans = (bags_sat S + bags_sun) * cans_per_bag) : 
  S = 3 :=
by sorry

end bags_filled_on_saturday_l61_61387


namespace horses_lcm_l61_61857

theorem horses_lcm :
  let horse_times := [2, 3, 4, 5, 6, 7, 8, 9]
  let lcm_six := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
  let time_T := lcm_six
  lcm_six = 420 ∧ (Nat.digits 10 time_T).sum = 6 := by
    let horse_times := [2, 3, 4, 5, 6, 7, 8, 9]
    let lcm_six := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
    let time_T := lcm_six
    have h1 : lcm_six = 420 := sorry
    have h2 : (Nat.digits 10 time_T).sum = 6 := sorry
    exact ⟨h1, h2⟩

end horses_lcm_l61_61857


namespace largest_divisor_of_product_of_5_consecutive_integers_l61_61242

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l61_61242


namespace always_negative_l61_61123

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (x ^ 2 + 1) - x) - Real.sin x

theorem always_negative (a b : ℝ) (ha : a ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
                     (hb : b ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
                     (hab : a + b ≠ 0) : 
  (f a + f b) / (a + b) < 0 := 
sorry

end always_negative_l61_61123


namespace algebra_expression_value_l61_61676

theorem algebra_expression_value (a b : ℝ) (h : (30^3) * a + 30 * b - 7 = 9) :
  (-30^3) * a + (-30) * b + 2 = -14 := 
by
  sorry

end algebra_expression_value_l61_61676


namespace sum_of_four_triangles_l61_61729

theorem sum_of_four_triangles (x y : ℝ) (h1 : 3 * x + 2 * y = 27) (h2 : 2 * x + 3 * y = 23) : 4 * y = 12 :=
sorry

end sum_of_four_triangles_l61_61729


namespace smallest_number_divisible_by_18_70_100_84_increased_by_3_l61_61732

theorem smallest_number_divisible_by_18_70_100_84_increased_by_3 :
  ∃ n : ℕ, (n + 3) % 18 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 100 = 0 ∧ (n + 3) % 84 = 0 ∧ n = 6297 :=
by
  sorry

end smallest_number_divisible_by_18_70_100_84_increased_by_3_l61_61732


namespace maximal_possible_degree_difference_l61_61795

theorem maximal_possible_degree_difference (n_vertices : ℕ) (n_edges : ℕ) (disjoint_edge_pairs : ℕ) 
    (h1 : n_vertices = 30) (h2 : n_edges = 105) (h3 : disjoint_edge_pairs = 4822) : 
    ∃ (max_diff : ℕ), max_diff = 22 :=
by
  sorry

end maximal_possible_degree_difference_l61_61795


namespace rightmost_three_digits_of_7_pow_2023_l61_61713

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 343 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l61_61713


namespace time_of_free_fall_l61_61307

theorem time_of_free_fall (h : ℝ) (t : ℝ) (height_fall_eq : h = 4.9 * t^2) (initial_height : h = 490) : t = 10 :=
by
  -- Proof is omitted
  sorry

end time_of_free_fall_l61_61307


namespace work_completion_time_l61_61959

theorem work_completion_time 
(w : ℝ)  -- total amount of work
(A B : ℝ)  -- work rate of a and b per day
(h1 : A + B = w / 30)  -- combined work rate
(h2 : 20 * (A + B) + 20 * A = w) : 
  (1 / A = 60) :=
sorry

end work_completion_time_l61_61959


namespace not_divisible_by_121_l61_61989

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 3 * n + 5)) :=
by
  sorry

end not_divisible_by_121_l61_61989


namespace find_roots_l61_61228

theorem find_roots : 
  ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (x + 2) = 0 ↔ (x = -2 ∨ x = 2 ∨ x = 3) := by
  sorry

end find_roots_l61_61228


namespace Roger_years_to_retire_l61_61103

noncomputable def Peter : ℕ := 12
noncomputable def Robert : ℕ := Peter - 4
noncomputable def Mike : ℕ := Robert - 2
noncomputable def Tom : ℕ := 2 * Robert
noncomputable def Roger : ℕ := Peter + Tom + Robert + Mike

theorem Roger_years_to_retire :
  Roger = 42 → 50 - Roger = 8 := by
sorry

end Roger_years_to_retire_l61_61103


namespace floyd_infinite_jumps_l61_61062

def sum_of_digits (n: Nat) : Nat := 
  n.digits 10 |>.sum 

noncomputable def jumpable (a b: Nat) : Prop := 
  b > a ∧ b ≤ 2 * a 

theorem floyd_infinite_jumps :
  ∃ f : ℕ → ℕ, 
    (∀ n : ℕ, jumpable (f n) (f (n + 1))) ∧
    (∀ m n : ℕ, m ≠ n → sum_of_digits (f m) ≠ sum_of_digits (f n)) :=
sorry

end floyd_infinite_jumps_l61_61062


namespace remainder_2017_div_89_l61_61262

theorem remainder_2017_div_89 : 2017 % 89 = 59 :=
by
  sorry

end remainder_2017_div_89_l61_61262


namespace sum_of_roots_l61_61741

theorem sum_of_roots (a b c : ℝ) (x1 x2 x3 : ℝ) (h_eq: 6*x1^3 + 7*x2^2 - 12*x3 = 0) :
  (x1 + x2 + x3) = -1.17 :=
sorry

end sum_of_roots_l61_61741


namespace probability_two_queens_or_at_least_one_king_l61_61720

/-- Prove that the probability of either drawing two queens or drawing at least one king 
    when 2 cards are selected randomly from a standard deck of 52 cards is 2/13. -/
theorem probability_two_queens_or_at_least_one_king :
  (∃ (kq pk pq : ℚ), kq = 4 ∧
                     pk = 4 ∧
                     pq = 52 ∧
                     (∃ (p : ℚ), p = (kq*(kq-1))/(pq*(pq-1)) + (pk/pq)*(pq-pk)/(pq-1) + (kq*(kq-1))/(pq*(pq-1)) ∧
                            p = 2/13)) :=
by {
  sorry
}

end probability_two_queens_or_at_least_one_king_l61_61720


namespace marble_arrangement_count_l61_61702
noncomputable def countValidMarbleArrangements : Nat := 
  let totalArrangements := 120
  let restrictedPairsCount := 24
  totalArrangements - restrictedPairsCount

theorem marble_arrangement_count :
  countValidMarbleArrangements = 96 :=
  by
    sorry

end marble_arrangement_count_l61_61702


namespace soccer_lineup_count_l61_61110

theorem soccer_lineup_count :
  let total_players : ℕ := 16
  let total_starters : ℕ := 7
  let m_j_players : ℕ := 2 -- Michael and John
  let other_players := total_players - m_j_players
  let total_ways : ℕ :=
    2 * Nat.choose other_players (total_starters - 1) + Nat.choose other_players (total_starters - 2)
  total_ways = 8008
:= sorry

end soccer_lineup_count_l61_61110


namespace one_eighth_of_N_l61_61254

theorem one_eighth_of_N
  (N : ℝ)
  (h : (6 / 11) * N = 48) : (1 / 8) * N = 11 :=
sorry

end one_eighth_of_N_l61_61254


namespace slope_OA_l61_61648

-- Definitions for the given conditions
def ellipse (a b : ℝ) := {P : ℝ × ℝ | (P.1^2) / a^2 + (P.2^2) / b^2 = 1}

def C1 := ellipse 2 1  -- ∑(x^2 / 4 + y^2 = 1)
def C2 := ellipse 2 4  -- ∑(y^2 / 16 + x^2 / 4 = 1)

variable {P₁ P₂ : ℝ × ℝ}  -- Points A and B
variable (h1 : P₁ ∈ C1)
variable (h2 : P₂ ∈ C2)
variable (h_rel : P₂.1 = 2 * P₁.1 ∧ P₂.2 = 2 * P₁.2)  -- ∑(x₂ = 2x₁, y₂ = 2y₁)

-- Proof that the slope of ray OA is ±1
theorem slope_OA : ∃ (m : ℝ), (m = 1 ∨ m = -1) :=
sorry

end slope_OA_l61_61648


namespace Chrysler_Building_floors_l61_61077

variable (C L : ℕ)

theorem Chrysler_Building_floors :
  (C = L + 11) → (C + L = 35) → (C = 23) :=
by
  intro h1 h2
  sorry

end Chrysler_Building_floors_l61_61077


namespace no_quadratic_polynomials_f_g_l61_61502

theorem no_quadratic_polynomials_f_g (f g : ℝ → ℝ) 
  (hf : ∃ a b c, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e h, ∀ x, g x = d * x^2 + e * x + h) : 
  ¬ (∀ x, f (g x) = x^4 - 3 * x^3 + 3 * x^2 - x) :=
by
  sorry

end no_quadratic_polynomials_f_g_l61_61502


namespace proof_problem_l61_61079

noncomputable def g (x : ℝ) : ℝ := 2^(2*x - 1) + x - 1

theorem proof_problem
  (x1 x2 : ℝ)
  (h1 : g x1 = 0)  -- x1 is the root of the equation
  (h2 : 2 * x2 - 1 = 0)  -- x2 is the zero point of f(x) = 2x - 1
  : |x1 - x2| ≤ 1/4 :=
sorry

end proof_problem_l61_61079


namespace exists_polynomial_distinct_powers_of_2_l61_61188

open Polynomial

variable (n : ℕ) (hn : n > 0)

theorem exists_polynomial_distinct_powers_of_2 :
  ∃ P : Polynomial ℤ, P.degree = n ∧ (∃ (k : Fin (n + 1) → ℕ), ∀ i j : Fin (n + 1), i ≠ j → 2 ^ k i ≠ 2 ^ k j ∧ (∀ i, P.eval i.val = 2 ^ k i)) :=
sorry

end exists_polynomial_distinct_powers_of_2_l61_61188


namespace triangle_area_l61_61465

theorem triangle_area 
  (DE EL EF : ℝ)
  (hDE : DE = 14)
  (hEL : EL = 9)
  (hEF : EF = 17)
  (DL : ℝ)
  (hDL : DE^2 = DL^2 + EL^2)
  (hDL_val : DL = Real.sqrt 115):
  (1/2) * EF * DL = 17 * Real.sqrt 115 / 2 :=
by
  -- Sorry, as the proof is not required.
  sorry

end triangle_area_l61_61465


namespace volume_of_pyramid_l61_61731

/--
Rectangle ABCD is the base of pyramid PABCD. Let AB = 10, BC = 6, PA is perpendicular to AB, and PB = 20. 
If PA makes an angle θ = 30° with the diagonal AC of the base, prove the volume of the pyramid PABCD is 200 cubic units.
-/
theorem volume_of_pyramid (AB BC PB : ℝ) (θ : ℝ) (hAB : AB = 10) (hBC : BC = 6)
  (hPB : PB = 20) (hθ : θ = 30) (PA_is_perpendicular_to_AB : true) (PA_makes_angle_with_AC : true) : 
  ∃ V, V = 1 / 3 * (AB * BC) * 10 ∧ V = 200 := 
by
  exists 1 / 3 * (AB * BC) * 10
  sorry

end volume_of_pyramid_l61_61731


namespace fruit_seller_original_apples_l61_61725

variable (x : ℝ)

theorem fruit_seller_original_apples (h : 0.60 * x = 420) : x = 700 := by
  sorry

end fruit_seller_original_apples_l61_61725


namespace fraction_numerator_greater_than_denominator_l61_61088

theorem fraction_numerator_greater_than_denominator (x : ℝ) :
  (4 * x + 2 > 8 - 3 * x) ↔ (6 / 7 < x ∧ x ≤ 3) :=
by
  sorry

end fraction_numerator_greater_than_denominator_l61_61088


namespace rectangle_area_l61_61790

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) 
  : l * w = 1600 := 
by 
  sorry

end rectangle_area_l61_61790


namespace sculpture_cost_in_inr_l61_61721

def convert_currency (n_cost : ℕ) (n_to_b_rate : ℕ) (b_to_i_rate : ℕ) : ℕ := 
  (n_cost / n_to_b_rate) * b_to_i_rate

theorem sculpture_cost_in_inr (n_cost : ℕ) (n_to_b_rate : ℕ) (b_to_i_rate : ℕ) :
  n_cost = 360 → 
  n_to_b_rate = 18 → 
  b_to_i_rate = 20 →
  convert_currency n_cost n_to_b_rate b_to_i_rate = 400 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- turns 360 / 18 * 20 = 400
  sorry

end sculpture_cost_in_inr_l61_61721


namespace Kiarra_age_l61_61750

variable (Kiarra Bea Job Figaro Harry : ℕ)

theorem Kiarra_age 
  (h1 : Kiarra = 2 * Bea)
  (h2 : Job = 3 * Bea)
  (h3 : Figaro = Job + 7)
  (h4 : Harry = Figaro / 2)
  (h5 : Harry = 26) : 
  Kiarra = 30 := sorry

end Kiarra_age_l61_61750


namespace no_natural_pairs_exist_l61_61612

theorem no_natural_pairs_exist (n m : ℕ) : ¬(n + 1) * (2 * n + 1) = 18 * m ^ 2 :=
by
  sorry

end no_natural_pairs_exist_l61_61612


namespace find_f2_l61_61316

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f2 (h : ∀ x : ℝ, f x + 3 * f (8 - x) = x) : f 2 = 2 :=
by
  sorry

end find_f2_l61_61316


namespace anya_hair_growth_l61_61787

theorem anya_hair_growth (wash_loss : ℕ) (brush_loss : ℕ) (total_loss : ℕ) : wash_loss = 32 → brush_loss = wash_loss / 2 → total_loss = wash_loss + brush_loss → total_loss + 1 = 49 :=
by
  sorry

end anya_hair_growth_l61_61787


namespace find_fourth_number_l61_61221

theorem find_fourth_number : 
  ∃ (x : ℝ), (217 + 2.017 + 0.217 + x = 221.2357) ∧ (x = 2.0017) :=
by
  sorry

end find_fourth_number_l61_61221


namespace evaluate_expression_l61_61507

theorem evaluate_expression : (1.2^3 - (0.9^3 / 1.2^2) + 1.08 + 0.9^2 = 3.11175) :=
by
  sorry -- Proof goes here

end evaluate_expression_l61_61507


namespace only_one_positive_integer_n_l61_61963

theorem only_one_positive_integer_n (k : ℕ) (hk : 0 < k) (m : ℕ) (hm : k + 2 ≤ m) :
  ∃! (n : ℕ), 0 < n ∧ n^m ∣ 5^(n^k) + 1 :=
sorry

end only_one_positive_integer_n_l61_61963


namespace problem_1_problem_2_l61_61679

open Set

variables {U : Type*} [TopologicalSpace U] (a x : ℝ)

def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def N (a : ℝ) : Set ℝ := { x | a + 1 ≤ x ∧ x ≤ 2 * a + 1 }

noncomputable def complement_N (a : ℝ) : Set ℝ := { x | x < a + 1 ∨ 2 * a + 1 < x }

theorem problem_1 (h : a = 2) :
  M ∩ (complement_N a) = { x | -2 ≤ x ∧ x < 3 } :=
sorry

theorem problem_2 (h : M ∪ N a = M) :
  a ≤ 2 :=
sorry

end problem_1_problem_2_l61_61679


namespace intersection_M_N_l61_61882

-- Define the universal set U, and subsets M and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x < 1}

-- Prove that the intersection of M and N is as stated
theorem intersection_M_N :
  M ∩ N = {x | -2 ≤ x ∧ x < 1} :=
by
  -- This is where the proof would go
  sorry

end intersection_M_N_l61_61882


namespace relationship_between_a_b_c_l61_61418

theorem relationship_between_a_b_c (a b c : ℕ) (h1 : a = 2^40) (h2 : b = 3^32) (h3 : c = 4^24) : a < c ∧ c < b := by
  -- Definitions as per conditions
  have ha : a = 32^8 := by sorry
  have hb : b = 81^8 := by sorry
  have hc : c = 64^8 := by sorry
  -- Comparisons involving the bases
  have h : 32 < 64 := by sorry
  have h' : 64 < 81 := by sorry
  -- Resultant comparison
  exact ⟨by sorry, by sorry⟩

end relationship_between_a_b_c_l61_61418


namespace solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l61_61952

variable (a b : ℝ)

theorem solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0 :
  (∀ x : ℝ, (|x - 2| > 1 ↔ x^2 + a * x + b > 0)) → a + b = -1 :=
by
  sorry

end solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l61_61952


namespace dyed_pink_correct_l61_61335

def silk_dyed_green := 61921
def total_yards_dyed := 111421
def yards_dyed_pink := total_yards_dyed - silk_dyed_green

theorem dyed_pink_correct : yards_dyed_pink = 49500 := by 
  sorry

end dyed_pink_correct_l61_61335


namespace crescent_moon_area_l61_61956

theorem crescent_moon_area :
  let big_quarter_circle := (4 * 4 * Real.pi) / 4
  let small_semi_circle := (2 * 2 * Real.pi) / 2
  let crescent_area := big_quarter_circle - small_semi_circle
  crescent_area = 2 * Real.pi :=
by
  let big_quarter_circle := (4 * 4 * Real.pi) / 4
  let small_semi_circle := (2 * 2 * Real.pi) / 2
  let crescent_area := big_quarter_circle - small_semi_circle
  have h_bqc : big_quarter_circle = 4 * Real.pi := by
    sorry
  have h_ssc : small_semi_circle = 2 * Real.pi := by
    sorry
  have h_ca : crescent_area = 2 * Real.pi := by
    sorry
  exact h_ca

end crescent_moon_area_l61_61956


namespace clocks_sync_again_in_lcm_days_l61_61826

-- Defining the given conditions based on the problem statement.

-- Arthur's clock gains 15 minutes per day, taking 48 days to gain 12 hours (720 minutes).
def arthur_days : ℕ := 48

-- Oleg's clock gains 12 minutes per day, taking 60 days to gain 12 hours (720 minutes).
def oleg_days : ℕ := 60

-- The problem asks to prove that the situation repeats after 240 days, which is the LCM of 48 and 60.
theorem clocks_sync_again_in_lcm_days : Nat.lcm arthur_days oleg_days = 240 := 
by 
  sorry

end clocks_sync_again_in_lcm_days_l61_61826


namespace simplify_expression_l61_61486

theorem simplify_expression :
  (49 * 91^3 + 338 * 343^2) / (66^3 - 176 * 121) / (39^3 * 7^5 / 1331000) = 125 / 13 :=
by
  sorry

end simplify_expression_l61_61486


namespace right_triangle_exists_with_area_ab_l61_61638

theorem right_triangle_exists_with_area_ab (a b c d : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
    (h1 : a * b = c * d) (h2 : a + b = c - d) :
    ∃ (x y z : ℕ), x^2 + y^2 = z^2 ∧ (x * y / 2 = a * b) := sorry

end right_triangle_exists_with_area_ab_l61_61638


namespace smallest_even_integer_l61_61401

theorem smallest_even_integer :
  ∃ (x : ℤ), |3 * x - 4| ≤ 20 ∧ (∀ (y : ℤ), |3 * y - 4| ≤ 20 → (2 ∣ y) → x ≤ y) ∧ (2 ∣ x) :=
by
  use -4
  sorry

end smallest_even_integer_l61_61401


namespace stream_current_rate_l61_61059

theorem stream_current_rate (r c : ℝ) (h1 : 20 / (r + c) + 6 = 20 / (r - c)) (h2 : 20 / (3 * r + c) + 1.5 = 20 / (3 * r - c)) 
  : c = 3 :=
  sorry

end stream_current_rate_l61_61059


namespace intersection_complement_l61_61353

def U : Set ℤ := Set.univ
def M : Set ℤ := {1, 2}
def P : Set ℤ := {-2, -1, 0, 1, 2}
def CUM : Set ℤ := {x : ℤ | x ∉ M}

theorem intersection_complement :
  P ∩ CUM = {-2, -1, 0} :=
by
  sorry

end intersection_complement_l61_61353


namespace total_birds_on_fence_l61_61332

-- Definitions based on conditions.
def initial_birds : ℕ := 12
def additional_birds : ℕ := 8

-- Theorem corresponding to the problem statement.
theorem total_birds_on_fence : initial_birds + additional_birds = 20 := by 
  sorry

end total_birds_on_fence_l61_61332


namespace sum_of_units_digits_eq_0_l61_61225

-- Units digit function definition
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement in Lean 
theorem sum_of_units_digits_eq_0 :
  units_digit (units_digit (17 * 34) + units_digit (19 * 28)) = 0 :=
by
  sorry

end sum_of_units_digits_eq_0_l61_61225


namespace number_of_exclusive_students_l61_61626

-- Definitions from the conditions
def S_both : ℕ := 16
def S_alg : ℕ := 36
def S_geo_only : ℕ := 15

-- Theorem to prove the number of students taking algebra or geometry but not both
theorem number_of_exclusive_students : (S_alg - S_both) + S_geo_only = 35 :=
by
  sorry

end number_of_exclusive_students_l61_61626


namespace inequality_holds_l61_61205

theorem inequality_holds (c : ℝ) : (∀ x : ℝ, 3 * Real.sin x - 4 * Real.cos x + c > 0) → c > 5 := by sorry

end inequality_holds_l61_61205


namespace problem_ABC_sum_l61_61140

-- Let A, B, and C be positive integers such that A and C, B and C, and A and B
-- have no common factor greater than 1.
-- If they satisfy the equation A * log_100 5 + B * log_100 4 = C,
-- then we need to prove that A + B + C = 4.

theorem problem_ABC_sum (A B C : ℕ) (h1 : 1 < A ∧ 1 < B ∧ 1 < C)
    (h2 : A.gcd B = 1 ∧ B.gcd C = 1 ∧ A.gcd C = 1)
    (h3 : A * Real.log 5 / Real.log 100 + B * Real.log 4 / Real.log 100 = C) :
    A + B + C = 4 :=
sorry

end problem_ABC_sum_l61_61140


namespace fraction_geq_81_l61_61314

theorem fraction_geq_81 {p q r s : ℝ} (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) :
  ((p^2 + p + 1) * (q^2 + q + 1) * (r^2 + r + 1) * (s^2 + s + 1)) / (p * q * r * s) ≥ 81 :=
by
  sorry

end fraction_geq_81_l61_61314


namespace maximum_value_a_over_b_plus_c_l61_61965

open Real

noncomputable def max_frac_a_over_b_plus_c (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_eq: a * (a + b + c) = b * c) : ℝ :=
  if (b = c) then (Real.sqrt 2 - 1) / 2 else -1 -- placeholder for irrelevant case

theorem maximum_value_a_over_b_plus_c 
  (a b c : ℝ) 
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq: a * (a + b + c) = b * c) :
  max_frac_a_over_b_plus_c a b c h_pos h_eq = (Real.sqrt 2 - 1) / 2 :=
sorry

end maximum_value_a_over_b_plus_c_l61_61965


namespace smallest_a_plus_b_l61_61696

theorem smallest_a_plus_b : ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2^3 * 3^7 * 7^2 = a^b ∧ a + b = 380 :=
sorry

end smallest_a_plus_b_l61_61696


namespace parallelogram_side_length_l61_61870

theorem parallelogram_side_length (x y : ℚ) (h1 : 3 * x + 2 = 12) (h2 : 5 * y - 3 = 9) : x + y = 86 / 15 :=
by 
  sorry

end parallelogram_side_length_l61_61870


namespace votes_cast_is_330_l61_61339

variable (T A F : ℝ)

theorem votes_cast_is_330
  (h1 : A = 0.40 * T)
  (h2 : F = A + 66)
  (h3 : T = F + A) :
  T = 330 :=
by
  sorry

end votes_cast_is_330_l61_61339


namespace total_songs_l61_61245

-- Define the number of albums Faye bought and the number of songs per album
def country_albums : ℕ := 2
def pop_albums : ℕ := 3
def songs_per_album : ℕ := 6

-- Define the total number of albums Faye bought
def total_albums : ℕ := country_albums + pop_albums

-- Prove that the total number of songs Faye bought is 30
theorem total_songs : total_albums * songs_per_album = 30 := by
  sorry

end total_songs_l61_61245


namespace total_participants_l61_61915

theorem total_participants (x : ℕ) (h1 : 800 / x + 60 = 800 / (x - 3)) : x = 8 :=
sorry

end total_participants_l61_61915


namespace perfect_square_of_d_l61_61960

theorem perfect_square_of_d (a b c d : ℤ) (h : d = (a + (2:ℝ)^(1/3) * b + (4:ℝ)^(1/3) * c)^2) : ∃ k : ℤ, d = k^2 :=
by
  sorry

end perfect_square_of_d_l61_61960


namespace rectangle_fraction_l61_61804

noncomputable def side_of_square : ℝ := Real.sqrt 900
noncomputable def radius_of_circle : ℝ := side_of_square
noncomputable def area_of_rectangle : ℝ := 120
noncomputable def breadth_of_rectangle : ℝ := 10
noncomputable def length_of_rectangle : ℝ := area_of_rectangle / breadth_of_rectangle
noncomputable def fraction : ℝ := length_of_rectangle / radius_of_circle

theorem rectangle_fraction :
  (length_of_rectangle / radius_of_circle) = (2 / 5) :=
by
  sorry

end rectangle_fraction_l61_61804


namespace eggs_in_two_boxes_l61_61831

theorem eggs_in_two_boxes (eggs_per_box : ℕ) (number_of_boxes : ℕ) (total_eggs : ℕ) 
  (h1 : eggs_per_box = 3)
  (h2 : number_of_boxes = 2) :
  total_eggs = eggs_per_box * number_of_boxes :=
sorry

end eggs_in_two_boxes_l61_61831


namespace first_of_five_consecutive_sums_60_l61_61580

theorem first_of_five_consecutive_sums_60 (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 60) : n = 10 :=
by {
  sorry
}

end first_of_five_consecutive_sums_60_l61_61580


namespace dividend_is_5336_l61_61426

theorem dividend_is_5336 (D Q R : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) : 
  D * Q + R = 5336 := 
by sorry

end dividend_is_5336_l61_61426


namespace maria_towels_l61_61949

theorem maria_towels (green_towels white_towels given_towels : ℕ) (h1 : green_towels = 35) (h2 : white_towels = 21) (h3 : given_towels = 34) :
  green_towels + white_towels - given_towels = 22 :=
by
  sorry

end maria_towels_l61_61949


namespace inequality_chain_l61_61260

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem inequality_chain (a b : ℝ) (h1 : b > a) (h2 : a > 3) :
  f b < f ((a + b) / 2) ∧ f ((a + b) / 2) < f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) < f a :=
by
  sorry

end inequality_chain_l61_61260


namespace second_coloring_book_pictures_l61_61097

-- Let P1 be the number of pictures in the first coloring book.
def P1 := 23

-- Let P2 be the number of pictures in the second coloring book.
variable (P2 : Nat)

-- Let colored_pics be the number of pictures Rachel colored.
def colored_pics := 44

-- Let remaining_pics be the number of pictures Rachel still has to color.
def remaining_pics := 11

-- Total number of pictures in both coloring books.
def total_pics := colored_pics + remaining_pics

theorem second_coloring_book_pictures :
  P2 = total_pics - P1 :=
by
  -- We need to prove that P2 = 32.
  sorry

end second_coloring_book_pictures_l61_61097


namespace students_before_Yoongi_l61_61819

theorem students_before_Yoongi (total_students : ℕ) (students_after_Yoongi : ℕ) 
  (condition1 : total_students = 20) (condition2 : students_after_Yoongi = 11) :
  total_students - students_after_Yoongi - 1 = 8 :=
by 
  sorry

end students_before_Yoongi_l61_61819


namespace sum_mod_9_l61_61886

theorem sum_mod_9 :
  (8 + 77 + 666 + 5555 + 44444 + 333333 + 2222222 + 11111111) % 9 = 3 := 
by sorry

end sum_mod_9_l61_61886


namespace cubic_inequality_solution_l61_61420

theorem cubic_inequality_solution (x : ℝ) (h : 0 ≤ x) : 
  x^3 - 9*x^2 - 16*x > 0 ↔ 16 < x := 
by 
  sorry

end cubic_inequality_solution_l61_61420


namespace part_one_part_one_equality_part_two_l61_61337

-- Given constants and their properties
variables (a b c d : ℝ)

-- Statement for the first problem
theorem part_one : a^6 + b^6 + c^6 + d^6 - 6 * a * b * c * d ≥ -2 :=
sorry

-- Statement for the equality condition in the first problem
theorem part_one_equality (h : |a| = 1 ∧ |b| = 1 ∧ |c| = 1 ∧ |d| = 1) : 
  a^6 + b^6 + c^6 + d^6 - 6 * a * b * c * d = -2 :=
sorry

-- Statement for the second problem (existence of Mk for k >= 4 and odd)
theorem part_two (k : ℕ) (hk1 : 4 ≤ k) (hk2 : k % 2 = 1) : ∃ Mk : ℝ, ∀ a b c d : ℝ, a^k + b^k + c^k + d^k - k * a * b * c * d ≥ Mk :=
sorry

end part_one_part_one_equality_part_two_l61_61337


namespace finite_fraction_n_iff_l61_61176

theorem finite_fraction_n_iff (n : ℕ) (h_pos : 0 < n) :
  (∃ (a b : ℕ), n * (n + 1) = 2^a * 5^b) ↔ (n = 1 ∨ n = 4) :=
by
  sorry

end finite_fraction_n_iff_l61_61176


namespace sum_of_values_of_N_l61_61325

theorem sum_of_values_of_N (N : ℂ) : (N * (N - 8) = 12) → (∃ x y : ℂ, N = x ∨ N = y ∧ x + y = 8) :=
by
  sorry

end sum_of_values_of_N_l61_61325


namespace total_cost_correct_l61_61986

noncomputable def camera_old_cost : ℝ := 4000
noncomputable def camera_new_cost := camera_old_cost * 1.30
noncomputable def lens_cost := 400
noncomputable def lens_discount := 200
noncomputable def lens_discounted_price := lens_cost - lens_discount
noncomputable def total_cost := camera_new_cost + lens_discounted_price

theorem total_cost_correct :
  total_cost = 5400 := by
  sorry

end total_cost_correct_l61_61986


namespace money_made_per_minute_l61_61409

theorem money_made_per_minute (total_tshirts : ℕ) (time_minutes : ℕ) (black_tshirt_price white_tshirt_price : ℕ) (num_black num_white : ℕ) :
  total_tshirts = 200 →
  time_minutes = 25 →
  black_tshirt_price = 30 →
  white_tshirt_price = 25 →
  num_black = total_tshirts / 2 →
  num_white = total_tshirts / 2 →
  (num_black * black_tshirt_price + num_white * white_tshirt_price) / time_minutes = 220 :=
by
  sorry

end money_made_per_minute_l61_61409


namespace parabola_focus_distance_l61_61357

noncomputable def parabolic_distance (x y : ℝ) : ℝ :=
  x + x / 2

theorem parabola_focus_distance : 
  (∃ y : ℝ, (1 : ℝ) = (1 / 2) * y^2) → 
  parabolic_distance 1 y = 3 / 2 :=
by 
  intros hy
  obtain ⟨y, hy⟩ := hy
  unfold parabolic_distance
  have hx : 1 = (1 / 2) * y^2 := hy
  sorry

end parabola_focus_distance_l61_61357


namespace farmer_rows_of_tomatoes_l61_61042

def num_rows (total_tomatoes yield_per_plant plants_per_row : ℕ) : ℕ :=
  (total_tomatoes / yield_per_plant) / plants_per_row

theorem farmer_rows_of_tomatoes (total_tomatoes yield_per_plant plants_per_row : ℕ)
    (ht : total_tomatoes = 6000)
    (hy : yield_per_plant = 20)
    (hp : plants_per_row = 10) :
    num_rows total_tomatoes yield_per_plant plants_per_row = 30 := 
by
  sorry

end farmer_rows_of_tomatoes_l61_61042


namespace math_club_problem_l61_61950

theorem math_club_problem :
  ((7^2 - 5^2) - 2)^3 = 10648 := by
  sorry

end math_club_problem_l61_61950


namespace find_a_l61_61970

theorem find_a (x y a : ℤ) (h1 : x = 1) (h2 : y = 3) (h3 : a * x - 2 * y = 4) : a = 10 :=
by {
  sorry
}

end find_a_l61_61970


namespace least_three_digit_multiple_of_8_l61_61036

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end least_three_digit_multiple_of_8_l61_61036


namespace compute_g_five_times_l61_61146

def g (x : ℤ) : ℤ :=
  if x ≥ 0 then -x^3 else x + 6

theorem compute_g_five_times :
  g (g (g (g (g 1)))) = -113 :=
  by sorry

end compute_g_five_times_l61_61146


namespace twenty_second_entry_l61_61130

-- Definition of r_9 which is the remainder left when n is divided by 9
def r_9 (n : ℕ) : ℕ := n % 9

-- Statement to prove that the 22nd entry in the ordered list of all nonnegative integers
-- that satisfy r_9(5n) ≤ 4 is 38
theorem twenty_second_entry (n : ℕ) (hn : 5 * n % 9 ≤ 4) :
  ∃ m : ℕ, m = 22 ∧ n = 38 :=
sorry

end twenty_second_entry_l61_61130


namespace sufficient_condition_for_reciprocal_inequality_l61_61374

theorem sufficient_condition_for_reciprocal_inequality (a b : ℝ) (h : b < a ∧ a < 0) : (1 / a) < (1 / b) :=
sorry

end sufficient_condition_for_reciprocal_inequality_l61_61374


namespace sum_of_solutions_eq_zero_l61_61534

theorem sum_of_solutions_eq_zero :
  let p := 6
  let q := 150
  (∃ x1 x2 : ℝ, p * x1 = q / x1 ∧ p * x2 = q / x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 0) :=
sorry

end sum_of_solutions_eq_zero_l61_61534


namespace moli_bought_7_clips_l61_61355

theorem moli_bought_7_clips (R C S x : ℝ) 
  (h1 : 3*R + x*C + S = 120) 
  (h2 : 4*R + 10*C + S = 164) 
  (h3 : R + C + S = 32) : 
  x = 7 := 
by
  sorry

end moli_bought_7_clips_l61_61355


namespace product_of_square_roots_of_nine_l61_61453

theorem product_of_square_roots_of_nine (a b : ℝ) (ha : a^2 = 9) (hb : b^2 = 9) : a * b = -9 :=
sorry

end product_of_square_roots_of_nine_l61_61453


namespace probability_volleyball_is_one_third_l61_61291

-- Define the total number of test items
def total_test_items : ℕ := 3

-- Define the number of favorable outcomes for hitting the wall with a volleyball
def favorable_outcomes_volleyball : ℕ := 1

-- Define the probability calculation
def probability_hitting_wall_with_volleyball : ℚ :=
  favorable_outcomes_volleyball / total_test_items

-- Prove the probability is 1/3
theorem probability_volleyball_is_one_third :
  probability_hitting_wall_with_volleyball = 1 / 3 := 
sorry

end probability_volleyball_is_one_third_l61_61291


namespace nat_number_of_the_form_l61_61094

theorem nat_number_of_the_form (a b : ℕ) (h : ∃ (a b : ℕ), a * a * 3 + b * b * 32 = n) :
  ∃ (a' b' : ℕ), a' * a' * 3 + b' * b' * 32 = 97 * n  :=
  sorry

end nat_number_of_the_form_l61_61094


namespace train_speed_kmph_l61_61052

theorem train_speed_kmph (length : ℝ) (time : ℝ) (speed_conversion : ℝ) (speed_kmph : ℝ) :
  length = 100.008 → time = 4 → speed_conversion = 3.6 →
  speed_kmph = (length / time) * speed_conversion → speed_kmph = 90.0072 :=
by
  sorry

end train_speed_kmph_l61_61052


namespace max_min_z_l61_61336

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 + 4*y^2 = 4*x

-- Define the function z
def z (x y : ℝ) : ℝ :=
  x^2 - y^2

-- Define the required points
def P1 (x y : ℝ) :=
  x = 4 ∧ y = 0

def P2 (x y : ℝ) :=
  x = 2/5 ∧ (y = 3/5 ∨ y = -3/5)

-- Theorem stating the required conditions
theorem max_min_z (x y : ℝ) (h : on_ellipse x y) :
  (P1 x y → z x y = 16) ∧ (P2 x y → z x y = -1/5) :=
by
  sorry

end max_min_z_l61_61336


namespace water_consumption_l61_61858

theorem water_consumption (x y : ℝ)
  (h1 : 120 + 20 * x = 3200000 * y)
  (h2 : 120 + 15 * x = 3000000 * y) :
  x = 200 ∧ y = 50 :=
by
  sorry

end water_consumption_l61_61858


namespace find_c_l61_61585

theorem find_c (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 310) : C = 10 :=
by
  sorry

end find_c_l61_61585


namespace remainder_equality_l61_61026

variables (A B D : ℕ) (S S' s s' : ℕ)

theorem remainder_equality 
  (h1 : A > B) 
  (h2 : (A + 3) % D = S) 
  (h3 : (B - 2) % D = S') 
  (h4 : ((A + 3) * (B - 2)) % D = s) 
  (h5 : (S * S') % D = s') : 
  s = s' := 
sorry

end remainder_equality_l61_61026


namespace problem_discussion_organization_l61_61091

theorem problem_discussion_organization 
    (students : Fin 20 → Finset (Fin 20))
    (problems : Fin 20 → Finset (Fin 20))
    (h1 : ∀ s, (students s).card = 2)
    (h2 : ∀ p, (problems p).card = 2)
    (h3 : ∀ s p, s ∈ problems p ↔ p ∈ students s) : 
    ∃ (discussion : Fin 20 → Fin 20), 
        (∀ s, discussion s ∈ students s) ∧ 
        (Finset.univ.image discussion).card = 20 :=
by
  -- proof goes here
  sorry

end problem_discussion_organization_l61_61091


namespace trigonometric_identity_l61_61142

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin θ * Real.sin (π / 2 - θ)) / (Real.sin θ ^ 2 + Real.cos (2 * θ) + Real.cos θ ^ 2) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l61_61142


namespace washing_machine_capacity_l61_61983

-- Definitions of the conditions
def total_pounds_per_day : ℕ := 200
def number_of_machines : ℕ := 8

-- Main theorem to prove the question == answer given the conditions
theorem washing_machine_capacity :
  total_pounds_per_day / number_of_machines = 25 :=
by
  sorry

end washing_machine_capacity_l61_61983


namespace quadratic_solution_eq_l61_61921

theorem quadratic_solution_eq (c d : ℝ) 
  (h_eq : ∀ x : ℝ, x^2 - 6*x + 11 = 25 ↔ (x = c ∨ x = d))
  (h_order : c ≥ d) :
  c + 2*d = 9 - Real.sqrt 23 :=
sorry

end quadratic_solution_eq_l61_61921


namespace find_cost_price_l61_61628

-- Define the known data
def cost_price_80kg (C : ℝ) := 80 * C
def cost_price_20kg := 20 * 20
def selling_price_mixed := 2000
def total_cost_price_mixed (C : ℝ) := cost_price_80kg C + cost_price_20kg

-- Using the condition for 25% profit
def selling_price_of_mixed (C : ℝ) := 1.25 * total_cost_price_mixed C

-- The main theorem
theorem find_cost_price (C : ℝ) : selling_price_of_mixed C = selling_price_mixed → C = 15 :=
by
  sorry

end find_cost_price_l61_61628


namespace A_beats_B_by_160_meters_l61_61764

-- Definitions used in conditions
def distance_A := 400 -- meters
def time_A := 60 -- seconds
def distance_B := 400 -- meters
def time_B := 100 -- seconds
def speed_B := distance_B / time_B -- B's speed in meters/second
def time_for_B_in_A_time := time_A -- B's time for the duration A took to finish the race
def distance_B_in_A_time := speed_B * time_for_B_in_A_time -- Distance B covers in A's time

-- Statement to prove
theorem A_beats_B_by_160_meters : distance_A - distance_B_in_A_time = 160 :=
by
  -- This is a placeholder for an eventual proof
  sorry

end A_beats_B_by_160_meters_l61_61764


namespace booksReadPerDay_l61_61883

-- Mrs. Hilt read 14 books in a week.
def totalBooksReadInWeek : ℕ := 14

-- There are 7 days in a week.
def daysInWeek : ℕ := 7

-- We need to prove that the number of books read per day is 2.
theorem booksReadPerDay :
  totalBooksReadInWeek / daysInWeek = 2 :=
by
  sorry

end booksReadPerDay_l61_61883


namespace digit_d_for_5678d_is_multiple_of_9_l61_61258

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem digit_d_for_5678d_is_multiple_of_9 : 
  ∃ d : ℕ, d < 10 ∧ is_multiple_of_9 (56780 + d) ∧ d = 1 :=
by
  sorry

end digit_d_for_5678d_is_multiple_of_9_l61_61258


namespace people_came_in_first_hour_l61_61490
-- Import the entirety of the necessary library

-- Lean 4 statement for the given problem
theorem people_came_in_first_hour (X : ℕ) (net_change_first_hour : ℕ) (net_change_second_hour : ℕ) (people_after_2_hours : ℕ) : 
    (net_change_first_hour = X - 27) → 
    (net_change_second_hour = 18 - 9) →
    (people_after_2_hours = 76) → 
    (X - 27 + 9 = 76) → 
    X = 94 :=
by 
    intros h1 h2 h3 h4 
    sorry -- Proof is not required by instructions

end people_came_in_first_hour_l61_61490


namespace complex_pure_imaginary_l61_61571

theorem complex_pure_imaginary (a : ℝ) : 
  ((a^2 - 3*a + 2) = 0) → (a = 2) := 
  by 
  sorry

end complex_pure_imaginary_l61_61571


namespace concert_revenue_l61_61057

-- Defining the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := ticket_price_adult / 2
def attendees_adults : ℕ := 183
def attendees_children : ℕ := 28

-- Defining the total revenue calculation based on the conditions
def total_revenue : ℕ :=
  attendees_adults * ticket_price_adult +
  attendees_children * ticket_price_child

-- The theorem to prove the total revenue
theorem concert_revenue : total_revenue = 5122 := by
  sorry

end concert_revenue_l61_61057


namespace right_triangle_angle_l61_61595

theorem right_triangle_angle (x : ℝ) (h1 : x + 5 * x = 90) : 5 * x = 75 :=
by
  sorry

end right_triangle_angle_l61_61595


namespace third_side_triangle_max_l61_61751

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l61_61751


namespace product_a3_a10_a17_l61_61313

-- Let's define the problem setup
variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a r : α) (n : ℕ) : α := a * r ^ (n - 1)

theorem product_a3_a10_a17 
  (a r : α)
  (h1 : geometric_sequence a r 2 + geometric_sequence a r 18 = -15) 
  (h2 : geometric_sequence a r 2 * geometric_sequence a r 18 = 16) 
  (ha2pos : geometric_sequence a r 18 ≠ 0) 
  (h3 : r < 0) :
  geometric_sequence a r 3 * geometric_sequence a r 10 * geometric_sequence a r 17 = -64 :=
sorry

end product_a3_a10_a17_l61_61313


namespace cylinder_volume_triple_quadruple_l61_61714

theorem cylinder_volume_triple_quadruple (r h : ℝ) (V : ℝ) (π : ℝ) (original_volume : V = π * r^2 * h) 
                                         (original_volume_value : V = 8):
  ∃ V', V' = π * (3 * r)^2 * (4 * h) ∧ V' = 288 :=
by
  sorry

end cylinder_volume_triple_quadruple_l61_61714


namespace calculate_expression_l61_61979

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 :=
  by
  sorry

end calculate_expression_l61_61979


namespace polynomial_coeff_sum_eq_neg_two_l61_61643

/-- If (1 - 2 * x) ^ 9 = a₉ * x ^ 9 + a₈ * x ^ 8 + ... + a₂ * x ^ 2 + a₁ * x + a₀, 
then a₁ + a₂ + ... + a₈ + a₉ = -2. -/
theorem polynomial_coeff_sum_eq_neg_two 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℤ) 
  (h : (1 - 2 * x) ^ 9 = a₉ * x ^ 9 + a₈ * x ^ 8 + a₇ * x ^ 7 + a₆ * x ^ 6 + a₅ * x ^ 5 + a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀) : 
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by sorry

end polynomial_coeff_sum_eq_neg_two_l61_61643


namespace jerky_remaining_after_giving_half_l61_61943

-- Define the main conditions as variables
def days := 5
def initial_jerky := 40
def jerky_per_day := 1 + 1 + 2

-- Calculate total consumption
def total_consumption := jerky_per_day * days

-- Calculate remaining jerky
def remaining_jerky := initial_jerky - total_consumption

-- Calculate final jerky after giving half to her brother
def jerky_left := remaining_jerky / 2

-- Statement to be proved
theorem jerky_remaining_after_giving_half :
  jerky_left = 10 :=
by
  -- Proof will go here
  sorry

end jerky_remaining_after_giving_half_l61_61943


namespace inequality_proof_l61_61848

variable (x y z : ℝ)

theorem inequality_proof
  (h : x + 2*y + 3*z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 14 :=
by
  sorry

end inequality_proof_l61_61848


namespace cookie_calories_l61_61449

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

end cookie_calories_l61_61449


namespace functional_relationship_optimizing_profit_l61_61297

-- Define the scope of the problem with conditions and proof statements

variables (x : ℝ) (y : ℝ)

-- Conditions
def price_condition := 44 ≤ x ∧ x ≤ 52
def sales_function := y = -10 * x + 740
def profit_function (x : ℝ) := -10 * x^2 + 1140 * x - 29600

-- Lean statement to prove the first part
theorem functional_relationship (h₁ : 44 ≤ x) (h₂ : x ≤ 52) : y = -10 * x + 740 := by
  sorry

-- Lean statement to prove the second part
theorem optimizing_profit (h₃ : 44 ≤ x) (h₄ : x ≤ 52) : (profit_function 52 = 2640 ∧ (∀ x, (44 ≤ x ∧ x ≤ 52) → profit_function x ≤ 2640)) := by
  sorry

end functional_relationship_optimizing_profit_l61_61297


namespace wine_with_cork_cost_is_2_10_l61_61581

noncomputable def cork_cost : ℝ := 0.05
noncomputable def wine_without_cork_cost : ℝ := cork_cost + 2.00
noncomputable def wine_with_cork_cost : ℝ := wine_without_cork_cost + cork_cost

theorem wine_with_cork_cost_is_2_10 : wine_with_cork_cost = 2.10 :=
by
  -- skipped proof
  sorry

end wine_with_cork_cost_is_2_10_l61_61581


namespace solution_opposite_numbers_l61_61030

theorem solution_opposite_numbers (x y : ℤ) (h1 : 2 * x + 3 * y - 4 = 0) (h2 : x = -y) : x = -4 ∧ y = 4 :=
by
  sorry

end solution_opposite_numbers_l61_61030


namespace series_sum_equals_seven_ninths_l61_61437

noncomputable def infinite_series_sum : ℝ :=
  ∑' n, 1 / (n * (n + 3))

theorem series_sum_equals_seven_ninths : infinite_series_sum = (7 / 9) :=
by sorry

end series_sum_equals_seven_ninths_l61_61437


namespace compute_value_l61_61799

open Nat Real

theorem compute_value (A B : ℝ × ℝ) (hA : A = (15, 10)) (hB : B = (-5, 6)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∃ (x y : ℝ), C = (x, y) ∧ 2 * x - 4 * y = -22 := by
  sorry

end compute_value_l61_61799


namespace rotated_squares_overlap_area_l61_61181

noncomputable def total_overlap_area (side_length : ℝ) : ℝ :=
  let base_area := side_length ^ 2
  3 * base_area

theorem rotated_squares_overlap_area : total_overlap_area 8 = 192 := by
  sorry

end rotated_squares_overlap_area_l61_61181


namespace modulus_of_z_l61_61086

open Complex

theorem modulus_of_z 
  (z : ℂ) 
  (h : (1 - I) * z = 2 * I) : 
  abs z = Real.sqrt 2 := 
sorry

end modulus_of_z_l61_61086


namespace range_of_a_l61_61001

theorem range_of_a (a : ℝ) (h : ∀ x, a ≤ x ∧ x ≤ a + 2 → |x + a| ≥ 2 * |x|) : a ≤ -3 / 2 := 
by
  sorry

end range_of_a_l61_61001


namespace point_coordinates_l61_61541

theorem point_coordinates (m : ℝ) 
  (h1 : dist (0 : ℝ) (Real.sqrt m) = 4) : 
  (-m, Real.sqrt m) = (-16, 4) := 
by
  -- The proof will use the conditions and solve for m to find the coordinates
  sorry

end point_coordinates_l61_61541


namespace john_took_more_chickens_l61_61577

theorem john_took_more_chickens (john_chickens mary_chickens ray_chickens : ℕ)
  (h_ray : ray_chickens = 10)
  (h_mary : mary_chickens = ray_chickens + 6)
  (h_john : john_chickens = mary_chickens + 5) :
  john_chickens - ray_chickens = 11 := by
  sorry

end john_took_more_chickens_l61_61577


namespace intersection_complement_is_singleton_l61_61024

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 2, 5}

theorem intersection_complement_is_singleton : (U \ M) ∩ N = {1} := by
  sorry

end intersection_complement_is_singleton_l61_61024


namespace find_number_l61_61233

theorem find_number (N : ℕ) (k : ℕ) (Q : ℕ)
  (h1 : N = 9 * k)
  (h2 : Q = 25 * 9 + 7)
  (h3 : N / 9 = Q) :
  N = 2088 :=
by
  sorry

end find_number_l61_61233


namespace find_b_c_d_sum_l61_61516

theorem find_b_c_d_sum :
  ∃ (b c d : ℤ), (∀ n : ℕ, n > 0 → 
    a_n = b * (⌊(n : ℝ)^(1/3)⌋.natAbs : ℤ) + d ∧
    b = 2 ∧ c = 0 ∧ d = 0) ∧ (b + c + d = 2) :=
sorry

end find_b_c_d_sum_l61_61516


namespace paint_used_l61_61737

theorem paint_used (total_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) 
  (first_week_paint : ℚ) (remaining_paint : ℚ) (second_week_paint : ℚ) (total_used_paint : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1/6 →
  second_week_fraction = 1/5 →
  first_week_paint = first_week_fraction * total_paint →
  remaining_paint = total_paint - first_week_paint →
  second_week_paint = second_week_fraction * remaining_paint →
  total_used_paint = first_week_paint + second_week_paint →
  total_used_paint = 120 := sorry

end paint_used_l61_61737


namespace min_square_value_l61_61152

theorem min_square_value (a b : ℤ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ r : ℤ, r^2 = 15 * a + 16 * b)
  (h2 : ∃ s : ℤ, s^2 = 16 * a - 15 * b) : 
  231361 ≤ min (15 * a + 16 * b) (16 * a - 15 * b) :=
sorry

end min_square_value_l61_61152


namespace arithmetic_mean_twice_y_l61_61835

theorem arithmetic_mean_twice_y (y x : ℝ) (h1 : (8 + y + 24 + 6 + x) / 5 = 12) (h2 : x = 2 * y) :
  y = 22 / 3 ∧ x = 44 / 3 :=
by
  sorry

end arithmetic_mean_twice_y_l61_61835


namespace jar_filled_fraction_l61_61263

variable (S L : ℝ)

-- Conditions
axiom h1 : S * (1/3) = L * (1/2)

-- Statement of the problem
theorem jar_filled_fraction :
  (L * (1/2)) + (S * (1/3)) = L := by
sorry

end jar_filled_fraction_l61_61263


namespace ratio_pow_eq_l61_61425

variable (a b c d e f p q r : ℝ)
variable (n : ℕ)
variable (h : a / b = c / d)
variable (h1 : a / b = e / f)
variable (h2 : p ≠ 0 ∨ q ≠ 0 ∨ r ≠ 0)

theorem ratio_pow_eq
  (h : a / b = c / d)
  (h1 : a / b = e / f)
  (h2 : p ≠ 0 ∨ q ≠ 0 ∨ r ≠ 0)
  (n_ne_zero : n ≠ 0):
  (a / b) ^ n = (p * a ^ n + q * c ^ n + r * e ^ n) / (p * b ^ n + q * d ^ n + r * f ^ n) :=
by
  sorry

end ratio_pow_eq_l61_61425


namespace ratio_of_black_to_white_areas_l61_61270

theorem ratio_of_black_to_white_areas :
  let π := Real.pi
  let radii := [2, 4, 6, 8]
  let areas := [π * (radii[0])^2, π * (radii[1])^2, π * (radii[2])^2, π * (radii[3])^2]
  let black_areas := [areas[0], areas[2] - areas[1]]
  let white_areas := [areas[1] - areas[0], areas[3] - areas[2]]
  let total_black_area := black_areas.sum
  let total_white_area := white_areas.sum
  let ratio := total_black_area / total_white_area
  ratio = 3 / 5 := sorry

end ratio_of_black_to_white_areas_l61_61270


namespace fraction_halfway_between_l61_61881

theorem fraction_halfway_between : 
  ∃ (x : ℚ), (x = (1 / 6 + 1 / 4) / 2) ∧ x = 5 / 24 :=
by
  sorry

end fraction_halfway_between_l61_61881


namespace ivan_total_money_l61_61056

-- Define values of the coins
def penny_value : ℝ := 0.01
def dime_value : ℝ := 0.1
def nickel_value : ℝ := 0.05
def quarter_value : ℝ := 0.25

-- Define number of each type of coin in each piggy bank
def first_piggybank_pennies := 100
def first_piggybank_dimes := 50
def first_piggybank_nickels := 20
def first_piggybank_quarters := 10

def second_piggybank_pennies := 150
def second_piggybank_dimes := 30
def second_piggybank_nickels := 40
def second_piggybank_quarters := 15

def third_piggybank_pennies := 200
def third_piggybank_dimes := 60
def third_piggybank_nickels := 10
def third_piggybank_quarters := 20

-- Calculate the total value of each piggy bank
def first_piggybank_value : ℝ :=
  (first_piggybank_pennies * penny_value) +
  (first_piggybank_dimes * dime_value) +
  (first_piggybank_nickels * nickel_value) +
  (first_piggybank_quarters * quarter_value)

def second_piggybank_value : ℝ :=
  (second_piggybank_pennies * penny_value) +
  (second_piggybank_dimes * dime_value) +
  (second_piggybank_nickels * nickel_value) +
  (second_piggybank_quarters * quarter_value)

def third_piggybank_value : ℝ :=
  (third_piggybank_pennies * penny_value) +
  (third_piggybank_dimes * dime_value) +
  (third_piggybank_nickels * nickel_value) +
  (third_piggybank_quarters * quarter_value)

-- Calculate the total amount of money Ivan has
def total_value : ℝ :=
  first_piggybank_value + second_piggybank_value + third_piggybank_value

-- The theorem to prove
theorem ivan_total_money :
  total_value = 33.25 :=
by
  sorry

end ivan_total_money_l61_61056


namespace max_f_geq_l61_61996

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * x) + Real.sin (3 * x)

theorem max_f_geq (x : ℝ) : ∃ x, f x ≥ (3 + Real.sqrt 3) / 2 := sorry

end max_f_geq_l61_61996


namespace min_angle_B_l61_61905

-- Definitions using conditions from part a)
def triangle (A B C : ℝ) : Prop := A + B + C = Real.pi
def arithmetic_sequence_prop (A B C : ℝ) : Prop := 
  Real.tan A + Real.tan C = 2 * (1 + Real.sqrt 2) * Real.tan B

-- Main theorem to prove
theorem min_angle_B (A B C : ℝ) (h1 : triangle A B C) (h2 : arithmetic_sequence_prop A B C) :
  B ≥ Real.pi / 4 :=
sorry

end min_angle_B_l61_61905


namespace systematic_sampling_l61_61908

theorem systematic_sampling (N : ℕ) (k : ℕ) (interval : ℕ) (seq : List ℕ) : 
  N = 70 → k = 7 → interval = 10 → 
  seq = [3, 13, 23, 33, 43, 53, 63] := 
by 
  intros hN hk hInt;
  sorry

end systematic_sampling_l61_61908


namespace trapezoid_rectangle_ratio_l61_61442

noncomputable def area_ratio (a1 a2 r : ℝ) : ℝ := 
  if a2 = 0 then 0 else a1 / a2

theorem trapezoid_rectangle_ratio 
  (radius : ℝ) (AD BC : ℝ)
  (trapezoid_area rectangle_area : ℝ) :
  radius = 13 →
  AD = 10 →
  BC = 24 →
  area_ratio trapezoid_area rectangle_area = 1 / 2 ∨
  area_ratio trapezoid_area rectangle_area = 289 / 338 :=
  sorry

end trapezoid_rectangle_ratio_l61_61442


namespace min_guests_at_banquet_l61_61728

theorem min_guests_at_banquet (total_food : ℕ) (max_food_per_guest : ℕ) : 
  total_food = 323 ∧ max_food_per_guest = 2 → 
  (∀ guests : ℕ, guests * max_food_per_guest >= total_food) → 
  (∃ g : ℕ, g = 162) :=
by
  -- Assuming total food and max food per guest
  intro h_cons
  -- Mathematical proof steps would go here, skipping with sorry
  sorry

end min_guests_at_banquet_l61_61728


namespace max_y_for_f_eq_0_l61_61624

-- Define f(x, y, z) as the remainder when (x - y)! is divided by (x + z).
def f (x y z : ℕ) : ℕ :=
  Nat.factorial (x - y) % (x + z)

-- Conditions given in the problem
variable (x y z : ℕ)
variable (hx : x = 100)
variable (hz : z = 50)

theorem max_y_for_f_eq_0 : 
  f x y z = 0 → y ≤ 75 :=
by
  rw [hx, hz]
  sorry

end max_y_for_f_eq_0_l61_61624


namespace arithmetic_to_geometric_progression_l61_61972

theorem arithmetic_to_geometric_progression (x y z : ℝ) 
  (hAP : 2 * y^2 - y * x = z^2) : 
  z^2 = y * (2 * y - x) := 
  by 
  sorry

end arithmetic_to_geometric_progression_l61_61972


namespace original_fish_count_l61_61503

def initial_fish_count (fish_taken_out : ℕ) (current_fish : ℕ) : ℕ :=
  fish_taken_out + current_fish

theorem original_fish_count :
  initial_fish_count 16 3 = 19 :=
by
  sorry

end original_fish_count_l61_61503


namespace simplify_fraction_l61_61929

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h_cond : y^3 - 1/x ≠ 0) :
  (x^3 - 1/y) / (y^3 - 1/x) = x / y :=
by
  sorry

end simplify_fraction_l61_61929


namespace dan_time_second_hour_tshirts_l61_61837

-- Definition of conditions
def t_shirts_in_first_hour (rate1 : ℕ) (time : ℕ) : ℕ := time / rate1
def total_t_shirts (hour1_ts hour2_ts : ℕ) : ℕ := hour1_ts + hour2_ts
def time_per_t_shirt_in_second_hour (time : ℕ) (hour2_ts : ℕ) : ℕ := time / hour2_ts

-- Main theorem statement (without proof)
theorem dan_time_second_hour_tshirts
  (rate1 : ℕ) (hour1_time : ℕ) (total_ts : ℕ) (hour_time : ℕ)
  (hour1_ts := t_shirts_in_first_hour rate1 hour1_time)
  (hour2_ts := total_ts - hour1_ts) :
  rate1 = 12 → 
  hour1_time = 60 → 
  total_ts = 15 → 
  hour_time = 60 →
  time_per_t_shirt_in_second_hour hour_time hour2_ts = 6 :=
by
  intros rate1_eq hour1_time_eq total_ts_eq hour_time_eq
  sorry

end dan_time_second_hour_tshirts_l61_61837


namespace speed_in_still_water_l61_61662

theorem speed_in_still_water (upstream downstream : ℝ) 
  (h_up : upstream = 25) 
  (h_down : downstream = 45) : 
  (upstream + downstream) / 2 = 35 := 
by 
  -- Proof will go here
  sorry

end speed_in_still_water_l61_61662


namespace betty_total_cost_l61_61360

theorem betty_total_cost :
    (6 * 2.5) + (4 * 1.25) + (8 * 3) = 44 :=
by
    sorry

end betty_total_cost_l61_61360


namespace work_done_in_one_day_l61_61066

theorem work_done_in_one_day (A_days B_days : ℝ) (hA : A_days = 6) (hB : B_days = A_days / 2) : 
  (1 / A_days + 1 / B_days) = 1 / 2 := by
  sorry

end work_done_in_one_day_l61_61066


namespace intersection_eq_l61_61064

-- Define sets A and B
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {0, 1, 2}

-- The theorem to be proved
theorem intersection_eq : A ∩ B = {2} := 
by sorry

end intersection_eq_l61_61064


namespace g_50_zero_l61_61466

noncomputable def g : ℕ → ℝ → ℝ
| 0, x     => x + |x - 50| - |x + 50|
| (n+1), x => |g n x| - 2

theorem g_50_zero :
  ∃! x : ℝ, g 50 x = 0 :=
sorry

end g_50_zero_l61_61466


namespace coronavirus_transmission_l61_61981

theorem coronavirus_transmission:
  (∃ x: ℝ, (1 + x)^2 = 225) :=
by
  sorry

end coronavirus_transmission_l61_61981


namespace word_sum_problems_l61_61527

theorem word_sum_problems (J M O I : Fin 10) (h_distinct : J ≠ M ∧ J ≠ O ∧ J ≠ I ∧ M ≠ O ∧ M ≠ I ∧ O ≠ I) 
  (h_nonzero_J : J ≠ 0) (h_nonzero_I : I ≠ 0) :
  let JMO := 100 * J + 10 * M + O
  let IMO := 100 * I + 10 * M + O
  (JMO + JMO + JMO = IMO) → 
  (JMO = 150 ∧ IMO = 450) ∨ (JMO = 250 ∧ IMO = 750) :=
sorry

end word_sum_problems_l61_61527


namespace find_percentage_l61_61872

theorem find_percentage (P : ℝ) (h1 : (P / 100) * 200 = 30 + 0.60 * 50) : P = 30 :=
by
  sorry

end find_percentage_l61_61872


namespace percent_problem_l61_61992

theorem percent_problem (x : ℝ) (h : 0.30 * 0.40 * x = 36) : 0.40 * 0.30 * x = 36 :=
by
  sorry

end percent_problem_l61_61992


namespace largest_t_value_l61_61046

theorem largest_t_value : 
  ∃ t : ℝ, 
    (∃ s : ℝ, s > 0 ∧ t = 3 ∧
    ∀ u : ℝ, 
      (u = 3 →
        (15 * u^2 - 40 * u + 18) / (4 * u - 3) + 3 * u = 4 * u + 2 ∧
        u ≤ 3) ∧
      (u ≠ 3 → 
        (15 * u^2 - 40 * u + 18) / (4 * u - 3) + 3 * u = 4 * u + 2 → 
        u ≤ 3)) :=
sorry

end largest_t_value_l61_61046


namespace isosceles_triangle_three_times_ce_l61_61456

/-!
# Problem statement
In the isosceles triangle \( ABC \) with \( \overline{AC} = \overline{BC} \), 
\( D \) is the foot of the altitude through \( C \) and \( M \) is 
the midpoint of segment \( CD \). The line \( BM \) intersects \( AC \) 
at \( E \). Prove that \( AC \) is three times as long as \( CE \).
-/

-- Definition of isosceles triangle and related points
variables {A B C D E M : Type} 

-- Assume necessary conditions
variables (triangle_isosceles : A = B)
variables (D_foot : true) -- Placeholder, replace with proper definition if needed
variables (M_midpoint : true) -- Placeholder, replace with proper definition if needed
variables (BM_intersects_AC : true) -- Placeholder, replace with proper definition if needed

-- Main statement to prove
theorem isosceles_triangle_three_times_ce (h1 : A = B)
    (h2 : true) (h3 : true) (h4 : true) : 
    AC = 3 * CE :=
by
  sorry

end isosceles_triangle_three_times_ce_l61_61456


namespace find_value_of_fraction_l61_61971

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : (x / y) + (y / x) = 8)

theorem find_value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l61_61971


namespace even_numbers_average_l61_61736

theorem even_numbers_average (n : ℕ) (h : (n / 2 * (2 + 2 * n)) / n = 16) : n = 15 :=
by
  have hn : n ≠ 0 := sorry -- n > 0 because the first some even numbers were mentioned
  have hn_pos : 0 < n / 2 * (2 + 2 * n) := sorry -- n / 2 * (2 + 2n) > 0
  sorry

end even_numbers_average_l61_61736


namespace days_to_clear_land_l61_61324

-- Definitions of all the conditions
def length_of_land := 200
def width_of_land := 900
def area_cleared_by_one_rabbit_per_day_square_yards := 10
def number_of_rabbits := 100
def conversion_square_yards_to_square_feet := 9
def total_area_of_land := length_of_land * width_of_land
def area_cleared_by_one_rabbit_per_day_square_feet := area_cleared_by_one_rabbit_per_day_square_yards * conversion_square_yards_to_square_feet
def area_cleared_by_all_rabbits_per_day := number_of_rabbits * area_cleared_by_one_rabbit_per_day_square_feet

-- Theorem to prove the number of days required to clear the land
theorem days_to_clear_land :
  total_area_of_land / area_cleared_by_all_rabbits_per_day = 20 := by
  sorry

end days_to_clear_land_l61_61324


namespace total_cost_correct_l61_61347

noncomputable def cost_4_canvases : ℕ := 40
noncomputable def cost_paints : ℕ := cost_4_canvases / 2
noncomputable def cost_easel : ℕ := 15
noncomputable def cost_paintbrushes : ℕ := 15
noncomputable def total_cost : ℕ := cost_4_canvases + cost_paints + cost_easel + cost_paintbrushes

theorem total_cost_correct : total_cost = 90 :=
by
  unfold total_cost
  unfold cost_4_canvases
  unfold cost_paints
  unfold cost_easel
  unfold cost_paintbrushes
  simp
  sorry

end total_cost_correct_l61_61347


namespace min_value_of_expression_l61_61294

noncomputable def min_expression_value (a b c d : ℝ) : ℝ :=
  (a ^ 8) / ((a ^ 2 + b) * (a ^ 2 + c) * (a ^ 2 + d)) +
  (b ^ 8) / ((b ^ 2 + c) * (b ^ 2 + d) * (b ^ 2 + a)) +
  (c ^ 8) / ((c ^ 2 + d) * (c ^ 2 + a) * (c ^ 2 + b)) +
  (d ^ 8) / ((d ^ 2 + a) * (d ^ 2 + b) * (d ^ 2 + c))

theorem min_value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_expression_value a b c d = 1 / 2 :=
by
  -- Proof is omitted.
  sorry

end min_value_of_expression_l61_61294


namespace reggie_loses_by_21_points_l61_61493

-- Define the points for each type of shot.
def layup_points := 1
def free_throw_points := 2
def three_pointer_points := 3
def half_court_points := 5

-- Define Reggie's shot counts.
def reggie_layups := 4
def reggie_free_throws := 3
def reggie_three_pointers := 2
def reggie_half_court_shots := 1

-- Define Reggie's brother's shot counts.
def brother_layups := 3
def brother_free_throws := 2
def brother_three_pointers := 5
def brother_half_court_shots := 4

-- Calculate Reggie's total points.
def reggie_total_points :=
  reggie_layups * layup_points +
  reggie_free_throws * free_throw_points +
  reggie_three_pointers * three_pointer_points +
  reggie_half_court_shots * half_court_points

-- Calculate Reggie's brother's total points.
def brother_total_points :=
  brother_layups * layup_points +
  brother_free_throws * free_throw_points +
  brother_three_pointers * three_pointer_points +
  brother_half_court_shots * half_court_points

-- Calculate the difference in points.
def point_difference := brother_total_points - reggie_total_points

-- Prove that the difference in points Reggie lost by is 21.
theorem reggie_loses_by_21_points : point_difference = 21 := by
  sorry

end reggie_loses_by_21_points_l61_61493


namespace hallie_net_earnings_correct_l61_61319

noncomputable def hallieNetEarnings : ℚ :=
  let monday_hours := 7
  let monday_rate := 10
  let monday_tips := 18
  let tuesday_hours := 5
  let tuesday_rate := 12
  let tuesday_tips := 12
  let wednesday_hours := 7
  let wednesday_rate := 10
  let wednesday_tips := 20
  let thursday_hours := 8
  let thursday_rate := 11
  let thursday_tips := 25
  let thursday_discount := 0.10
  let friday_hours := 6
  let friday_rate := 9
  let friday_tips := 15
  let income_tax := 0.05

  let monday_earnings := monday_hours * monday_rate
  let tuesday_earnings := tuesday_hours * tuesday_rate
  let wednesday_earnings := wednesday_hours * wednesday_rate
  let thursday_earnings := thursday_hours * thursday_rate
  let thursday_earnings_after_discount := thursday_earnings * (1 - thursday_discount)
  let friday_earnings := friday_hours * friday_rate

  let total_hourly_earnings := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings + friday_earnings
  let total_tips := monday_tips + tuesday_tips + wednesday_tips + thursday_tips + friday_tips

  let total_tax := total_hourly_earnings * income_tax
  
  let net_earnings := (total_hourly_earnings - total_tax) - (thursday_earnings - thursday_earnings_after_discount) + total_tips
  net_earnings

theorem hallie_net_earnings_correct : hallieNetEarnings = 406.10 := by
  sorry

end hallie_net_earnings_correct_l61_61319


namespace garden_ratio_length_to_width_l61_61198

theorem garden_ratio_length_to_width (width length : ℕ) (area : ℕ) 
  (h1 : area = 507) 
  (h2 : width = 13) 
  (h3 : length * width = area) :
  length / width = 3 :=
by
  -- Proof to be filled in.
  sorry

end garden_ratio_length_to_width_l61_61198


namespace fly_travel_time_to_opposite_vertex_l61_61143

noncomputable def cube_side_length (a : ℝ) := 
  a

noncomputable def fly_travel_time_base := 4 -- minutes

noncomputable def fly_speed (a : ℝ) := 
  4 * a / fly_travel_time_base

noncomputable def space_diagonal_length (a : ℝ) := 
  a * Real.sqrt 3

theorem fly_travel_time_to_opposite_vertex (a : ℝ) : 
  fly_speed a ≠ 0 -> 
  space_diagonal_length a / fly_speed a = Real.sqrt 3 :=
by
  intro h
  sorry

end fly_travel_time_to_opposite_vertex_l61_61143


namespace no_real_solutions_l61_61661

theorem no_real_solutions (x : ℝ) (h_nonzero : x ≠ 0) (h_pos : 0 < x):
  (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 12 * x^9 :=
by
-- Proof will go here.
sorry

end no_real_solutions_l61_61661


namespace giselle_paint_l61_61869

theorem giselle_paint (x : ℚ) (h1 : 5/7 = x/21) : x = 15 :=
by
  sorry

end giselle_paint_l61_61869


namespace solve_for_nabla_l61_61563

theorem solve_for_nabla (nabla mu : ℤ) (h1 : 5 * (-3) = nabla + mu - 3) (h2 : mu = 4) : 
  nabla = -16 := 
by
  sorry

end solve_for_nabla_l61_61563


namespace compute_fraction_square_l61_61012

theorem compute_fraction_square : 6 * (3 / 7) ^ 2 = 54 / 49 :=
by 
  sorry

end compute_fraction_square_l61_61012


namespace max_value_of_largest_integer_l61_61647

theorem max_value_of_largest_integer (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h1 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 560) (h2 : a7 - a1 = 20) : a7 ≤ 21 :=
sorry

end max_value_of_largest_integer_l61_61647


namespace no_integers_satisfy_eq_l61_61588

theorem no_integers_satisfy_eq (m n : ℤ) : m^2 ≠ n^5 - 4 := 
by {
  sorry
}

end no_integers_satisfy_eq_l61_61588


namespace xiao_li_profit_l61_61193

noncomputable def original_price_per_share : ℝ := 21 / 1.05
noncomputable def closing_price_first_day : ℝ := original_price_per_share * 0.94
noncomputable def selling_price_second_day : ℝ := closing_price_first_day * 1.10
noncomputable def total_profit : ℝ := (selling_price_second_day - 21) * 5000

theorem xiao_li_profit :
  total_profit = 600 := sorry

end xiao_li_profit_l61_61193


namespace probability_five_common_correct_l61_61698

-- Define the conditions
def compulsory_subjects : ℕ := 3  -- Chinese, Mathematics, and English
def elective_from_physics_history : ℕ := 1  -- Physics and History
def elective_from_four : ℕ := 4  -- Politics, Geography, Chemistry, Biology

def chosen_subjects_by_xiaoming_xiaofang : ℕ := 2  -- two subjects from the four electives

-- Calculate total combinations
noncomputable def total_combinations : ℕ := Nat.choose 4 2 * Nat.choose 4 2

-- Calculate combinations to have exactly five subjects in common
noncomputable def combinations_five_common : ℕ := Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 2 1

-- Calculate the probability
noncomputable def probability_five_common : ℚ := combinations_five_common / total_combinations

-- The theorem to be proved
theorem probability_five_common_correct : probability_five_common = 2 / 3 := by
  sorry

end probability_five_common_correct_l61_61698


namespace difference_of_two_numbers_l61_61781

theorem difference_of_two_numbers (a b : ℕ) (h₀ : a + b = 25800) (h₁ : b = 12 * a) (h₂ : b % 10 = 0) (h₃ : b / 10 = a) : b - a = 21824 :=
by 
  -- sorry to skip the proof
  sorry

end difference_of_two_numbers_l61_61781


namespace triangle_DEF_area_l61_61251

noncomputable def point := (ℝ × ℝ)

def D : point := (-2, 2)
def E : point := (8, 2)
def F : point := (6, -4)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_DEF_area : area_of_triangle D E F = 30 := by
  sorry

end triangle_DEF_area_l61_61251


namespace money_together_l61_61689

variable (Billy Sam : ℕ)

theorem money_together (h1 : Billy = 2 * Sam - 25) (h2 : Sam = 75) : Billy + Sam = 200 := by
  sorry

end money_together_l61_61689


namespace find_x_condition_l61_61891

theorem find_x_condition (x : ℝ) (h : 0.75 / x = 5 / 11) : x = 1.65 := 
by
  sorry

end find_x_condition_l61_61891


namespace crayons_given_proof_l61_61037

def initial_crayons : ℕ := 110
def total_lost_crayons : ℕ := 412
def more_lost_than_given : ℕ := 322

def G : ℕ := 45 -- This is the given correct answer to prove.

theorem crayons_given_proof :
  ∃ G : ℕ, (G + (G + more_lost_than_given)) = total_lost_crayons ∧ G = 45 :=
by
  sorry

end crayons_given_proof_l61_61037


namespace muffin_sum_l61_61738

theorem muffin_sum (N : ℕ) : 
  (N % 13 = 3) → 
  (N % 8 = 5) → 
  (N < 120) → 
  (N = 16 ∨ N = 81 ∨ N = 107) → 
  (16 + 81 + 107 = 204) := 
by sorry

end muffin_sum_l61_61738


namespace percentage_increase_of_kim_l61_61636

variables (S P K : ℝ)
variables (h1 : S = 0.80 * P) (h2 : S + P = 1.80) (h3 : K = 1.12)

theorem percentage_increase_of_kim (hK : K = 1.12) (hS : S = 0.80 * P) (hSP : S + P = 1.80) :
  ((K - S) / S * 100) = 40 :=
sorry

end percentage_increase_of_kim_l61_61636


namespace probability_open_lock_l61_61298

/-- Given 5 keys and only 2 can open the lock, the probability of opening the lock by selecting one key randomly is 0.4. -/
theorem probability_open_lock (k : Finset ℕ) (h₁ : k.card = 5) (s : Finset ℕ) (h₂ : s.card = 2 ∧ s ⊆ k) :
  ∃ p : ℚ, p = 0.4 :=
by
  sorry

end probability_open_lock_l61_61298


namespace constant_term_correct_l61_61218

variable (x : ℝ)

noncomputable def constant_term_expansion : ℝ :=
  let term := λ (r : ℕ) => (Nat.choose 9 r) * (-2)^r * x^((9 - 9 * r) / 2)
  term 1

theorem constant_term_correct : 
  constant_term_expansion x = -18 :=
sorry

end constant_term_correct_l61_61218


namespace inequality_proof_l61_61602

variable {a b : ℕ → ℝ}

-- Conditions: {a_n} is a geometric sequence with positive terms, {b_n} is an arithmetic sequence, a_6 = b_8
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

axiom a_pos_terms : ∀ n : ℕ, a n > 0
axiom a_geom_seq : is_geometric a
axiom b_arith_seq : is_arithmetic b
axiom a6_eq_b8 : a 6 = b 8

-- Prove: a_3 + a_9 ≥ b_9 + b_7
theorem inequality_proof : a 3 + a 9 ≥ b 9 + b 7 :=
by sorry

end inequality_proof_l61_61602


namespace field_area_l61_61429

-- Define a rectangular field
structure RectangularField where
  length : ℕ
  width : ℕ
  fencing : ℕ := 2 * width + length
  
-- Given conditions
def field_conditions (L W F : ℕ) : Prop :=
  L = 30 ∧ 2 * W + L = F

-- Theorem stating the required proof
theorem field_area : ∀ (L W F : ℕ), field_conditions L W F → F = 84 → (L * W) = 810 :=
by
  intros L W F h1 h2
  sorry

end field_area_l61_61429


namespace original_price_is_1611_11_l61_61322

theorem original_price_is_1611_11 (profit: ℝ) (rate: ℝ) (original_price: ℝ) (selling_price: ℝ) 
(h1: profit = 725) (h2: rate = 0.45) (h3: profit = rate * original_price) : 
original_price = 725 / 0.45 := 
sorry

end original_price_is_1611_11_l61_61322


namespace find_set_of_points_B_l61_61789

noncomputable def is_incenter (A B C I : Point) : Prop :=
  -- define the incenter condition
  sorry

noncomputable def angle_less_than (A B C : Point) (α : ℝ) : Prop :=
  -- define the condition that all angles of triangle ABC are less than α
  sorry

theorem find_set_of_points_B (A I : Point) (α : ℝ) (hα1 : 60 < α) (hα2 : α < 90) :
  ∃ B : Point, ∃ C : Point,
    is_incenter A B C I ∧ angle_less_than A B C α :=
by
  -- The proof will go here
  sorry

end find_set_of_points_B_l61_61789


namespace maximum_value_of_f_l61_61786

noncomputable def f (a x : ℝ) : ℝ := (1 + x) ^ a - a * x

theorem maximum_value_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  ∃ x : ℝ, x > -1 ∧ ∀ y : ℝ, y > -1 → f a y ≤ f a x ∧ f a x = 1 :=
by {
  sorry
}

end maximum_value_of_f_l61_61786


namespace gcd_282_470_l61_61115

theorem gcd_282_470 : Int.gcd 282 470 = 94 := by
  sorry

end gcd_282_470_l61_61115


namespace stuart_initially_had_20_l61_61865

variable (B T S : ℕ) -- Initial number of marbles for Betty, Tom, and Susan
variable (S_after : ℕ) -- Number of marbles Stuart has after receiving from Betty

-- Given conditions
axiom betty_initially : B = 150
axiom tom_initially : T = 30
axiom susan_initially : S = 20

axiom betty_to_tom : (0.20 : ℚ) * B = 30
axiom betty_to_susan : (0.10 : ℚ) * B = 15
axiom betty_to_stuart : (0.40 : ℚ) * B = 60
axiom stuart_after_receiving : S_after = 80

-- Theorem to prove Stuart initially had 20 marbles
theorem stuart_initially_had_20 : ∃ S_initial : ℕ, S_after - 60 = S_initial ∧ S_initial = 20 :=
by {
  sorry
}

end stuart_initially_had_20_l61_61865


namespace train_speed_is_36_kph_l61_61615

-- Define the given conditions
def distance_meters : ℕ := 1800
def time_minutes : ℕ := 3

-- Convert distance from meters to kilometers
def distance_kilometers : ℕ -> ℕ := fun d => d / 1000
-- Convert time from minutes to hours
def time_hours : ℕ -> ℚ := fun t => (t : ℚ) / 60

-- Calculate speed in kilometers per hour
def speed_kph (d : ℕ) (t : ℕ) : ℚ :=
  let d_km := d / 1000
  let t_hr := (t : ℚ) / 60
  d_km / t_hr

-- The theorem to prove the speed
theorem train_speed_is_36_kph :
  speed_kph distance_meters time_minutes = 36 := by
  sorry

end train_speed_is_36_kph_l61_61615


namespace graph_description_l61_61740

theorem graph_description : ∀ x y : ℝ, (x + y)^2 = 2 * (x^2 + y^2) → x = 0 ∧ y = 0 :=
by 
  sorry

end graph_description_l61_61740


namespace xiao_wang_ways_to_make_8_cents_l61_61127

theorem xiao_wang_ways_to_make_8_cents :
  let one_cent_coins := 8
  let two_cent_coins := 4
  let five_cent_coin := 1
  ∃ ways, ways = 7 ∧ (
       (ways = 8 ∧ one_cent_coins >= 8) ∨
       (ways = 4 ∧ two_cent_coins >= 4) ∨
       (ways = 2 ∧ one_cent_coins >= 2 ∧ two_cent_coins >= 3) ∨
       (ways = 4 ∧ one_cent_coins >= 4 ∧ two_cent_coins >= 2) ∨
       (ways = 6 ∧ one_cent_coins >= 6 ∧ two_cent_coins >= 1) ∨
       (ways = 3 ∧ one_cent_coins >= 3 ∧ five_cent_coin >= 1) ∨
       (ways = 1 ∧ one_cent_coins >= 1 ∧ two_cent_coins >= 1 ∧ five_cent_coin >= 1)
   ) :=
  sorry

end xiao_wang_ways_to_make_8_cents_l61_61127


namespace num_ways_distribute_balls_l61_61044

-- Definition of the combinatorial problem
def indistinguishableBallsIntoBoxes : ℕ := 11

-- Main theorem statement
theorem num_ways_distribute_balls : indistinguishableBallsIntoBoxes = 11 := by
  sorry

end num_ways_distribute_balls_l61_61044


namespace determine_g_2023_l61_61230

noncomputable def g (x : ℕ) : ℝ := sorry

axiom g_pos (x : ℕ) (hx : x > 0) : g x > 0

axiom g_property (x y : ℕ) (h1 : x > 2 * y) (h2 : 0 < y) : 
  g (x - y) = Real.sqrt (g (x / y) + 3)

theorem determine_g_2023 : g 2023 = (1 + Real.sqrt 13) / 2 :=
by
  sorry

end determine_g_2023_l61_61230


namespace average_score_l61_61051

variable (K M : ℕ) (E : ℕ)

theorem average_score (h1 : (K + M) / 2 = 86) (h2 : E = 98) :
  (K + M + E) / 3 = 90 :=
by
  sorry

end average_score_l61_61051


namespace calculate_expression_l61_61523

-- Defining the main theorem to prove
theorem calculate_expression (a b : ℝ) : 
  3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by 
  sorry

end calculate_expression_l61_61523


namespace total_cost_of_books_l61_61646

-- Conditions from the problem
def C1 : ℝ := 350
def loss_percent : ℝ := 0.15
def gain_percent : ℝ := 0.19
def SP1 : ℝ := C1 - (loss_percent * C1) -- Selling price of the book sold at a loss
def SP2 : ℝ := SP1 -- Selling price of the book sold at a gain

-- Statement to prove the total cost
theorem total_cost_of_books : C1 + (SP2 / (1 + gain_percent)) = 600 := by
  sorry

end total_cost_of_books_l61_61646


namespace find_m_l61_61108

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 2}
noncomputable def N (m : ℝ) : Set ℝ := {x | x*x - m*x < 0}
noncomputable def M_inter_N (m : ℝ) : Set ℝ := {x | 0 < x ∧ x < 1}

theorem find_m (m : ℝ) (h : M ∩ (N m) = M_inter_N m) : m = 1 :=
by sorry

end find_m_l61_61108


namespace find_m_l61_61727

theorem find_m (m : ℤ) (h1 : -180 ≤ m ∧ m ≤ 180) (h2 : Real.sin (m * Real.pi / 180) = Real.cos (810 * Real.pi / 180)) :
  m = 0 ∨ m = 180 :=
sorry

end find_m_l61_61727


namespace avg_seven_consecutive_integers_l61_61484

variable (c d : ℕ)
variable (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7)

theorem avg_seven_consecutive_integers (c d : ℕ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 6 :=
sorry

end avg_seven_consecutive_integers_l61_61484


namespace no_three_distinct_positive_perfect_squares_sum_to_100_l61_61428

theorem no_three_distinct_positive_perfect_squares_sum_to_100 :
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ (m n p : ℕ), a = m^2 ∧ b = n^2 ∧ c = p^2) ∧ a + b + c = 100 :=
by
  sorry

end no_three_distinct_positive_perfect_squares_sum_to_100_l61_61428


namespace equal_roots_of_quadratic_l61_61922

theorem equal_roots_of_quadratic (k : ℝ) : 
  (∃ x, (x^2 + 2 * x + k = 0) ∧ (x^2 + 2 * x + k) = 0) → k = 1 :=
by
  sorry

end equal_roots_of_quadratic_l61_61922


namespace perpendicular_lines_sum_l61_61330

theorem perpendicular_lines_sum (a b : ℝ) :
  (∃ (x y : ℝ), 2 * x - 5 * y + b = 0 ∧ a * x + 4 * y - 2 = 0 ∧ x = 1 ∧ y = -2) ∧
  (-a / 4) * (2 / 5) = -1 →
  a + b = -2 :=
by
  sorry

end perpendicular_lines_sum_l61_61330


namespace bear_cubs_count_l61_61906

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

end bear_cubs_count_l61_61906


namespace circle_center_radius_l61_61947

theorem circle_center_radius (x y : ℝ) :
  (x ^ 2 + y ^ 2 + 2 * x - 4 * y - 6 = 0) →
  ((x + 1) ^ 2 + (y - 2) ^ 2 = 11) :=
by sorry

end circle_center_radius_l61_61947


namespace range_of_x_l61_61633

theorem range_of_x (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_increasing : ∀ {a b : ℝ}, a ≤ b → b ≤ 0 → f a ≤ f b) :
  (∀ x : ℝ, f (2^(2*x^2 - x - 1)) ≥ f (-4)) → ∀ x, x ∈ Set.Icc (-1 : ℝ) (3/2 : ℝ) :=
by 
  sorry

end range_of_x_l61_61633


namespace find_x_l61_61909

theorem find_x (x : ℝ) : 0.6 * x = (x / 3) + 110 → x = 412.5 := 
by
  intro h
  sorry

end find_x_l61_61909


namespace james_spent_6_dollars_l61_61860

-- Define the cost of items
def cost_milk : ℕ := 3
def cost_bananas : ℕ := 2

-- Define the sales tax rate as a decimal
def sales_tax_rate : ℚ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℕ := cost_milk + cost_bananas

-- Define the sales tax amount
def sales_tax_amount : ℚ := sales_tax_rate * total_cost_before_tax

-- Define the total amount spent
def total_amount_spent : ℚ := total_cost_before_tax + sales_tax_amount

-- The proof statement
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l61_61860


namespace solve_exponential_equation_l61_61815

theorem solve_exponential_equation (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3^x + 4^y = 5^z ↔ x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end solve_exponential_equation_l61_61815


namespace bicyclist_speed_remainder_l61_61925

theorem bicyclist_speed_remainder (total_distance first_distance remainder_distance first_speed avg_speed remainder_speed time_total time_first time_remainder : ℝ) 
  (H1 : total_distance = 350)
  (H2 : first_distance = 200)
  (H3 : remainder_distance = total_distance - first_distance)
  (H4 : first_speed = 20)
  (H5 : avg_speed = 17.5)
  (H6 : time_total = total_distance / avg_speed)
  (H7 : time_first = first_distance / first_speed)
  (H8 : time_remainder = time_total - time_first)
  (H9 : remainder_speed = remainder_distance / time_remainder) :
  remainder_speed = 15 := 
sorry

end bicyclist_speed_remainder_l61_61925


namespace xyz_inequality_l61_61573

theorem xyz_inequality (x y z : ℝ) (h : x + y + z = 0) : 
  6 * (x^3 + y^3 + z^3)^2 ≤ (x^2 + y^2 + z^2)^3 := 
by sorry

end xyz_inequality_l61_61573


namespace tina_mother_age_l61_61609

variable {x : ℕ}

theorem tina_mother_age (h1 : 10 + x = 2 * x - 20) : 2010 + x = 2040 :=
by 
  sorry

end tina_mother_age_l61_61609


namespace exists_n_not_perfect_square_l61_61469

theorem exists_n_not_perfect_square (a b : ℤ) (h1 : a > 1) (h2 : b > 1) (h3 : a ≠ b) : 
  ∃ (n : ℕ), (n > 0) ∧ ¬∃ (k : ℤ), (a^n - 1) * (b^n - 1) = k^2 :=
by sorry

end exists_n_not_perfect_square_l61_61469


namespace determine_a_and_theta_l61_61669

noncomputable def f (a θ : ℝ) (x : ℝ) : ℝ := 2 * a * Real.sin (2 * x + θ)

theorem determine_a_and_theta :
  (∃ a θ : ℝ, 0 < θ ∧ θ < π ∧ a ≠ 0 ∧ (∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f a θ x ∈ Set.Icc (-2 : ℝ) 2) ∧ 
  (∀ (x1 x2 : ℝ), x1 ∈ Set.Icc (-5 * π / 12) (π / 12) → x2 ∈ Set.Icc (-5 * π / 12) (π / 12) → x1 < x2 → f a θ x1 > f a θ x2)) →
  (a = -1) ∧ (θ = π / 3) :=
sorry

end determine_a_and_theta_l61_61669


namespace brian_tape_needed_l61_61041

-- Definitions of conditions
def tape_needed_for_box (short_side: ℕ) (long_side: ℕ) : ℕ := 
  2 * short_side + long_side

def total_tape_needed (num_short_long_boxes: ℕ) (short_side: ℕ) (long_side: ℕ) (num_square_boxes: ℕ) (side: ℕ) : ℕ := 
  (num_short_long_boxes * tape_needed_for_box short_side long_side) + (num_square_boxes * 3 * side)

-- Theorem statement
theorem brian_tape_needed : total_tape_needed 5 15 30 2 40 = 540 := 
by 
  sorry

end brian_tape_needed_l61_61041


namespace number_of_charms_l61_61853

-- Let x be the number of charms used to make each necklace
variable (x : ℕ)

-- Each charm costs $15
variable (cost_per_charm : ℕ)
axiom cost_per_charm_is_15 : cost_per_charm = 15

-- Tim sells each necklace for $200
variable (selling_price : ℕ)
axiom selling_price_is_200 : selling_price = 200

-- Tim makes a profit of $1500 if he sells 30 necklaces
variable (total_profit : ℕ)
axiom total_profit_is_1500 : total_profit = 1500

theorem number_of_charms (h : 30 * (selling_price - cost_per_charm * x) = total_profit) : x = 10 :=
sorry

end number_of_charms_l61_61853


namespace inequality_proof_l61_61068

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (z + x) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) :=
by
  sorry

end inequality_proof_l61_61068


namespace average_speed_v2_l61_61280

theorem average_speed_v2 (v1 : ℝ) (t : ℝ) (S1 : ℝ) (S2 : ℝ) : 
  (v1 = 30) → (t = 30) → (S1 = 800) → (S2 = 200) → 
  (v2 = (v1 - (S1 - S2) / t) ∨ v2 = (v1 + (S1 - S2) / t)) :=
by
  intros h1 h2 h3 h4
  sorry

end average_speed_v2_l61_61280


namespace points_satisfy_l61_61253

theorem points_satisfy (x y : ℝ) : 
  (y^2 - y = x^2 - x) ↔ (y = x ∨ y = 1 - x) :=
by sorry

end points_satisfy_l61_61253


namespace find_ab_sum_l61_61817

theorem find_ab_sum 
  (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) 
  : a + b = -14 := by
  sorry

end find_ab_sum_l61_61817


namespace identify_A_B_l61_61239

variable {Person : Type}
variable (isTruthful isLiar : Person → Prop)
variable (isBoy isGirl : Person → Prop)

variables (A B : Person)

-- Conditions
axiom truthful_or_liar : ∀ x : Person, isTruthful x ∨ isLiar x
axiom boy_or_girl : ∀ x : Person, isBoy x ∨ isGirl x
axiom not_both_truthful_and_liar : ∀ x : Person, ¬(isTruthful x ∧ isLiar x)
axiom not_both_boy_and_girl : ∀ x : Person, ¬(isBoy x ∧ isGirl x)

-- Statements made by A and B
axiom A_statement : isTruthful A → isLiar B 
axiom B_statement : isBoy B → isGirl A 

-- Goal: prove the identities of A and B
theorem identify_A_B : isTruthful A ∧ isBoy A ∧ isLiar B ∧ isBoy B :=
by {
  sorry
}

end identify_A_B_l61_61239


namespace choose_5_with_exactly_one_twin_l61_61859

theorem choose_5_with_exactly_one_twin :
  let total_players := 12
  let twins := 2
  let players_to_choose := 5
  let remaining_players_after_one_twin := total_players - twins + 1 -- 11 players to choose from
  (2 * Nat.choose remaining_players_after_one_twin (players_to_choose - 1)) = 420 := 
by
  sorry

end choose_5_with_exactly_one_twin_l61_61859


namespace min_value_of_squares_find_p_l61_61897

open Real

theorem min_value_of_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (eqn : a + sqrt 2 * b + sqrt 3 * c = 2 * sqrt 3) :
  a^2 + b^2 + c^2 = 2 :=
by sorry

theorem find_p (m : ℝ) (hm : m = 2) (p q : ℝ) :
  (∀ x, |x - 3| ≥ m ↔ x^2 + p * x + q ≥ 0) → p = -6 :=
by sorry

end min_value_of_squares_find_p_l61_61897


namespace fraction_subtraction_l61_61384

theorem fraction_subtraction : 
  (3 + 6 + 9) = 18 ∧ (2 + 5 + 8) = 15 ∧ (2 + 5 + 8) = 15 ∧ (3 + 6 + 9) = 18 →
  (18 / 15 - 15 / 18) = 11 / 30 :=
by
  intro h
  sorry

end fraction_subtraction_l61_61384


namespace correct_average_marks_l61_61614

theorem correct_average_marks 
  (avg_marks : ℝ) 
  (num_students : ℕ) 
  (incorrect_marks : ℕ → (ℝ × ℝ)) :
  avg_marks = 85 →
  num_students = 50 →
  incorrect_marks 0 = (95, 45) →
  incorrect_marks 1 = (78, 58) →
  incorrect_marks 2 = (120, 80) →
  (∃ corrected_avg_marks : ℝ, corrected_avg_marks = 82.8) :=
by
  sorry

end correct_average_marks_l61_61614


namespace isosceles_base_lines_l61_61266
open Real

theorem isosceles_base_lines {x y : ℝ} (h1 : 7 * x - y - 9 = 0) (h2 : x + y - 7 = 0) (hx : x = 3) (hy : y = -8) :
  (x - 3 * y - 27 = 0) ∨ (3 * x + y - 1 = 0) :=
sorry

end isosceles_base_lines_l61_61266


namespace geometric_sequence_general_term_geometric_sequence_sum_n_l61_61380

theorem geometric_sequence_general_term (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) : 
  a n = 48 * (1 / 2) ^ (n - 1) := 
by {
  sorry
}

theorem geometric_sequence_sum_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) 
  (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) 
  (h4 : ∀ (n : ℕ), S n = 48 * (1 - (1 / 2) ^ n) / (1 - 1 / 2))
  (h5 : S 5 = 93) : 
  5 = 5 := 
by {
  sorry
}

end geometric_sequence_general_term_geometric_sequence_sum_n_l61_61380


namespace fraction_of_loss_is_correct_l61_61168

-- Definitions based on the conditions
def selling_price : ℕ := 18
def cost_price : ℕ := 19

-- Calculating the loss
def loss : ℕ := cost_price - selling_price

-- Fraction of the loss compared to the cost price
def fraction_of_loss : ℚ := loss / cost_price

-- The theorem we want to prove
theorem fraction_of_loss_is_correct : fraction_of_loss = 1 / 19 := by
  sorry

end fraction_of_loss_is_correct_l61_61168


namespace bees_second_day_l61_61763

-- Define the number of bees on the first day
def bees_on_first_day : ℕ := 144 

-- Define the multiplier for the second day
def multiplier : ℕ := 3

-- Define the number of bees on the second day
def bees_on_second_day : ℕ := bees_on_first_day * multiplier

-- Theorem stating the number of bees seen on the second day
theorem bees_second_day : bees_on_second_day = 432 := by
  -- Proof is pending.
  sorry

end bees_second_day_l61_61763


namespace circle_E_radius_sum_l61_61707

noncomputable def radius_A := 15
noncomputable def radius_B := 5
noncomputable def radius_C := 3
noncomputable def radius_D := 3

-- We need to find that the sum of m and n for the radius of circle E is 131.
theorem circle_E_radius_sum (m n : ℕ) (h1 : Nat.gcd m n = 1) (radius_E : ℚ := (m / n)) :
  m + n = 131 :=
  sorry

end circle_E_radius_sum_l61_61707


namespace proj_v_onto_w_l61_61315

open Real

noncomputable def v : ℝ × ℝ := (8, -4)
noncomputable def w : ℝ × ℝ := (2, 3)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let coeff := dot_product v w / dot_product w w
  (coeff * w.1, coeff * w.2)

theorem proj_v_onto_w :
  projection v w = (8 / 13, 12 / 13) :=
by
  sorry

end proj_v_onto_w_l61_61315


namespace probability_of_odd_product_is_zero_l61_61663

-- Define the spinners
def spinnerC : List ℕ := [1, 3, 5, 7]
def spinnerD : List ℕ := [2, 4, 6]

-- Define the condition that the odds and evens have a specific product property
axiom odd_times_even_is_even {a b : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 0) : (a * b) % 2 = 0

-- Define the probability of getting an odd product
noncomputable def probability_odd_product : ℕ :=
  if ∃ a ∈ spinnerC, ∃ b ∈ spinnerD, (a * b) % 2 = 1 then 1 else 0

-- Main theorem
theorem probability_of_odd_product_is_zero : probability_odd_product = 0 := by
  sorry

end probability_of_odd_product_is_zero_l61_61663


namespace distance_midpoint_to_origin_l61_61672

variables {a b c d m k l n : ℝ}

theorem distance_midpoint_to_origin (h1 : b = m * a + k) (h2 : d = m * c + k) (h3 : n = -1 / m) :
  dist (0, 0) ( ((a + c) / 2), ((m * (a + c) + 2 * k) / 2) ) = (1 / 2) * Real.sqrt ((1 + m^2) * (a + c)^2 + 4 * k^2 + 4 * m * (a + c) * k) :=
by
  sorry

end distance_midpoint_to_origin_l61_61672


namespace evaluate_M_l61_61082

noncomputable def M : ℝ := 
  (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)

theorem evaluate_M : M = (1 + Real.sqrt 3 + Real.sqrt 5 + 3 * Real.sqrt 2) / 3 :=
by
  sorry

end evaluate_M_l61_61082


namespace julia_height_in_cm_l61_61243

def height_in_feet : ℕ := 5
def height_in_inches : ℕ := 4
def feet_to_inches : ℕ := 12
def inch_to_cm : ℝ := 2.54

theorem julia_height_in_cm : (height_in_feet * feet_to_inches + height_in_inches) * inch_to_cm = 162.6 :=
sorry

end julia_height_in_cm_l61_61243


namespace distribute_candy_bars_l61_61482

theorem distribute_candy_bars (candies bags : ℕ) (h1 : candies = 15) (h2 : bags = 5) :
  candies / bags = 3 :=
by
  sorry

end distribute_candy_bars_l61_61482


namespace total_amount_divided_l61_61321

theorem total_amount_divided (A B C : ℝ) (h1 : A = 2/3 * (B + C)) (h2 : B = 2/3 * (A + C)) (h3 : A = 80) : 
  A + B + C = 200 :=
by
  sorry

end total_amount_divided_l61_61321


namespace min_visible_sum_of_prism_faces_l61_61234

theorem min_visible_sum_of_prism_faces :
  let corners := 8
  let edges := 8
  let face_centers := 12
  let min_corner_sum := 6 -- Each corner dice can show 1, 2, and 3
  let min_edge_sum := 3    -- Each edge dice can show 1 and 2
  let min_face_center_sum := 1 -- Each face center dice can show 1
  let total_sum := corners * min_corner_sum + edges * min_edge_sum + face_centers * min_face_center_sum
  total_sum = 84 := 
by
  -- The proof is omitted
  sorry

end min_visible_sum_of_prism_faces_l61_61234


namespace f_15_equals_227_l61_61149

def f (n : ℕ) : ℕ := n^2 - n + 17

theorem f_15_equals_227 : f 15 = 227 := by
  sorry

end f_15_equals_227_l61_61149


namespace winner_more_votes_l61_61235

variable (totalStudents : ℕ) (votingPercentage : ℤ) (winnerPercentage : ℤ) (loserPercentage : ℤ)

theorem winner_more_votes
    (h1 : totalStudents = 2000)
    (h2 : votingPercentage = 25)
    (h3 : winnerPercentage = 55)
    (h4 : loserPercentage = 100 - winnerPercentage)
    (h5 : votingStudents = votingPercentage * totalStudents / 100)
    (h6 : winnerVotes = winnerPercentage * votingStudents / 100)
    (h7 : loserVotes = loserPercentage * votingStudents / 100)
    : winnerVotes - loserVotes = 50 := by
  sorry

end winner_more_votes_l61_61235


namespace point_on_line_l61_61128

theorem point_on_line :
  ∃ a b : ℝ, (a ≠ 0) ∧
  (∀ x y : ℝ, (x = 4 ∧ y = 5) ∨ (x = 8 ∧ y = 17) ∨ (x = 12 ∧ y = 29) → y = a * x + b) →
  (∃ t : ℝ, (15, t) ∈ {(x, y) | y = a * x + b} ∧ t = 38) :=
by
  sorry

end point_on_line_l61_61128


namespace instantaneous_velocity_at_3_l61_61473

-- Define the position function s(t)
def s (t : ℝ) : ℝ := 1 - t + t^2

-- The main statement we need to prove
theorem instantaneous_velocity_at_3 : (deriv s 3) = 5 :=
by 
  -- The theorem requires a proof which we mark as sorry for now.
  sorry

end instantaneous_velocity_at_3_l61_61473


namespace expected_value_is_7_l61_61985

def win (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * (10 - n) else 10 - n

def fair_die_values := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def expected_value (values : List ℕ) (win : ℕ → ℕ) : ℚ :=
  (values.map (λ n => win n)).sum / values.length

theorem expected_value_is_7 :
  expected_value fair_die_values win = 7 := 
sorry

end expected_value_is_7_l61_61985


namespace angle_bisector_relation_l61_61340

theorem angle_bisector_relation (a b : ℝ) (h : a = -b ∨ a = -b) : a = -b :=
sorry

end angle_bisector_relation_l61_61340


namespace my_op_evaluation_l61_61311

def my_op (x y : Int) : Int := x * y - 3 * x + y

theorem my_op_evaluation : my_op 5 3 - my_op 3 5 = -8 := by 
  sorry

end my_op_evaluation_l61_61311


namespace spider_has_eight_legs_l61_61506

-- Define the number of legs a human has
def human_legs : ℕ := 2

-- Define the number of legs for a spider, based on the given condition
def spider_legs : ℕ := 2 * (2 * human_legs)

-- The theorem to be proven, that the spider has 8 legs
theorem spider_has_eight_legs : spider_legs = 8 :=
by
  sorry

end spider_has_eight_legs_l61_61506


namespace find_f_2017_l61_61295

def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_f_shifted : ∀ x : ℝ, f (1 - x) = f (x + 1)
axiom f_neg_one : f (-1) = 2

theorem find_f_2017 : f 2017 = -2 :=
by
  sorry

end find_f_2017_l61_61295


namespace perfect_square_tens_digits_l61_61391

theorem perfect_square_tens_digits
  (a b : ℕ)
  (is_square_a : ∃ k : ℕ, a = k * k)
  (is_square_b : ∃ k : ℕ, b = k * k)
  (units_digit_a : a % 10 = 1)
  (tens_digit_a : ∃ x : ℕ, a / 10 % 10 = x)
  (units_digit_b : b % 10 = 6)
  (tens_digit_b : ∃ y : ℕ, b / 10 % 10 = y) :
  ∃ x y : ℕ, (a / 10 % 10 = x) ∧ (b / 10 % 10 = y) ∧ (x % 2 = 0) ∧ (y % 2 = 1) :=
sorry

end perfect_square_tens_digits_l61_61391


namespace number_is_more_than_sum_l61_61759

theorem number_is_more_than_sum : 20.2 + 33.8 - 5.1 = 48.9 :=
by
  sorry

end number_is_more_than_sum_l61_61759


namespace problem_statement_l61_61752

variable (m n : ℝ)
noncomputable def sqrt_2_minus_1_inv := (Real.sqrt 2 - 1)⁻¹
noncomputable def sqrt_2_plus_1_inv := (Real.sqrt 2 + 1)⁻¹

theorem problem_statement 
  (hm : m = sqrt_2_minus_1_inv) 
  (hn : n = sqrt_2_plus_1_inv) : 
  m + n = 2 * Real.sqrt 2 := 
sorry

end problem_statement_l61_61752


namespace percentage_increase_l61_61982

theorem percentage_increase 
    (P : ℝ)
    (buying_price : ℝ) (h1 : buying_price = 0.80 * P)
    (selling_price : ℝ) (h2 : selling_price = 1.24 * P) :
    ((selling_price - buying_price) / buying_price) * 100 = 55 := by 
  sorry

end percentage_increase_l61_61982


namespace triplet_unique_solution_l61_61934

theorem triplet_unique_solution {x y z : ℝ} :
  x^2 - 2*x - 4*z = 3 →
  y^2 - 2*y - 2*x = -14 →
  z^2 - 4*y - 4*z = -18 →
  (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry

end triplet_unique_solution_l61_61934


namespace geometric_seq_sum_S40_l61_61320

noncomputable def geometric_seq_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q ≠ 1 then a1 * (1 - q^n) / (1 - q) else a1 * n

theorem geometric_seq_sum_S40 :
  ∃ (a1 q : ℝ), (0 < q ∧ q ≠ 1) ∧ 
                geometric_seq_sum a1 q 10 = 10 ∧
                geometric_seq_sum a1 q 30 = 70 ∧
                geometric_seq_sum a1 q 40 = 150 :=
by
  sorry

end geometric_seq_sum_S40_l61_61320


namespace valid_operation_l61_61229

theorem valid_operation :
  ∀ x : ℝ, x^2 + x^3 ≠ x^5 ∧
  ∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2 ∧
  ∀ m : ℝ, (|m| = m ↔ m ≥ 0) :=
by
  sorry

end valid_operation_l61_61229


namespace even_of_square_even_l61_61378

theorem even_of_square_even (a : Int) (h1 : ∃ n : Int, a = 2 * n) (h2 : Even (a ^ 2)) : Even a := 
sorry

end even_of_square_even_l61_61378


namespace boat_distance_downstream_l61_61570

-- Let v_s be the speed of the stream in km/h
-- Condition 1: In one hour, a boat goes 5 km against the stream.
-- Condition 2: The speed of the boat in still water is 8 km/h.

theorem boat_distance_downstream (v_s : ℝ) :
  (8 - v_s = 5) →
  (distance : ℝ) →
  8 + v_s = distance →
  distance = 11 := by
  sorry

end boat_distance_downstream_l61_61570


namespace subtract_base8_l61_61159

theorem subtract_base8 (a b : ℕ) (h₁ : a = 0o2101) (h₂ : b = 0o1245) :
  a - b = 0o634 := sorry

end subtract_base8_l61_61159


namespace coin_flip_probability_l61_61430

theorem coin_flip_probability (P : ℕ → ℕ → ℚ) (n : ℕ) :
  (∀ k, P k 0 = 1/2) →
  (∀ k, P k 1 = 1/2) →
  (∀ k m, P k m = 1/2) →
  n = 3 →
  P 0 0 * P 1 1 * P 2 1 = 1/8 :=
by
  intros h0 h1 h_indep hn
  sorry

end coin_flip_probability_l61_61430


namespace friends_came_over_later_l61_61145

def original_friends : ℕ := 4
def total_people : ℕ := 7

theorem friends_came_over_later : (total_people - original_friends = 3) :=
sorry

end friends_came_over_later_l61_61145


namespace date_behind_D_correct_l61_61058

noncomputable def date_behind_B : ℕ := sorry
noncomputable def date_behind_E : ℕ := date_behind_B + 2
noncomputable def date_behind_F : ℕ := date_behind_B + 15
noncomputable def date_behind_D : ℕ := sorry

theorem date_behind_D_correct :
  date_behind_B + date_behind_D = date_behind_E + date_behind_F := sorry

end date_behind_D_correct_l61_61058


namespace inequality_not_always_hold_l61_61927

variable (a b c : ℝ)

theorem inequality_not_always_hold (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) : ¬ (∀ (a b : ℝ), |a - b| + 1 / (a - b) ≥ 2) :=
by
  sorry

end inequality_not_always_hold_l61_61927


namespace probability_green_face_l61_61611

def faces : ℕ := 6
def green_faces : ℕ := 3

theorem probability_green_face : (green_faces : ℚ) / (faces : ℚ) = 1 / 2 := by
  sorry

end probability_green_face_l61_61611


namespace solve_real_roots_in_intervals_l61_61955

noncomputable def real_roots_intervals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∃ x₁ x₂ : ℝ,
    (3 * x₁^2 - 2 * (a - b) * x₁ - a * b = 0) ∧
    (3 * x₂^2 - 2 * (a - b) * x₂ - a * b = 0) ∧
    (a / 3 < x₁ ∧ x₁ < 2 * a / 3) ∧
    (-2 * b / 3 < x₂ ∧ x₂ < -b / 3)

-- Statement of the problem:
theorem solve_real_roots_in_intervals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  real_roots_intervals a b ha hb :=
sorry

end solve_real_roots_in_intervals_l61_61955


namespace larger_integer_is_7sqrt14_l61_61605

theorem larger_integer_is_7sqrt14 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a / b = 7 / 3) (h2 : a * b = 294) : max a b = 7 * Real.sqrt 14 :=
by 
  sorry

end larger_integer_is_7sqrt14_l61_61605


namespace each_girl_brought_2_cups_l61_61018

-- Here we define all the given conditions
def total_students : ℕ := 30
def num_boys : ℕ := 10
def cups_per_boy : ℕ := 5
def total_cups : ℕ := 90

-- Define the conditions as Lean definitions
def num_girls : ℕ := total_students - num_boys -- From condition 3
def cups_by_boys : ℕ := num_boys * cups_per_boy
def cups_by_girls : ℕ := total_cups - cups_by_boys

-- Define the final question and expected answer
def cups_per_girl : ℕ := cups_by_girls / num_girls

-- Final problem statement to prove
theorem each_girl_brought_2_cups :
  cups_per_girl = 2 :=
by
  have h1 : num_girls = 20 := by sorry
  have h2 : cups_by_boys = 50 := by sorry
  have h3 : cups_by_girls = 40 := by sorry
  have h4 : cups_per_girl = 2 := by sorry
  exact h4

end each_girl_brought_2_cups_l61_61018


namespace money_spent_l61_61444

def initial_money (Henry : Type) : ℤ := 11
def birthday_money (Henry : Type) : ℤ := 18
def final_money (Henry : Type) : ℤ := 19

theorem money_spent (Henry : Type) : (initial_money Henry + birthday_money Henry - final_money Henry = 10) := 
by sorry

end money_spent_l61_61444


namespace max_value_of_z_l61_61246

theorem max_value_of_z 
    (x y : ℝ) 
    (h1 : |2 * x + y + 1| ≤ |x + 2 * y + 2|)
    (h2 : -1 ≤ y ∧ y ≤ 1) : 
    2 * x + y ≤ 5 := 
sorry

end max_value_of_z_l61_61246


namespace ken_summit_time_l61_61264

variables (t : ℕ) (s : ℕ)

/--
Sari and Ken climb up a mountain. 
Ken climbs at a constant pace of 500 meters per hour,
and reaches the summit after \( t \) hours starting from 10:00.
Sari starts climbing 2 hours before Ken at 08:00 and is 50 meters behind Ken when he reaches the summit.
Sari is already 700 meters ahead of Ken when he starts climbing.
Prove that Ken reaches the summit at 15:00.
-/
theorem ken_summit_time (h1 : 500 * t = s * (t + 2) + 50)
  (h2 : s * 2 = 700) : t + 10 = 15 :=

sorry

end ken_summit_time_l61_61264


namespace age_of_15th_student_l61_61700

theorem age_of_15th_student (avg_age_all : ℝ) (avg_age_4 : ℝ) (avg_age_10 : ℝ) 
  (total_students : ℕ) (group_4_students : ℕ) (group_10_students : ℕ) 
  (h1 : avg_age_all = 15) (h2 : avg_age_4 = 14) (h3 : avg_age_10 = 16) 
  (h4 : total_students = 15) (h5 : group_4_students = 4) (h6 : group_10_students = 10) : 
  ∃ x : ℝ, x = 9 := 
by 
  sorry

end age_of_15th_student_l61_61700


namespace max_stamps_l61_61562

-- Definitions based on conditions
def price_of_stamp := 28 -- in cents
def total_money := 3600 -- in cents

-- The theorem statement
theorem max_stamps (price_of_stamp total_money : ℕ) : (total_money / price_of_stamp) = 128 := by
  sorry

end max_stamps_l61_61562


namespace find_x_squared_plus_y_squared_plus_z_squared_l61_61006

theorem find_x_squared_plus_y_squared_plus_z_squared
  (x y z : ℤ)
  (h1 : x + y + z = 3)
  (h2 : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 57 :=
by
  sorry

end find_x_squared_plus_y_squared_plus_z_squared_l61_61006


namespace sqrt_last_digit_l61_61195

-- Definitions related to the problem
def is_p_adic_number (α : ℕ) (p : ℕ) := true -- assume this definition captures p-adic number system

-- Problem statement in Lean 4
theorem sqrt_last_digit (p α a1 b1 : ℕ) (hα : is_p_adic_number α p) (h_last_digit_α : α % p = a1)
  (h_sqrt : (b1 * b1) % p = α % p) :
  (b1 * b1) % p = a1 :=
by sorry

end sqrt_last_digit_l61_61195


namespace range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3_l61_61348

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3 :
  {x : ℝ | f (2 * x) > f (x + 3)} = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3_l61_61348


namespace f_shift_l61_61156

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem f_shift (x : ℝ) : f (x - 1) = x^2 - 4 * x + 3 := by
  sorry

end f_shift_l61_61156


namespace point_in_second_quadrant_l61_61274

theorem point_in_second_quadrant (a : ℝ) (h1 : 2 * a + 1 < 0) (h2 : 1 - a > 0) : a < -1 / 2 := 
sorry

end point_in_second_quadrant_l61_61274


namespace area_of_inscribed_square_l61_61121

theorem area_of_inscribed_square
    (r : ℝ)
    (h : ∀ A : ℝ × ℝ, (A.1 = r - 1 ∨ A.1 = -(r - 1)) ∧ (A.2 = r - 2 ∨ A.2 = -(r - 2)) → A.1^2 + A.2^2 = r^2) :
    4 * r^2 = 100 := by
  -- proof would go here
  sorry

end area_of_inscribed_square_l61_61121


namespace proof_stops_with_two_pizzas_l61_61461

/-- The number of stops with orders of two pizzas. -/
def stops_with_two_pizzas : ℕ := 2

theorem proof_stops_with_two_pizzas
  (total_pizzas : ℕ)
  (single_stops : ℕ)
  (two_pizza_stops : ℕ)
  (average_time : ℕ)
  (total_time : ℕ)
  (h1 : total_pizzas = 12)
  (h2 : two_pizza_stops * 2 + single_stops = total_pizzas)
  (h3 : total_time = 40)
  (h4 : average_time = 4)
  (h5 : two_pizza_stops + single_stops = total_time / average_time) :
  two_pizza_stops = stops_with_two_pizzas := 
sorry

end proof_stops_with_two_pizzas_l61_61461


namespace sequence_form_l61_61342

-- Defining the sequence a_n as a function f
def seq (f : ℕ → ℕ) : Prop :=
  ∃ c : ℝ, (0 < c) ∧ ∀ m n : ℕ, Nat.gcd (f m + n) (f n + m) > (c * (m + n))

-- Proving that if there exists such a sequence, then it is of the form n + c
theorem sequence_form (f : ℕ → ℕ) (h : seq f) :
  ∃ c : ℤ, ∀ n : ℕ, f n = n + c :=
sorry

end sequence_form_l61_61342


namespace linda_original_amount_l61_61711

theorem linda_original_amount (L L2 : ℕ) 
  (h1 : L = 20) 
  (h2 : L - 5 = L2) : 
  L2 + 5 = 15 := 
sorry

end linda_original_amount_l61_61711


namespace find_x_values_l61_61610

theorem find_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 3 / 8) :
  x = 4 ∨ x = 8 :=
by
  sorry

end find_x_values_l61_61610


namespace a_2018_value_l61_61640

theorem a_2018_value (S a : ℕ -> ℕ) (h₁ : S 1 = a 1) (h₂ : a 1 = 1) (h₃ : ∀ n : ℕ, n > 0 -> S (n + 1) = 3 * S n) :
  a 2018 = 2 * 3 ^ 2016 :=
sorry

end a_2018_value_l61_61640


namespace binary_to_base4_conversion_l61_61709

theorem binary_to_base4_conversion :
  let b := 110110100
  let b_2 := Nat.ofDigits 2 [1, 1, 0, 1, 1, 0, 1, 0, 0]
  let b_4 := Nat.ofDigits 4 [3, 1, 2, 2, 0]
  b_2 = b → b_4 = 31220 :=
by
  intros b b_2 b_4 h
  sorry

end binary_to_base4_conversion_l61_61709


namespace arithmetic_sequence_fifth_term_l61_61532

theorem arithmetic_sequence_fifth_term :
  ∀ (a₁ d n : ℕ), a₁ = 3 → d = 4 → n = 5 → a₁ + (n - 1) * d = 19 :=
by
  intros a₁ d n ha₁ hd hn
  sorry

end arithmetic_sequence_fifth_term_l61_61532


namespace lemonade_syrup_parts_l61_61308

theorem lemonade_syrup_parts (L : ℝ) :
  (L = 2 / 0.75) →
  (L = 2.6666666666666665) :=
by
  sorry

end lemonade_syrup_parts_l61_61308


namespace problem_statement_l61_61839

variable {x y : ℝ}

theorem problem_statement (h1 : x * y = -3) (h2 : x + y = -4) : x^2 + 3 * x * y + y^2 = 13 := sorry

end problem_statement_l61_61839


namespace part1_part2_l61_61047

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 6)

theorem part1 (k : ℝ) :
  (∀ x : ℝ, (f x > k) ↔ (x < -3 ∨ x > -2)) ↔ k = -2/5 :=
by
  sorry

theorem part2 (t : ℝ) :
  (∀ x : ℝ, (x > 0) → (f x ≤ t)) ↔ t ∈ (Set.Ici (Real.sqrt 6 / 6)) :=
by
  sorry

end part1_part2_l61_61047


namespace min_coins_for_less_than_1_dollar_l61_61133

theorem min_coins_for_less_than_1_dollar :
  ∃ (p n q h : ℕ), 1*p + 5*n + 25*q + 50*h ≥ 1 ∧ 1*p + 5*n + 25*q + 50*h < 100 ∧ p + n + q + h = 8 :=
by 
  sorry

end min_coins_for_less_than_1_dollar_l61_61133


namespace min_queries_to_determine_parity_l61_61938

def num_bags := 100
def num_queries := 3
def bags := Finset (Fin num_bags)

def can_query_parity (bags : Finset (Fin num_bags)) : Prop :=
  bags.card = 15

theorem min_queries_to_determine_parity :
  ∀ (query : Fin num_queries → Finset (Fin num_bags)),
  (∀ i, can_query_parity (query i)) →
  (∀ i j k, query i ∪ query j ∪ query k = {a : Fin num_bags | a.val = 1}) →
  num_queries ≥ 3 :=
  sorry

end min_queries_to_determine_parity_l61_61938


namespace add_to_make_divisible_l61_61376

theorem add_to_make_divisible :
  ∃ n, n = 34 ∧ ∃ k : ℕ, 758492136547 + n = 51 * k := by
  sorry

end add_to_make_divisible_l61_61376


namespace sum_adjacent_to_49_l61_61884

noncomputable def sum_of_adjacent_divisors : ℕ :=
  let divisors := [5, 7, 35, 49, 245]
  -- We assume an arrangement such that adjacent pairs to 49 are {35, 245}
  35 + 245

theorem sum_adjacent_to_49 : sum_of_adjacent_divisors = 280 := by
  sorry

end sum_adjacent_to_49_l61_61884


namespace ellipse_range_of_k_l61_61345

theorem ellipse_range_of_k (k : ℝ) :
  (1 - k > 0) ∧ (1 + k > 0) ∧ (1 - k ≠ 1 + k) ↔ (-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1) :=
by
  sorry

end ellipse_range_of_k_l61_61345


namespace units_digit_of_a_l61_61463

theorem units_digit_of_a (a : ℕ) (ha : (∃ b : ℕ, 1 ≤ b ∧ b ≤ 9 ∧ (a*a / 10^1) % 10 = b)) : 
  ((a % 10 = 4) ∨ (a % 10 = 6)) :=
sorry

end units_digit_of_a_l61_61463


namespace length_of_room_l61_61999

theorem length_of_room (b : ℕ) (t : ℕ) (L : ℕ) (blue_tiles : ℕ) (tile_area : ℕ) (total_area : ℕ) (effective_area : ℕ) (blue_area : ℕ) :
  b = 10 →
  t = 2 →
  blue_tiles = 16 →
  tile_area = t * t →
  total_area = (L - 4) * (b - 4) →
  blue_area = blue_tiles * tile_area →
  2 * blue_area = 3 * total_area →
  L = 20 :=
by
  intros h_b h_t h_blue_tiles h_tile_area h_total_area h_blue_area h_proportion
  sorry

end length_of_room_l61_61999


namespace simplify_expression_l61_61903

theorem simplify_expression (x : ℝ) :
  4 * x - 8 * x ^ 2 + 10 - (5 - 4 * x + 8 * x ^ 2) = -16 * x ^ 2 + 8 * x + 5 :=
by
  sorry

end simplify_expression_l61_61903


namespace sum_of_first_4_terms_l61_61485

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem sum_of_first_4_terms (a r : ℝ) 
  (h1 : a * (1 + r + r^2) = 13) (h2 : a * (1 + r + r^2 + r^3 + r^4) = 121) : 
  a * (1 + r + r^2 + r^3) = 27.857 :=
by
  sorry

end sum_of_first_4_terms_l61_61485


namespace min_value_inequality_l61_61508

open Real

theorem min_value_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 9) :
  ( (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ) ≥ 9 :=
sorry

end min_value_inequality_l61_61508


namespace derivative_given_limit_l61_61973

open Real

variable {f : ℝ → ℝ} {x₀ : ℝ}

theorem derivative_given_limit (h : ∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (x₀ - 2 * Δx) - f x₀) / Δx + 2) < ε) :
  deriv f x₀ = -1 := by
  sorry

end derivative_given_limit_l61_61973


namespace principal_amount_l61_61932

theorem principal_amount (P : ℝ) (h : (P * 0.1236) - (P * 0.12) = 36) : P = 10000 := 
sorry

end principal_amount_l61_61932


namespace find_larger_number_l61_61215

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 :=
sorry

end find_larger_number_l61_61215


namespace intersection_A_B_l61_61754

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < 2 - x ∧ 2 - x < 3 }

theorem intersection_A_B :
  A ∩ B = {0, 1} := sorry

end intersection_A_B_l61_61754


namespace rectangle_length_width_difference_l61_61688

noncomputable def difference_between_length_and_width : ℝ :=
  let x := by sorry
  let y := by sorry
  (x - y)

theorem rectangle_length_width_difference {x y : ℝ}
  (h₁ : 2 * (x + y) = 20) (h₂ : x^2 + y^2 = 10^2) :
  difference_between_length_and_width = 10 :=
  by sorry

end rectangle_length_width_difference_l61_61688


namespace vertical_asymptote_x_value_l61_61843

theorem vertical_asymptote_x_value (x : ℝ) : 
  4 * x - 6 = 0 ↔ x = 3 / 2 :=
by
  sorry

end vertical_asymptote_x_value_l61_61843


namespace projection_matrix_ratio_l61_61590

theorem projection_matrix_ratio
  (x y : ℚ)
  (h1 : (4/29) * x - (10/29) * y = x)
  (h2 : -(10/29) * x + (25/29) * y = y) :
  y / x = -5/2 :=
by
  sorry

end projection_matrix_ratio_l61_61590


namespace total_dog_food_amount_l61_61632

def initial_dog_food : ℝ := 15
def first_purchase : ℝ := 15
def second_purchase : ℝ := 10

theorem total_dog_food_amount : initial_dog_food + first_purchase + second_purchase = 40 := 
by 
  sorry

end total_dog_food_amount_l61_61632


namespace serving_guests_possible_iff_even_l61_61285

theorem serving_guests_possible_iff_even (n : ℕ) : 
  (∀ seats : Finset ℕ, ∀ p : ℕ → ℕ, (∀ i : ℕ, i < n → p i ∈ seats) → 
    (∀ i j : ℕ, i < j → p i ≠ p j) → (n % 2 = 0)) = (n % 2 = 0) :=
by sorry

end serving_guests_possible_iff_even_l61_61285


namespace intercepts_congruence_l61_61621

theorem intercepts_congruence (m : ℕ) (h : m = 29) (x0 y0 : ℕ) (hx : 0 ≤ x0 ∧ x0 < m) (hy : 0 ≤ y0 ∧ y0 < m) 
  (h1 : 5 * x0 % m = (2 * 0 + 3) % m)  (h2 : (5 * 0) % m = (2 * y0 + 3) % m) : 
  x0 + y0 = 31 := by
  sorry

end intercepts_congruence_l61_61621


namespace graph_translation_l61_61536

theorem graph_translation (f : ℝ → ℝ) (x : ℝ) (h : f 1 = -1) :
  f (x - 1) - 1 = -2 :=
by
  sorry

end graph_translation_l61_61536


namespace parabola_intersect_l61_61413

theorem parabola_intersect (b c m p q x1 x2 : ℝ)
  (h_intersect1 : x1^2 + b * x1 + c = 0)
  (h_intersect2 : x2^2 + b * x2 + c = 0)
  (h_order : m < x1)
  (h_middle : x1 < x2)
  (h_range : x2 < m + 1)
  (h_valm : p = m^2 + b * m + c)
  (h_valm1 : q = (m + 1)^2 + b * (m + 1) + c) :
  p < 1 / 4 ∧ q < 1 / 4 :=
sorry

end parabola_intersect_l61_61413


namespace allowance_is_14_l61_61364

def initial := 11
def spent := 3
def final := 22

def allowance := final - (initial - spent)

theorem allowance_is_14 : allowance = 14 := by
  -- proof goes here
  sorry

end allowance_is_14_l61_61364


namespace value_of_m_l61_61154

theorem value_of_m 
  (m : ℝ)
  (h1 : |m - 1| = 1)
  (h2 : m - 2 ≠ 0) : 
  m = 0 :=
  sorry

end value_of_m_l61_61154


namespace specific_clothing_choice_probability_l61_61045

noncomputable def probability_of_specific_clothing_choice : ℚ :=
  let total_clothing := 4 + 5 + 6
  let total_ways_to_choose_3 := Nat.choose 15 3
  let ways_to_choose_specific_3 := 4 * 5 * 6
  let probability := ways_to_choose_specific_3 / total_ways_to_choose_3
  probability

theorem specific_clothing_choice_probability :
  probability_of_specific_clothing_choice = 24 / 91 :=
by
  -- proof here 
  sorry

end specific_clothing_choice_probability_l61_61045


namespace bus_driver_total_hours_l61_61443

def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
def total_compensation : ℝ := 976
def max_regular_hours : ℝ := 40

theorem bus_driver_total_hours :
  ∃ (hours_worked : ℝ), 
  (hours_worked = max_regular_hours + (total_compensation - (regular_rate * max_regular_hours)) / overtime_rate) ∧
  hours_worked = 52 :=
by
  sorry

end bus_driver_total_hours_l61_61443


namespace find_f_values_find_f_expression_l61_61816

variable (f : ℕ+ → ℤ)

-- Conditions in Lean
def is_increasing (f : ℕ+ → ℤ) : Prop :=
  ∀ {m n : ℕ+}, m < n → f m < f n

axiom h1 : is_increasing f
axiom h2 : f 4 = 5
axiom h3 : ∀ n : ℕ+, ∃ k : ℕ, f n = k
axiom h4 : ∀ m n : ℕ+, f m * f n = f (m * n) + f (m + n - 1)

-- Proof in Lean 4
theorem find_f_values : f 1 = 2 ∧ f 2 = 3 ∧ f 3 = 4 :=
by
  sorry

theorem find_f_expression : ∀ n : ℕ+, f n = n + 1 :=
by
  sorry

end find_f_values_find_f_expression_l61_61816


namespace inclination_angle_of_y_axis_l61_61222

theorem inclination_angle_of_y_axis : 
  ∀ (l : ℝ), l = 90 :=
sorry

end inclination_angle_of_y_axis_l61_61222


namespace smallest_k_remainder_2_l61_61017

theorem smallest_k_remainder_2 (k : ℕ) :
  k > 1 ∧
  k % 13 = 2 ∧
  k % 7 = 2 ∧
  k % 3 = 2 →
  k = 275 :=
by sorry

end smallest_k_remainder_2_l61_61017


namespace no_internal_angle_less_than_60_l61_61299

-- Define the concept of a Δ-curve
def delta_curve (K : Type) : Prop := sorry

-- Define the concept of a bicentric Δ-curve
def bicentric_delta_curve (K : Type) : Prop := sorry

-- Define the concept of internal angles of a Δ-curve
def has_internal_angle (K : Type) (A : ℝ) : Prop := sorry

-- The Lean statement for the problem
theorem no_internal_angle_less_than_60 (K : Type) 
  (h1 : delta_curve K) 
  (h2 : has_internal_angle K 60 ↔ bicentric_delta_curve K) :
  (∀ A < 60, ¬has_internal_angle K A) ∧ (has_internal_angle K 60 → bicentric_delta_curve K) := 
sorry

end no_internal_angle_less_than_60_l61_61299


namespace largest_constant_inequality_l61_61813

theorem largest_constant_inequality (C : ℝ) (h : ∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) : 
  C ≤ 2 / Real.sqrt 3 :=
sorry

end largest_constant_inequality_l61_61813


namespace divisibility_check_l61_61073

variable (d : ℕ) (h1 : d % 2 = 1) (h2 : d % 5 ≠ 0)
variable (δ : ℕ) (h3 : ∃ m : ℕ, 10 * δ + 1 = m * d)
variable (N : ℕ)

def last_digit (N : ℕ) : ℕ := N % 10
def remove_last_digit (N : ℕ) : ℕ := N / 10

theorem divisibility_check (h4 : ∃ N' u : ℕ, N = 10 * N' + u ∧ N = N' * 10 + u ∧ N' = remove_last_digit N ∧ u = last_digit N)
  (N' : ℕ) (u : ℕ) (N1 : ℕ) (h5 : N1 = N' - δ * u) :
  d ∣ N1 → d ∣ N := by
  sorry

end divisibility_check_l61_61073


namespace arithmetic_series_sum_l61_61760

theorem arithmetic_series_sum :
  let a1 : ℚ := 22
  let d : ℚ := 3 / 7
  let an : ℚ := 73
  let n := (an - a1) / d + 1
  let S := n * (a1 + an) / 2
  S = 5700 := by
  sorry

end arithmetic_series_sum_l61_61760


namespace find_numbers_l61_61090

theorem find_numbers (a b : ℕ) 
  (h1 : a / b * 6 = 10)
  (h2 : a - b + 4 = 10) :
  a = 15 ∧ b = 9 := by
  sorry

end find_numbers_l61_61090


namespace find_result_l61_61278

theorem find_result : ∀ (x : ℝ), x = 1 / 3 → 5 - 7 * x = 8 / 3 := by
  intros x hx
  sorry

end find_result_l61_61278


namespace find_p_l61_61680

theorem find_p (p : ℝ) : 
  (Nat.choose 5 3) * p^3 = 80 → p = 2 :=
by
  intro h
  sorry

end find_p_l61_61680


namespace value_of_p_l61_61433

-- Let us assume the conditions given, and the existence of positive values p and q such that p + q = 1,
-- and the second term and fourth term of the polynomial expansion (x + y)^10 are equal when x = p and y = q.

theorem value_of_p (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (h_sum : p + q = 1) (h_eq_terms : 10 * p ^ 9 * q = 120 * p ^ 7 * q ^ 3) :
    p = Real.sqrt (12 / 13) :=
    by sorry

end value_of_p_l61_61433


namespace shaded_region_area_l61_61694

noncomputable def line1 (x : ℝ) : ℝ := -(3 / 10) * x + 5
noncomputable def line2 (x : ℝ) : ℝ := -(5 / 7) * x + 47 / 7

noncomputable def intersection_x : ℝ := 17 / 5

noncomputable def area_under_curve (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

theorem shaded_region_area : 
  area_under_curve line1 line2 0 intersection_x = 1.91 :=
sorry

end shaded_region_area_l61_61694


namespace shift_quadratic_function_left_l61_61967

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the shifted quadratic function
def shifted_function (x : ℝ) : ℝ := (x + 1)^2

-- Theorem statement
theorem shift_quadratic_function_left :
  ∀ x : ℝ, shifted_function x = original_function (x + 1) := by
  sorry

end shift_quadratic_function_left_l61_61967


namespace inscribed_circle_radius_l61_61293

noncomputable def radius_inscribed_circle (DE DF EF : ℝ) : ℝ := 
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem inscribed_circle_radius :
  radius_inscribed_circle 8 5 9 = 6 * Real.sqrt 11 / 11 :=
by
  sorry

end inscribed_circle_radius_l61_61293


namespace evaluate_expression_l61_61161

theorem evaluate_expression : 2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end evaluate_expression_l61_61161


namespace meaningful_expr_l61_61202

theorem meaningful_expr (x : ℝ) : 
    (x + 1 ≥ 0 ∧ x - 2 ≠ 0) → (x ≥ -1 ∧ x ≠ 2) := by
  sorry

end meaningful_expr_l61_61202


namespace total_books_l61_61438

-- Definitions based on the conditions
def TimBooks : ℕ := 44
def SamBooks : ℕ := 52
def AlexBooks : ℕ := 65

-- Theorem to be proven
theorem total_books : TimBooks + SamBooks + AlexBooks = 161 := by
  sorry

end total_books_l61_61438


namespace find_t_l61_61256

def vector := (ℝ × ℝ)

def a : vector := (-3, 4)
def b : vector := (-1, 5)
def c : vector := (2, 3)

def parallel (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem find_t (t : ℝ) : 
  parallel (a.1 - c.1, a.2 - c.2) ((2 * t) + b.1, (3 * t) + b.2) ↔ t = -24 / 17 :=
by
  sorry

end find_t_l61_61256


namespace max_sum_a_b_c_d_l61_61114

theorem max_sum_a_b_c_d (a c d : ℤ) (b : ℕ) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  a + b + c + d = -5 := 
sorry

end max_sum_a_b_c_d_l61_61114


namespace ellipse_chord_line_eq_l61_61677

noncomputable def chord_line (x y : ℝ) : ℝ := 2 * x + 4 * y - 3

theorem ellipse_chord_line_eq :
  ∀ (x y : ℝ),
    (x ^ 2 / 2 + y ^ 2 = 1) ∧ (x + y = 1) → (chord_line x y = 0) :=
by
  intros x y h
  sorry

end ellipse_chord_line_eq_l61_61677


namespace integer_part_M_is_4_l61_61544

-- Define the variables and conditions based on the problem statement
variable (a b c : ℝ)

-- This non-computable definition includes the main mathematical expression we need to evaluate
noncomputable def M (a b c : ℝ) := Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1)

-- The theorem we need to prove
theorem integer_part_M_is_4 (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : 
  ⌊M a b c⌋ = 4 := 
by 
  sorry

end integer_part_M_is_4_l61_61544


namespace find_years_simple_interest_l61_61828

variable (R T : ℝ)
variable (P : ℝ := 6000)
variable (additional_interest : ℝ := 360)
variable (rate_diff : ℝ := 2)
variable (H : P * ((R + rate_diff) / 100) * T = P * (R / 100) * T + additional_interest)

theorem find_years_simple_interest (h : P = 6000) (h₁ : P * ((R + 2) / 100) * T = P * (R / 100) * T + 360) : 
T = 3 :=
sorry

end find_years_simple_interest_l61_61828


namespace class_average_weight_l61_61849

theorem class_average_weight (n_A n_B : ℕ) (w_A w_B : ℝ) (h1 : n_A = 50) (h2 : n_B = 40) (h3 : w_A = 50) (h4 : w_B = 70) :
  (n_A * w_A + n_B * w_B) / (n_A + n_B) = 58.89 :=
by
  sorry

end class_average_weight_l61_61849


namespace jerry_cut_pine_trees_l61_61237

theorem jerry_cut_pine_trees (P : ℕ)
  (h1 : 3 * 60 = 180)
  (h2 : 4 * 100 = 400)
  (h3 : 80 * P + 180 + 400 = 1220) :
  P = 8 :=
by {
  sorry -- Proof not required as per the instructions
}

end jerry_cut_pine_trees_l61_61237


namespace max_value_g_l61_61977

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_g : ∃ M, ∀ n, g n ≤ M ∧ M = 23 :=
  sorry

end max_value_g_l61_61977


namespace households_used_both_brands_l61_61667

/-- 
A marketing firm determined that, of 160 households surveyed, 80 used neither brand A nor brand B soap.
60 used only brand A soap and for every household that used both brands of soap, 3 used only brand B soap.
--/
theorem households_used_both_brands (X: ℕ) (H: 4*X + 140 = 160): X = 5 :=
by
  sorry

end households_used_both_brands_l61_61667


namespace compute_HHHH_of_3_l61_61184

def H (x : ℝ) : ℝ := -0.5 * x^2 + 3 * x

theorem compute_HHHH_of_3 :
  H (H (H (H 3))) = 2.689453125 := by
  sorry

end compute_HHHH_of_3_l61_61184


namespace find_multiplier_l61_61896

-- Define the variables x and y
variables (x y : ℕ)

-- Define the conditions
def condition1 := (x / 6) * y = 12
def condition2 := x = 6

-- State the theorem to prove
theorem find_multiplier (h1 : condition1 x y) (h2 : condition2 x) : y = 12 :=
sorry

end find_multiplier_l61_61896


namespace integer_for_finitely_many_n_l61_61913

theorem integer_for_finitely_many_n (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ N : ℕ, ∀ n : ℕ, N < n → ¬ ∃ k : ℤ, (a + 1 / 2) ^ n + (b + 1 / 2) ^ n = k := 
sorry

end integer_for_finitely_many_n_l61_61913


namespace work_time_B_l61_61780

theorem work_time_B (A_efficiency : ℕ) (B_efficiency : ℕ) (days_together : ℕ) (total_work : ℕ) :
  (A_efficiency = 2 * B_efficiency) →
  (days_together = 5) →
  (total_work = (A_efficiency + B_efficiency) * days_together) →
  (total_work / B_efficiency = 15) :=
by
  intros
  sorry

end work_time_B_l61_61780


namespace urn_marbles_100_white_l61_61008

theorem urn_marbles_100_white 
(initial_white initial_black final_white final_black : ℕ) 
(h_initial : initial_white = 150 ∧ initial_black = 50)
(h_operations : 
  (∀ n, (initial_white - 3 * n + 2 * n = final_white ∧ initial_black + n = final_black) ∨
  (initial_white - 2 * n - 1 = initial_white ∧ initial_black = final_black) ∨
  (initial_white - 1 * n - 2 = final_white ∧ initial_black - 1 * n = final_black) ∨
  (initial_white - 3 * n + 2 = final_white ∧ initial_black + 1 * n = final_black)) →
  ((initial_white = 150 ∧ initial_black = 50) →
   ∃ m: ℕ, final_white = 100)) :
∃ n: ℕ, initial_white - 3 * n + 2 * n = 100 ∧ initial_black + n = final_black :=
sorry

end urn_marbles_100_white_l61_61008


namespace probability_of_choosing_perfect_square_is_0_08_l61_61529

-- Definitions for the conditions
def n : ℕ := 100
def p : ℚ := 1 / 200
def probability (m : ℕ) : ℚ := if m ≤ 50 then p else 3 * p
def perfect_squares_before_50 : Finset ℕ := {1, 4, 9, 16, 25, 36, 49}
def perfect_squares_between_51_and_100 : Finset ℕ := {64, 81, 100}
def total_perfect_squares : Finset ℕ := perfect_squares_before_50 ∪ perfect_squares_between_51_and_100

-- Statement to prove that the probability of selecting a perfect square is 0.08
theorem probability_of_choosing_perfect_square_is_0_08 :
  (perfect_squares_before_50.card * p + perfect_squares_between_51_and_100.card * 3 * p) = 0.08 := 
by
  -- Adding sorry to skip the proof
  sorry

end probability_of_choosing_perfect_square_is_0_08_l61_61529


namespace sum_of_digits_of_smallest_N_l61_61273

-- Defining the conditions
def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k
def P (N : ℕ) : ℚ := ((2/3 : ℚ) * N * (1/3 : ℚ) * N) / ((N + 2) * (N + 3))
def S (n : ℕ) : ℕ := (n % 10) + ((n / 10) % 10) + (n / 100)

-- The statement of the problem
theorem sum_of_digits_of_smallest_N :
  ∃ N : ℕ, is_multiple_of_6 N ∧ P N < (4/5 : ℚ) ∧ S N = 6 :=
sorry

end sum_of_digits_of_smallest_N_l61_61273


namespace tax_percentage_l61_61916

theorem tax_percentage (C T : ℝ) (h1 : C + 10 = 90) (h2 : 1 = 90 - C - T * 90) : T = 0.1 := 
by 
  -- We provide the conditions using sorry to indicate the steps would go here
  sorry

end tax_percentage_l61_61916


namespace part1_part2_l61_61568

variables {R : Type} [LinearOrderedField R]

def setA := {x : R | -1 < x ∧ x ≤ 5}
def setB (m : R) := {x : R | x^2 - 2*x - m < 0}
def complementB (m : R) := {x : R | x ≤ -1 ∨ x ≥ 3}

theorem part1 : 
  {x : R | 6 / (x + 1) ≥ 1} = setA := 
by 
  sorry

theorem part2 (m : R) (hm : m = 3) : 
  setA ∩ complementB m = {x : R | 3 ≤ x ∧ x ≤ 5} := 
by 
  sorry

end part1_part2_l61_61568


namespace plane_through_intersection_l61_61666

def plane1 (x y z : ℝ) : Prop := x + y + 5 * z - 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 3 * y - z + 2 = 0
def pointM (x y z : ℝ) : Prop := (x, y, z) = (3, 2, 1)

theorem plane_through_intersection (x y z : ℝ) :
  plane1 x y z ∧ plane2 x y z ∧ pointM x y z → 5 * x + 14 * y - 74 * z + 31 = 0 := by
  intro h
  sorry

end plane_through_intersection_l61_61666


namespace Marty_combinations_l61_61392

theorem Marty_combinations :
  let colors := 5
  let methods := 4
  let patterns := 3
  colors * methods * patterns = 60 :=
by
  sorry

end Marty_combinations_l61_61392


namespace b4_minus_a4_l61_61629

-- Given quadratic equation and specified root, prove the difference of fourth powers.
theorem b4_minus_a4 (a b : ℝ) (h_root : (a^2 - b^2)^2 = x) (h_equation : x^2 + 4 * a^2 * b^2 * x = 4) : b^4 - a^4 = 2 ∨ b^4 - a^4 = -2 :=
sorry

end b4_minus_a4_l61_61629


namespace remainder_expr_div_by_5_l61_61306

theorem remainder_expr_div_by_5 (n : ℤ) : 
  (7 - 2 * n + (n + 5)) % 5 = (-n + 2) % 5 := 
sorry

end remainder_expr_div_by_5_l61_61306


namespace probability_of_first_three_red_cards_l61_61678

def total_cards : ℕ := 104
def suits : ℕ := 4
def cards_per_suit : ℕ := 26
def red_suits : ℕ := 2
def black_suits : ℕ := 2
def total_red_cards : ℕ := 52
def total_black_cards : ℕ := 52

noncomputable def probability_first_three_red : ℚ :=
  (total_red_cards / total_cards) * ((total_red_cards - 1) / (total_cards - 1)) * ((total_red_cards - 2) / (total_cards - 2))

theorem probability_of_first_three_red_cards :
  probability_first_three_red = 425 / 3502 :=
sorry

end probability_of_first_three_red_cards_l61_61678


namespace sum_of_numbers_in_third_column_is_96_l61_61350

theorem sum_of_numbers_in_third_column_is_96 :
  ∃ (a : ℕ), (136 = a + 16 * a) ∧ (272 = 2 * a + 32 * a) ∧ (12 * a = 96) :=
by
  let a := 8
  have h1 : 136 = a + 16 * a := by sorry  -- Proof here that 136 = 8 + 16 * 8
  have h2 : 272 = 2 * a + 32 * a := by sorry  -- Proof here that 272 = 2 * 8 + 32 * 8
  have h3 : 12 * a = 96 := by sorry  -- Proof here that 12 * 8 = 96
  existsi a
  exact ⟨h1, h2, h3⟩

end sum_of_numbers_in_third_column_is_96_l61_61350


namespace arctan_sum_property_l61_61153

open Real

theorem arctan_sum_property (x y z : ℝ) :
  arctan x + arctan y + arctan z = π / 2 → x * y + y * z + x * z = 1 :=
by
  sorry

end arctan_sum_property_l61_61153


namespace number_property_l61_61385

theorem number_property : ∀ n : ℕ, (∀ q : ℕ, q > 0 → n % q^2 < q^(q^2) / 2) ↔ n = 1 ∨ n = 4 :=
by sorry

end number_property_l61_61385


namespace square_area_dimensions_l61_61770

theorem square_area_dimensions (x : ℝ) (n : ℝ) : 
  (x^2 + (x + 12)^2 = 2120) → 
  (n = x + 12) → 
  (x = 26) → 
  (n = 38) := 
by
  sorry

end square_area_dimensions_l61_61770


namespace solve_inequality_l61_61825

theorem solve_inequality (a x : ℝ) : 
  (a < 0 → (x ≤ 3 / a ∨ x ≥ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a = 0 → (x ≥ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (0 < a ∧ a < 3 → (1 ≤ x ∧ x ≤ 3 / a) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a = 3 → (x = 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a > 3 → (3 / a ≤ x ∧ x ≤ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) :=
  sorry

end solve_inequality_l61_61825


namespace find_numbers_l61_61388

noncomputable def sum_nat (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem find_numbers : 
  {n : ℕ | ∃ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = sum_nat a b} = {14, 26, 37, 48, 59} :=
by {
  sorry
}

end find_numbers_l61_61388


namespace non_real_roots_b_range_l61_61941

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_l61_61941


namespace PropositionA_necessary_not_sufficient_l61_61931

variable (a : ℝ)

def PropositionA : Prop := a < 2
def PropositionB : Prop := a^2 < 4

theorem PropositionA_necessary_not_sufficient : 
  (PropositionA a → PropositionB a) ∧ ¬ (PropositionB a → PropositionA a) :=
sorry

end PropositionA_necessary_not_sufficient_l61_61931


namespace unaccounted_bottles_l61_61681

theorem unaccounted_bottles :
  let total_bottles := 254
  let football_bottles := 11 * 6
  let soccer_bottles := 53
  let lacrosse_bottles := football_bottles + 12
  let rugby_bottles := 49
  let team_bottles := football_bottles + soccer_bottles + lacrosse_bottles + rugby_bottles
  total_bottles - team_bottles = 8 :=
by
  rfl

end unaccounted_bottles_l61_61681


namespace certain_number_is_45_l61_61300

-- Define the variables and condition
def x : ℝ := 45
axiom h : x * 7 = 0.35 * 900

-- The statement we need to prove
theorem certain_number_is_45 : x = 45 :=
by
  sorry

end certain_number_is_45_l61_61300


namespace Sn_minimum_value_l61_61255

theorem Sn_minimum_value {a : ℕ → ℤ} (n : ℕ) (S : ℕ → ℤ)
  (h1 : a 1 = -11)
  (h2 : a 4 + a 6 = -6)
  (S_def : ∀ n, S n = n * (-12 + n)) :
  ∃ n, S n = S 6 :=
sorry

end Sn_minimum_value_l61_61255


namespace gain_percent_is_50_l61_61511

theorem gain_percent_is_50
  (C : ℕ) (S : ℕ) (hC : C = 10) (hS : S = 15) : ((S - C) / C : ℚ) * 100 = 50 := by
  sorry

end gain_percent_is_50_l61_61511


namespace find_analytical_expression_of_f_l61_61043

-- Define the function f satisfying the condition
def f (x : ℝ) : ℝ := sorry

-- Lean 4 theorem statement
theorem find_analytical_expression_of_f :
  (∀ x : ℝ, f (x + 1) = x^2 + 2*x + 2) → (∀ x : ℝ, f x = x^2 + 1) :=
by
  -- The initial f definition and theorem statement are created
  -- The proof is omitted since the focus is on translating the problem
  sorry

end find_analytical_expression_of_f_l61_61043


namespace arrow_sequence_correct_l61_61101

variable (A B C D E F G : ℕ)
variable (square : ℕ → ℕ)

-- Definitions based on given conditions
def conditions : Prop :=
  square 1 = 1 ∧ square 9 = 9 ∧
  square A = 6 ∧ square B = 2 ∧ square C = 4 ∧
  square D = 5 ∧ square E = 3 ∧ square F = 8 ∧ square G = 7 ∧
  (∀ x, (x = 1 → square 2 = B) ∧ (x = 2 → square 3 = E) ∧
       (x = 3 → square 4 = C) ∧ (x = 4 → square 5 = D) ∧
       (x = 5 → square 6 = A) ∧ (x = 6 → square 7 = G) ∧
       (x = 7 → square 8 = F) ∧ (x = 8 → square 9 = 9))

theorem arrow_sequence_correct :
  conditions A B C D E F G square → 
  ∀ x, square (x + 1) = 1 + x :=
by sorry

end arrow_sequence_correct_l61_61101


namespace stamp_distribution_correct_l61_61106

variables {W : ℕ} -- We use ℕ (natural numbers) for simplicity but this can be any type representing weight.

-- Number of envelopes that weigh less than W and need 2 stamps each
def envelopes_lt_W : ℕ := 6

-- Number of stamps per envelope if the envelope weighs less than W
def stamps_lt_W : ℕ := 2

-- Number of envelopes in total
def total_envelopes : ℕ := 14

-- Number of stamps for the envelopes that weigh less
def total_stamps_lt_W : ℕ := envelopes_lt_W * stamps_lt_W

-- Total stamps bought by Micah
def total_stamps_bought : ℕ := 52

-- Stamps left for envelopes that weigh more than W
def stamps_remaining : ℕ := total_stamps_bought - total_stamps_lt_W

-- Remaining envelopes that need stamps (those that weigh more than W)
def envelopes_gt_W : ℕ := total_envelopes - envelopes_lt_W

-- Number of stamps required per envelope that weighs more than W
def stamps_gt_W : ℕ := 5

-- Total stamps needed for the envelopes that weigh more than W
def total_stamps_needed_gt_W : ℕ := envelopes_gt_W * stamps_gt_W

theorem stamp_distribution_correct :
  total_stamps_bought = (total_stamps_lt_W + total_stamps_needed_gt_W) :=
by
  sorry

end stamp_distribution_correct_l61_61106


namespace proportion_solution_l61_61686

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 7 / 8) : x = 6 / 7 :=
by {
  sorry
}

end proportion_solution_l61_61686


namespace inner_rectangle_length_is_4_l61_61439

-- Define the conditions
def inner_rectangle_width : ℝ := 2
def shaded_region_width : ℝ := 2

-- Define the lengths and areas of the respective regions
def inner_rectangle_length (x : ℝ) : ℝ := x
def second_rectangle_dimensions (x : ℝ) : (ℝ × ℝ) := (x + 4, 6)
def largest_rectangle_dimensions (x : ℝ) : (ℝ × ℝ) := (x + 8, 10)

def inner_rectangle_area (x : ℝ) : ℝ := inner_rectangle_length x * inner_rectangle_width
def second_rectangle_area (x : ℝ) : ℝ := (second_rectangle_dimensions x).1 * (second_rectangle_dimensions x).2
def largest_rectangle_area (x : ℝ) : ℝ := (largest_rectangle_dimensions x).1 * (largest_rectangle_dimensions x).2

def first_shaded_region_area (x : ℝ) : ℝ := second_rectangle_area x - inner_rectangle_area x
def second_shaded_region_area (x : ℝ) : ℝ := largest_rectangle_area x - second_rectangle_area x

-- Define the arithmetic progression condition
def arithmetic_progression (x : ℝ) : Prop :=
  (first_shaded_region_area x - inner_rectangle_area x) = (second_shaded_region_area x - first_shaded_region_area x)

-- State the theorem
theorem inner_rectangle_length_is_4 :
  ∃ x : ℝ, arithmetic_progression x ∧ inner_rectangle_length x = 4 := 
by
  use 4
  -- Proof goes here
  sorry

end inner_rectangle_length_is_4_l61_61439


namespace plant_ways_count_l61_61196

theorem plant_ways_count :
  ∃ (solutions : Finset (Fin 7 → ℕ)), 
    (∀ x ∈ solutions, (x 0 + x 1 + x 2 + x 3 + x 4 + x 5 = 10) ∧ 
                       (100 * x 0 + 200 * x 1 + 300 * x 2 + 150 * x 3 + 125 * x 4 + 125 * x 5 = 2500)) ∧
    (solutions.card = 8) :=
sorry

end plant_ways_count_l61_61196


namespace minimum_value_of_f_l61_61944

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 1/x + 1/(x^2 + 1/x)

theorem minimum_value_of_f : 
  ∃ x > 0, f x = 2.5 :=
by 
  sorry

end minimum_value_of_f_l61_61944


namespace min_rows_needed_l61_61248

-- Define the basic conditions
def total_students := 2016
def seats_per_row := 168
def max_students_per_school := 40

-- Define the minimum number of rows required to accommodate all conditions
noncomputable def min_required_rows (students : ℕ) (seats : ℕ) (max_per_school : ℕ) : ℕ := 15

-- Lean theorem asserting the truth of the above definition under given conditions
theorem min_rows_needed : min_required_rows total_students seats_per_row max_students_per_school = 15 :=
by
  -- Proof omitted
  sorry

end min_rows_needed_l61_61248


namespace milkshake_cost_is_five_l61_61855

def initial_amount : ℝ := 132
def hamburger_cost : ℝ := 4
def num_hamburgers : ℕ := 8
def num_milkshakes : ℕ := 6
def amount_left : ℝ := 70

theorem milkshake_cost_is_five (M : ℝ) (h : initial_amount - (num_hamburgers * hamburger_cost + num_milkshakes * M) = amount_left) : 
  M = 5 :=
by
  sorry

end milkshake_cost_is_five_l61_61855


namespace smallest_number_of_integers_l61_61862

theorem smallest_number_of_integers (a b n : ℕ) 
  (h_avg_original : 89 * n = 73 * a + 111 * b) 
  (h_group_sum : a + b = n)
  (h_ratio : 8 * a = 11 * b) : 
  n = 19 :=
sorry

end smallest_number_of_integers_l61_61862


namespace lying_dwarf_possible_numbers_l61_61995

theorem lying_dwarf_possible_numbers (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
  (h2 : a_2 = a_1) 
  (h3 : a_3 = a_1 + a_2) 
  (h4 : a_4 = a_1 + a_2 + a_3) 
  (h5 : a_5 = a_1 + a_2 + a_3 + a_4) 
  (h6 : a_6 = a_1 + a_2 + a_3 + a_4 + a_5) 
  (h7 : a_7 = a_1 + a_2 + a_3 + a_4 + a_5 + a_6)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 58)
  (h_lied : ∃ (d : ℕ), d ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] ∧ (d ≠ a_1 ∧ d ≠ a_2 ∧ d ≠ a_3 ∧ d ≠ a_4 ∧ d ≠ a_5 ∧ d ≠ a_6 ∧ d ≠ a_7)) : 
  ∃ x : ℕ, x = 13 ∨ x = 26 :=
by
  sorry

end lying_dwarf_possible_numbers_l61_61995


namespace lampshire_parade_group_max_members_l61_61775

theorem lampshire_parade_group_max_members 
  (n : ℕ) 
  (h1 : 30 * n % 31 = 7)
  (h2 : 30 * n % 17 = 0)
  (h3 : 30 * n < 1500) :
  30 * n = 1020 :=
sorry

end lampshire_parade_group_max_members_l61_61775


namespace volume_of_cube_l61_61021

theorem volume_of_cube (d : ℝ) (h : d = 5 * Real.sqrt 3) : ∃ (V : ℝ), V = 125 := by
  sorry

end volume_of_cube_l61_61021


namespace combined_weight_of_emma_and_henry_l61_61654

variables (e f g h : ℕ)

theorem combined_weight_of_emma_and_henry 
  (h1 : e + f = 310)
  (h2 : f + g = 265)
  (h3 : g + h = 280) : e + h = 325 :=
by
  sorry

end combined_weight_of_emma_and_henry_l61_61654


namespace find_A_l61_61065

def heartsuit (A B : ℤ) : ℤ := 4 * A + A * B + 3 * B + 6

theorem find_A (A : ℤ) : heartsuit A 3 = 75 ↔ A = 60 / 7 := sorry

end find_A_l61_61065


namespace initial_percentage_of_alcohol_l61_61286

theorem initial_percentage_of_alcohol :
  ∃ P : ℝ, (P / 100 * 11) = (33 / 100 * 14) :=
by
  use 42
  sorry

end initial_percentage_of_alcohol_l61_61286


namespace vertical_asymptote_sum_l61_61619

theorem vertical_asymptote_sum :
  (∀ x : ℝ, 4*x^2 + 6*x + 3 = 0 → x = -1 / 2 ∨ x = -1) →
  (-1 / 2 + -1) = -3 / 2 :=
by
  intro h
  sorry

end vertical_asymptote_sum_l61_61619


namespace gcd_f_x_l61_61125

def f (x : ℤ) : ℤ := (5 * x + 3) * (11 * x + 2) * (14 * x + 7) * (3 * x + 8)

theorem gcd_f_x (x : ℤ) (hx : x % 3456 = 0) : Int.gcd (f x) x = 48 := by
  sorry

end gcd_f_x_l61_61125


namespace necessary_but_not_sufficient_l61_61644

variables (A B : Prop)

theorem necessary_but_not_sufficient 
  (h1 : ¬ B → ¬ A)  -- Condition: ¬ B → ¬ A is true
  (h2 : ¬ (¬ A → ¬ B))  -- Condition: ¬ A → ¬ B is false
  : (A → B) ∧ ¬ (B → A) := -- Conclusion: A → B and not (B → A)
by
  -- Proof is not required, so we place sorry
  sorry

end necessary_but_not_sufficient_l61_61644


namespace greatest_b_value_l61_61329

theorem greatest_b_value (b : ℝ) : 
  (-b^3 + b^2 + 7 * b - 10 ≥ 0) ↔ b ≤ 4 + Real.sqrt 6 :=
sorry

end greatest_b_value_l61_61329


namespace game_a_greater_than_game_c_l61_61520

-- Definitions of probabilities for heads and tails
def prob_heads : ℚ := 3 / 4
def prob_tails : ℚ := 1 / 4

-- Define the probabilities for Game A and Game C based on given conditions
def prob_game_a : ℚ := (prob_heads ^ 4) + (prob_tails ^ 4)
def prob_game_c : ℚ :=
  (prob_heads ^ 5) +
  (prob_tails ^ 5) +
  (prob_heads ^ 3 * prob_tails ^ 2) +
  (prob_tails ^ 3 * prob_heads ^ 2)

-- Define the difference
def prob_difference : ℚ := prob_game_a - prob_game_c

-- The theorem to be proved
theorem game_a_greater_than_game_c :
  prob_difference = 3 / 64 :=
by
  sorry

end game_a_greater_than_game_c_l61_61520


namespace solve_equation_1_solve_equation_2_l61_61910

theorem solve_equation_1 (x : ℝ) (h₁ : x - 4 = -5) : x = -1 :=
sorry

theorem solve_equation_2 (x : ℝ) (h₂ : (1/2) * x + 2 = 6) : x = 8 :=
sorry

end solve_equation_1_solve_equation_2_l61_61910


namespace fraction_equality_l61_61224

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 5 * b) / (b + 5 * a) = 2) : a / b = 3 / 5 :=
by
  sorry

end fraction_equality_l61_61224


namespace john_new_total_lifting_capacity_is_correct_l61_61147

def initial_clean_and_jerk : ℕ := 80
def initial_snatch : ℕ := 50

def new_clean_and_jerk : ℕ := 2 * initial_clean_and_jerk
def new_snatch : ℕ := initial_snatch + (initial_snatch * 8 / 10)

def new_combined_total_capacity : ℕ := new_clean_and_jerk + new_snatch

theorem john_new_total_lifting_capacity_is_correct : 
  new_combined_total_capacity = 250 := by
  sorry

end john_new_total_lifting_capacity_is_correct_l61_61147


namespace stratified_sampling_by_edu_stage_is_reasonable_l61_61495

variable (visionConditions : String → Type) -- visionConditions for different sampling methods
variable (primaryVision : Type) -- vision condition for primary school
variable (juniorVision : Type) -- vision condition for junior high school
variable (seniorVision : Type) -- vision condition for senior high school
variable (insignificantDiffGender : Prop) -- insignificant differences between boys and girls

-- Given conditions
variable (sigDiffEduStage : Prop) -- significant differences between educational stages

-- Stating the theorem
theorem stratified_sampling_by_edu_stage_is_reasonable (h1 : sigDiffEduStage) (h2 : insignificantDiffGender) : 
  visionConditions "Stratified_sampling_by_educational_stage" = visionConditions C :=
sorry

end stratified_sampling_by_edu_stage_is_reasonable_l61_61495


namespace inequality_solution_set_l61_61634

noncomputable def solution_set := {x : ℝ | x^2 + 2 * x - 3 ≥ 0}

theorem inequality_solution_set :
  (solution_set = {x : ℝ | x ≤ -3 ∨ x ≥ 1}) :=
sorry

end inequality_solution_set_l61_61634


namespace simplify_sqrt_90000_l61_61954

theorem simplify_sqrt_90000 : Real.sqrt 90000 = 300 :=
by
  /- Proof goes here -/
  sorry

end simplify_sqrt_90000_l61_61954


namespace solve_equation_and_find_c_d_l61_61533

theorem solve_equation_and_find_c_d : 
  ∃ (c d : ℕ), (∃ x : ℝ, x^2 + 14 * x = 84 ∧ x = Real.sqrt c - d) ∧ c + d = 140 := 
sorry

end solve_equation_and_find_c_d_l61_61533


namespace axis_of_symmetry_l61_61561

-- Definitions for conditions
variable (ω : ℝ) (φ : ℝ) (A B : ℝ)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- Hypotheses
axiom ω_pos : ω > 0
axiom φ_bound : 0 ≤ φ ∧ φ < Real.pi
axiom even_func : ∀ x, f x = f (-x)
axiom dist_AB : abs (B - A) = 4 * Real.sqrt 2

-- Proof statement
theorem axis_of_symmetry : ∃ x : ℝ, x = 4 := 
sorry

end axis_of_symmetry_l61_61561


namespace div_d_a_value_l61_61351

variable {a b c d : ℚ}

theorem div_d_a_value (h1 : a / b = 3) (h2 : b / c = 5 / 3) (h3 : c / d = 2) : d / a = 1 / 10 := by
  sorry

end div_d_a_value_l61_61351


namespace train_speed_l61_61976

def train_length : ℝ := 400  -- Length of the train in meters
def crossing_time : ℝ := 40  -- Time to cross the electric pole in seconds

theorem train_speed : train_length / crossing_time = 10 := by
  sorry  -- Proof to be completed

end train_speed_l61_61976


namespace complex_powers_l61_61864

theorem complex_powers (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^(23 : ℕ) + i^(58 : ℕ) = -1 - i :=
by sorry

end complex_powers_l61_61864


namespace isosceles_triangle_base_angle_l61_61252

theorem isosceles_triangle_base_angle (a b h θ : ℝ)
  (h1 : a^2 = 4 * b^2 * h)
  (h_b : b = 2 * a * Real.cos θ)
  (h_h : h = a * Real.sin θ) :
  θ = Real.arccos (1/4) :=
by
  sorry

end isosceles_triangle_base_angle_l61_61252


namespace books_read_l61_61400

-- Definitions
def total_books : ℕ := 13
def unread_books : ℕ := 4

-- Theorem
theorem books_read : total_books - unread_books = 9 :=
by
  sorry

end books_read_l61_61400


namespace distance_between_vertices_l61_61850

/-
Problem statement:
Prove that the distance between the vertices of the hyperbola
\(\frac{x^2}{144} - \frac{y^2}{64} = 1\) is 24.
-/

/-- 
We define the given hyperbola equation:
\frac{x^2}{144} - \frac{y^2}{64} = 1
-/
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 64 = 1

/--
We establish that the distance between the vertices of the hyperbola is 24.
-/
theorem distance_between_vertices : 
  (∀ x y : ℝ, hyperbola x y → dist (12, 0) (-12, 0) = 24) :=
by
  sorry

end distance_between_vertices_l61_61850


namespace sin_alpha_expression_l61_61399

theorem sin_alpha_expression (α : ℝ) 
  (h_tan : Real.tan α = -3 / 4) : 
  Real.sin α * (Real.sin α - Real.cos α) = 21 / 25 := 
sorry

end sin_alpha_expression_l61_61399


namespace value_of_expression_l61_61670

theorem value_of_expression : 1 + 2 / (1 + 2 / (2 * 2)) = 7 / 3 := 
by 
  -- proof to be filled in
  sorry

end value_of_expression_l61_61670


namespace boat_speed_in_still_water_l61_61652

variable (B S : ℝ)

theorem boat_speed_in_still_water :
  (B + S = 38) ∧ (B - S = 16) → B = 27 :=
by
  sorry

end boat_speed_in_still_water_l61_61652


namespace b_ne_d_l61_61604

-- Conditions
def P (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def Q (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

def PQ_eq_QP_no_real_roots (a b c d : ℝ) : Prop := 
  ∀ (x : ℝ), P (Q x c d) a b ≠ Q (P x a b) c d

-- Goal
theorem b_ne_d (a b c d : ℝ) (h : PQ_eq_QP_no_real_roots a b c d) : b ≠ d := 
sorry

end b_ne_d_l61_61604


namespace exponent_division_l61_61553

variable (a : ℝ) (m n : ℝ)
-- Conditions
def condition1 : Prop := a^m = 2
def condition2 : Prop := a^n = 16

-- Theorem Statement
theorem exponent_division (h1 : condition1 a m) (h2 : condition2 a n) : a^(m - n) = 1 / 8 := by
  sorry

end exponent_division_l61_61553


namespace square_area_multiplier_l61_61499

theorem square_area_multiplier 
  (perimeter_square : ℝ) (length_rectangle : ℝ) (width_rectangle : ℝ)
  (perimeter_square_eq : perimeter_square = 800) 
  (length_rectangle_eq : length_rectangle = 125) 
  (width_rectangle_eq : width_rectangle = 64)
  : (perimeter_square / 4) ^ 2 / (length_rectangle * width_rectangle) = 5 := 
by
  sorry

end square_area_multiplier_l61_61499


namespace find_a_l61_61467

def possible_scores : List ℕ := [103, 104, 105, 106, 107, 108, 109, 110]

def is_possible_score (a : ℕ) (n : ℕ) : Prop :=
  ∃ (k8 k0 ka : ℕ), k8 * 8 + ka * a + k0 * 0 = n

def is_impossible_score (a : ℕ) (n : ℕ) : Prop :=
  ¬ is_possible_score a n

theorem find_a : ∀ (a : ℕ), a ≠ 0 → a ≠ 8 →
  (∀ n ∈ possible_scores, is_possible_score a n) →
  is_impossible_score a 83 →
  a = 13 := by
  intros a ha1 ha2 hpossible himpossible
  sorry

end find_a_l61_61467


namespace janet_saves_time_l61_61365

theorem janet_saves_time (looking_time_per_day : ℕ := 8) (complaining_time_per_day : ℕ := 3) (days_per_week : ℕ := 7) :
  (looking_time_per_day + complaining_time_per_day) * days_per_week = 77 := 
sorry

end janet_saves_time_l61_61365


namespace neither_rain_nor_snow_l61_61576

theorem neither_rain_nor_snow 
  (p_rain : ℚ)
  (p_snow : ℚ)
  (independent : Prop) 
  (h_rain : p_rain = 4/10)
  (h_snow : p_snow = 1/5)
  (h_independent : independent)
  : (1 - p_rain) * (1 - p_snow) = 12 / 25 := 
by
  sorry

end neither_rain_nor_snow_l61_61576


namespace alcohol_percentage_after_adding_water_l61_61105

variables (initial_volume : ℕ) (initial_percentage : ℕ) (added_volume : ℕ)
def initial_alcohol_volume := initial_volume * initial_percentage / 100
def final_volume := initial_volume + added_volume
def final_percentage := initial_alcohol_volume * 100 / final_volume

theorem alcohol_percentage_after_adding_water :
  initial_volume = 15 →
  initial_percentage = 20 →
  added_volume = 5 →
  final_percentage = 15 := by
sorry

end alcohol_percentage_after_adding_water_l61_61105


namespace tournament_key_player_l61_61656

theorem tournament_key_player (n : ℕ) (plays : Fin n → Fin n → Bool) (wins : ∀ i j, plays i j → ¬plays j i) :
  ∃ X, ∀ (Y : Fin n), Y ≠ X → (plays X Y ∨ ∃ Z, plays X Z ∧ plays Z Y) :=
by
  sorry

end tournament_key_player_l61_61656


namespace lisa_need_add_pure_juice_l61_61491

theorem lisa_need_add_pure_juice
  (x : ℝ) 
  (total_volume : ℝ := 2)
  (initial_pure_juice_fraction : ℝ := 0.10)
  (desired_pure_juice_fraction : ℝ := 0.25) 
  (added_pure_juice : ℝ := x) 
  (initial_pure_juice_amount : ℝ := total_volume * initial_pure_juice_fraction)
  (final_pure_juice_amount : ℝ := initial_pure_juice_amount + added_pure_juice)
  (final_volume : ℝ := total_volume + added_pure_juice) :
  (final_pure_juice_amount / final_volume) = desired_pure_juice_fraction → x = 0.4 :=
by
  intro h
  sorry

end lisa_need_add_pure_juice_l61_61491


namespace persons_in_first_group_l61_61441

-- Define the given conditions
def first_group_work_done (P : ℕ) : ℕ := P * 12 * 10
def second_group_work_done : ℕ := 30 * 26 * 6

-- Define the proof problem statement
theorem persons_in_first_group (P : ℕ) (h : first_group_work_done P = second_group_work_done) : P = 39 :=
by
  unfold first_group_work_done second_group_work_done at h
  sorry

end persons_in_first_group_l61_61441


namespace max_areas_in_disk_l61_61942

noncomputable def max_non_overlapping_areas (n : ℕ) : ℕ := 5 * n + 1

theorem max_areas_in_disk (n : ℕ) : 
  let disk_divided_by_2n_radii_and_two_secant_lines_areas  := (5 * n + 1)
  disk_divided_by_2n_radii_and_two_secant_lines_areas = max_non_overlapping_areas n := by sorry

end max_areas_in_disk_l61_61942


namespace servings_in_box_l61_61089

def totalCereal : ℕ := 18
def servingSize : ℕ := 2

theorem servings_in_box : totalCereal / servingSize = 9 := by
  sorry

end servings_in_box_l61_61089


namespace apples_shared_l61_61791

-- Definitions and conditions based on problem statement
def initial_apples : ℕ := 89
def remaining_apples : ℕ := 84

-- The goal to prove that Ruth shared 5 apples with Peter
theorem apples_shared : initial_apples - remaining_apples = 5 := by
  sorry

end apples_shared_l61_61791


namespace payment_proof_l61_61918

theorem payment_proof (X Y : ℝ) 
  (h₁ : X + Y = 572) 
  (h₂ : X = 1.20 * Y) 
  : Y = 260 := 
by 
  sorry

end payment_proof_l61_61918


namespace B_work_rate_l61_61564

theorem B_work_rate (A_rate C_rate combined_rate : ℝ) (B_days : ℝ) (hA : A_rate = 1 / 4) (hC : C_rate = 1 / 8) (hCombined : A_rate + 1 / B_days + C_rate = 1 / 2) : B_days = 8 :=
by
  sorry

end B_work_rate_l61_61564


namespace determine_a_l61_61389

def quadratic_condition (a : ℝ) (x : ℝ) : Prop := 
  abs (x^2 + 2 * a * x + 3 * a) ≤ 2

theorem determine_a : {a : ℝ | ∃! x : ℝ, quadratic_condition a x} = {1, 2} :=
sorry

end determine_a_l61_61389


namespace scouts_attended_l61_61912

def chocolate_bar_cost : ℝ := 1.50
def total_spent : ℝ := 15
def sections_per_bar : ℕ := 3
def smores_per_scout : ℕ := 2

theorem scouts_attended (bars : ℝ) (sections : ℕ) (smores : ℕ) (scouts : ℕ) :
  bars = total_spent / chocolate_bar_cost →
  sections = bars * sections_per_bar →
  smores = sections →
  scouts = smores / smores_per_scout →
  scouts = 15 :=
by
  intro h1 h2 h3 h4
  sorry

end scouts_attended_l61_61912


namespace nine_cubed_expansion_l61_61871

theorem nine_cubed_expansion : 9^3 + 3 * 9^2 + 3 * 9 + 1 = 1000 := 
by 
  sorry

end nine_cubed_expansion_l61_61871


namespace inequality_proof_l61_61847

noncomputable def sum_expression (a b c : ℝ) : ℝ :=
  (1 / (b * c + a + 1 / a)) + (1 / (c * a + b + 1 / b)) + (1 / (a * b + c + 1 / c))

theorem inequality_proof (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  sum_expression a b c ≤ 27 / 31 :=
by
  sorry

end inequality_proof_l61_61847


namespace sheilas_hours_mwf_is_24_l61_61177

-- Define Sheila's earning conditions and working hours
def sheilas_hours_mwf (H : ℕ) : Prop :=
  let hours_tu_th := 6 * 2
  let earnings_tu_th := hours_tu_th * 14
  let earnings_mwf := 504 - earnings_tu_th
  H = earnings_mwf / 14

-- The theorem to state that Sheila works 24 hours on Monday, Wednesday, and Friday
theorem sheilas_hours_mwf_is_24 : sheilas_hours_mwf 24 :=
by
  -- Proof is omitted
  sorry

end sheilas_hours_mwf_is_24_l61_61177


namespace bullet_speed_difference_l61_61878

theorem bullet_speed_difference (speed_horse speed_bullet : ℕ) 
    (h_horse : speed_horse = 20) (h_bullet : speed_bullet = 400) :
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    speed_same_direction - speed_opposite_direction = 40 :=
    by
    -- Define the speeds in terms of the given conditions.
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    -- State the equality to prove.
    show speed_same_direction - speed_opposite_direction = 40;
    -- Proof (skipped here).
    -- sorry is used to denote where the formal proof steps would go.
    sorry

end bullet_speed_difference_l61_61878


namespace specificTriangle_perimeter_l61_61830

-- Assume a type to represent triangle sides
structure IsoscelesTriangle (a b : ℕ) : Prop :=
  (equal_sides : a = b ∨ a + b > max a b)

-- Define the condition where we have specific sides
def specificTriangle : Prop :=
  IsoscelesTriangle 5 2

-- Prove that given the specific sides, the perimeter is 12
theorem specificTriangle_perimeter : specificTriangle → 5 + 5 + 2 = 12 :=
by
  intro h
  cases h
  sorry

end specificTriangle_perimeter_l61_61830


namespace cost_to_marked_price_ratio_l61_61935

variables (p : ℝ) (discount : ℝ := 0.20) (cost_ratio : ℝ := 0.60)

theorem cost_to_marked_price_ratio :
  (cost_ratio * (1 - discount) * p) / p = 0.48 :=
by sorry

end cost_to_marked_price_ratio_l61_61935


namespace total_stickers_l61_61589

-- Definitions for the given conditions
def stickers_per_page : ℕ := 10
def number_of_pages : ℕ := 22

-- The theorem to be proven
theorem total_stickers : stickers_per_page * number_of_pages = 220 := by
  sorry

end total_stickers_l61_61589


namespace line_circle_interaction_l61_61404

theorem line_circle_interaction (a : ℝ) :
  let r := 10
  let d := |a| / 5
  let intersects := -50 < a ∧ a < 50 
  let tangent := a = 50 ∨ a = -50 
  let separate := a < -50 ∨ a > 50 
  (d < r ↔ intersects) ∧ (d = r ↔ tangent) ∧ (d > r ↔ separate) :=
by sorry

end line_circle_interaction_l61_61404


namespace school_club_profit_l61_61879

def price_per_bar_buy : ℚ := 5 / 6
def price_per_bar_sell : ℚ := 2 / 3
def total_bars : ℕ := 1200
def total_cost : ℚ := total_bars * price_per_bar_buy
def total_revenue : ℚ := total_bars * price_per_bar_sell
def profit : ℚ := total_revenue - total_cost

theorem school_club_profit : profit = -200 := by
  sorry

end school_club_profit_l61_61879


namespace binom_15_4_l61_61620

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l61_61620


namespace distance_dormitory_to_city_l61_61768

variable (D : ℝ)
variable (c : ℝ := 12)
variable (f := (1/5) * D)
variable (b := (2/3) * D)

theorem distance_dormitory_to_city (h : f + b + c = D) : D = 90 := by
  sorry

end distance_dormitory_to_city_l61_61768


namespace area_to_be_painted_l61_61840

variable (h_wall : ℕ) (l_wall : ℕ)
variable (h_window : ℕ) (l_window : ℕ)
variable (h_door : ℕ) (l_door : ℕ)

theorem area_to_be_painted :
  ∀ (h_wall : ℕ) (l_wall : ℕ) (h_window : ℕ) (l_window : ℕ) (h_door : ℕ) (l_door : ℕ),
  h_wall = 10 → l_wall = 15 →
  h_window = 3 → l_window = 5 →
  h_door = 2 → l_door = 3 →
  (h_wall * l_wall) - ((h_window * l_window) + (h_door * l_door)) = 129 :=
by
  intros
  sorry

end area_to_be_painted_l61_61840


namespace stone_breadth_5_l61_61668

theorem stone_breadth_5 (hall_length_m hall_breadth_m stone_length_dm num_stones b₁ b₂ : ℝ) 
  (h1 : hall_length_m = 36) 
  (h2 : hall_breadth_m = 15) 
  (h3 : stone_length_dm = 3) 
  (h4 : num_stones = 3600)
  (h5 : hall_length_m * 10 * hall_breadth_m * 10 = 54000)
  (h6 : stone_length_dm * b₁ * num_stones = hall_length_m * 10 * hall_breadth_m * 10) :
  b₂ = 5 := 
  sorry

end stone_breadth_5_l61_61668


namespace unit_trip_to_expo_l61_61792

theorem unit_trip_to_expo (n : ℕ) (cost : ℕ) (total_cost : ℕ) :
  (n ≤ 30 → cost = 120) ∧ 
  (n > 30 → cost = 120 - 2 * (n - 30) ∧ cost ≥ 90) →
  (total_cost = 4000) →
  (total_cost = n * cost) →
  n = 40 :=
by
  sorry

end unit_trip_to_expo_l61_61792


namespace words_difference_l61_61305

-- Definitions based on conditions.
def right_hand_speed (words_per_minute : ℕ) := 10
def left_hand_speed (words_per_minute : ℕ) := 7
def time_duration (minutes : ℕ) := 5

-- Problem statement
theorem words_difference :
  let right_hand_words := right_hand_speed 0 * time_duration 0
  let left_hand_words := left_hand_speed 0 * time_duration 0
  (right_hand_words - left_hand_words) = 15 :=
by
  sorry

end words_difference_l61_61305


namespace system_consistent_and_solution_l61_61214

theorem system_consistent_and_solution (a x : ℝ) : 
  (a = -10 ∧ x = -1/3) ∨ (a = -8 ∧ x = -1) ∨ (a = 4 ∧ x = -2) ↔ 
  3 * x^2 - x - a - 10 = 0 ∧ (a + 4) * x + a + 12 = 0 := by
  sorry

end system_consistent_and_solution_l61_61214


namespace sum_of_values_l61_61136

theorem sum_of_values (N : ℝ) (R : ℝ) (h : N ≠ 0) (h_eq : N + 5 / N = R) : N = R := 
sorry

end sum_of_values_l61_61136


namespace average_speed_monday_to_wednesday_l61_61767

theorem average_speed_monday_to_wednesday :
  ∃ x : ℝ, (∀ (total_hours total_distance thursday_friday_distance : ℝ),
    total_hours = 2 * 5 ∧
    thursday_friday_distance = 9 * 2 * 2 ∧
    total_distance = 108 ∧
    total_distance - thursday_friday_distance = x * (2 * 3))
    → x = 12 :=
sorry

end average_speed_monday_to_wednesday_l61_61767


namespace last_digit_1993_2002_plus_1995_2002_l61_61081

theorem last_digit_1993_2002_plus_1995_2002 :
  (1993 ^ 2002 + 1995 ^ 2002) % 10 = 4 :=
by sorry

end last_digit_1993_2002_plus_1995_2002_l61_61081


namespace total_amount_l61_61489

-- Conditions as given definitions
def ratio_a : Nat := 2
def ratio_b : Nat := 3
def ratio_c : Nat := 4
def share_b : Nat := 1500

-- The final statement
theorem total_amount (parts_b := 3) (one_part := share_b / parts_b) :
  (2 * one_part) + (3 * one_part) + (4 * one_part) = 4500 :=
by
  sorry

end total_amount_l61_61489


namespace sum_of_a_b_l61_61431

theorem sum_of_a_b (a b : ℝ) (h₁ : a^3 - 3 * a^2 + 5 * a = 1) (h₂ : b^3 - 3 * b^2 + 5 * b = 5) : a + b = 2 :=
sorry

end sum_of_a_b_l61_61431


namespace number_proportion_l61_61171

theorem number_proportion (number : ℚ) :
  (number : ℚ) / 12 = 9 / 360 →
  number = 0.3 :=
by
  intro h
  sorry

end number_proportion_l61_61171


namespace range_of_a_l61_61755

variable (a : ℝ)

def discriminant (a : ℝ) : ℝ := 4 * a ^ 2 - 12

theorem range_of_a
  (h : discriminant a > 0) :
  a < -Real.sqrt 3 ∨ a > Real.sqrt 3 :=
sorry

end range_of_a_l61_61755


namespace arithmetic_sequence_S30_l61_61363

theorem arithmetic_sequence_S30
  (S : ℕ → ℕ)
  (h_arith_seq: ∀ m : ℕ, 2 * (S (2 * m) - S m) = S m + S (3 * m) - S (2 * m))
  (h_S10: S 10 = 4)
  (h_S20: S 20 = 20) :
  S 30 = 48 := 
by
  sorry

end arithmetic_sequence_S30_l61_61363


namespace eggs_per_chicken_per_day_l61_61978

-- Define the conditions
def chickens : ℕ := 8
def price_per_dozen : ℕ := 5
def total_revenue : ℕ := 280
def weeks : ℕ := 4
def eggs_per_dozen : ℕ := 12
def days_per_week : ℕ := 7

-- Theorem statement on how many eggs each chicken lays per day
theorem eggs_per_chicken_per_day :
  (chickens * ((total_revenue / price_per_dozen * eggs_per_dozen) / (weeks * days_per_week))) / chickens = 3 :=
by
  sorry

end eggs_per_chicken_per_day_l61_61978


namespace total_foreign_objects_l61_61407

-- Definitions based on the conditions
def burrs := 12
def ticks := 6 * burrs

-- Theorem to prove the total number of foreign objects
theorem total_foreign_objects : burrs + ticks = 84 :=
by
  sorry -- Proof omitted

end total_foreign_objects_l61_61407


namespace const_seq_is_arithmetic_not_geometric_l61_61607

-- Define the sequence
def const_seq (n : ℕ) : ℕ := 0

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, seq (n + 1) = seq n + d

-- Define what it means for a sequence to be geometric
def is_geometric_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = seq n * r

-- The proof statement
theorem const_seq_is_arithmetic_not_geometric :
  is_arithmetic_sequence const_seq ∧ ¬ is_geometric_sequence const_seq :=
by
  sorry

end const_seq_is_arithmetic_not_geometric_l61_61607


namespace multiple_choice_options_l61_61358

-- Define the problem conditions
def num_true_false_combinations : ℕ := 14
def num_possible_keys (n : ℕ) : ℕ := num_true_false_combinations * n^2
def total_keys : ℕ := 224

-- The theorem problem
theorem multiple_choice_options : ∃ n : ℕ, num_possible_keys n = total_keys ∧ n = 4 := by
  -- We don't need to provide the proof, so we use sorry. 
  sorry

end multiple_choice_options_l61_61358


namespace sample_size_proof_l61_61560

theorem sample_size_proof (p : ℝ) (N : ℤ) (n : ℤ) (h1 : N = 200) (h2 : p = 0.25) : n = 50 :=
by
  sorry

end sample_size_proof_l61_61560


namespace initial_money_jennifer_l61_61863

theorem initial_money_jennifer (M : ℝ) (h1 : (1/5) * M + (1/6) * M + (1/2) * M + 12 = M) : M = 90 :=
sorry

end initial_money_jennifer_l61_61863


namespace pedro_furniture_area_l61_61349

theorem pedro_furniture_area :
  let width : ℝ := 2
  let length : ℝ := 2.5
  let door_arc_area := (1 / 4) * Real.pi * (0.5 ^ 2)
  let window_arc_area := 2 * (1 / 2) * Real.pi * (0.5 ^ 2)
  let room_area := width * length
  room_area - door_arc_area - window_arc_area = (80 - 9 * Real.pi) / 16 := 
by
  sorry

end pedro_furniture_area_l61_61349


namespace total_roses_planted_three_days_l61_61845

-- Definitions based on conditions
def susan_roses_two_days_ago : ℕ := 10
def maria_roses_two_days_ago : ℕ := 2 * susan_roses_two_days_ago
def john_roses_two_days_ago : ℕ := susan_roses_two_days_ago + 10
def roses_two_days_ago : ℕ := susan_roses_two_days_ago + maria_roses_two_days_ago + john_roses_two_days_ago

def roses_yesterday : ℕ := roses_two_days_ago + 20
def susan_roses_yesterday : ℕ := susan_roses_two_days_ago * roses_yesterday / roses_two_days_ago
def maria_roses_yesterday : ℕ := maria_roses_two_days_ago * roses_yesterday / roses_two_days_ago
def john_roses_yesterday : ℕ := john_roses_two_days_ago * roses_yesterday / roses_two_days_ago

def roses_today : ℕ := 2 * roses_two_days_ago
def susan_roses_today : ℕ := susan_roses_two_days_ago
def maria_roses_today : ℕ := maria_roses_two_days_ago + (maria_roses_two_days_ago * 25 / 100)
def john_roses_today : ℕ := john_roses_two_days_ago - (john_roses_two_days_ago * 10 / 100)

def total_roses_planted : ℕ := 
  (susan_roses_two_days_ago + maria_roses_two_days_ago + john_roses_two_days_ago) +
  (susan_roses_yesterday + maria_roses_yesterday + john_roses_yesterday) +
  (susan_roses_today + maria_roses_today + john_roses_today)

-- The statement that needs to be proved
theorem total_roses_planted_three_days : total_roses_planted = 173 := by 
  sorry

end total_roses_planted_three_days_l61_61845


namespace arithmetic_sequence_general_formula_l61_61398

noncomputable def arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n, a n = 2 * n - 3

theorem arithmetic_sequence_general_formula
    (a : ℕ → ℤ)
    (h1 : (a 2 + a 6) / 2 = 5)
    (h2 : (a 3 + a 7) / 2 = 7) :
  arithmetic_sequence a :=
by
  sorry

end arithmetic_sequence_general_formula_l61_61398


namespace tank_length_is_25_l61_61395

noncomputable def cost_to_paise (cost_in_rupees : ℕ) : ℕ :=
  cost_in_rupees * 100

noncomputable def total_area_plastered (total_cost_in_paise : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  total_cost_in_paise / cost_per_sq_m

noncomputable def length_of_tank (width height cost_in_rupees rate : ℕ) : ℕ :=
  let total_cost_in_paise := cost_to_paise cost_in_rupees
  let total_area := total_area_plastered total_cost_in_paise rate
  let area_eq := total_area = (2 * (height * width) + 2 * (6 * height) + (height * width))
  let simplified_eq := total_area - 144 = 24 * height
  (total_area - 144) / 24

theorem tank_length_is_25 (width height cost_in_rupees rate : ℕ) : 
  width = 12 → height = 6 → cost_in_rupees = 186 → rate = 25 → length_of_tank width height cost_in_rupees rate = 25 :=
  by
    intros hwidth hheight hcost hrate
    unfold length_of_tank
    rw [hwidth, hheight, hcost, hrate]
    simp
    sorry

end tank_length_is_25_l61_61395


namespace even_function_iff_b_zero_l61_61148

theorem even_function_iff_b_zero (b c : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + c) = ((-x)^2 + b * (-x) + c)) ↔ b = 0 :=
by
  sorry

end even_function_iff_b_zero_l61_61148


namespace prism_volume_l61_61396

theorem prism_volume (a b c : ℝ) (h1 : a * b = 45) (h2 : b * c = 49) (h3 : a * c = 56) : a * b * c = 1470 := by
  sorry

end prism_volume_l61_61396


namespace shelves_in_room_l61_61555

theorem shelves_in_room
  (n_action_figures_per_shelf : ℕ)
  (total_action_figures : ℕ)
  (h1 : n_action_figures_per_shelf = 10)
  (h2 : total_action_figures = 80) :
  total_action_figures / n_action_figures_per_shelf = 8 := by
  sorry

end shelves_in_room_l61_61555


namespace wax_current_amount_l61_61158

theorem wax_current_amount (wax_needed wax_total : ℕ) (h : wax_needed + 11 = wax_total) : 11 = wax_total - wax_needed :=
by
  sorry

end wax_current_amount_l61_61158


namespace sum_of_731_and_one_fifth_l61_61005

theorem sum_of_731_and_one_fifth :
  (7.31 + (1 / 5) = 7.51) :=
sorry

end sum_of_731_and_one_fifth_l61_61005


namespace biased_coin_prob_three_heads_l61_61744

def prob_heads := 1/3

theorem biased_coin_prob_three_heads : prob_heads^3 = 1/27 :=
by
  sorry

end biased_coin_prob_three_heads_l61_61744


namespace dividend_rate_of_stock_l61_61782

variable (MarketPrice : ℝ) (YieldPercent : ℝ) (DividendPercent : ℝ)
variable (NominalValue : ℝ) (AnnualDividend : ℝ)

def stock_dividend_rate_condition (YieldPercent MarketPrice NominalValue DividendPercent : ℝ) 
  (AnnualDividend : ℝ) : Prop :=
  YieldPercent = 20 ∧ MarketPrice = 125 ∧ DividendPercent = 0.25 ∧ NominalValue = 100 ∧
  AnnualDividend = (YieldPercent / 100) * MarketPrice

theorem dividend_rate_of_stock :
  stock_dividend_rate_condition 20 125 100 0.25 25 → (DividendPercent * NominalValue) = 25 :=
by 
  sorry

end dividend_rate_of_stock_l61_61782


namespace upper_bound_expression_4n_plus_7_l61_61579

theorem upper_bound_expression_4n_plus_7 (U : ℤ) :
  (∃ (n : ℕ),  4 * n + 7 > 1) ∧
  (∀ (n : ℕ), 4 * n + 7 < U → ∃ (k : ℕ), k ≤ 19 ∧ k = n) ∧
  (∃ (n_min n_max : ℕ), n_max = n_min + 19 ∧ 4 * n_max + 7 < U) →
  U = 84 := sorry

end upper_bound_expression_4n_plus_7_l61_61579


namespace initialPersonsCount_l61_61283

noncomputable def numberOfPersonsInitially (increaseInAverageWeight kg_diff : ℝ) : ℝ :=
  kg_diff / increaseInAverageWeight

theorem initialPersonsCount :
  numberOfPersonsInitially 2.5 20 = 8 := by
  sorry

end initialPersonsCount_l61_61283


namespace product_of_points_l61_61823

def f (n : ℕ) : ℕ :=
  if n % 6 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 1

def allie_rolls := [5, 6, 1, 2, 3]
def betty_rolls := [6, 1, 1, 2, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.foldl (fun acc n => acc + f n) 0

theorem product_of_points :
  total_points allie_rolls * total_points betty_rolls = 169 :=
by
  sorry

end product_of_points_l61_61823


namespace tommy_nickels_l61_61025

-- Definitions of given conditions
def pennies (quarters : Nat) : Nat := 10 * quarters  -- Tommy has 10 times as many pennies as quarters
def dimes (pennies : Nat) : Nat := pennies + 10      -- Tommy has 10 more dimes than pennies
def nickels (dimes : Nat) : Nat := 2 * dimes         -- Tommy has twice as many nickels as dimes

theorem tommy_nickels (quarters : Nat) (P : Nat) (D : Nat) (N : Nat) 
  (h1 : quarters = 4) 
  (h2 : P = pennies quarters) 
  (h3 : D = dimes P) 
  (h4 : N = nickels D) : 
  N = 100 := 
by
  -- sorry allows us to skip the proof
  sorry

end tommy_nickels_l61_61025


namespace farmer_planting_problem_l61_61885

theorem farmer_planting_problem (total_acres : ℕ) (flax_acres : ℕ) (sunflower_acres : ℕ)
  (h1 : total_acres = 240)
  (h2 : flax_acres = 80)
  (h3 : sunflower_acres = total_acres - flax_acres) :
  sunflower_acres - flax_acres = 80 := by
  sorry

end farmer_planting_problem_l61_61885


namespace incorrect_vertex_is_false_l61_61777

-- Definition of the given parabola
def parabola (x : ℝ) : ℝ := -2 * (x - 2)^2 + 1

-- Define the incorrect hypothesis: Vertex at (-2, 1)
def incorrect_vertex (x y : ℝ) : Prop := (x, y) = (-2, 1)

-- Proposition to prove that the vertex is not at (-2, 1)
theorem incorrect_vertex_is_false : ¬ ∃ x y, (x, y) = (-2, 1) ∧ parabola x = y :=
by
  sorry

end incorrect_vertex_is_false_l61_61777


namespace average_price_per_book_l61_61038

-- Definitions of the conditions
def books_shop1 := 65
def cost_shop1 := 1480
def books_shop2 := 55
def cost_shop2 := 920

-- Definition of total values
def total_books := books_shop1 + books_shop2
def total_cost := cost_shop1 + cost_shop2

-- Proof statement
theorem average_price_per_book : (total_cost / total_books) = 20 := by
  sorry

end average_price_per_book_l61_61038


namespace find_multiple_l61_61793

-- Define the conditions
def ReetaPencils : ℕ := 20
def TotalPencils : ℕ := 64

-- Define the question and proof statement
theorem find_multiple (AnikaPencils : ℕ) (M : ℕ) :
  AnikaPencils = ReetaPencils * M + 4 →
  AnikaPencils + ReetaPencils = TotalPencils →
  M = 2 :=
by
  intros hAnika hTotal
  -- Skip the proof
  sorry

end find_multiple_l61_61793


namespace option_B_coplanar_l61_61877

-- Define the three vectors in Option B.
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (-2, -4, 6)
def c : ℝ × ℝ × ℝ := (1, 0, 5)

-- Define the coplanarity condition for vectors a, b, and c.
def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = k • a

-- Prove that the vectors in Option B are coplanar.
theorem option_B_coplanar : coplanar a b c :=
sorry

end option_B_coplanar_l61_61877


namespace parallel_x_axis_implies_conditions_l61_61029

variable (a b : ℝ)

theorem parallel_x_axis_implies_conditions (h1 : (5, a) ≠ (b, -2)) (h2 : (5, -2) = (5, a)) : a = -2 ∧ b ≠ 5 :=
sorry

end parallel_x_axis_implies_conditions_l61_61029


namespace find_constants_a_b_l61_61900

variables (x a b : ℝ)

theorem find_constants_a_b (h : (x - a) / (x + b) = (x^2 - 45 * x + 504) / (x^2 + 66 * x - 1080)) :
  a + b = 48 :=
sorry

end find_constants_a_b_l61_61900


namespace simplify_expression_correct_l61_61370

noncomputable def simplify_expression : ℝ :=
  2 - 2 / (2 + Real.sqrt 5) - 2 / (2 - Real.sqrt 5)

theorem simplify_expression_correct : simplify_expression = 10 := by
  sorry

end simplify_expression_correct_l61_61370


namespace domain_of_composite_function_l61_61450

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x → x ≤ 2 → f x = f x) →
  (∀ (x : ℝ), -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 → f (x^2) = f (x^2)) :=
by
  sorry

end domain_of_composite_function_l61_61450


namespace number_of_cars_lifted_l61_61841

def total_cars_lifted : ℕ := 6

theorem number_of_cars_lifted : total_cars_lifted = 6 := by
  sorry

end number_of_cars_lifted_l61_61841


namespace mean_visits_between_200_and_300_l61_61292

def monday_visits := 300
def tuesday_visits := 400
def wednesday_visits := 300
def thursday_visits := 200
def friday_visits := 200

def total_visits := monday_visits + tuesday_visits + wednesday_visits + thursday_visits + friday_visits
def number_of_days := 5
def mean_visits_per_day := total_visits / number_of_days

theorem mean_visits_between_200_and_300 : 200 ≤ mean_visits_per_day ∧ mean_visits_per_day ≤ 300 :=
by sorry

end mean_visits_between_200_and_300_l61_61292


namespace number_of_undeveloped_sections_l61_61072

def undeveloped_sections (total_area section_area : ℕ) : ℕ :=
  total_area / section_area

theorem number_of_undeveloped_sections :
  undeveloped_sections 7305 2435 = 3 :=
by
  unfold undeveloped_sections
  exact rfl

end number_of_undeveloped_sections_l61_61072


namespace total_bouquets_sold_l61_61530

-- Define the conditions as variables
def monday_bouquets : ℕ := 12
def tuesday_bouquets : ℕ := 3 * monday_bouquets
def wednesday_bouquets : ℕ := tuesday_bouquets / 3

-- The statement to prove
theorem total_bouquets_sold : 
  monday_bouquets + tuesday_bouquets + wednesday_bouquets = 60 :=
by
  -- The proof is omitted using sorry
  sorry

end total_bouquets_sold_l61_61530


namespace total_number_of_subjects_l61_61015

-- Definitions from conditions
def average_marks_5_subjects (total_marks : ℕ) : Prop :=
  74 * 5 = total_marks

def marks_in_last_subject (marks : ℕ) : Prop :=
  marks = 74

def total_average_marks (n : ℕ) (total_marks : ℕ) : Prop :=
  74 * n = total_marks

-- Lean 4 statement
theorem total_number_of_subjects (n total_marks total_marks_5 last_subject_marks : ℕ)
  (h1 : total_average_marks n total_marks)
  (h2 : average_marks_5_subjects total_marks_5)
  (h3 : marks_in_last_subject last_subject_marks)
  (h4 : total_marks = total_marks_5 + last_subject_marks) :
  n = 6 :=
sorry

end total_number_of_subjects_l61_61015


namespace correct_option_C_l61_61566

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem correct_option_C : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → x1 * f x1 < x2 * f x2 :=
by
  intro x1 x2 hx1 hx12
  sorry

end correct_option_C_l61_61566


namespace uncle_jerry_total_tomatoes_l61_61213

def tomatoes_reaped_yesterday : ℕ := 120
def tomatoes_reaped_more_today : ℕ := 50

theorem uncle_jerry_total_tomatoes : 
  tomatoes_reaped_yesterday + (tomatoes_reaped_yesterday + tomatoes_reaped_more_today) = 290 :=
by 
  sorry

end uncle_jerry_total_tomatoes_l61_61213


namespace total_oil_leakage_l61_61951

def oil_leaked_before : ℕ := 6522
def oil_leaked_during : ℕ := 5165
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem total_oil_leakage : total_oil_leaked = 11687 := by
  sorry

end total_oil_leakage_l61_61951


namespace minimum_cups_needed_l61_61080

theorem minimum_cups_needed (container_capacity cup_capacity : ℕ) (h1 : container_capacity = 980) (h2 : cup_capacity = 80) : 
  Nat.ceil (container_capacity / cup_capacity : ℚ) = 13 :=
by
  sorry

end minimum_cups_needed_l61_61080


namespace annual_growth_rate_l61_61785

theorem annual_growth_rate (u_2021 u_2023 : ℝ) (x : ℝ) : 
    u_2021 = 1 ∧ u_2023 = 1.69 ∧ x > 0 → (u_2023 / u_2021) = (1 + x)^2 → x * 100 = 30 :=
by
  intros h1 h2
  sorry

end annual_growth_rate_l61_61785


namespace points_on_line_possible_l61_61421

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end points_on_line_possible_l61_61421


namespace final_quarters_l61_61191

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 760
def first_spent : ℕ := 418
def second_spent : ℕ := 192

-- Define the final amount of quarters Sally should have
theorem final_quarters (initial_quarters first_spent second_spent : ℕ) : initial_quarters - first_spent - second_spent = 150 :=
by
  sorry

end final_quarters_l61_61191


namespace coneCannotBeQuadrilateral_l61_61227

-- Define types for our geometric solids
inductive Solid
| Cylinder
| Cone
| FrustumCone
| Prism

-- Define a predicate for whether the cross-section can be a quadrilateral
def canBeQuadrilateral (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => true
  | Solid.Cone => false
  | Solid.FrustumCone => true
  | Solid.Prism => true

-- The theorem we need to prove
theorem coneCannotBeQuadrilateral : canBeQuadrilateral Solid.Cone = false := by
  sorry

end coneCannotBeQuadrilateral_l61_61227


namespace train1_speed_l61_61844

noncomputable def total_distance_in_kilometers : ℝ :=
  (630 + 100 + 200) / 1000

noncomputable def time_in_hours : ℝ :=
  13.998880089592832 / 3600

noncomputable def relative_speed : ℝ :=
  total_distance_in_kilometers / time_in_hours

noncomputable def speed_of_train2 : ℝ :=
  72

noncomputable def speed_of_train1 : ℝ :=
  relative_speed - speed_of_train2

theorem train1_speed : speed_of_train1 = 167.076 := by 
  sorry

end train1_speed_l61_61844


namespace find_x_for_parallel_l61_61539

-- Definitions for vector components and parallel condition.
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem find_x_for_parallel :
  ∃ x : ℝ, parallel a (b x) ∧ x = -3 / 2 :=
by
  -- The statement to be proven
  sorry

end find_x_for_parallel_l61_61539


namespace together_finish_work_in_10_days_l61_61657

theorem together_finish_work_in_10_days (x_days y_days : ℕ) (hx : x_days = 15) (hy : y_days = 30) :
  let x_rate := 1 / (x_days : ℚ)
  let y_rate := 1 / (y_days : ℚ)
  let combined_rate := x_rate + y_rate
  let total_days := 1 / combined_rate
  total_days = 10 :=
by
  sorry

end together_finish_work_in_10_days_l61_61657


namespace total_cost_of_commodities_l61_61545

theorem total_cost_of_commodities (a b : ℕ) (h₁ : a = 477) (h₂ : a - b = 127) : a + b = 827 :=
by
  sorry

end total_cost_of_commodities_l61_61545


namespace gcd_36_n_eq_12_l61_61220

theorem gcd_36_n_eq_12 (n : ℕ) (h1 : 80 ≤ n) (h2 : n ≤ 100) (h3 : Int.gcd 36 n = 12) : n = 84 ∨ n = 96 :=
by
  sorry

end gcd_36_n_eq_12_l61_61220


namespace ellipse_condition_necessary_but_not_sufficient_l61_61641

-- Define the conditions and proof statement in Lean 4
theorem ellipse_condition (m : ℝ) (h₁ : 2 < m) (h₂ : m < 6) : 
  (6 - m ≠ m - 2) -> 
  (∃ x y : ℝ, (x^2) / (m - 2) + (y^2) / (6 - m)= 1) :=
by
  sorry

theorem necessary_but_not_sufficient : (2 < m ∧ m < 6) ↔ (2 < m ∧ m < 6 ∧ m ≠ 4) :=
by
  sorry

end ellipse_condition_necessary_but_not_sufficient_l61_61641


namespace quadratic_inequality_condition_l61_61998

theorem quadratic_inequality_condition
  (a b c : ℝ)
  (h1 : b^2 - 4 * a * c < 0)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c < 0) :
  False :=
sorry

end quadratic_inequality_condition_l61_61998


namespace find_m_value_l61_61343

theorem find_m_value (m x y : ℝ) (hx : x = 2) (hy : y = -1) (h_eq : m * x - y = 3) : m = 1 :=
by
  sorry

end find_m_value_l61_61343


namespace initial_tomatoes_l61_61515

/-- 
Given the conditions:
  - The farmer picked 134 tomatoes yesterday.
  - The farmer picked 30 tomatoes today.
  - The farmer will have 7 tomatoes left after today.
Prove that the initial number of tomatoes in the farmer's garden was 171.
--/

theorem initial_tomatoes (picked_yesterday : ℕ) (picked_today : ℕ) (left_tomatoes : ℕ)
  (h1 : picked_yesterday = 134)
  (h2 : picked_today = 30)
  (h3 : left_tomatoes = 7) :
  (picked_yesterday + picked_today + left_tomatoes) = 171 :=
by 
  sorry

end initial_tomatoes_l61_61515


namespace four_disjoint_subsets_with_equal_sums_l61_61141

theorem four_disjoint_subsets_with_equal_sums :
  ∀ (S : Finset ℕ), 
  (∀ x ∈ S, 100 ≤ x ∧ x ≤ 999) ∧ S.card = 117 → 
  ∃ A B C D : Finset ℕ, 
    (A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧ D ⊆ S) ∧ 
    (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ A ∩ D = ∅ ∧ B ∩ C = ∅ ∧ B ∩ D = ∅ ∧ C ∩ D = ∅) ∧ 
    (A.sum id = B.sum id ∧ B.sum id = C.sum id ∧ C.sum id = D.sum id) := by
  sorry

end four_disjoint_subsets_with_equal_sums_l61_61141


namespace number_of_sad_children_l61_61208

-- Definitions of the given conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def neither_happy_nor_sad_children : ℕ := 20

-- The main statement to be proved
theorem number_of_sad_children : 
  total_children - happy_children - neither_happy_nor_sad_children = 10 := 
by 
  sorry

end number_of_sad_children_l61_61208


namespace distance_from_P_to_focus_l61_61893

-- Definition of a parabola y^2 = 8x
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Definition of distance from P to y-axis
def distance_to_y_axis (x : ℝ) : ℝ := abs x

-- Definition of the focus of the parabola y^2 = 8x
def focus : (ℝ × ℝ) := (2, 0)

-- Definition of Euclidean distance
def euclidean_distance (P₁ P₂ : ℝ × ℝ) : ℝ :=
  (P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2 

theorem distance_from_P_to_focus (x y : ℝ) (h₁ : parabola x y) (h₂ : distance_to_y_axis x = 4) :
  abs (euclidean_distance (x, y) focus) = 6 :=
sorry

end distance_from_P_to_focus_l61_61893


namespace length_of_XY_correct_l61_61559

noncomputable def length_of_XY (XZ : ℝ) (angleY : ℝ) (angleZ : ℝ) :=
  if angleZ = 90 ∧ angleY = 30 then 8 * Real.sqrt 3 else panic! "Invalid triangle angles"

theorem length_of_XY_correct : length_of_XY 12 30 90 = 8 * Real.sqrt 3 :=
by
  sorry

end length_of_XY_correct_l61_61559


namespace find_f_3_l61_61701

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : f (x + y) = f x + f y
axiom f_4_eq_6 : f 4 = 6

theorem find_f_3 : f 3 = 9 / 2 :=
by sorry

end find_f_3_l61_61701


namespace ab_geq_3_plus_cd_l61_61591

theorem ab_geq_3_plus_cd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 13) (h5 : a^2 + b^2 + c^2 + d^2 = 43) :
  a * b ≥ 3 + c * d := 
sorry

end ab_geq_3_plus_cd_l61_61591


namespace athletes_and_probability_l61_61706

-- Given conditions and parameters
def total_athletes_a := 27
def total_athletes_b := 9
def total_athletes_c := 18
def total_selected := 6
def athletes := ["A1", "A2", "A3", "A4", "A5", "A6"]

-- Definitions based on given conditions and solution steps
def selection_ratio := total_selected / (total_athletes_a + total_athletes_b + total_athletes_c)

def selected_from_a := total_athletes_a * selection_ratio
def selected_from_b := total_athletes_b * selection_ratio
def selected_from_c := total_athletes_c * selection_ratio

def pairs (l : List String) : List (String × String) :=
  (List.bind l (λ x => List.map (λ y => (x, y)) l)).filter (λ (x,y) => x < y)

def all_pairs := pairs athletes

def event_A (pair : String × String) : Bool :=
  pair.fst = "A5" ∨ pair.snd = "A5" ∨ pair.fst = "A6" ∨ pair.snd = "A6"

def favorable_event_A := all_pairs.filter event_A

noncomputable def probability_event_A := favorable_event_A.length / all_pairs.length

-- The main theorem: Number of athletes selected from each association and probability of event A
theorem athletes_and_probability : selected_from_a = 3 ∧ selected_from_b = 1 ∧ selected_from_c = 2 ∧ probability_event_A = 3/5 := by
  sorry

end athletes_and_probability_l61_61706


namespace juan_marbles_eq_64_l61_61504

def connie_marbles : ℕ := 39
def juan_extra_marbles : ℕ := 25

theorem juan_marbles_eq_64 : (connie_marbles + juan_extra_marbles) = 64 :=
by
  -- definition and conditions handled above
  sorry

end juan_marbles_eq_64_l61_61504


namespace croissants_left_l61_61138

-- Definitions based on conditions
def total_croissants : ℕ := 17
def vegans : ℕ := 3
def allergic_to_chocolate : ℕ := 2
def any_type : ℕ := 2
def guests : ℕ := 7
def plain_needed : ℕ := vegans + allergic_to_chocolate
def plain_baked : ℕ := plain_needed
def choc_baked : ℕ := total_croissants - plain_baked

-- Assuming choc_baked > plain_baked as given
axiom croissants_greater_condition : choc_baked > plain_baked

-- Theorem to prove
theorem croissants_left (total_croissants vegans allergic_to_chocolate any_type guests : ℕ) 
    (plain_needed plain_baked choc_baked : ℕ) 
    (croissants_greater_condition : choc_baked > plain_baked) : 
    (choc_baked - guests + any_type) = 3 := 
by sorry

end croissants_left_l61_61138


namespace surface_area_of_cube_edge_8_l61_61574

-- Definition of surface area of a cube
def surface_area_of_cube (edge_length : ℕ) : ℕ :=
  6 * (edge_length * edge_length)

-- Theorem to prove the surface area for a cube with edge length of 8 cm is 384 cm²
theorem surface_area_of_cube_edge_8 : surface_area_of_cube 8 = 384 :=
by
  -- The proof will be inserted here. We use sorry to indicate the missing proof.
  sorry

end surface_area_of_cube_edge_8_l61_61574


namespace find_a4_l61_61776

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

axiom hyp1 : is_arithmetic_sequence a d
axiom hyp2 : a 5 = 9
axiom hyp3 : a 7 + a 8 = 28

-- Goal
theorem find_a4 : a 4 = 7 :=
by
  sorry

end find_a4_l61_61776


namespace find_u_minus_v_l61_61179

theorem find_u_minus_v (u v : ℚ) (h1 : 5 * u - 6 * v = 31) (h2 : 3 * u + 5 * v = 4) : u - v = 5.3 := by
  sorry

end find_u_minus_v_l61_61179


namespace lcm_gcd_product_l61_61003

def a : ℕ := 20 -- Defining the first number as 20
def b : ℕ := 90 -- Defining the second number as 90

theorem lcm_gcd_product : Nat.lcm a b * Nat.gcd a b = 1800 := 
by 
  -- Computation and proof steps would go here
  sorry -- Replace with actual proof

end lcm_gcd_product_l61_61003


namespace greatest_divisor_of_arithmetic_sequence_l61_61169

theorem greatest_divisor_of_arithmetic_sequence (x c : ℤ) (h_odd : x % 2 = 1) (h_even : c % 2 = 0) :
  15 ∣ (15 * (x + 7 * c)) :=
sorry

end greatest_divisor_of_arithmetic_sequence_l61_61169


namespace cat_litter_cost_l61_61957

theorem cat_litter_cost 
    (container_weight : ℕ) (container_cost : ℕ)
    (litter_box_capacity : ℕ) (change_interval : ℕ) 
    (days_needed : ℕ) (cost : ℕ) :
  container_weight = 45 → 
  container_cost = 21 → 
  litter_box_capacity = 15 → 
  change_interval = 7 →
  days_needed = 210 → 
  cost = 210 :=
by
  intros h1 h2 h3 h4 h5
  /- Here we would add the proof steps, but this is not required. -/
  sorry

end cat_litter_cost_l61_61957


namespace jaclyn_constant_term_l61_61144

variable {R : Type*} [CommRing R] (P Q : Polynomial R)

theorem jaclyn_constant_term (hP : P.leadingCoeff = 1) (hQ : Q.leadingCoeff = 1)
  (deg_P : P.degree = 4) (deg_Q : Q.degree = 4)
  (constant_terms_eq : P.coeff 0 = Q.coeff 0)
  (coeff_z_eq : P.coeff 1 = Q.coeff 1)
  (product_eq : P * Q = Polynomial.C 1 * 
    Polynomial.C 1 * Polynomial.C 1 * Polynomial.C (-1) *
    Polynomial.C 1) :
  Jaclyn's_constant_term = 3 :=
sorry

end jaclyn_constant_term_l61_61144


namespace evaluate_expression_l61_61269

theorem evaluate_expression : 3^(2 + 3 + 4) - (3^2 * 3^3 + 3^4) = 19359 :=
by
  sorry

end evaluate_expression_l61_61269


namespace second_vote_difference_l61_61063

-- Define the total number of members
def total_members : ℕ := 300

-- Define the votes for and against in the initial vote
structure votes_initial :=
  (a : ℕ) (b : ℕ) (h : a + b = total_members) (rejected : b > a)

-- Define the votes for and against in the second vote
structure votes_second :=
  (a' : ℕ) (b' : ℕ) (h : a' + b' = total_members)

-- Define the margin and condition of passage by three times the margin
def margin (vi : votes_initial) : ℕ := vi.b - vi.a

def passage_by_margin (vi : votes_initial) (vs : votes_second) : Prop :=
  vs.a' - vs.b' = 3 * margin vi

-- Define the condition that a' is 7/6 times b
def proportion (vs : votes_second) (vi : votes_initial) : Prop :=
  vs.a' = (7 * vi.b) / 6

-- The final proof statement
theorem second_vote_difference (vi : votes_initial) (vs : votes_second)
  (h_margin : passage_by_margin vi vs)
  (h_proportion : proportion vs vi) :
  vs.a' - vi.a = 55 :=
by
  sorry  -- This is where the proof would go

end second_vote_difference_l61_61063


namespace tan_315_degrees_l61_61923

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l61_61923


namespace original_price_of_apples_l61_61538

-- Define the conditions and problem
theorem original_price_of_apples 
  (discounted_price : ℝ := 0.60 * original_price)
  (total_cost : ℝ := 30)
  (weight : ℝ := 10) :
  original_price = 5 :=
by
  -- This is the point where the proof steps would go.
  sorry

end original_price_of_apples_l61_61538


namespace kim_money_l61_61526

theorem kim_money (S P K : ℝ) (h1 : K = 1.40 * S) (h2 : S = 0.80 * P) (h3 : S + P = 1.80) : K = 1.12 :=
by sorry

end kim_money_l61_61526


namespace correct_propositions_l61_61761

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def period_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

def symmetry_about_points (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x, f (x + k) = f (x - k)

theorem correct_propositions (h1: is_odd_function f) (h2 : ∀ x, f (x + 1) = f (x -1)) :
  period_2 f ∧ (∀ k : ℤ, symmetry_about_points f k) :=
by
  sorry

end correct_propositions_l61_61761


namespace find_smaller_number_l61_61231

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end find_smaller_number_l61_61231


namespace find_divisor_l61_61479

theorem find_divisor (x : ℝ) (h : 740 / x - 175 = 10) : x = 4 := by
  sorry

end find_divisor_l61_61479


namespace total_games_in_season_is_correct_l61_61758

-- Definitions based on given conditions
def games_per_month : ℕ := 7
def season_months : ℕ := 2

-- The theorem to prove
theorem total_games_in_season_is_correct : 
  (games_per_month * season_months = 14) :=
by
  sorry

end total_games_in_season_is_correct_l61_61758


namespace polygon_sides_l61_61100

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
sorry

end polygon_sides_l61_61100


namespace complex_expression_simplification_l61_61821

-- Given: i is the imaginary unit
def i := Complex.I

-- Prove that the expression simplifies to -1
theorem complex_expression_simplification : (i^3 * (i + 1)) / (i - 1) = -1 := by
  -- We are skipping the proof and adding sorry for now
  sorry

end complex_expression_simplification_l61_61821


namespace range_omega_l61_61109

noncomputable def f (ω x : ℝ) := Real.cos (ω * x + Real.pi / 6)

theorem range_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi → -1 ≤ f ω x ∧ f ω x ≤ Real.sqrt 3 / 2) →
  ω ∈ Set.Icc (5 / 6) (5 / 3) :=
  sorry

end range_omega_l61_61109


namespace complex_problem_proof_l61_61060

open Complex

noncomputable def z : ℂ := (1 - I)^2 + 1 + 3 * I

theorem complex_problem_proof : z = 1 + I ∧ abs (z - 2 * I) = Real.sqrt 2 ∧ (∀ a b : ℝ, (z^2 + a + b = 1 - I) → (a = -3 ∧ b = 4)) := 
by
  have h1 : z = (1 - I)^2 + 1 + 3 * I := rfl
  have h2 : z = 1 + I := sorry
  have h3 : abs (z - 2 * I) = Real.sqrt 2 := sorry
  have h4 : (∀ a b : ℝ, (z^2 + a + b = 1 - I) → (a = -3 ∧ b = 4)) := sorry
  exact ⟨h2, h3, h4⟩

end complex_problem_proof_l61_61060


namespace russel_carousel_rides_l61_61265

variable (tickets_used : Nat) (tickets_shooting : Nat) (tickets_carousel : Nat)
variable (total_tickets : Nat)
variable (times_shooting : Nat)

theorem russel_carousel_rides :
    times_shooting = 2 →
    tickets_shooting = 5 →
    tickets_carousel = 3 →
    total_tickets = 19 →
    tickets_used = total_tickets - (times_shooting * tickets_shooting) →
    tickets_used / tickets_carousel = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end russel_carousel_rides_l61_61265


namespace rate_of_current_l61_61183

theorem rate_of_current (c : ℝ) : 
  (∀ t : ℝ, t = 0.4 → ∀ d : ℝ, d = 9.6 → ∀ b : ℝ, b = 20 →
  d = (b + c) * t → c = 4) :=
sorry

end rate_of_current_l61_61183


namespace factor_x_squared_minus_sixtyfour_l61_61722

theorem factor_x_squared_minus_sixtyfour (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_sixtyfour_l61_61722


namespace number_of_children_in_group_l61_61366

-- Definitions based on the conditions
def num_adults : ℕ := 55
def meal_for_adults : ℕ := 70
def meal_for_children : ℕ := 90
def remaining_children_after_adults : ℕ := 81
def num_adults_eaten : ℕ := 7
def ratio_adult_to_child : ℚ := (70 : ℚ) / 90

-- Statement of the problem to prove number of children in the group
theorem number_of_children_in_group : 
  ∃ C : ℕ, 
    (meal_for_adults - num_adults_eaten) * (ratio_adult_to_child) = (remaining_children_after_adults) ∧
    C = remaining_children_after_adults := 
sorry

end number_of_children_in_group_l61_61366


namespace gcd_4004_10010_l61_61567

theorem gcd_4004_10010 : Nat.gcd 4004 10010 = 2002 :=
by
  have h1 : 4004 = 4 * 1001 := by norm_num
  have h2 : 10010 = 10 * 1001 := by norm_num
  sorry

end gcd_4004_10010_l61_61567


namespace find_big_bonsai_cost_l61_61287

-- Given definitions based on conditions
def small_bonsai_cost : ℕ := 30
def num_small_bonsai_sold : ℕ := 3
def num_big_bonsai_sold : ℕ := 5
def total_earnings : ℕ := 190

-- Define the function to calculate total earnings from bonsai sales
def calculate_total_earnings (big_bonsai_cost: ℕ) : ℕ :=
  (num_small_bonsai_sold * small_bonsai_cost) + (num_big_bonsai_sold * big_bonsai_cost)

-- The theorem state
theorem find_big_bonsai_cost (B : ℕ) : calculate_total_earnings B = total_earnings → B = 20 :=
by
  sorry

end find_big_bonsai_cost_l61_61287


namespace kyunghoon_time_to_go_down_l61_61600

theorem kyunghoon_time_to_go_down (d : ℕ) (t_up t_down total_time : ℕ) : 
  ((t_up = d / 3) ∧ (t_down = (d + 2) / 4) ∧ (total_time = 4) → (t_up + t_down = total_time) → (t_down = 2)) := 
by
  sorry

end kyunghoon_time_to_go_down_l61_61600


namespace shot_put_distance_l61_61223

theorem shot_put_distance :
  (∃ x : ℝ, (y = - 1 / 12 * x^2 + 2 / 3 * x + 5 / 3) ∧ y = 0) ↔ x = 10 := 
by
  sorry

end shot_put_distance_l61_61223


namespace maximum_height_l61_61194

noncomputable def h (t : ℝ) : ℝ :=
  -20 * t ^ 2 + 100 * t + 30

theorem maximum_height : 
  ∃ t : ℝ, h t = 155 ∧ ∀ t' : ℝ, h t' ≤ 155 := 
sorry

end maximum_height_l61_61194


namespace complex_z_modulus_l61_61275

noncomputable def i : ℂ := Complex.I

theorem complex_z_modulus (z : ℂ) (h : (1 + i) * z = 2 * i) : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end complex_z_modulus_l61_61275


namespace fiona_hoodies_l61_61685

theorem fiona_hoodies (F C : ℕ) (h1 : F + C = 8) (h2 : C = F + 2) : F = 3 :=
by
  sorry

end fiona_hoodies_l61_61685


namespace satisfies_conditions_l61_61162

open Real

def point_P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

def condition1 (a : ℝ) : Prop := (point_P a).fst = 0

def condition2 (a : ℝ) : Prop := (point_P a).snd = 5

def condition3 (a : ℝ) : Prop := abs ((point_P a).fst) = abs ((point_P a).snd)

theorem satisfies_conditions :
  ∃ P : ℝ × ℝ, P = (12, 12) ∨ P = (-12, -12) ∨ P = (4, -4) ∨ P = (-4, 4) :=
by
  sorry

end satisfies_conditions_l61_61162


namespace set_intersection_l61_61178

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l61_61178


namespace james_owns_145_l61_61211

theorem james_owns_145 (total : ℝ) (diff : ℝ) (james_and_ali : total = 250) (james_more_than_ali : diff = 40):
  ∃ (james ali : ℝ), ali + diff = james ∧ ali + james = total ∧ james = 145 :=
by
  sorry

end james_owns_145_l61_61211


namespace D_is_largest_l61_61432

def D := (2008 / 2007) + (2008 / 2009)
def E := (2008 / 2009) + (2010 / 2009)
def F := (2009 / 2008) + (2009 / 2010) - (1 / 2009)

theorem D_is_largest : D > E ∧ D > F := by
  sorry

end D_is_largest_l61_61432


namespace multiplier_is_three_l61_61411

theorem multiplier_is_three (n m : ℝ) (h₁ : n = 3) (h₂ : 7 * n = m * n + 12) : m = 3 := 
by
  -- Skipping the proof using sorry
  sorry 

end multiplier_is_three_l61_61411


namespace student_average_less_than_actual_average_l61_61733

variable {a b c : ℝ}

theorem student_average_less_than_actual_average (h : a < b) (h2 : b < c) :
  (a + (b + c) / 2) / 2 < (a + b + c) / 3 :=
by
  sorry

end student_average_less_than_actual_average_l61_61733


namespace second_train_speed_l61_61552

theorem second_train_speed :
  ∃ v : ℝ, 
  (∀ t : ℝ, 20 * t = v * t + 50) ∧
  (∃ t : ℝ, 20 * t + v * t = 450) →
  v = 16 :=
by
  sorry

end second_train_speed_l61_61552


namespace find_a_l61_61726

def setA (a : ℤ) : Set ℤ := {a, 0}

def setB : Set ℤ := {x : ℤ | 3 * x^2 - 10 * x < 0}

theorem find_a (a : ℤ) (h : (setA a ∩ setB).Nonempty) : a = 1 ∨ a = 2 ∨ a = 3 :=
sorry

end find_a_l61_61726


namespace find_side_b_in_triangle_l61_61584

theorem find_side_b_in_triangle 
  (A B : ℝ) (a : ℝ)
  (h_cosA : Real.cos A = -1/2)
  (h_B : B = Real.pi / 4)
  (h_a : a = 3) :
  ∃ b, b = Real.sqrt 6 :=
by
  sorry

end find_side_b_in_triangle_l61_61584


namespace best_model_l61_61606

theorem best_model (R1 R2 R3 R4 : ℝ) (h1 : R1 = 0.55) (h2 : R2 = 0.65) (h3 : R3 = 0.79) (h4 : R4 = 0.95) :
  R4 > R3 ∧ R4 > R2 ∧ R4 > R1 :=
by {
  sorry
}

end best_model_l61_61606


namespace initial_marbles_count_l61_61575

theorem initial_marbles_count (g y : ℕ) 
  (h1 : (g + 3) * 4 = g + y + 3) 
  (h2 : 3 * g = g + y + 4) : 
  g + y = 8 := 
by 
  -- The proof will go here
  sorry

end initial_marbles_count_l61_61575


namespace find_m_l61_61739

theorem find_m (S : ℕ → ℝ) (m : ℝ) (h : ∀ n, S n = m * 2^(n-1) - 3) : m = 6 :=
by
  sorry

end find_m_l61_61739


namespace exponential_simplification_l61_61000

theorem exponential_simplification : 
  (10^0.25) * (10^0.25) * (10^0.5) * (10^0.5) * (10^0.75) * (10^0.75) = 1000 := 
by 
  sorry

end exponential_simplification_l61_61000


namespace inequality_abc_l61_61969

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1) * (b + 1) * (a + c) * (b + c) ≥ 16 * a * b * c :=
by
  sorry

end inequality_abc_l61_61969


namespace spherical_to_rectangular_conversion_l61_61338

noncomputable def convert_spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  convert_spherical_to_rectangular 8 (5 * Real.pi / 4) (Real.pi / 4) = (-4, -4, 4 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l61_61338


namespace probability_of_winning_set_l61_61692

def winning_probability : ℚ :=
  let total_cards := 9
  let total_draws := 3
  let same_color_sets := 3
  let same_letter_sets := 3
  let total_ways_to_draw := Nat.choose total_cards total_draws
  let total_favorable_outcomes := same_color_sets + same_letter_sets
  let probability := total_favorable_outcomes / total_ways_to_draw
  probability

theorem probability_of_winning_set :
  winning_probability = 1 / 14 :=
by
  sorry

end probability_of_winning_set_l61_61692


namespace num_four_digit_integers_with_3_and_6_l61_61440

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end num_four_digit_integers_with_3_and_6_l61_61440


namespace x_lt_1_nec_not_suff_l61_61210

theorem x_lt_1_nec_not_suff (x : ℝ) : (x < 1 → x^2 < 1) ∧ (¬(x < 1) → x^2 < 1) := 
by {
  sorry
}

end x_lt_1_nec_not_suff_l61_61210


namespace math_problem_l61_61034

theorem math_problem :
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 5000 := 
by
  sorry

end math_problem_l61_61034


namespace child_l61_61002

-- Definitions of the given conditions
def total_money : ℕ := 35
def adult_ticket_cost : ℕ := 8
def number_of_children : ℕ := 9

-- Statement of the math proof problem
theorem child's_ticket_cost : ∃ C : ℕ, total_money - adult_ticket_cost = C * number_of_children ∧ C = 3 :=
by
  sorry

end child_l61_61002


namespace tax_rate_for_remaining_l61_61206

variable (total_earnings deductions first_tax_rate total_tax taxed_amount remaining_taxable_income rem_tax_rate : ℝ)

def taxable_income (total_earnings deductions : ℝ) := total_earnings - deductions

def tax_on_first_portion (portion tax_rate : ℝ) := portion * tax_rate

def remaining_taxable (total_taxable first_portion : ℝ) := total_taxable - first_portion

def total_tax_payable (tax_first tax_remaining : ℝ) := tax_first + tax_remaining

theorem tax_rate_for_remaining :
  total_earnings = 100000 ∧ 
  deductions = 30000 ∧ 
  first_tax_rate = 0.10 ∧
  total_tax = 12000 ∧
  tax_on_first_portion 20000 first_tax_rate = 2000 ∧
  taxed_amount = 2000 ∧
  remaining_taxable_income = taxable_income total_earnings deductions - 20000 ∧
  total_tax_payable taxed_amount (remaining_taxable_income * rem_tax_rate) = total_tax →
  rem_tax_rate = 0.20 := 
sorry

end tax_rate_for_remaining_l61_61206


namespace percentage_passed_both_l61_61690

-- Define the percentages of failures
def percentage_failed_hindi : ℕ := 34
def percentage_failed_english : ℕ := 44
def percentage_failed_both : ℕ := 22

-- Statement to prove
theorem percentage_passed_both : 
  (100 - (percentage_failed_hindi + percentage_failed_english - percentage_failed_both)) = 44 := by
  sorry

end percentage_passed_both_l61_61690


namespace fractions_addition_l61_61618

theorem fractions_addition : (1 / 6 - 5 / 12 + 3 / 8) = 1 / 8 :=
by
  sorry

end fractions_addition_l61_61618


namespace triangle_perimeter_l61_61303

theorem triangle_perimeter (a b c : ℝ) (h1 : a = 2) (h2 : (b-2)^2 + |c-3| = 0) : a + b + c = 7 :=
by
  sorry

end triangle_perimeter_l61_61303


namespace seven_circle_divisors_exists_non_adjacent_divisors_l61_61684

theorem seven_circle_divisors_exists_non_adjacent_divisors (a : Fin 7 → ℕ)
  (h_adj : ∀ i : Fin 7, a i ∣ a (i + 1) % 7 ∨ a (i + 1) % 7 ∣ a i) :
  ∃ (i j : Fin 7), i ≠ j ∧ j ≠ i + 1 % 7 ∧ j ≠ i + 6 % 7 ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
  sorry

end seven_circle_divisors_exists_non_adjacent_divisors_l61_61684


namespace number_of_windows_davids_house_l61_61874

theorem number_of_windows_davids_house
  (windows_per_minute : ℕ → ℕ)
  (h1 : ∀ t, windows_per_minute t = (4 * t) / 10)
  (h2 : windows_per_minute 160 = w)
  : w = 64 :=
by
  sorry

end number_of_windows_davids_house_l61_61874


namespace find_a_in_triangle_l61_61107

variable (a b c B : ℝ)

theorem find_a_in_triangle (h1 : b = Real.sqrt 3) (h2 : c = 3) (h3 : B = 30) :
    a = 2 * Real.sqrt 3 := by
  sorry

end find_a_in_triangle_l61_61107


namespace geometric_implies_b_squared_eq_ac_not_geometric_if_all_zero_sufficient_but_not_necessary_condition_l61_61771

variable (a b c : ℝ)

theorem geometric_implies_b_squared_eq_ac
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ∃ r : ℝ, b = r * a ∧ c = r * b) :
  b^2 = a * c :=
by
  sorry

theorem not_geometric_if_all_zero 
  (hz : a = 0 ∧ b = 0 ∧ c = 0) : 
  ¬(∃ r : ℝ, b = r * a ∧ c = r * b) :=
by
  sorry

theorem sufficient_but_not_necessary_condition :
  (∃ r : ℝ, b = r * a ∧ c = r * b → b^2 = a * c) ∧ ¬(b^2 = a * c → ∃ r : ℝ, b = r * a ∧ c = r * b) :=
by
  sorry

end geometric_implies_b_squared_eq_ac_not_geometric_if_all_zero_sufficient_but_not_necessary_condition_l61_61771


namespace angle_BAC_is_105_or_35_l61_61023

-- Definitions based on conditions
def arcAB : ℝ := 110
def arcAC : ℝ := 40
def arcBC_major : ℝ := 360 - (arcAB + arcAC)
def arcBC_minor : ℝ := arcAB - arcAC

-- The conjecture: proving that the inscribed angle ∠BAC is 105° or 35° given the conditions.
theorem angle_BAC_is_105_or_35
  (h1 : 0 < arcAB ∧ arcAB < 360)
  (h2 : 0 < arcAC ∧ arcAC < 360)
  (h3 : arcAB + arcAC < 360) :
  (arcBC_major / 2 = 105) ∨ (arcBC_minor / 2 = 35) :=
  sorry

end angle_BAC_is_105_or_35_l61_61023


namespace right_triangle_congruence_l61_61937

theorem right_triangle_congruence (A B C D : Prop) :
  (A → true) → (C → true) → (D → true) → (¬ B) → B :=
by
sorry

end right_triangle_congruence_l61_61937


namespace solve_for_f_2012_l61_61873

noncomputable def f : ℝ → ℝ := sorry -- as the exact function definition isn't provided

variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = -f x)
variable (functional_eqn : ∀ x, f (x + 2) = f x + f 2)
variable (f_one : f 1 = 2)

theorem solve_for_f_2012 : f 2012 = 4024 :=
sorry

end solve_for_f_2012_l61_61873


namespace train_length_l61_61359

-- Definitions of the conditions as Lean terms/functions
def V (L : ℕ) := (L + 170) / 15
def U (L : ℕ) := (L + 250) / 20

-- The theorem to prove that the length of the train is 70 meters.
theorem train_length : ∃ L : ℕ, (V L = U L) → L = 70 := by
  sorry

end train_length_l61_61359


namespace total_listening_days_l61_61594

-- Definitions
variables {x y z t : ℕ}

-- Problem statement
theorem total_listening_days (x y z t : ℕ) : (x + y + z) * t = ((x + y + z) * t) :=
by sorry

end total_listening_days_l61_61594


namespace value_of_x_in_logarithm_equation_l61_61166

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem value_of_x_in_logarithm_equation (n : ℝ) (h1 : n = 343) : 
  ∃ (x : ℝ), log_base x n + log_base 7 n = log_base 1 n :=
by
  sorry

end value_of_x_in_logarithm_equation_l61_61166


namespace decreasing_exponential_iff_l61_61546

theorem decreasing_exponential_iff {a : ℝ} :
  (∀ x y : ℝ, x < y → (a - 1)^y < (a - 1)^x) ↔ (1 < a ∧ a < 2) :=
by 
  sorry

end decreasing_exponential_iff_l61_61546


namespace fraction_doubled_l61_61712

theorem fraction_doubled (x y : ℝ) (h_nonzero : x + y ≠ 0) : (4 * x^2) / (2 * (x + y)) = 2 * (x^2 / (x + y)) :=
by
  sorry

end fraction_doubled_l61_61712


namespace exists_plane_perpendicular_l61_61512

-- Definitions of line, plane and perpendicularity intersection etc.
variables (Point : Type) (Line Plane : Type)
variables (l : Line) (α : Plane) (intersects : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop) (perpendicular_planes : Plane → Plane → Prop)
variables (β : Plane) (subset : Line → Plane → Prop)

-- Conditions
axiom line_intersects_plane (h1 : intersects l α) : Prop
axiom line_not_perpendicular_plane (h2 : ¬perpendicular l α) : Prop

-- The main statement to prove
theorem exists_plane_perpendicular (h1 : intersects l α) (h2 : ¬perpendicular l α) :
  ∃ (β : Plane), (subset l β) ∧ (perpendicular_planes β α) :=
sorry

end exists_plane_perpendicular_l61_61512


namespace problem_1_problem_2_l61_61098

-- Definitions of conditions
variables {a b : ℝ}
axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_sum : a + b = 1

-- The statements to prove
theorem problem_1 : 
  (1 / (a^2)) + (1 / (b^2)) ≥ 8 := 
sorry

theorem problem_2 : 
  (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 := 
sorry

end problem_1_problem_2_l61_61098


namespace chip_final_balance_l61_61288

noncomputable def finalBalance : ℝ := 
  let initialBalance := 50.0
  let month1InterestRate := 0.20
  let month2NewCharges := 20.0
  let month2InterestRate := 0.20
  let month3NewCharges := 30.0
  let month3Payment := 10.0
  let month3InterestRate := 0.25
  let month4NewCharges := 40.0
  let month4Payment := 20.0
  let month4InterestRate := 0.15

  -- Month 1
  let month1InterestFee := initialBalance * month1InterestRate
  let balanceMonth1 := initialBalance + month1InterestFee

  -- Month 2
  let balanceMonth2BeforeInterest := balanceMonth1 + month2NewCharges
  let month2InterestFee := balanceMonth2BeforeInterest * month2InterestRate
  let balanceMonth2 := balanceMonth2BeforeInterest + month2InterestFee

  -- Month 3
  let balanceMonth3BeforeInterest := balanceMonth2 + month3NewCharges
  let balanceMonth3AfterPayment := balanceMonth3BeforeInterest - month3Payment
  let month3InterestFee := balanceMonth3AfterPayment * month3InterestRate
  let balanceMonth3 := balanceMonth3AfterPayment + month3InterestFee

  -- Month 4
  let balanceMonth4BeforeInterest := balanceMonth3 + month4NewCharges
  let balanceMonth4AfterPayment := balanceMonth4BeforeInterest - month4Payment
  let month4InterestFee := balanceMonth4AfterPayment * month4InterestRate
  let balanceMonth4 := balanceMonth4AfterPayment + month4InterestFee

  balanceMonth4

theorem chip_final_balance : finalBalance = 189.75 := by sorry

end chip_final_balance_l61_61288


namespace months_b_after_a_started_business_l61_61448

theorem months_b_after_a_started_business
  (A_initial : ℝ)
  (B_initial : ℝ)
  (profit_ratio : ℝ)
  (A_investment_time : ℕ)
  (B_investment_time : ℕ)
  (investment_ratio : A_initial * A_investment_time / (B_initial * B_investment_time) = profit_ratio) :
  B_investment_time = 6 :=
by
  -- Given:
  -- A_initial = 3500
  -- B_initial = 10500
  -- profit_ratio = 2 / 3
  -- A_investment_time = 12 months
  -- B_investment_time = 12 - x months
  -- We need to prove that x = 6 months such that investment ratio matches profit ratio.
  sorry

end months_b_after_a_started_business_l61_61448


namespace kopeechka_items_l61_61011

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l61_61011


namespace compute_expression_l61_61880

theorem compute_expression (x : ℕ) (h : x = 3) : (x^8 + 8 * x^4 + 16) / (x^4 - 4) = 93 :=
by
  rw [h]
  sorry

end compute_expression_l61_61880


namespace delta_value_l61_61383

theorem delta_value (Delta : ℤ) (h : 5 * (-3) = Delta - 3) : Delta = -12 := 
by 
  sorry

end delta_value_l61_61383


namespace paving_time_together_l61_61547

/-- Define the rate at which Mary alone paves the driveway -/
noncomputable def Mary_rate : ℝ := 1 / 4

/-- Define the rate at which Hillary alone paves the driveway -/
noncomputable def Hillary_rate : ℝ := 1 / 3

/-- Define the increased rate of Mary when working together -/
noncomputable def Mary_rate_increased := Mary_rate + (0.3333 * Mary_rate)

/-- Define the decreased rate of Hillary when working together -/
noncomputable def Hillary_rate_decreased := Hillary_rate - (0.5 * Hillary_rate)

/-- Combine their rates when working together -/
noncomputable def combined_rate := Mary_rate_increased + Hillary_rate_decreased

/-- Prove that the time taken to pave the driveway together is approximately 2 hours -/
theorem paving_time_together : abs ((1 / combined_rate) - 2) < 0.0001 :=
by
  sorry

end paving_time_together_l61_61547


namespace find_2a_plus_b_l61_61899

open Real

theorem find_2a_plus_b (a b : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2)
    (h1 : 4 * (cos a)^3 - 3 * (cos b)^3 = 2) 
    (h2 : 4 * cos (2 * a) + 3 * cos (2 * b) = 1) : 
    2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l61_61899


namespace arithmetic_expression_l61_61565

theorem arithmetic_expression :
  10 + 4 * (5 + 3)^3 = 2058 :=
by
  sorry

end arithmetic_expression_l61_61565


namespace find_equation_line_l61_61984

noncomputable def line_through_point_area (A : Real × Real) (S : Real) : Prop :=
  ∃ (k : Real), (k < 0) ∧ (2 * A.1 + A.2 - 4 = 0) ∧
    (1 / 2 * (2 - k) * (1 - 2 / k) = S)

theorem find_equation_line (A : ℝ × ℝ) (S : ℝ) (hA : A = (1, 2)) (hS : S = 4) :
  line_through_point_area A S →
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, y = l x ↔ 2 * x + y - 4 = 0 :=
by
  sorry

end find_equation_line_l61_61984


namespace inscribed_sphere_radius_eq_l61_61244

-- Define the parameters for the right cone
structure RightCone where
  base_radius : ℝ
  height : ℝ

-- Given the right cone conditions
def givenCone : RightCone := { base_radius := 15, height := 40 }

-- Define the properties for inscribed sphere
def inscribedSphereRadius (c : RightCone) : ℝ := sorry

-- The theorem statement for the radius of the inscribed sphere
theorem inscribed_sphere_radius_eq (c : RightCone) : ∃ (b d : ℝ), 
  inscribedSphereRadius c = b * Real.sqrt d - b ∧ (b + d = 14) :=
by
  use 5, 9
  sorry

end inscribed_sphere_radius_eq_l61_61244


namespace original_data_props_l61_61435

variable {n : ℕ}
variable {x : Fin n → ℝ}
variable {new_x : Fin n → ℝ} 

noncomputable def average (data : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i => data i)) / n

noncomputable def variance (data : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i => (data i - average data) ^ 2)) / n

-- Conditions
def condition1 (x new_x : Fin n → ℝ) (h : ∀ i, new_x i = x i - 80) : Prop := true

def condition2 (new_x : Fin n → ℝ) : Prop :=
  average new_x = 1.2

def condition3 (new_x : Fin n → ℝ) : Prop :=
  variance new_x = 4.4

theorem original_data_props (h : ∀ i, new_x i = x i - 80)
  (h_avg : average new_x = 1.2) 
  (h_var : variance new_x = 4.4) :
  average x = 81.2 ∧ variance x = 4.4 :=
sorry

end original_data_props_l61_61435


namespace production_line_B_units_l61_61716

theorem production_line_B_units (total_units : ℕ) (A_units B_units C_units : ℕ) 
  (h1 : total_units = 16800)
  (h2 : ∃ d : ℕ, A_units + d = B_units ∧ B_units + d = C_units) :
  B_units = 5600 := 
sorry

end production_line_B_units_l61_61716


namespace sum_of_a_for_one_solution_l61_61480

theorem sum_of_a_for_one_solution (a : ℝ) :
  (∀ x : ℝ, 3 * x^2 + (a + 15) * x + 18 = 0 ↔ (a + 15) ^ 2 - 4 * 3 * 18 = 0) →
  a = -15 + 6 * Real.sqrt 6 ∨ a = -15 - 6 * Real.sqrt 6 → a + (-15 + 6 * Real.sqrt 6) + (-15 - 6 * Real.sqrt 6) = -30 :=
by
  intros h1 h2
  have hsum : (-15 + 6 * Real.sqrt 6) + (-15 - 6 * Real.sqrt 6) = -30 := by linarith [Real.sqrt 6]
  sorry

end sum_of_a_for_one_solution_l61_61480


namespace list_price_correct_l61_61838

noncomputable def list_price_satisfied : Prop :=
∃ x : ℝ, 0.25 * (x - 25) + 0.05 * (x - 5) = 0.15 * (x - 15) ∧ x = 28.33

theorem list_price_correct : list_price_satisfied :=
sorry

end list_price_correct_l61_61838


namespace number_of_girls_l61_61616

theorem number_of_girls (B G : ℕ) (h1 : B * 5 = G * 8) (h2 : B + G = 1040) : G = 400 :=
by
  sorry

end number_of_girls_l61_61616


namespace original_speed_of_Person_A_l61_61810

variable (v_A v_B : ℝ)

-- Define the conditions
def condition1 : Prop := v_B = 2 * v_A
def condition2 : Prop := v_A + 10 = 4 * (v_B - 5)

-- Define the theorem to prove
theorem original_speed_of_Person_A (h1 : condition1 v_A v_B) (h2 : condition2 v_A v_B) : v_A = 18 := 
by
  sorry

end original_speed_of_Person_A_l61_61810


namespace vertex_of_parabola_l61_61048

theorem vertex_of_parabola :
  (∃ x y : ℝ, y = -3*x^2 + 6*x + 1 ∧ (x, y) = (1, 4)) :=
sorry

end vertex_of_parabola_l61_61048


namespace vector_on_line_l61_61087

theorem vector_on_line (t : ℝ) (x y : ℝ) : 
  (x = 3 * t + 1) → (y = 2 * t + 3) → 
  ∃ t, (∃ x y, (x = 3 * t + 1) ∧ (y = 2 * t + 3) ∧ (x = 23 / 2) ∧ (y = 10)) :=
  by
  sorry

end vector_on_line_l61_61087


namespace find_p_l61_61423

variables (m n p : ℝ)

def line_equation (x y : ℝ) : Prop :=
  x = y / 3 - 2 / 5

theorem find_p
  (h1 : line_equation m n)
  (h2 : line_equation (m + p) (n + 9)) :
  p = 3 :=
by
  sorry

end find_p_l61_61423


namespace tan_negative_angle_l61_61135

theorem tan_negative_angle (m : ℝ) (h1 : m = Real.cos (80 * Real.pi / 180)) (h2 : m = Real.sin (10 * Real.pi / 180)) :
  Real.tan (-440 * Real.pi / 180) = - (Real.sqrt (1 - m^2)) / m :=
by
  sorry

end tan_negative_angle_l61_61135


namespace fraction_equals_decimal_l61_61390

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l61_61390


namespace quadratic_no_real_roots_l61_61061

theorem quadratic_no_real_roots (m : ℝ) : (4 + 4 * m < 0) → (m < -1) :=
by
  intro h
  linarith

end quadratic_no_real_roots_l61_61061


namespace min_containers_needed_l61_61693

theorem min_containers_needed 
  (total_boxes1 : ℕ) 
  (weight_box1 : ℕ) 
  (total_boxes2 : ℕ) 
  (weight_box2 : ℕ) 
  (weight_limit : ℕ) :
  total_boxes1 = 90000 →
  weight_box1 = 3300 →
  total_boxes2 = 5000 →
  weight_box2 = 200 →
  weight_limit = 100000 →
  (total_boxes1 * weight_box1 + total_boxes2 * weight_box2 + weight_limit - 1) / weight_limit = 3000 :=
by
  sorry

end min_containers_needed_l61_61693


namespace real_distance_between_cities_l61_61027

-- Condition: the map distance between Goteborg and Jonkoping
def map_distance_cm : ℝ := 88

-- Condition: the map scale
def map_scale_km_per_cm : ℝ := 15

-- The real distance to be proven
theorem real_distance_between_cities :
  (map_distance_cm * map_scale_km_per_cm) = 1320 := by
  sorry

end real_distance_between_cities_l61_61027


namespace max_y_difference_eq_l61_61074

theorem max_y_difference_eq (x y p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h : x * y = p * x + q * y) : y - x = (p - 1) * (q + 1) :=
sorry

end max_y_difference_eq_l61_61074


namespace Ava_watch_minutes_l61_61284

theorem Ava_watch_minutes (hours_watched : ℕ) (minutes_per_hour : ℕ) (h : hours_watched = 4) (m : minutes_per_hour = 60) : 
  hours_watched * minutes_per_hour = 240 :=
by
  sorry

end Ava_watch_minutes_l61_61284


namespace no_snow_probability_l61_61519

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 2 / 3) 
  (h2 : p2 = 3 / 4) 
  (h3 : p3 = 5 / 6) 
  (h4 : p4 = 1 / 2) : 
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1 / 144 :=
by
  sorry

end no_snow_probability_l61_61519


namespace number_of_students_on_wednesday_l61_61412

-- Define the problem conditions
variables (W T : ℕ)

-- Define the given conditions
def condition1 : Prop := T = W - 9
def condition2 : Prop := W + T = 65

-- Define the theorem to prove
theorem number_of_students_on_wednesday (h1 : condition1 W T) (h2 : condition2 W T) : W = 37 :=
by
  sorry

end number_of_students_on_wednesday_l61_61412


namespace simplify_and_calculate_expression_l61_61354

theorem simplify_and_calculate_expression (a b : ℤ) (ha : a = -1) (hb : b = -2) :
  (2 * a + b) * (b - 2 * a) - (a - 3 * b) ^ 2 = -25 :=
by 
  -- We can use 'by' to start the proof and 'sorry' to skip it
  sorry

end simplify_and_calculate_expression_l61_61354


namespace rectangular_region_area_l61_61513

theorem rectangular_region_area :
  ∀ (s : ℝ), 18 * s * s = (15 * Real.sqrt 2) * (7.5 * Real.sqrt 2) :=
by
  intro s
  have h := 5 ^ 2 = 2 * s ^ 2
  have s := Real.sqrt (25 / 2)
  exact sorry

end rectangular_region_area_l61_61513


namespace div_neg_forty_five_l61_61050

theorem div_neg_forty_five : (-40 / 5) = -8 :=
by
  sorry

end div_neg_forty_five_l61_61050


namespace opposite_of_6_is_neg_6_l61_61190

theorem opposite_of_6_is_neg_6 : -6 = -6 := by
  sorry

end opposite_of_6_is_neg_6_l61_61190


namespace cars_already_parked_l61_61040

-- Define the levels and their parking spaces based on given conditions
def first_level_spaces : Nat := 90
def second_level_spaces : Nat := first_level_spaces + 8
def third_level_spaces : Nat := second_level_spaces + 12
def fourth_level_spaces : Nat := third_level_spaces - 9

-- Compute total spaces in the garage
def total_spaces : Nat := first_level_spaces + second_level_spaces + third_level_spaces + fourth_level_spaces

-- Define the available spaces for more cars
def available_spaces : Nat := 299

-- Prove the number of cars already parked
theorem cars_already_parked : total_spaces - available_spaces = 100 :=
by
  exact Nat.sub_eq_of_eq_add sorry -- Fill in with the actual proof step

end cars_already_parked_l61_61040


namespace no_third_degree_polynomial_exists_l61_61801

theorem no_third_degree_polynomial_exists (a b c d : ℤ) (h : a ≠ 0) :
  ¬(p 15 = 3 ∧ p 21 = 12 ∧ p = λ x => a * x ^ 3 + b * x ^ 2 + c * x + d) :=
sorry

end no_third_degree_polynomial_exists_l61_61801


namespace leila_armchairs_l61_61405

theorem leila_armchairs :
  ∀ {sofa_price armchair_price coffee_table_price total_invoice armchairs : ℕ},
  sofa_price = 1250 →
  armchair_price = 425 →
  coffee_table_price = 330 →
  total_invoice = 2430 →
  1 * sofa_price + armchairs * armchair_price + 1 * coffee_table_price = total_invoice →
  armchairs = 2 :=
by
  intros sofa_price armchair_price coffee_table_price total_invoice armchairs
  intros h1 h2 h3 h4 h_eq
  sorry

end leila_armchairs_l61_61405


namespace finite_points_outside_unit_circle_l61_61522

noncomputable def centroid (x y z : ℝ × ℝ) : ℝ × ℝ := 
  ((x.1 + y.1 + z.1) / 3, (x.2 + y.2 + z.2) / 3)

theorem finite_points_outside_unit_circle
  (A₁ B₁ C₁ D₁ : ℝ × ℝ)
  (A : ℕ → ℝ × ℝ)
  (B : ℕ → ℝ × ℝ)
  (C : ℕ → ℝ × ℝ)
  (D : ℕ → ℝ × ℝ)
  (hA : ∀ n, A (n + 1) = centroid (B n) (C n) (D n))
  (hB : ∀ n, B (n + 1) = centroid (A n) (C n) (D n))
  (hC : ∀ n, C (n + 1) = centroid (A n) (B n) (D n))
  (hD : ∀ n, D (n + 1) = centroid (A n) (B n) (C n))
  (h₀ : A 1 = A₁ ∧ B 1 = B₁ ∧ C 1 = C₁ ∧ D 1 = D₁)
  : ∃ N : ℕ, ∀ n > N, (A n).1 * (A n).1 + (A n).2 * (A n).2 ≤ 1 :=
sorry

end finite_points_outside_unit_circle_l61_61522


namespace percentage_of_Muscovy_ducks_l61_61126

theorem percentage_of_Muscovy_ducks
  (N : ℕ) (M : ℝ) (female_percentage : ℝ) (female_Muscovy : ℕ)
  (hN : N = 40)
  (hfemale_percentage : female_percentage = 0.30)
  (hfemale_Muscovy : female_Muscovy = 6)
  (hcondition : female_percentage * M * N = female_Muscovy) 
  : M = 0.5 := 
sorry

end percentage_of_Muscovy_ducks_l61_61126


namespace range_of_a_l61_61173

variable {x : ℝ} {a : ℝ}

theorem range_of_a (h : ∀ x : ℝ, ¬ (x^2 - 5*x + (5/4)*a > 0)) : 5 < a :=
by
  sorry

end range_of_a_l61_61173


namespace correct_subtraction_l61_61174

/-- Given a number n where subtracting 63 results in 8,
we aim to find the result of subtracting 36 from n
and proving that the result is 35. -/
theorem correct_subtraction (n : ℕ) (h : n - 63 = 8) : n - 36 = 35 :=
by
  sorry

end correct_subtraction_l61_61174


namespace expression_parity_l61_61092

theorem expression_parity (p m : ℤ) (hp : Odd p) : (Odd (p^3 + m * p)) ↔ Even m := by
  sorry

end expression_parity_l61_61092


namespace intersection_of_domains_l61_61071

def A_domain : Set ℝ := { x : ℝ | 4 - x^2 ≥ 0 }
def B_domain : Set ℝ := { x : ℝ | 1 - x > 0 }

theorem intersection_of_domains :
  (A_domain ∩ B_domain) = { x : ℝ | -2 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_of_domains_l61_61071


namespace algebraic_identity_specific_case_l61_61362

theorem algebraic_identity (a b : ℝ) : (a - b)^2 = a^2 + b^2 - 2 * a * b :=
by sorry

theorem specific_case : 2021^2 - 2021 * 4034 + 2017^2 = 16 :=
by sorry

end algebraic_identity_specific_case_l61_61362


namespace find_second_bag_weight_l61_61165

variable (initialWeight : ℕ) (firstBagWeight : ℕ) (totalWeight : ℕ)

theorem find_second_bag_weight 
  (h1: initialWeight = 15)
  (h2: firstBagWeight = 15)
  (h3: totalWeight = 40) :
  totalWeight - (initialWeight + firstBagWeight) = 10 :=
  sorry

end find_second_bag_weight_l61_61165


namespace problem_statement_l61_61035

noncomputable def AB2_AC2_BC2_eq_4 (l m n k : ℝ) : Prop :=
  let D := (l+k, 0, 0)
  let E := (0, m+k, 0)
  let F := (0, 0, n+k)
  let AB_sq := 4 * (n+k)^2
  let AC_sq := 4 * (m+k)^2
  let BC_sq := 4 * (l+k)^2
  AB_sq + AC_sq + BC_sq = 4 * ((l+k)^2 + (m+k)^2 + (n+k)^2)

theorem problem_statement (l m n k : ℝ) : 
  AB2_AC2_BC2_eq_4 l m n k :=
by
  sorry

end problem_statement_l61_61035


namespace same_solutions_a_value_l61_61524

theorem same_solutions_a_value (a x : ℝ) (h1 : 2 * x + 1 = 3) (h2 : 3 - (a - x) / 3 = 1) : a = 7 := by
  sorry

end same_solutions_a_value_l61_61524


namespace trig_identity_tangent_l61_61312

variable {θ : ℝ}

theorem trig_identity_tangent (h : Real.tan θ = 2) : 
  (Real.sin θ * (Real.cos θ * Real.cos θ - Real.sin θ * Real.sin θ)) / (Real.cos θ - Real.sin θ) = 6 / 5 := 
sorry

end trig_identity_tangent_l61_61312


namespace calculate_final_speed_l61_61085

noncomputable def final_speed : ℝ :=
  let v1 : ℝ := (150 * 1.60934 * 1000) / 3600
  let v2 : ℝ := (170 * 1000) / 3600
  let v_decreased : ℝ := v1 - v2
  let a : ℝ := (500000 * 0.01) / 60
  v_decreased + a * (30 * 60)

theorem calculate_final_speed : final_speed = 150013.45 :=
by
  sorry

end calculate_final_speed_l61_61085


namespace road_trip_mileage_base10_l61_61535

-- Defining the base 8 number 3452
def base8_to_base10 (n : Nat) : Nat :=
  3 * 8^3 + 4 * 8^2 + 5 * 8^1 + 2 * 8^0

-- Stating the problem as a theorem
theorem road_trip_mileage_base10 : base8_to_base10 3452 = 1834 := by
  sorry

end road_trip_mileage_base10_l61_61535


namespace faster_speed_l61_61608

theorem faster_speed (x : ℝ) (h1 : 40 = 8 * 5) (h2 : 60 = x * 5) : x = 12 :=
sorry

end faster_speed_l61_61608


namespace video_game_cost_l61_61542

theorem video_game_cost :
  let september_saving : ℕ := 50
  let october_saving : ℕ := 37
  let november_saving : ℕ := 11
  let mom_gift : ℕ := 25
  let remaining_money : ℕ := 36
  let total_savings : ℕ := september_saving + october_saving + november_saving
  let total_with_gift : ℕ := total_savings + mom_gift
  let game_cost : ℕ := total_with_gift - remaining_money
  game_cost = 87 :=
by
  sorry

end video_game_cost_l61_61542


namespace Bomi_change_l61_61635

def candy_cost : ℕ := 350
def chocolate_cost : ℕ := 500
def total_paid : ℕ := 1000
def total_cost := candy_cost + chocolate_cost
def change := total_paid - total_cost

theorem Bomi_change : change = 150 :=
by
  -- Here we would normally provide the proof steps.
  sorry

end Bomi_change_l61_61635


namespace percentage_of_boys_l61_61304

theorem percentage_of_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (total_students_eq : total_students = 42)
  (ratio_eq : boy_ratio = 3 ∧ girl_ratio = 4) :
  (boy_ratio + girl_ratio) = 7 ∧ (total_students / 7 * boy_ratio * 100 / total_students : ℚ) = 42.86 :=
by
  sorry

end percentage_of_boys_l61_61304


namespace rosa_called_pages_sum_l61_61403

theorem rosa_called_pages_sum :
  let week1_pages := 10.2
  let week2_pages := 8.6
  let week3_pages := 12.4
  week1_pages + week2_pages + week3_pages = 31.2 :=
by
  let week1_pages := 10.2
  let week2_pages := 8.6
  let week3_pages := 12.4
  sorry  -- proof will be done here

end rosa_called_pages_sum_l61_61403


namespace shaina_chocolate_amount_l61_61010

variable (total_chocolate : ℚ) (num_piles : ℕ) (fraction_kept : ℚ)
variable (eq_total_chocolate : total_chocolate = 72 / 7)
variable (eq_num_piles : num_piles = 6)
variable (eq_fraction_kept : fraction_kept = 1 / 3)

theorem shaina_chocolate_amount :
  (total_chocolate / num_piles) * (1 - fraction_kept) = 8 / 7 :=
by
  sorry

end shaina_chocolate_amount_l61_61010


namespace field_dimension_m_l61_61745

theorem field_dimension_m (m : ℝ) (h : (3 * m + 8) * (m - 3) = 80) : m = 6.057 := by
  sorry

end field_dimension_m_l61_61745


namespace age_of_youngest_child_l61_61124

theorem age_of_youngest_child
  (total_bill : ℝ)
  (mother_charge : ℝ)
  (child_charge_per_year : ℝ)
  (children_total_years : ℝ)
  (twins_age : ℕ)
  (youngest_child_age : ℕ)
  (h_total_bill : total_bill = 13.00)
  (h_mother_charge : mother_charge = 6.50)
  (h_child_charge_per_year : child_charge_per_year = 0.65)
  (h_children_bill : total_bill - mother_charge = children_total_years * child_charge_per_year)
  (h_children_age : children_total_years = 10)
  (h_youngest_child : youngest_child_age = 10 - 2 * twins_age) :
  youngest_child_age = 2 ∨ youngest_child_age = 4 :=
by
  sorry

end age_of_youngest_child_l61_61124


namespace curve_touches_x_axis_at_most_three_times_l61_61131

theorem curve_touches_x_axis_at_most_three_times
  (a b c d : ℝ) :
  ∃ (x : ℝ), (x^4 - x^5 + a * x^3 + b * x^2 + c * x + d = 0) → ∃ (y : ℝ), (y = 0) → 
  ∃(n : ℕ), (n ≤ 3) :=
by sorry

end curve_touches_x_axis_at_most_three_times_l61_61131


namespace area_of_black_region_l61_61990

def side_length_square : ℝ := 10
def length_rectangle : ℝ := 5
def width_rectangle : ℝ := 2

theorem area_of_black_region :
  (side_length_square * side_length_square) - (length_rectangle * width_rectangle) = 90 := by
sorry

end area_of_black_region_l61_61990


namespace age_ratio_in_years_l61_61249

variable (s d x : ℕ)

theorem age_ratio_in_years (h1 : s - 3 = 2 * (d - 3)) (h2 : s - 7 = 3 * (d - 7)) (hx : (s + x) = 3 * (d + x) / 2) : x = 5 := sorry

end age_ratio_in_years_l61_61249


namespace two_pow_2001_mod_127_l61_61773

theorem two_pow_2001_mod_127 : (2^2001) % 127 = 64 := 
by
  sorry

end two_pow_2001_mod_127_l61_61773


namespace arithmetic_sequence_tenth_term_l61_61459

theorem arithmetic_sequence_tenth_term (a d : ℤ)
  (h1 : a + 2 * d = 5)
  (h2 : a + 6 * d = 13) :
  a + 9 * d = 19 := 
sorry

end arithmetic_sequence_tenth_term_l61_61459


namespace binom_12_3_equal_220_l61_61500

theorem binom_12_3_equal_220 : Nat.choose 12 3 = 220 := by sorry

end binom_12_3_equal_220_l61_61500


namespace smallest_n_satisfies_conditions_l61_61186

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l61_61186


namespace power_function_value_l61_61993

theorem power_function_value (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ (1 / 2)) (H : f 9 = 3) : f 25 = 5 :=
by
  sorry

end power_function_value_l61_61993


namespace find_b_exists_l61_61067

theorem find_b_exists (N : ℕ) (hN : N ≠ 1) : ∃ (a c d : ℕ), a > 1 ∧ c > 1 ∧ d > 1 ∧
  (N : ℝ) ^ (1/a + 1/(a*4) + 1/(a*4*c) + 1/(a*4*c*d)) = (N : ℝ) ^ (37/48) :=
by
  sorry

end find_b_exists_l61_61067


namespace Marta_books_directly_from_bookstore_l61_61302

theorem Marta_books_directly_from_bookstore :
  let total_books_sale := 5
  let price_per_book_sale := 10
  let total_books_online := 2
  let total_cost_online := 40
  let total_spent := 210
  let cost_of_books_directly := 3 * total_cost_online
  let total_cost_sale := total_books_sale * price_per_book_sale
  let cost_per_book_directly := cost_of_books_directly / (total_cost_online / total_books_online)
  total_spent = total_cost_sale + total_cost_online + cost_of_books_directly ∧ (cost_of_books_directly / cost_per_book_directly) = 2 :=
by
  sorry

end Marta_books_directly_from_bookstore_l61_61302


namespace part_I_solution_part_II_solution_l61_61201

-- Definitions for the problem
def f (x a : ℝ) : ℝ := |x - a| + |x - 1|

-- Part I: When a = 2, solve the inequality f(x) < 4
theorem part_I_solution (x : ℝ) : f x 2 < 4 ↔ x > -1/2 ∧ x < 7/2 :=
by sorry

-- Part II: Range of values for a such that f(x) ≥ 2 for all x
theorem part_II_solution (a : ℝ) : (∀ x, f x a ≥ 2) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by sorry

end part_I_solution_part_II_solution_l61_61201


namespace product_of_coprime_numbers_l61_61268

variable {a b c : ℕ}

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem product_of_coprime_numbers (h1 : coprime a b) (h2 : a * b = c) : Nat.lcm a b = c := by
  sorry

end product_of_coprime_numbers_l61_61268


namespace measure_of_angle_A_l61_61765

-- Defining the measures of angles
def angle_B : ℝ := 50
def angle_C : ℝ := 40
def angle_D : ℝ := 30

-- Prove that measure of angle A is 120 degrees given the conditions
theorem measure_of_angle_A (B C D : ℝ) (hB : B = angle_B) (hC : C = angle_C) (hD : D = angle_D) : B + C + D + 60 = 180 -> 180 - (B + C + D + 60) = 120 :=
by sorry

end measure_of_angle_A_l61_61765


namespace translation_m_n_l61_61682

theorem translation_m_n (m n : ℤ) (P Q : ℤ × ℤ) (hP : P = (-1, -3)) (hQ : Q = (-2, 0))
(hx : P.1 - m = Q.1) (hy : P.2 + n = Q.2) :
  m + n = 4 :=
by
  sorry

end translation_m_n_l61_61682


namespace portia_high_school_students_l61_61209

variables (P L M : ℕ)
axiom h1 : P = 4 * L
axiom h2 : P = 2 * M
axiom h3 : P + L + M = 4800

theorem portia_high_school_students : P = 2740 :=
by sorry

end portia_high_school_students_l61_61209


namespace number_of_teachers_l61_61172

theorem number_of_teachers (total_people : ℕ) (sampled_individuals : ℕ) (sampled_students : ℕ) 
    (school_total : total_people = 2400) 
    (sample_total : sampled_individuals = 160) 
    (sample_students : sampled_students = 150) : 
    ∃ teachers : ℕ, teachers = 150 := 
by
  -- Proof omitted
  sorry

end number_of_teachers_l61_61172


namespace range_of_a_l61_61797

variable {x a : ℝ}

theorem range_of_a (h1 : x > 1) (h2 : a ≤ x + 1 / (x - 1)) : a ≤ 3 :=
sorry

end range_of_a_l61_61797


namespace circle_area_l61_61637

-- Define the conditions of the problem
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0

-- State the proof problem
theorem circle_area : (∀ (x y : ℝ), circle_equation x y) → (∀ r : ℝ, r = 2 → π * r^2 = 4 * π) :=
by
  sorry

end circle_area_l61_61637


namespace sin_alpha_of_point_P_l61_61033

theorem sin_alpha_of_point_P (α : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (Real.cos (π / 3), 1) ∧ P = (Real.cos α, Real.sin α) ) :
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
  sorry

end sin_alpha_of_point_P_l61_61033


namespace lily_final_balance_l61_61613

noncomputable def initial_balance : ℝ := 55
noncomputable def shirt_cost : ℝ := 7
noncomputable def shoes_cost : ℝ := 3 * shirt_cost
noncomputable def book_cost : ℝ := 4
noncomputable def books_amount : ℝ := 5
noncomputable def gift_fraction : ℝ := 0.20

noncomputable def remaining_balance : ℝ :=
  initial_balance - 
  shirt_cost - 
  shoes_cost - 
  books_amount * book_cost - 
  gift_fraction * (initial_balance - shirt_cost - shoes_cost - books_amount * book_cost)

theorem lily_final_balance : remaining_balance = 5.60 := 
by 
  sorry

end lily_final_balance_l61_61613


namespace philip_oranges_count_l61_61415

def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def betty_bill_oranges := betty_oranges + bill_oranges
def frank_oranges := 3 * betty_bill_oranges
def seeds_planted := frank_oranges * 2
def orange_trees := seeds_planted
def oranges_per_tree : ℕ := 5
def oranges_for_philip := orange_trees * oranges_per_tree

theorem philip_oranges_count : oranges_for_philip = 810 := by sorry

end philip_oranges_count_l61_61415


namespace original_area_area_after_translation_l61_61803

-- Defining vectors v, w, and t
def v : ℝ × ℝ := (6, -4)
def w : ℝ × ℝ := (-8, 3)
def t : ℝ × ℝ := (3, 2)

-- Function to compute the determinant of two vectors in R^2
def det (v w : ℝ × ℝ) : ℝ := v.1 * w.2 - v.2 * w.1

-- The area of a parallelogram is the absolute value of the determinant
def parallelogram_area (v w : ℝ × ℝ) : ℝ := |det v w|

-- Proving the original area is 14
theorem original_area : parallelogram_area v w = 14 := by
  sorry

-- Proving the area remains the same after translation
theorem area_after_translation : parallelogram_area v w = parallelogram_area (v.1 + t.1, v.2 + t.2) (w.1 + t.1, w.2 + t.2) := by
  sorry

end original_area_area_after_translation_l61_61803


namespace emir_needs_more_money_l61_61548

def dictionary_cost : ℕ := 5
def dinosaur_book_cost : ℕ := 11
def cookbook_cost : ℕ := 5
def saved_money : ℕ := 19
def total_cost : ℕ := dictionary_cost + dinosaur_book_cost + cookbook_cost
def additional_money_needed : ℕ := total_cost - saved_money

theorem emir_needs_more_money : additional_money_needed = 2 := by
  sorry

end emir_needs_more_money_l61_61548


namespace tan_half_alpha_third_quadrant_sine_cos_expression_l61_61328

-- Problem (1): Proof for tan(α/2) = -5 given the conditions
theorem tan_half_alpha_third_quadrant (α : ℝ) (h1 : α ∈ Set.Ioo π (3 * π / 2))
  (h2 : Real.sin α = -5/13) :
  Real.tan (α / 2) = -5 := by
  sorry

-- Problem (2): Proof for sin²(π - α) + 2sin(3π/2 + α)cos(π/2 + α) = 8/5 given the condition
theorem sine_cos_expression (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (π - α) ^ 2 + 2 * Real.sin (3 * π / 2 + α) * Real.cos (π / 2 + α) = 8 / 5 := by
  sorry

end tan_half_alpha_third_quadrant_sine_cos_expression_l61_61328


namespace trig_expression_value_l61_61009

theorem trig_expression_value {θ : Real} (h : Real.tan θ = 2) :
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + 2 * Real.cos θ) = 3 / 4 := 
by
  sorry

end trig_expression_value_l61_61009


namespace task_D_is_suitable_l61_61277

-- Definitions of the tasks
def task_A := "Investigating the age distribution of your classmates"
def task_B := "Understanding the ratio of male to female students in the eighth grade of your school"
def task_C := "Testing the urine samples of athletes who won championships at the Olympics"
def task_D := "Investigating the sleeping conditions of middle school students in Lishui City"

-- Definition of suitable_for_sampling_survey condition
def suitable_for_sampling_survey (task : String) : Prop :=
  task = task_D

-- Theorem statement
theorem task_D_is_suitable : suitable_for_sampling_survey task_D := by
  -- the proof is omitted
  sorry

end task_D_is_suitable_l61_61277


namespace floor_area_l61_61217

theorem floor_area (length_feet : ℝ) (width_feet : ℝ) (feet_to_meters : ℝ) 
  (h_length : length_feet = 15) (h_width : width_feet = 10) (h_conversion : feet_to_meters = 0.3048) :
  let length_meters := length_feet * feet_to_meters
  let width_meters := width_feet * feet_to_meters
  let area_meters := length_meters * width_meters
  area_meters = 13.93 := 
by
  sorry

end floor_area_l61_61217


namespace aluminum_carbonate_weight_l61_61455

-- Define the atomic weights
def Al : ℝ := 26.98
def C : ℝ := 12.01
def O : ℝ := 16.00

-- Define the molecular weight of aluminum carbonate
def molecularWeightAl2CO3 : ℝ := (2 * Al) + (3 * C) + (9 * O)

-- Define the number of moles
def moles : ℝ := 5

-- Calculate the total weight of 5 moles of aluminum carbonate
def totalWeight : ℝ := moles * molecularWeightAl2CO3

-- Statement to prove
theorem aluminum_carbonate_weight : totalWeight = 1169.95 :=
by {
  sorry
}

end aluminum_carbonate_weight_l61_61455


namespace even_property_of_f_when_a_zero_non_even_odd_property_of_f_when_a_nonzero_minimum_value_of_f_l61_61653

open Real

def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem even_property_of_f_when_a_zero : 
  ∀ x : ℝ, f 0 x = f 0 (-x) :=
by sorry

theorem non_even_odd_property_of_f_when_a_nonzero : 
  ∀ (a x : ℝ), a ≠ 0 → (f a x ≠ f a (-x) ∧ f a x ≠ -f a (-x)) :=
by sorry

theorem minimum_value_of_f :
  ∀ (a : ℝ), 
    (a ≤ -1/2 → ∃ x : ℝ, f a x = -a - 5/4) ∧ 
    (-1/2 < a ∧ a ≤ 1/2 → ∃ x : ℝ, f a x = a^2 - 1) ∧ 
    (a > 1/2 → ∃ x : ℝ, f a x = a - 5/4) :=
by sorry

end even_property_of_f_when_a_zero_non_even_odd_property_of_f_when_a_nonzero_minimum_value_of_f_l61_61653


namespace ratio_to_percent_l61_61118

theorem ratio_to_percent :
  (9 / 5 * 100) = 180 :=
by
  sorry

end ratio_to_percent_l61_61118


namespace find_x_l61_61769

theorem find_x (x : ℝ) (h : 61 + 5 * 12 / (x / 3) = 62) : x = 180 :=
by
  sorry

end find_x_l61_61769


namespace quadratic_one_solution_set_l61_61381

theorem quadratic_one_solution_set (a : ℝ) :
  (∃ x : ℝ, ax^2 + x + 1 = 0 ∧ (∀ y : ℝ, ax^2 + x + 1 = 0 → y = x)) ↔ (a = 0 ∨ a = 1 / 4) :=
by sorry

end quadratic_one_solution_set_l61_61381


namespace gcd_153_119_eq_17_l61_61796

theorem gcd_153_119_eq_17 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_eq_17_l61_61796


namespace sphere_radius_eq_3_l61_61477

theorem sphere_radius_eq_3 (r : ℝ) (h : (4/3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end sphere_radius_eq_3_l61_61477


namespace ellipse_eccentricity_l61_61946

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (B F A C : ℝ × ℝ) 
    (h3 : (B.1 ^ 2 / a ^ 2 + B.2 ^ 2 / b ^ 2 = 1))
    (h4 : (C.1 ^ 2 / a ^ 2 + C.2 ^ 2 / b ^ 2 = 1))
    (h5 : B.1 > 0 ∧ B.2 > 0)
    (h6 : C.1 > 0 ∧ C.2 > 0)
    (h7 : ∃ M : ℝ × ℝ, M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧ (F = M)) :
    ∃ e : ℝ, e = (1 / 3) := 
  sorry

end ellipse_eccentricity_l61_61946


namespace magnitude_a_minus_2b_l61_61327

noncomputable def magnitude_of_vector_difference : ℝ :=
  let a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
  let b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2)

theorem magnitude_a_minus_2b :
  let a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
  let b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 3 :=
by
  sorry

end magnitude_a_minus_2b_l61_61327


namespace min_value_fraction_l61_61964

theorem min_value_fraction (x : ℝ) (hx : x < 2) : ∃ y : ℝ, y = (5 - 4 * x + x^2) / (2 - x) ∧ y = 2 :=
by sorry

end min_value_fraction_l61_61964


namespace find_x_l61_61723

theorem find_x (x : ℤ) (h : (2 + 76 + x) / 3 = 5) : x = -63 := 
sorry

end find_x_l61_61723


namespace factor_expression_l61_61904

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l61_61904


namespace geometry_problem_l61_61650

/-- Given:
  DC = 5
  CB = 9
  AB = 1/3 * AD
  ED = 2/3 * AD
  Prove: FC = 10.6667 -/
theorem geometry_problem
  (DC CB AD FC : ℝ) (hDC : DC = 5) (hCB : CB = 9) (hAB : AB = 1 / 3 * AD) (hED : ED = 2 / 3 * AD)
  (AB ED: ℝ):
  FC = 10.6667 :=
by
  sorry

end geometry_problem_l61_61650


namespace non_empty_solution_set_l61_61593

theorem non_empty_solution_set (a : ℝ) (h : a > 0) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 :=
by
  sorry

end non_empty_solution_set_l61_61593


namespace GCD_17_51_LCM_17_51_GCD_6_8_LCM_8_9_l61_61892

noncomputable def GCD (a b : ℕ) : ℕ := Nat.gcd a b
noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCD_17_51 : GCD 17 51 = 17 := by
  sorry

theorem LCM_17_51 : LCM 17 51 = 51 := by
  sorry

theorem GCD_6_8 : GCD 6 8 = 2 := by
  sorry

theorem LCM_8_9 : LCM 8 9 = 72 := by
  sorry

end GCD_17_51_LCM_17_51_GCD_6_8_LCM_8_9_l61_61892


namespace pictures_left_after_deletion_l61_61261

variable (zoo museum deleted : ℕ)

def total_pictures_taken (zoo museum : ℕ) : ℕ := zoo + museum

def pictures_remaining (total deleted : ℕ) : ℕ := total - deleted

theorem pictures_left_after_deletion (h1 : zoo = 50) (h2 : museum = 8) (h3 : deleted = 38) :
  pictures_remaining (total_pictures_taken zoo museum) deleted = 20 :=
by
  sorry

end pictures_left_after_deletion_l61_61261


namespace sum_first_50_natural_numbers_l61_61422

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Prove that the sum of the first 50 natural numbers is 1275
theorem sum_first_50_natural_numbers : sum_natural 50 = 1275 := 
by
  -- Skipping proof details
  sorry

end sum_first_50_natural_numbers_l61_61422


namespace john_sixth_quiz_score_l61_61446

noncomputable def sixth_quiz_score_needed : ℤ :=
  let scores := [86, 91, 88, 84, 97]
  let desired_average := 95
  let number_of_quizzes := 6
  let total_score_needed := number_of_quizzes * desired_average
  let total_score_so_far := scores.sum
  total_score_needed - total_score_so_far

theorem john_sixth_quiz_score :
  sixth_quiz_score_needed = 124 := 
by
  sorry

end john_sixth_quiz_score_l61_61446


namespace simplify_expression_l61_61846

theorem simplify_expression (x : ℝ) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) :=
sorry

end simplify_expression_l61_61846


namespace number_of_ways_to_choose_water_polo_team_l61_61454

theorem number_of_ways_to_choose_water_polo_team :
  let total_members := 15
  let team_size := 7
  let goalie_positions := 1
  let player_choices := team_size - goalie_positions
  ∃ (total_ways : ℕ), 
  total_ways = total_members * Nat.choose (total_members - 1) player_choices ∧ 
  total_ways = 45045 :=
by
  let total_members := 15
  let team_size := 7
  let goalie_positions := 1
  let player_choices := team_size - goalie_positions
  have total_ways : ℕ := total_members * Nat.choose (total_members - 1) player_choices
  use total_ways
  sorry

end number_of_ways_to_choose_water_polo_team_l61_61454


namespace set_intersection_complement_l61_61182

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}
def complement_B : Set ℝ := U \ B
def expected_set : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem set_intersection_complement :
  A ∩ complement_B = expected_set :=
by
  sorry

end set_intersection_complement_l61_61182


namespace murtha_total_items_at_day_10_l61_61687

-- Define terms and conditions
def num_pebbles (n : ℕ) : ℕ := n
def num_seashells (n : ℕ) : ℕ := 1 + 2 * (n - 1)

def total_pebbles (n : ℕ) : ℕ :=
  (n * (1 + n)) / 2

def total_seashells (n : ℕ) : ℕ :=
  (n * (1 + num_seashells n)) / 2

-- Define main proposition
theorem murtha_total_items_at_day_10 : total_pebbles 10 + total_seashells 10 = 155 := by
  -- Placeholder for proof
  sorry

end murtha_total_items_at_day_10_l61_61687


namespace sum_of_prime_factors_1320_l61_61452

theorem sum_of_prime_factors_1320 : 
  let smallest_prime := 2
  let largest_prime := 11
  smallest_prime + largest_prime = 13 :=
by
  sorry

end sum_of_prime_factors_1320_l61_61452


namespace solve_system_of_inequalities_l61_61708

theorem solve_system_of_inequalities (x : ℝ) :
  ( (x - 2) / (x - 1) < 1 ) ∧ ( -x^2 + x + 2 < 0 ) → x > 2 :=
by
  sorry

end solve_system_of_inequalities_l61_61708


namespace hours_week3_and_4_l61_61659

variable (H3 H4 : Nat)

def hours_worked_week1_and_2 : Nat := 35 + 35
def extra_hours_worked_week3_and_4 : Nat := 26
def total_hours_week3_and_4 : Nat := hours_worked_week1_and_2 + extra_hours_worked_week3_and_4

theorem hours_week3_and_4 :
  H3 + H4 = total_hours_week3_and_4 := by
sorry

end hours_week3_and_4_l61_61659


namespace domain_of_function_l61_61078

theorem domain_of_function :
  ∀ x, (x - 2 > 0) ∧ (3 - x ≥ 0) ↔ 2 < x ∧ x ≤ 3 :=
by 
  intros x 
  simp only [and_imp, gt_iff_lt, sub_lt_iff_lt_add, sub_nonneg, le_iff_eq_or_lt, add_comm]
  exact sorry

end domain_of_function_l61_61078


namespace num_four_digit_integers_divisible_by_7_l61_61139

theorem num_four_digit_integers_divisible_by_7 :
  ∃ n : ℕ, n = 1286 ∧ ∀ k : ℕ, (1000 ≤ k ∧ k ≤ 9999) → (k % 7 = 0 ↔ ∃ m : ℕ, k = m * 7) :=
by {
  sorry
}

end num_four_digit_integers_divisible_by_7_l61_61139


namespace sum_of_integers_with_product_5_pow_4_l61_61596

theorem sum_of_integers_with_product_5_pow_4 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = 5^4 ∧
  a + b + c + d = 156 :=
by sorry

end sum_of_integers_with_product_5_pow_4_l61_61596


namespace amount_received_by_Sam_l61_61471

noncomputable def final_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem amount_received_by_Sam 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (hP : P = 12000) (hr : r = 0.10) (hn : n = 2) (ht : t = 1) :
  final_amount P r n t = 12607.50 :=
by
  sorry

end amount_received_by_Sam_l61_61471


namespace probability_black_ball_BoxB_higher_l61_61326

def boxA_red_balls : ℕ := 40
def boxA_black_balls : ℕ := 10
def boxB_red_balls : ℕ := 60
def boxB_black_balls : ℕ := 40
def boxB_white_balls : ℕ := 50

theorem probability_black_ball_BoxB_higher :
  (boxA_black_balls : ℚ) / (boxA_red_balls + boxA_black_balls) <
  (boxB_black_balls : ℚ) / (boxB_red_balls + boxB_black_balls + boxB_white_balls) :=
by
  sorry

end probability_black_ball_BoxB_higher_l61_61326


namespace triangle_area_l61_61371

noncomputable def area_triangle_ACD (t p : ℝ) : ℝ :=
  1 / 2 * p * (t - 2)

theorem triangle_area (t p : ℝ) (ht : 0 < t ∧ t < 12) (hp : 0 < p ∧ p < 12) :
  area_triangle_ACD t p = 1 / 2 * p * (t - 2) :=
sorry

end triangle_area_l61_61371


namespace minimize_expression_l61_61257

theorem minimize_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (cond1 : x + y > z) (cond2 : y + z > x) (cond3 : z + x > y) :
  (x + y + z) * (1 / (x + y - z) + 1 / (y + z - x) + 1 / (z + x - y)) ≥ 9 :=
by
  sorry

end minimize_expression_l61_61257


namespace least_number_to_add_l61_61558

theorem least_number_to_add (n : ℕ) (h : n = 28523) : 
  ∃ x, x + n = 29560 ∧ 3 ∣ (x + n) ∧ 5 ∣ (x + n) ∧ 7 ∣ (x + n) ∧ 8 ∣ (x + n) :=
by 
  sorry

end least_number_to_add_l61_61558


namespace sum_composite_l61_61377

theorem sum_composite (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 34 * a = 43 * b) : ∃ d : ℕ, d > 1 ∧ d < a + b ∧ d ∣ (a + b) :=
by
  sorry

end sum_composite_l61_61377


namespace max_servings_l61_61267

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end max_servings_l61_61267


namespace line_ellipse_tangent_l61_61475

theorem line_ellipse_tangent (m : ℝ) (h : ∃ x y : ℝ, y = 2 * m * x + 2 ∧ 2 * x^2 + 8 * y^2 = 8) :
  m^2 = 3 / 16 :=
sorry

end line_ellipse_tangent_l61_61475


namespace similar_triangle_leg_length_l61_61232

theorem similar_triangle_leg_length (a b c : ℝ) (h0 : a = 12) (h1 : b = 9) (h2 : c = 7.5) :
  ∃ y : ℝ, ((12 / 7.5) = (9 / y) → y = 5.625) :=
by
  use 5.625
  intro h
  linarith

end similar_triangle_leg_length_l61_61232


namespace save_percentage_l61_61259

theorem save_percentage (I S : ℝ) 
  (h1 : 1.5 * I - 2 * S + (I - S) = 2 * (I - S))
  (h2 : I ≠ 0) : 
  S / I = 0.5 :=
by sorry

end save_percentage_l61_61259


namespace simplify_fraction_l61_61509

theorem simplify_fraction (x : ℚ) : 
  (↑(x + 2) / 4 + ↑(3 - 4 * x) / 3 : ℚ) = ((-13 * x + 18) / 12 : ℚ) :=
by 
  sorry

end simplify_fraction_l61_61509


namespace calories_burned_l61_61076

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

end calories_burned_l61_61076


namespace skipping_ropes_l61_61814

theorem skipping_ropes (length1 length2 : ℕ) (h1 : length1 = 18) (h2 : length2 = 24) :
  ∃ (max_length : ℕ) (num_ropes : ℕ),
    max_length = Nat.gcd length1 length2 ∧
    max_length = 6 ∧
    num_ropes = length1 / max_length + length2 / max_length ∧
    num_ropes = 7 :=
by
  have max_length : ℕ := Nat.gcd length1 length2
  have num_ropes : ℕ := length1 / max_length + length2 / max_length
  use max_length, num_ropes
  sorry

end skipping_ropes_l61_61814


namespace binomial_coefficients_sum_l61_61631

theorem binomial_coefficients_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ),
  (1 - 2 * 0)^5 = a_0 + a_1 * (1 + 0) + a_2 * (1 + 0)^2 + a_3 * (1 + 0)^3 + a_4 * (1 + 0)^4 + a_5 * (1 + 0)^5 →
  (1 - 2 * 1)^5 = (-1)^5 * a_5 →
  a_0 + a_1 + a_2 + a_3 + a_4 = 33 :=
by sorry

end binomial_coefficients_sum_l61_61631


namespace no_integer_roots_l61_61549

def cubic_polynomial (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots (a b c d : ℤ) (h1 : cubic_polynomial a b c d 1 = 2015) (h2 : cubic_polynomial a b c d 2 = 2017) :
  ∀ x : ℤ, cubic_polynomial a b c d x ≠ 2016 :=
by
  sorry

end no_integer_roots_l61_61549


namespace trigonometric_identity_proof_l61_61111

theorem trigonometric_identity_proof (α : ℝ) :
  3.3998 * (Real.cos α) ^ 4 - 4 * (Real.cos α) ^ 3 - 8 * (Real.cos α) ^ 2 + 3 * Real.cos α + 1 =
  -2 * Real.sin (7 * α / 2) * Real.sin (α / 2) :=
by
  sorry

end trigonometric_identity_proof_l61_61111


namespace parabola_focus_l61_61953

noncomputable def focus_of_parabola (a b : ℝ) : ℝ × ℝ :=
  (0, 1 / (4 * a) - b)

theorem parabola_focus : focus_of_parabola 4 3 = (0, -47 / 16) :=
by
  -- Function definition: focus_of_parabola a b gives the focus of y = ax^2 - b
  -- Given: a = 4, b = 3
  -- Focus: (0, 1 / (4 * 4) - 3)
  -- Proof: Skipping detailed algebraic manipulation, assume function correctness
  sorry

end parabola_focus_l61_61953


namespace k_ge_a_l61_61170

theorem k_ge_a (a k : ℕ) (h_pos_a : 0 < a) (h_pos_k : 0 < k) 
  (h_div : (a ^ 2 + k) ∣ (a - 1) * a * (a + 1)) : k ≥ a := 
sorry

end k_ge_a_l61_61170


namespace period_f_2pi_max_value_f_exists_max_f_l61_61809

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem period_f_2pi : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

theorem max_value_f : ∀ x : ℝ, f x ≤ Real.sin 1 + 1 := by
  sorry

-- Optional: Existence of the maximum value.
theorem exists_max_f : ∃ x : ℝ, f x = Real.sin 1 + 1 := by
  sorry

end period_f_2pi_max_value_f_exists_max_f_l61_61809


namespace linear_system_k_value_l61_61724

theorem linear_system_k_value (x y k : ℝ) (h1 : x + 3 * y = 2 * k + 1) (h2 : x - y = 1) (h3 : x = -y) : k = -1 :=
sorry

end linear_system_k_value_l61_61724


namespace Mr_Deane_filled_today_l61_61888

theorem Mr_Deane_filled_today :
  ∀ (x : ℝ),
    (25 * (1.4 - 0.4) + 1.4 * x = 39) →
    x = 10 :=
by
  intros x h
  sorry

end Mr_Deane_filled_today_l61_61888


namespace ratio_of_siblings_l61_61204

/-- Let's define the sibling relationships and prove the ratio of Janet's to Masud's siblings is 3 to 1. -/
theorem ratio_of_siblings (masud_siblings : ℕ) (carlos_siblings janet_siblings : ℕ)
  (h1 : masud_siblings = 60)
  (h2 : carlos_siblings = 3 * masud_siblings / 4)
  (h3 : janet_siblings = carlos_siblings + 135) 
  (h4 : janet_siblings < some_mul * masud_siblings) : 
  janet_siblings / masud_siblings = 3 :=
by
  sorry

end ratio_of_siblings_l61_61204


namespace proof_problem_l61_61117

def diamond (a b : ℚ) := a - (1 / b)

theorem proof_problem :
  ((diamond (diamond 2 4) 5) - (diamond 2 (diamond 4 5))) = (-71 / 380) := by
  sorry

end proof_problem_l61_61117


namespace shaded_region_area_l61_61317

theorem shaded_region_area (RS : ℝ) (n_shaded : ℕ)
  (h1 : RS = 10) (h2 : n_shaded = 20) :
  (20 * (RS / (2 * Real.sqrt 2))^2) = 250 :=
by
  sorry

end shaded_region_area_l61_61317


namespace avg_of_14_23_y_is_21_l61_61333

theorem avg_of_14_23_y_is_21 (y : ℝ) (h : (14 + 23 + y) / 3 = 21) : y = 26 :=
by
  sorry

end avg_of_14_23_y_is_21_l61_61333


namespace ratio_of_Frederick_to_Tyson_l61_61683

-- Definitions of the ages based on given conditions
def Kyle : Nat := 25
def Tyson : Nat := 20
def Julian : Nat := Kyle - 5
def Frederick : Nat := Julian + 20

-- The ratio of Frederick's age to Tyson's age
def ratio : Nat × Nat := (Frederick / Nat.gcd Frederick Tyson, Tyson / Nat.gcd Frederick Tyson)

-- Proving the ratio is 2:1
theorem ratio_of_Frederick_to_Tyson : ratio = (2, 1) := by
  sorry

end ratio_of_Frederick_to_Tyson_l61_61683


namespace find_divisor_l61_61887

theorem find_divisor (d : ℕ) : 15 = (d * 4) + 3 → d = 3 := by
  intros h
  have h1 : 15 - 3 = 4 * d := by
    linarith
  have h2 : 12 = 4 * d := by
    linarith
  have h3 : d = 3 := by
    linarith
  exact h3

end find_divisor_l61_61887


namespace pizza_consumption_order_l61_61800

theorem pizza_consumption_order :
  let e := 1/6
  let s := 1/4
  let n := 1/3
  let o := 1/8
  let j := 1 - e - s - n - o
  (n > s) ∧ (s > e) ∧ (e = j) ∧ (j > o) :=
by
  sorry

end pizza_consumption_order_l61_61800


namespace proportion_solution_l61_61120

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 :=
by
  sorry

end proportion_solution_l61_61120


namespace quadratic_equation_roots_l61_61718

theorem quadratic_equation_roots {x y : ℝ}
  (h1 : x + y = 10)
  (h2 : |x - y| = 4)
  (h3 : x * y = 21) : (x - 7) * (x - 3) = 0 ∨ (x - 3) * (x - 7) = 0 :=
by
  sorry

end quadratic_equation_roots_l61_61718


namespace counting_unit_of_0_75_l61_61013

def decimal_places (n : ℝ) : ℕ := 
  by sorry  -- Assume this function correctly calculates the number of decimal places of n

def counting_unit (n : ℝ) : ℝ :=
  by sorry  -- Assume this function correctly determines the counting unit based on decimal places

theorem counting_unit_of_0_75 : counting_unit 0.75 = 0.01 :=
  by sorry


end counting_unit_of_0_75_l61_61013


namespace simplify_expression_l61_61501

theorem simplify_expression (x y z : ℝ) : (x + (y + z)) - ((x + z) + y) = 0 :=
by
  sorry

end simplify_expression_l61_61501


namespace reciprocal_of_neg_2023_l61_61717

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l61_61717


namespace time_passed_since_midnight_l61_61715

theorem time_passed_since_midnight (h : ℝ) :
  h = (12 - h) + (2/5) * h → h = 7.5 :=
by
  sorry

end time_passed_since_midnight_l61_61715


namespace ones_digit_power_sum_l61_61301

noncomputable def ones_digit_of_power_sum_is_5 : Prop :=
  (1^2010 + 2^2010 + 3^2010 + 4^2010 + 5^2010 + 6^2010 + 7^2010 + 8^2010 + 9^2010 + 10^2010) % 10 = 5

theorem ones_digit_power_sum : ones_digit_of_power_sum_is_5 :=
  sorry

end ones_digit_power_sum_l61_61301


namespace dice_probability_l61_61369

def first_die_prob : ℚ := 3 / 8
def second_die_prob : ℚ := 3 / 4
def combined_prob : ℚ := first_die_prob * second_die_prob

theorem dice_probability :
  combined_prob = 9 / 32 :=
by
  -- Here we write the proof steps.
  sorry

end dice_probability_l61_61369


namespace triangle_DEF_area_l61_61554

theorem triangle_DEF_area (DE height : ℝ) (hDE : DE = 12) (hHeight : height = 15) : 
  (1/2) * DE * height = 90 :=
by
  rw [hDE, hHeight]
  norm_num

end triangle_DEF_area_l61_61554


namespace total_cookies_l61_61151

theorem total_cookies (x y : Nat) (h1 : x = 137) (h2 : y = 251) : x * y = 34387 := by
  sorry

end total_cookies_l61_61151


namespace possible_values_l61_61112

theorem possible_values (a : ℝ) (h : a > 1) : ∃ (v : ℝ), (v = 5 ∨ v = 6 ∨ v = 7) ∧ (a + 4 / (a - 1) = v) :=
sorry

end possible_values_l61_61112


namespace complex_quadrant_l61_61557

def z1 := Complex.mk 1 (-2)
def z2 := Complex.mk 2 1
def z := z1 * z2

theorem complex_quadrant : z = Complex.mk 4 (-3) ∧ z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l61_61557


namespace gcd_of_lcm_l61_61902

noncomputable def gcd (A B C : ℕ) : ℕ := Nat.gcd (Nat.gcd A B) C
noncomputable def lcm (A B C : ℕ) : ℕ := Nat.lcm (Nat.lcm A B) C

theorem gcd_of_lcm (A B C : ℕ) (LCM_ABC : ℕ) (Product_ABC : ℕ) :
  lcm A B C = LCM_ABC →
  A * B * C = Product_ABC →
  gcd A B C = 20 :=
by
  intros lcm_eq product_eq
  sorry

end gcd_of_lcm_l61_61902


namespace problem_statement_l61_61675

theorem problem_statement (x : ℕ) (h : x = 2016) : (x^2 - x) - (x^2 - 2 * x + 1) = 2015 := by
  sorry

end problem_statement_l61_61675


namespace sphere_intersection_circle_radius_l61_61743

theorem sphere_intersection_circle_radius
  (x1 y1 z1: ℝ) (x2 y2 z2: ℝ) (r1 r2: ℝ)
  (hyp1: x1 = 3) (hyp2: y1 = 5) (hyp3: z1 = 0) 
  (hyp4: r1 = 2) 
  (hyp5: x2 = 0) (hyp6: y2 = 5) (hyp7: z2 = -8) :
  r2 = Real.sqrt 59 := 
by
  sorry

end sphere_intersection_circle_radius_l61_61743


namespace students_wearing_blue_lipstick_l61_61075

theorem students_wearing_blue_lipstick
  (total_students : ℕ)
  (half_students_wore_lipstick : total_students / 2 = 180)
  (red_fraction : ℚ)
  (pink_fraction : ℚ)
  (purple_fraction : ℚ)
  (green_fraction : ℚ)
  (students_wearing_red : red_fraction * 180 = 45)
  (students_wearing_pink : pink_fraction * 180 = 60)
  (students_wearing_purple : purple_fraction * 180 = 30)
  (students_wearing_green : green_fraction * 180 = 15)
  (total_red_fraction : red_fraction = 1 / 4)
  (total_pink_fraction : pink_fraction = 1 / 3)
  (total_purple_fraction : purple_fraction = 1 / 6)
  (total_green_fraction : green_fraction = 1 / 12) :
  (180 - (45 + 60 + 30 + 15) = 30) :=
by sorry

end students_wearing_blue_lipstick_l61_61075


namespace num_ways_to_assign_grades_l61_61417

theorem num_ways_to_assign_grades : (4 ^ 12) = 16777216 := by
  sorry

end num_ways_to_assign_grades_l61_61417


namespace min_omega_l61_61022

noncomputable def f (ω φ : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem min_omega (ω φ : ℝ) (hω : ω > 0)
  (h_sym : ∀ x : ℝ, f ω φ (2 * (π / 3) - x) = f ω φ x)
  (h_val : f ω φ (π / 12) = 0) :
  ω = 2 :=
sorry

end min_omega_l61_61022


namespace roots_real_l61_61794

variable {x p q k : ℝ}
variable {x1 x2 : ℝ}

theorem roots_real 
  (h1 : x^2 + p * x + q = 0) 
  (h2 : p = -(x1 + x2)) 
  (h3 : q = x1 * x2) 
  (h4 : x1 ≠ x2) 
  (h5 :  x1^2 - 2*x1*x2 + x2^2 + 4*q = 0):
  (∃ y1 y2, y1 = k * x1 + (1 / k) * x2 ∧ y2 = k * x2 + (1 / k) * x1 ∧ 
    (y1^2 + (k + 1/k) * p * y1 + (p^2 + q * ((k - 1/k)^2)) = 0) ∧ 
    (y2^2 + (k + 1/k) * p * y2 + (p^2 + q * ((k - 1/k)^2)) = 0)) → 
  (∃ z1 z2, z1 = k * x1 ∧ z2 = 1/k * x2 ∧ 
    (z1^2 - y1 * z1 + q = 0) ∧ 
    (z2^2 - y2 * z2 + q = 0)) :=
sorry

end roots_real_l61_61794


namespace log_49_48_in_terms_of_a_and_b_l61_61281

-- Define the constants and hypotheses
variable (a b : ℝ)
variable (h1 : a = Real.logb 7 3)
variable (h2 : b = Real.logb 7 4)

-- Define the statement to be proved
theorem log_49_48_in_terms_of_a_and_b (a b : ℝ) (h1 : a = Real.logb 7 3) (h2 : b = Real.logb 7 4) :
  Real.logb 49 48 = (a + 2 * b) / 2 :=
by
  sorry

end log_49_48_in_terms_of_a_and_b_l61_61281


namespace range_of_a_l61_61674

def is_odd_function (f : ℝ → ℝ) := 
  ∀ x : ℝ, f (-x) = - f x

noncomputable def f (x : ℝ) :=
  if x ≥ 0 then x^2 + 2*x else -(x^2 + 2*(-x))

theorem range_of_a (a : ℝ) (h_odd : is_odd_function f) 
(hf_pos : ∀ x : ℝ, x ≥ 0 → f x = x^2 + 2*x) : 
  f (2 - a^2) > f a → -2 < a ∧ a < 1 :=
sorry

end range_of_a_l61_61674


namespace sequence_formula_l61_61474

-- Define the properties of the sequence
axiom seq_prop_1 (a : ℕ → ℝ) (m n : ℕ) (h : m > n) : a (m - n) = a m - a n

axiom seq_increasing (a : ℕ → ℝ) : ∀ n m : ℕ, n < m → a n < a m

-- Formulate the theorem to prove the general sequence formula
theorem sequence_formula (a : ℕ → ℝ) (h1 : ∀ m n : ℕ, m > n → a (m - n) = a m - a n)
    (h2 : ∀ n m : ℕ, n < m → a n < a m) :
    ∃ k > 0, ∀ n, a n = k * n :=
sorry

end sequence_formula_l61_61474


namespace line_interparabola_length_l61_61939

theorem line_interparabola_length :
  (∀ (x y : ℝ), y = x - 2 → y^2 = 4 * x) →
  ∃ (A B : ℝ × ℝ), (∃ (x1 y1 x2 y2 : ℝ), A = (x1, y1) ∧ B = (x2, y2)) →
  (dist A B = 4 * Real.sqrt 6) :=
by
  intros
  sorry

end line_interparabola_length_l61_61939


namespace average_age_before_new_students_l61_61334

theorem average_age_before_new_students
  (A : ℝ) (N : ℕ)
  (h1 : N = 15)
  (h2 : 15 * 32 + N * A = (N + 15) * (A - 4)) :
  A = 40 :=
by {
  sorry
}

end average_age_before_new_students_l61_61334


namespace triangle_side_length_BC_49_l61_61004

theorem triangle_side_length_BC_49
  (angle_A : ℝ)
  (AC : ℝ)
  (area_ABC : ℝ)
  (h1 : angle_A = 60)
  (h2 : AC = 16)
  (h3 : area_ABC = 220 * Real.sqrt 3) : 
  ∃ (BC : ℝ), BC = 49 :=
by
  sorry

end triangle_side_length_BC_49_l61_61004


namespace order_of_x_y_z_l61_61382

theorem order_of_x_y_z (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  let x : ℝ := (a + b) * (c + d)
  let y : ℝ := (a + c) * (b + d)
  let z : ℝ := (a + d) * (b + c)
  x < y ∧ y < z :=
by
  let x : ℝ := (a + b) * (c + d)
  let y : ℝ := (a + c) * (b + d)
  let z : ℝ := (a + d) * (b + c)
  sorry

end order_of_x_y_z_l61_61382


namespace surface_area_of_circumscribed_sphere_l61_61890

/-- 
  Problem: Determine the surface area of the sphere circumscribed about a cube with edge length 2.

  Given:
  - The edge length of the cube is 2.
  - The space diagonal of a cube with edge length \(a\) is given by \(d = \sqrt{3} \cdot a\).
  - The diameter of the circumscribed sphere is equal to the space diagonal of the cube.
  - The surface area \(S\) of a sphere with radius \(R\) is given by \(S = 4\pi R^2\).

  To Prove:
  - The surface area of the sphere circumscribed about the cube is \(12\pi\).
-/
theorem surface_area_of_circumscribed_sphere (a : ℝ) (π : ℝ) (h1 : a = 2) 
  (h2 : ∀ a, d = Real.sqrt 3 * a) (h3 : ∀ d, R = d / 2) (h4 : ∀ R, S = 4 * π * R^2) : 
  S = 12 * π := 
by
  sorry

end surface_area_of_circumscribed_sphere_l61_61890


namespace tank_capacity_l61_61808

variable (C : ℝ)

theorem tank_capacity (h : (3/4) * C + 9 = (7/8) * C) : C = 72 :=
by
  sorry

end tank_capacity_l61_61808


namespace parallelogram_area_l61_61705

theorem parallelogram_area (base height : ℝ) (h_base : base = 10) (h_height : height = 7) :
  base * height = 70 := by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l61_61705


namespace probability_no_more_than_10_seconds_l61_61410

noncomputable def total_cycle_time : ℕ := 80
noncomputable def green_time : ℕ := 30
noncomputable def yellow_time : ℕ := 10
noncomputable def red_time : ℕ := 40
noncomputable def can_proceed : ℕ := green_time + yellow_time + yellow_time

theorem probability_no_more_than_10_seconds : 
  can_proceed / total_cycle_time = 5 / 8 := 
  sorry

end probability_no_more_than_10_seconds_l61_61410


namespace problem_statement_l61_61175

noncomputable def a (n : ℕ) := n^2

theorem problem_statement (x : ℝ) (hx : x > 0) (n : ℕ) (hn : n > 0) :
  x + a n / x ^ n ≥ n + 1 :=
sorry

end problem_statement_l61_61175


namespace length_of_rectangle_l61_61419

-- Definitions based on conditions:
def side_length_square : ℝ := 4
def width_rectangle : ℝ := 8
def area_square (side : ℝ) : ℝ := side * side
def area_rectangle (width length : ℝ) : ℝ := width * length

-- The goal is to prove the length of the rectangle
theorem length_of_rectangle :
  (area_square side_length_square) = (area_rectangle width_rectangle 2) :=
by
  sorry

end length_of_rectangle_l61_61419


namespace product_of_areas_eq_square_of_volume_l61_61414

-- define the dimensions of the prism
variables (x y z : ℝ)

-- define the areas of the faces as conditions
def top_area := x * y
def back_area := y * z
def lateral_face_area := z * x

-- define the product of the areas of the top, back, and one lateral face
def product_of_areas := (top_area x y) * (back_area y z) * (lateral_face_area z x)

-- define the volume of the prism
def volume := x * y * z

-- theorem to prove: product of areas equals square of the volume
theorem product_of_areas_eq_square_of_volume 
  (ht: top_area x y = x * y)
  (hb: back_area y z = y * z)
  (hl: lateral_face_area z x = z * x) :
  product_of_areas x y z = (volume x y z) ^ 2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l61_61414


namespace total_population_expression_l61_61603

variables (b g t: ℕ)

-- Assuming the given conditions
def condition1 := b = 4 * g
def condition2 := g = 8 * t

-- The theorem to prove
theorem total_population_expression (h1 : condition1 b g) (h2 : condition2 g t) :
    b + g + t = 41 * b / 32 := sorry

end total_population_expression_l61_61603


namespace rate_of_interest_l61_61132

/-
Let P be the principal amount, SI be the simple interest paid, R be the rate of interest, and N be the number of years. 
The problem states:
- P = 1200
- SI = 432
- R = N

We need to prove that R = 6.
-/

theorem rate_of_interest (P SI R N : ℝ) (h1 : P = 1200) (h2 : SI = 432) (h3 : R = N) :
  R = 6 :=
  sorry

end rate_of_interest_l61_61132


namespace probability_two_blue_marbles_l61_61290

theorem probability_two_blue_marbles (h_red: ℕ := 3) (h_blue: ℕ := 4) (h_white: ℕ := 9) :
  (h_blue / (h_red + h_blue + h_white)) * ((h_blue - 1) / ((h_red + h_blue + h_white) - 1)) = 1 / 20 :=
by sorry

end probability_two_blue_marbles_l61_61290


namespace find_x_plus_y_l61_61691

noncomputable def det3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

noncomputable def det2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem find_x_plus_y (x y : ℝ) (h1 : x ≠ y)
  (h2 : det3x3 2 5 10 4 x y 4 y x = 0)
  (h3 : det2x2 x y y x = 16) : x + y = 30 := by
  sorry

end find_x_plus_y_l61_61691


namespace total_workers_is_22_l61_61861

-- Define constants and variables based on conditions
def avg_salary_all : ℝ := 850
def avg_salary_technicians : ℝ := 1000
def avg_salary_rest : ℝ := 780
def num_technicians : ℝ := 7

-- Define the necessary proof statement
theorem total_workers_is_22
  (W : ℝ)
  (h1 : W * avg_salary_all = num_technicians * avg_salary_technicians + (W - num_technicians) * avg_salary_rest) :
  W = 22 :=
by
  sorry

end total_workers_is_22_l61_61861


namespace teacher_age_l61_61488

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_with_teacher : ℕ) (num_total : ℕ) 
  (h1 : avg_age_students = 14) (h2 : num_students = 50) (h3 : avg_age_with_teacher = 15) (h4 : num_total = 51) :
  ∃ (teacher_age : ℕ), teacher_age = 65 :=
by sorry

end teacher_age_l61_61488


namespace sum_to_fraction_l61_61543

theorem sum_to_fraction :
  (2 / 10) + (3 / 100) + (4 / 1000) + (6 / 10000) + (7 / 100000) = 23467 / 100000 :=
by
  sorry

end sum_to_fraction_l61_61543


namespace triangle_area_l61_61470

theorem triangle_area : 
  let p1 := (3, 2)
  let p2 := (3, -4)
  let p3 := (12, 2)
  let height := |2 - (-4)|
  let base := |12 - 3|
  let area := (1 / 2) * base * height
  area = 27 := sorry

end triangle_area_l61_61470


namespace number_of_girls_in_class_l61_61361

variable (B S G : ℕ)

theorem number_of_girls_in_class
  (h1 : (3 / 4 : ℚ) * B = 18)
  (h2 : B = (2 / 3 : ℚ) * S) :
  G = S - B → G = 12 := by
  intro hg
  sorry

end number_of_girls_in_class_l61_61361


namespace sum_first_100_even_numbers_divisible_by_6_l61_61703

-- Define the sequence of even numbers divisible by 6 between 100 and 300 inclusive.
def even_numbers_divisible_by_6 (n : ℕ) : ℕ := 102 + n * 6

-- Define the sum of the first 100 even numbers divisible by 6.
def sum_even_numbers_divisible_by_6 (k : ℕ) : ℕ := k / 2 * (102 + (102 + (k - 1) * 6))

-- Define the problem statement as a theorem.
theorem sum_first_100_even_numbers_divisible_by_6 :
  sum_even_numbers_divisible_by_6 100 = 39900 :=
by
  sorry

end sum_first_100_even_numbers_divisible_by_6_l61_61703


namespace kangaroo_fiber_intake_l61_61836

-- Suppose kangaroos absorb only 30% of the fiber they eat
def absorption_rate : ℝ := 0.30

-- If a kangaroo absorbed 15 ounces of fiber in one day
def absorbed_fiber : ℝ := 15.0

-- Prove the kangaroo ate 50 ounces of fiber that day
theorem kangaroo_fiber_intake (x : ℝ) (hx : absorption_rate * x = absorbed_fiber) : x = 50 :=
by
  sorry

end kangaroo_fiber_intake_l61_61836


namespace proof_problem_l61_61748

variable {a b c : ℝ}

theorem proof_problem (h1 : ∀ x : ℝ, 4 * x^2 - 3 * x + 1 = a * (x - 1)^2 + b * (x - 1) + c) : 
  (4 * a + 2 * b + c = 28) := by
  -- The proof goes here. The goal statement is what we need.
  sorry

end proof_problem_l61_61748


namespace roger_total_miles_l61_61464

def morning_miles : ℕ := 2
def evening_multiplicative_factor : ℕ := 5
def evening_miles := evening_multiplicative_factor * morning_miles
def third_session_subtract : ℕ := 1
def third_session_miles := (2 * morning_miles) - third_session_subtract
def total_miles := morning_miles + evening_miles + third_session_miles

theorem roger_total_miles : total_miles = 15 := by
  sorry

end roger_total_miles_l61_61464


namespace angle_between_line_and_plane_l61_61742

open Real

def plane1 (x y z : ℝ) : Prop := 2*x - y - 3*z + 5 = 0
def plane2 (x y z : ℝ) : Prop := x + y - 2 = 0

def point_M : ℝ × ℝ × ℝ := (-2, 0, 3)
def point_N : ℝ × ℝ × ℝ := (0, 2, 2)
def point_K : ℝ × ℝ × ℝ := (3, -3, 1)

theorem angle_between_line_and_plane :
  ∃ α : ℝ, α = arcsin (22 / (3 * sqrt 102)) :=
by sorry

end angle_between_line_and_plane_l61_61742


namespace integer_solutions_l61_61053

def satisfies_equation (x y : ℤ) : Prop := x^2 = y^2 * (x + y^4 + 2 * y^2)

theorem integer_solutions :
  {p : ℤ × ℤ | satisfies_equation p.1 p.2} = { (0, 0), (12, 2), (-8, 2) } :=
by sorry

end integer_solutions_l61_61053


namespace apron_more_than_recipe_book_l61_61994

-- Define the prices and the total spent
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def total_ingredient_cost : ℕ := 5 * ingredient_cost
def total_spent : ℕ := 40

-- Define the condition that the total cost including the apron is $40
def total_without_apron : ℕ := recipe_book_cost + baking_dish_cost + total_ingredient_cost
def apron_cost : ℕ := total_spent - total_without_apron

-- Prove that the apron cost $1 more than the recipe book
theorem apron_more_than_recipe_book : apron_cost - recipe_book_cost = 1 := by
  -- The proof goes here
  sorry

end apron_more_than_recipe_book_l61_61994


namespace total_songs_correct_l61_61966

-- Define the conditions of the problem
def num_country_albums := 2
def songs_per_country_album := 12
def num_pop_albums := 8
def songs_per_pop_album := 7
def num_rock_albums := 5
def songs_per_rock_album := 10
def num_jazz_albums := 2
def songs_per_jazz_album := 15

-- Define the total number of songs
def total_songs :=
  num_country_albums * songs_per_country_album +
  num_pop_albums * songs_per_pop_album +
  num_rock_albums * songs_per_rock_album +
  num_jazz_albums * songs_per_jazz_album

-- Proposition stating the correct total number of songs
theorem total_songs_correct : total_songs = 160 :=
by {
  sorry -- Proof not required
}

end total_songs_correct_l61_61966


namespace remaining_laps_l61_61598

def track_length : ℕ := 9
def initial_laps : ℕ := 6
def total_distance : ℕ := 99

theorem remaining_laps : (total_distance - (initial_laps * track_length)) / track_length = 5 := by
  sorry

end remaining_laps_l61_61598


namespace train_crosses_signal_post_in_40_seconds_l61_61778

noncomputable def time_to_cross_signal_post : Nat := 40

theorem train_crosses_signal_post_in_40_seconds
  (train_length : Nat) -- Length of the train in meters
  (bridge_length_km : Nat) -- Length of the bridge in kilometers
  (bridge_cross_time_min : Nat) -- Time to cross the bridge in minutes
  (constant_speed : Prop) -- Assumption that the speed is constant
  (h1 : train_length = 600) -- Train is 600 meters long
  (h2 : bridge_length_km = 9) -- Bridge is 9 kilometers long
  (h3 : bridge_cross_time_min = 10) -- Time to cross the bridge is 10 minutes
  (h4 : constant_speed) -- The train's speed is constant
  : time_to_cross_signal_post = 40 :=
sorry

end train_crosses_signal_post_in_40_seconds_l61_61778


namespace value_of_x_plus_y_squared_l61_61481

variable (x y : ℝ)

def condition1 : Prop := x * (x + y) = 40
def condition2 : Prop := y * (x + y) = 90
def condition3 : Prop := x - y = 5

theorem value_of_x_plus_y_squared (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : (x + y) ^ 2 = 130 :=
by
  sorry

end value_of_x_plus_y_squared_l61_61481


namespace ship_with_highest_no_car_round_trip_percentage_l61_61157

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end ship_with_highest_no_car_round_trip_percentage_l61_61157


namespace train_pass_bridge_time_l61_61842

noncomputable def trainLength : ℝ := 360
noncomputable def trainSpeedKMH : ℝ := 45
noncomputable def bridgeLength : ℝ := 160
noncomputable def totalDistance : ℝ := trainLength + bridgeLength
noncomputable def trainSpeedMS : ℝ := trainSpeedKMH * (1000 / 3600)
noncomputable def timeToPassBridge : ℝ := totalDistance / trainSpeedMS

theorem train_pass_bridge_time : timeToPassBridge = 41.6 := sorry

end train_pass_bridge_time_l61_61842


namespace range_of_a_l61_61180

-- Definitions of sets and the problem conditions
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}
def condition (a : ℝ) : Prop := P ∪ M a = P

-- The theorem stating what needs to be proven
theorem range_of_a (a : ℝ) (h : condition a) : -1 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l61_61180


namespace marble_count_l61_61671

-- Definitions from conditions
variable (M P : ℕ)
def condition1 : Prop := M = 26 * P
def condition2 : Prop := M = 28 * (P - 1)

-- Theorem to be proved
theorem marble_count (h1 : condition1 M P) (h2 : condition2 M P) : M = 364 := 
by
  sorry

end marble_count_l61_61671


namespace best_fitting_model_is_model3_l61_61289

-- Definitions of the coefficients of determination for the models
def R2_model1 : ℝ := 0.60
def R2_model2 : ℝ := 0.90
def R2_model3 : ℝ := 0.98
def R2_model4 : ℝ := 0.25

-- The best fitting effect corresponds to the highest R^2 value
theorem best_fitting_model_is_model3 :
  R2_model3 = max (max R2_model1 R2_model2) (max R2_model3 R2_model4) :=
by {
  -- Proofblock is skipped, using sorry
  sorry
}

end best_fitting_model_is_model3_l61_61289


namespace Talia_father_age_l61_61968

def Talia_age (T : ℕ) : Prop := T + 7 = 20
def Talia_mom_age (M T : ℕ) : Prop := M = 3 * T
def Talia_father_age_in_3_years (F M : ℕ) : Prop := F + 3 = M

theorem Talia_father_age (T F M : ℕ) 
    (hT : Talia_age T)
    (hM : Talia_mom_age M T)
    (hF : Talia_father_age_in_3_years F M) :
    F = 36 :=
by 
  sorry

end Talia_father_age_l61_61968


namespace sarah_numbers_sum_l61_61586

-- Definition of x and y being integers with their respective ranges
def isTwoDigit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99
def isThreeDigit (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999

-- The condition relating x and y
def formedNumber (x y : ℕ) : Prop := 1000 * x + y = 7 * x * y

-- The Lean 4 statement for the proof problem
theorem sarah_numbers_sum (x y : ℕ) (H1 : isTwoDigit x) (H2 : isThreeDigit y) (H3 : formedNumber x y) : x + y = 1074 :=
  sorry

end sarah_numbers_sum_l61_61586


namespace inequality_solution_l61_61160

noncomputable def ratFunc (x : ℝ) : ℝ := 
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution (x : ℝ) : 
  (ratFunc x > 0) ↔ 
  ((x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) := 
by
  sorry

end inequality_solution_l61_61160


namespace sum_of_eight_numbers_l61_61525

theorem sum_of_eight_numbers (avg : ℝ) (num_of_items : ℕ) (h_avg : avg = 5.3) (h_items : num_of_items = 8) :
  avg * num_of_items = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l61_61525


namespace baskets_of_peaches_l61_61805

theorem baskets_of_peaches (n : ℕ) :
  (∀ x : ℕ, (n * 2 = 14) → (n = x)) := by
  sorry

end baskets_of_peaches_l61_61805


namespace coats_collected_in_total_l61_61207

def high_school_coats : Nat := 6922
def elementary_school_coats : Nat := 2515
def total_coats : Nat := 9437

theorem coats_collected_in_total : 
  high_school_coats + elementary_school_coats = total_coats := 
  by
  sorry

end coats_collected_in_total_l61_61207


namespace distribution_of_balls_l61_61734

-- Definition for the problem conditions
inductive Ball : Type
| one : Ball
| two : Ball
| three : Ball
| four : Ball

inductive Box : Type
| box1 : Box
| box2 : Box
| box3 : Box

-- Function to count the number of ways to distribute the balls according to the conditions
noncomputable def num_ways_to_distribute_balls : Nat := 18

-- Theorem statement
theorem distribution_of_balls :
  num_ways_to_distribute_balls = 18 := by
  sorry

end distribution_of_balls_l61_61734


namespace g_eval_1000_l61_61673

def g (n : ℕ) : ℕ := sorry
axiom g_comp (n : ℕ) : g (g n) = 2 * n
axiom g_form (n : ℕ) : g (3 * n + 1) = 3 * n + 2

theorem g_eval_1000 : g 1000 = 1008 :=
by
  sorry

end g_eval_1000_l61_61673


namespace intersecting_circles_range_l61_61483

theorem intersecting_circles_range {k : ℝ} (a b : ℝ) :
  (-36 : ℝ) ≤ k ∧ k ≤ 104 →
  (∃ (x y : ℝ), (x^2 + y^2 - 4 - 12 * x + 6 * y) = 0 ∧ (x^2 + y^2 = k + 4 * x + 12 * y)) →
  b - a = (140 : ℝ) :=
by
  intro hk hab
  sorry

end intersecting_circles_range_l61_61483


namespace tiles_touching_walls_of_room_l61_61625

theorem tiles_touching_walls_of_room (length width : Nat) 
    (hl : length = 10) (hw : width = 5) : 
    2 * length + 2 * width - 4 = 26 := by
  sorry

end tiles_touching_walls_of_room_l61_61625


namespace percentage_of_alcohol_in_first_vessel_l61_61514

variable (x : ℝ) -- percentage of alcohol in the first vessel in decimal form, i.e., x% is represented as x/100

-- conditions
variable (v1_capacity : ℝ := 2)
variable (v2_capacity : ℝ := 6)
variable (v2_alcohol_concentration : ℝ := 0.5)
variable (total_capacity : ℝ := 10)
variable (new_concentration : ℝ := 0.37)

theorem percentage_of_alcohol_in_first_vessel :
  (x / 100) * v1_capacity + v2_alcohol_concentration * v2_capacity = new_concentration * total_capacity -> x = 35 := 
by
  sorry

end percentage_of_alcohol_in_first_vessel_l61_61514


namespace value_of_a_plus_b_l61_61487

variables (a b : ℝ)

theorem value_of_a_plus_b (h1 : a + 4 * b = 33) (h2 : 6 * a + 3 * b = 51) : a + b = 12 := 
by
  sorry

end value_of_a_plus_b_l61_61487


namespace product_mk_through_point_l61_61093

theorem product_mk_through_point (k m : ℝ) (h : (2 : ℝ) ^ m * k = (1/4 : ℝ)) : m * k = -2 := 
sorry

end product_mk_through_point_l61_61093


namespace elaine_earnings_increase_l61_61851

variable (E P : ℝ)

theorem elaine_earnings_increase :
  (0.25 * (E * (1 + P / 100)) = 1.4375 * 0.20 * E) → P = 15 :=
by
  intro h
  -- Start an intermediate transformation here
  sorry

end elaine_earnings_increase_l61_61851


namespace intersection_of_sets_l61_61492

noncomputable def set_A := {x : ℝ | Real.log x ≥ 0}
noncomputable def set_B := {x : ℝ | x^2 < 9}

theorem intersection_of_sets :
  set_A ∩ set_B = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by {
  sorry
}

end intersection_of_sets_l61_61492


namespace line_intersects_x_axis_at_10_0_l61_61907

theorem line_intersects_x_axis_at_10_0 :
  let x1 := 9
  let y1 := 1
  let x2 := 5
  let y2 := 5
  let slope := (y2 - y1) / (x2 - x1)
  let y := 0
  ∃ x, (x - x1) * slope = y - y1 ∧ y = 0 → x = 10 := by
  sorry

end line_intersects_x_axis_at_10_0_l61_61907


namespace find_grazing_months_l61_61462

def oxen_months_A := 10 * 7
def oxen_months_B := 12 * 5
def total_rent := 175
def rent_C := 45

def proportion_equation (x : ℕ) : Prop :=
  45 / 175 = (15 * x) / (oxen_months_A + oxen_months_B + 15 * x)

theorem find_grazing_months (x : ℕ) (h : proportion_equation x) : x = 3 :=
by
  -- We will need to involve some calculations leading to x = 3
  sorry

end find_grazing_months_l61_61462


namespace quadrilateral_inequality_l61_61987

theorem quadrilateral_inequality 
  (A B C D : Type)
  (AB AC AD BC BD CD : ℝ)
  (hAB_pos : 0 < AB)
  (hBC_pos : 0 < BC)
  (hCD_pos : 0 < CD)
  (hDA_pos : 0 < DA)
  (hAC_pos : 0 < AC)
  (hBD_pos : 0 < BD): 
  AC * BD ≤ AB * CD + BC * AD := 
sorry

end quadrilateral_inequality_l61_61987


namespace solution_of_loginequality_l61_61096

-- Define the conditions as inequalities
def condition1 (x : ℝ) : Prop := 2 * x - 1 > 0
def condition2 (x : ℝ) : Prop := -x + 5 > 0
def condition3 (x : ℝ) : Prop := 2 * x - 1 > -x + 5

-- Define the final solution set
def solution_set (x : ℝ) : Prop := (2 < x) ∧ (x < 5)

-- The theorem stating that under the given conditions, the solution set holds
theorem solution_of_loginequality (x : ℝ) : condition1 x ∧ condition2 x ∧ condition3 x → solution_set x :=
by
  intro h
  sorry

end solution_of_loginequality_l61_61096


namespace domain_of_h_l61_61926

-- Definition of the function domain of f(x) and h(x)
def f_domain := Set.Icc (-10: ℝ) 6
def h_domain := Set.Icc (-2: ℝ) (10/3)

-- Definition of f and h
def f (x: ℝ) : ℝ := sorry  -- f is assumed to be defined on the interval [-10, 6]
def h (x: ℝ) : ℝ := f (-3 * x)

-- Theorem statement: Given the domain of f(x), the domain of h(x) is as follows
theorem domain_of_h :
  (∀ x, x ∈ f_domain ↔ (-3 * x) ∈ h_domain) :=
sorry

end domain_of_h_l61_61926


namespace no_intersection_l61_61167

def M := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }
def N (a : ℝ) := { p : ℝ × ℝ | abs (p.1 - 1) + abs (p.2 - 1) = a }

theorem no_intersection (a : ℝ) : M ∩ (N a) = ∅ ↔ a ∈ (Set.Ioo (2-Real.sqrt 2) (2+Real.sqrt 2)) := 
by 
  sorry

end no_intersection_l61_61167


namespace equalize_costs_l61_61346

theorem equalize_costs (A B : ℝ) (h_lt : A < B) :
  (B - A) / 2 = (A + B) / 2 - A :=
by sorry

end equalize_costs_l61_61346


namespace meeting_time_coincides_l61_61163

variables (distance_ab : ℕ) (speed_train_a : ℕ) (start_time_train_a : ℕ) (distance_at_9am : ℕ) (speed_train_b : ℕ) (start_time_train_b : ℕ)

def total_distance_ab := 465
def train_a_speed := 60
def train_b_speed := 75
def start_time_a := 8
def start_time_b := 9
def distance_train_a_by_9am := train_a_speed * (start_time_b - start_time_a)
def remaining_distance := total_distance_ab - distance_train_a_by_9am
def relative_speed := train_a_speed + train_b_speed
def time_to_meet := remaining_distance / relative_speed

theorem meeting_time_coincides :
  time_to_meet = 3 → (start_time_b + time_to_meet = 12) :=
by
  sorry

end meeting_time_coincides_l61_61163


namespace find_a_l61_61642

noncomputable def polynomial (a : ℝ) : ℝ → ℝ := λ x => a * x^2 + (a - 3) * x + 1

-- This is a statement without the actual computation or proof.
theorem find_a (a : ℝ) :
  (∀ x : ℝ, polynomial a x = 0 → (∃! x, polynomial a x = 0)) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_l61_61642


namespace fraction_of_p_l61_61241

theorem fraction_of_p (p q r f : ℝ) (hp : p = 49) (hqr : p = (2 * f * 49) + 35) : f = 1/7 :=
sorry

end fraction_of_p_l61_61241


namespace minimum_a_l61_61240

theorem minimum_a (a : ℝ) (h : a > 0) :
  (∀ (N : ℝ × ℝ), (N.1 - a)^2 + (N.2 + a - 3)^2 = 1 → 
   dist (N.1, N.2) (0, 0) ≥ 2) → a ≥ 3 :=
by
  sorry

end minimum_a_l61_61240


namespace find_k_value_l61_61397

theorem find_k_value
  (x y k : ℝ)
  (h1 : 4 * x + 3 * y = 1)
  (h2 : k * x + (k - 1) * y = 3)
  (h3 : x = y) :
  k = 11 :=
  sorry

end find_k_value_l61_61397


namespace product_of_0_25_and_0_75_is_0_1875_l61_61856

noncomputable def product_of_decimals : ℝ := 0.25 * 0.75

theorem product_of_0_25_and_0_75_is_0_1875 :
  product_of_decimals = 0.1875 :=
by
  sorry

end product_of_0_25_and_0_75_is_0_1875_l61_61856


namespace find_d_from_sine_wave_conditions_l61_61779

theorem find_d_from_sine_wave_conditions (a b d : ℝ) (h1 : d + a = 4) (h2 : d - a = -2) : d = 1 :=
by {
  sorry
}

end find_d_from_sine_wave_conditions_l61_61779


namespace ab_bc_ca_max_le_l61_61129

theorem ab_bc_ca_max_le (a b c : ℝ) :
  ab + bc + ca + max (abs (a - b)) (max (abs (b - c)) (abs (c - a))) ≤
  1 + (1 / 3) * (a + b + c)^2 :=
sorry

end ab_bc_ca_max_le_l61_61129


namespace percentage_decrease_is_17_point_14_l61_61645

-- Define the conditions given in the problem
variable (S : ℝ) -- original salary
variable (D : ℝ) -- percentage decrease

-- Given conditions
def given_conditions : Prop :=
  1.40 * S - (D / 100) * 1.40 * S = 1.16 * S

-- The required proof problem, where we assert D = 17.14
theorem percentage_decrease_is_17_point_14 (S : ℝ) (h : given_conditions S D) : D = 17.14 := 
  sorry

end percentage_decrease_is_17_point_14_l61_61645


namespace starting_number_l61_61039

theorem starting_number (n : ℕ) (h1 : n % 11 = 3) (h2 : (n + 11) % 11 = 3) (h3 : (n + 22) % 11 = 3) 
  (h4 : (n + 33) % 11 = 3) (h5 : (n + 44) % 11 = 3) (h6 : n + 44 ≤ 50) : n = 3 := 
sorry

end starting_number_l61_61039


namespace Ben_sales_value_l61_61894

noncomputable def value_of_sale (old_salary new_salary commission_ratio sales_required : ℝ) (diff_salary: ℝ) :=
  ∃ x : ℝ, 0.15 * x * sales_required = diff_salary ∧ x = 750

theorem Ben_sales_value (old_salary new_salary commission_ratio sales_required diff_salary: ℝ)
  (h1: old_salary = 75000)
  (h2: new_salary = 45000)
  (h3: commission_ratio = 0.15)
  (h4: sales_required = 266.67)
  (h5: diff_salary = old_salary - new_salary) :
  value_of_sale old_salary new_salary commission_ratio sales_required diff_salary :=
by
  sorry

end Ben_sales_value_l61_61894


namespace Sarah_consumed_one_sixth_l61_61991

theorem Sarah_consumed_one_sixth (total_slices : ℕ) (slices_sarah_ate : ℕ) (shared_slices : ℕ) :
  total_slices = 20 → slices_sarah_ate = 3 → shared_slices = 1 → 
  ((slices_sarah_ate + shared_slices / 3) / total_slices : ℚ) = 1 / 6 :=
by
  intros h1 h2 h3
  sorry

end Sarah_consumed_one_sixth_l61_61991


namespace cells_after_10_days_l61_61958

theorem cells_after_10_days :
  let a := 4
  let r := 2
  let n := 10 / 2
  let a_n := a * r ^ (n - 1)
  a_n = 64 :=
by
  let a := 4
  let r := 2
  let n := 10 / 2
  let a_n := a * r ^ (n - 1)
  show a_n = 64
  sorry

end cells_after_10_days_l61_61958


namespace bobby_initial_blocks_l61_61427

variable (b : ℕ)

theorem bobby_initial_blocks
  (h : b + 6 = 8) : b = 2 := by
  sorry

end bobby_initial_blocks_l61_61427


namespace water_needed_l61_61505

-- Definitions as per conditions
def heavy_wash : ℕ := 20
def regular_wash : ℕ := 10
def light_wash : ℕ := 2
def extra_light_wash (bleach : ℕ) : ℕ := bleach * light_wash

def num_heavy_washes : ℕ := 2
def num_regular_washes : ℕ := 3
def num_light_washes : ℕ := 1
def num_bleached_loads : ℕ := 2

-- Function to calculate total water usage
def total_water_used : ℕ :=
  (num_heavy_washes * heavy_wash) +
  (num_regular_washes * regular_wash) +
  (num_light_washes * light_wash) + 
  (extra_light_wash num_bleached_loads)

-- Theorem to be proved
theorem water_needed : total_water_used = 76 := by
  sorry

end water_needed_l61_61505


namespace cos_equality_l61_61436

noncomputable def degrees_to_radians (d : ℝ) : ℝ := d * (Real.pi / 180)

theorem cos_equality : ∃ n : ℝ, (0 ≤ n ∧ n ≤ 180) ∧ Real.cos (degrees_to_radians n) = Real.cos (degrees_to_radians 317) :=
by
  use 43
  simp [degrees_to_radians, Real.cos]
  sorry

end cos_equality_l61_61436


namespace hyperbola_real_axis_length_l61_61095

theorem hyperbola_real_axis_length :
  (∃ (a b : ℝ), (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧ a = 3) →
  2 * 3 = 6 :=
by
  sorry

end hyperbola_real_axis_length_l61_61095


namespace percent_increase_sales_l61_61889

theorem percent_increase_sales (sales_this_year sales_last_year : ℝ) (h1 : sales_this_year = 460) (h2 : sales_last_year = 320) :
  (sales_this_year - sales_last_year) / sales_last_year * 100 = 43.75 :=
by
  sorry

end percent_increase_sales_l61_61889


namespace triple_g_eq_nineteen_l61_61155

def g (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 3 else 2 * n + 1

theorem triple_g_eq_nineteen : g (g (g 1)) = 19 := by
  sorry

end triple_g_eq_nineteen_l61_61155


namespace twenty_cows_twenty_days_l61_61104

-- Defining the initial conditions as constants
def num_cows : ℕ := 20
def days_one_cow_eats_one_bag : ℕ := 20
def bags_eaten_by_one_cow_in_days (d : ℕ) : ℕ := if d = days_one_cow_eats_one_bag then 1 else 0

-- Defining the total bags eaten by all cows
def total_bags_eaten_by_cows (cows : ℕ) (days : ℕ) : ℕ :=
  cows * (days / days_one_cow_eats_one_bag)

-- Statement to be proved: In 20 days, 20 cows will eat 20 bags of husk
theorem twenty_cows_twenty_days :
  total_bags_eaten_by_cows num_cows days_one_cow_eats_one_bag = 20 := sorry

end twenty_cows_twenty_days_l61_61104


namespace probability_of_heads_or_five_tails_is_one_eighth_l61_61556

namespace coin_flip

def num_heads_or_at_least_five_tails : ℕ :=
1 + 6 + 1

def total_outcomes : ℕ :=
2^6

def probability_heads_or_five_tails : ℚ :=
num_heads_or_at_least_five_tails / total_outcomes

theorem probability_of_heads_or_five_tails_is_one_eighth :
  probability_heads_or_five_tails = 1 / 8 := by
  sorry

end coin_flip

end probability_of_heads_or_five_tails_is_one_eighth_l61_61556


namespace reaction_requires_two_moles_of_HNO3_l61_61236

def nitric_acid_reaction (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) 
  (reaction : HNO3 + NaHCO3 = NaNO3 + CO2 + H2O)
  (n_NaHCO3 : ℕ) : ℕ :=
  if n_NaHCO3 = 2 then 2 else sorry

theorem reaction_requires_two_moles_of_HNO3
  (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) 
  (reaction : HNO3 + NaHCO3 = NaNO3 + CO2 + H2O)
  (n_NaHCO3 : ℕ) :
  n_NaHCO3 = 2 → nitric_acid_reaction HNO3 NaHCO3 NaNO3 CO2 H2O reaction n_NaHCO3 = 2 :=
by sorry

end reaction_requires_two_moles_of_HNO3_l61_61236


namespace paint_cost_l61_61134

theorem paint_cost {width height : ℕ} (price_per_quart coverage_area : ℕ) (total_cost : ℕ) :
  width = 5 → height = 4 → price_per_quart = 2 → coverage_area = 4 → total_cost = 20 :=
by
  intros h1 h2 h3 h4
  have area_one_side : ℕ := width * height
  have total_area : ℕ := 2 * area_one_side
  have quarts_needed : ℕ := total_area / coverage_area
  have cost : ℕ := quarts_needed * price_per_quart
  sorry

end paint_cost_l61_61134


namespace find_a_b_find_tangent_line_l61_61710

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := 2 * x ^ 3 + 3 * a * x ^ 2 + 3 * b * x + 8

-- Define the derivative of the function f(x)
def f' (a b x : ℝ) : ℝ := 6 * x ^ 2 + 6 * a * x + 3 * b

-- Define the conditions for extreme values at x=1 and x=2
def extreme_conditions (a b : ℝ) : Prop :=
  f' a b 1 = 0 ∧ f' a b 2 = 0

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) (h : extreme_conditions a b) : a = -3 ∧ b = 4 :=
by sorry

-- Find the equation of the tangent line at x=0
def tangent_equation (a b : ℝ) (x y : ℝ) : Prop :=
  12 * x - y + 8 = 0

-- Prove the equation of the tangent line
theorem find_tangent_line (a b : ℝ) (h : extreme_conditions a b) : tangent_equation a b 0 8 :=
by sorry

end find_a_b_find_tangent_line_l61_61710


namespace product_of_sums_of_squares_l61_61587

-- Given conditions as definitions
def sum_of_squares (a b : ℤ) : ℤ := a^2 + b^2

-- Prove that the product of two sums of squares is also a sum of squares
theorem product_of_sums_of_squares (a b n k : ℤ) (K P : ℤ) (hK : K = sum_of_squares a b) (hP : P = sum_of_squares n k) :
    K * P = (a * n + b * k)^2 + (a * k - b * n)^2 := 
by
  sorry

end product_of_sums_of_squares_l61_61587


namespace largest_prime_divisor_of_sum_of_squares_l61_61749

def largest_prime_divisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_sum_of_squares :
  largest_prime_divisor (11^2 + 90^2) = 89 :=
by sorry

end largest_prime_divisor_of_sum_of_squares_l61_61749


namespace max_yes_answers_100_l61_61867

-- Define the maximum number of "Yes" answers that could be given in a lineup of n people
def maxYesAnswers (n : ℕ) : ℕ :=
  if n = 0 then 0 else 1 + (n - 2)

theorem max_yes_answers_100 : maxYesAnswers 100 = 99 :=
  by sorry

end max_yes_answers_100_l61_61867


namespace part1_part2_l61_61054

def f (x a : ℝ) : ℝ := |x + a| + |x - a^2|

theorem part1 (x : ℝ) : f x 1 ≥ 4 ↔ x ≤ -2 ∨ x ≥ 2 := sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, ∃ a : ℝ, -1 < a ∧ a < 3 ∧ m < f x a) ↔ m < 12 := sorry

end part1_part2_l61_61054


namespace union_of_A_and_B_l61_61868

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} :=
by
  sorry

end union_of_A_and_B_l61_61868


namespace find_beta_l61_61373

open Real

theorem find_beta 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h1 : sin α = sqrt 5 / 5)
  (h2 : sin (α - β) = - sqrt 10 / 10):
  β = π / 4 :=
sorry

end find_beta_l61_61373


namespace max_sum_x_y_l61_61344

theorem max_sum_x_y 
  (x y : ℝ)
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
sorry

end max_sum_x_y_l61_61344


namespace max_value_frac_l61_61476

theorem max_value_frac (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    ∃ (c : ℝ), c = 1/4 ∧ (∀ (x y z : ℝ), (0 < x) → (0 < y) → (0 < z) → (xyz * (x + y + z)) / ((x + z)^2 * (z + y)^2) ≤ c) := 
by
  sorry

end max_value_frac_l61_61476


namespace number_of_sets_l61_61113

theorem number_of_sets (weight_per_rep reps total_weight : ℕ) 
  (h_weight_per_rep : weight_per_rep = 15)
  (h_reps : reps = 10)
  (h_total_weight : total_weight = 450) :
  (total_weight / (weight_per_rep * reps)) = 3 :=
by
  sorry

end number_of_sets_l61_61113


namespace tilings_of_3_by_5_rectangle_l61_61356

def num_tilings_of_3_by_5_rectangle : ℕ := 96

theorem tilings_of_3_by_5_rectangle (h : ℕ := 96) :
  (∃ (tiles : List (ℕ × ℕ)),
    tiles = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)] ∧
    -- Whether we are counting tiles in the context of a 3x5 rectangle
    -- with all distinct rotations and reflections allowed.
    True
  ) → num_tilings_of_3_by_5_rectangle = h :=
by {
  sorry -- Proof goes here
}

end tilings_of_3_by_5_rectangle_l61_61356


namespace necessary_condition_for_acute_angle_l61_61496

-- Defining vectors a and b
def vec_a (x : ℝ) : ℝ × ℝ := (x - 3, 2)
def vec_b : ℝ × ℝ := (1, 1)

-- Condition for the dot product to be positive
def dot_product_positive (x : ℝ) : Prop :=
  let (ax1, ax2) := vec_a x
  let (bx1, bx2) := vec_b
  ax1 * bx1 + ax2 * bx2 > 0

-- Statement for necessary condition
theorem necessary_condition_for_acute_angle (x : ℝ) :
  (dot_product_positive x) → (1 < x) :=
sorry

end necessary_condition_for_acute_angle_l61_61496


namespace problem_l61_61070

theorem problem 
  (k a b c : ℝ)
  (h1 : (3 : ℝ)^2 - 7 * 3 + k = 0)
  (h2 : (a : ℝ)^2 - 7 * a + k = 0)
  (h3 : (b : ℝ)^2 - 8 * b + (k + 1) = 0)
  (h4 : (c : ℝ)^2 - 8 * c + (k + 1) = 0) :
  a + b * c = 17 := sorry

end problem_l61_61070


namespace find_N_l61_61788

theorem find_N (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) :
  (x + y) / 3 = 1.222222222222222 := 
by
  -- We state the conditions.
  -- Lean will check whether these assumptions are consistent 
  sorry

end find_N_l61_61788


namespace total_handshakes_l61_61083

theorem total_handshakes :
  let gremlins := 20
  let imps := 20
  let sprites := 10
  let handshakes_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_gremlins_imps := gremlins * imps
  let handshakes_imps_sprites := imps * sprites
  handshakes_gremlins + handshakes_gremlins_imps + handshakes_imps_sprites = 790 :=
by
  sorry

end total_handshakes_l61_61083


namespace constant_c_for_local_maximum_l61_61367

theorem constant_c_for_local_maximum (c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x * (x - c) ^ 2) (h2 : ∃ δ > 0, ∀ x, |x - 2| < δ → f x ≤ f 2) : c = 6 :=
sorry

end constant_c_for_local_maximum_l61_61367


namespace julia_money_left_l61_61812

def initial_amount : ℕ := 40

def amount_spent_on_game (initial : ℕ) : ℕ := initial / 2

def amount_left_after_game (initial : ℕ) (spent_game : ℕ) : ℕ := initial - spent_game

def amount_spent_on_in_game (left_after_game : ℕ) : ℕ := left_after_game / 4

def final_amount (left_after_game : ℕ) (spent_in_game : ℕ) : ℕ := left_after_game - spent_in_game

theorem julia_money_left (initial : ℕ) 
  (h_init : initial = initial_amount)
  (spent_game : ℕ)
  (h_spent_game : spent_game = amount_spent_on_game initial)
  (left_after_game : ℕ)
  (h_left_after_game : left_after_game = amount_left_after_game initial spent_game)
  (spent_in_game : ℕ)
  (h_spent_in_game : spent_in_game = amount_spent_on_in_game left_after_game)
  : final_amount left_after_game spent_in_game = 15 := by 
  sorry

end julia_money_left_l61_61812


namespace find_a_and_b_l61_61460

noncomputable def find_ab (a b : ℝ) : Prop :=
  (3 - 2 * a + b = 0) ∧
  (27 + 6 * a + b = 0)

theorem find_a_and_b :
  ∃ (a b : ℝ), (find_ab a b) ∧ (a = -3) ∧ (b = -9) :=
by
  sorry

end find_a_and_b_l61_61460


namespace problem1_problem2_l61_61854

variable {a b : ℝ}

-- Proof problem 1
-- Goal: (1)(2a^(2/3)b^(1/2))(-6a^(1/2)b^(1/3)) / (-3a^(1/6)b^(5/6)) = -12a
theorem problem1 (h1 : 0 < a) (h2 : 0 < b) : 
  (1 : ℝ) * (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = -12 * a := 
sorry

-- Proof problem 2
-- Goal: 2(log(sqrt(2)))^2 + log(sqrt(2)) * log(5) + sqrt((log(sqrt(2)))^2 - log(2) + 1) = 1 + (1 / 2) * log(5)
theorem problem2 : 
  2 * (Real.log (Real.sqrt 2))^2 + (Real.log (Real.sqrt 2)) * (Real.log 5) + 
  Real.sqrt ((Real.log (Real.sqrt 2))^2 - Real.log 2 + 1) = 
  1 + 0.5 * (Real.log 5) := 
sorry

end problem1_problem2_l61_61854


namespace basketball_rim_height_l61_61451

theorem basketball_rim_height
    (height_in_inches : ℕ)
    (reach_in_inches : ℕ)
    (jump_in_inches : ℕ)
    (above_rim_in_inches : ℕ) :
    height_in_inches = 72
    → reach_in_inches = 22
    → jump_in_inches = 32
    → above_rim_in_inches = 6
    → (height_in_inches + reach_in_inches + jump_in_inches - above_rim_in_inches) = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end basketball_rim_height_l61_61451


namespace sum_of_coordinates_of_C_parallelogram_l61_61116

-- Definitions that encapsulate the given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨-1, 0⟩
def D : Point := ⟨5, -4⟩

-- The theorem we need to prove
theorem sum_of_coordinates_of_C_parallelogram :
  ∃ C : Point, C.x + C.y = 7 ∧
  ∃ M : Point, M = ⟨(A.x + D.x) / 2, (A.y + D.y) / 2⟩ ∧
  (M = ⟨(B.x + C.x) / 2, (B.y + C.y) / 2⟩) :=
sorry

end sum_of_coordinates_of_C_parallelogram_l61_61116


namespace ratio_of_share_l61_61924

/-- A certain amount of money is divided amongst a, b, and c. 
The share of a is $122, and the total amount of money is $366. 
Prove that the ratio of a's share to the combined share of b and c is 1 / 2. -/
theorem ratio_of_share (a b c : ℝ) (total share_a : ℝ) (h1 : a + b + c = total) 
  (h2 : total = 366) (h3 : share_a = 122) : share_a / (total - share_a) = 1 / 2 := by
  sorry

end ratio_of_share_l61_61924


namespace tan_45_deg_l61_61762

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l61_61762


namespace sin_two_alpha_l61_61695

theorem sin_two_alpha (alpha : ℝ) (h : Real.cos (π / 4 - alpha) = 4 / 5) : 
  Real.sin (2 * alpha) = 7 / 25 :=
by
  sorry

end sin_two_alpha_l61_61695


namespace find_fraction_abs_l61_61980

-- Define the conditions and the main proof problem
theorem find_fraction_abs (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5 * x * y) :
  abs ((x + y) / (x - y)) = Real.sqrt ((7 : ℝ) / 3) :=
by
  sorry

end find_fraction_abs_l61_61980


namespace fish_left_in_sea_l61_61820

theorem fish_left_in_sea : 
  let westward_initial := 1800
  let eastward_initial := 3200
  let north_initial := 500
  let eastward_caught := (2 / 5) * eastward_initial
  let westward_caught := (3 / 4) * westward_initial
  let eastward_left := eastward_initial - eastward_caught
  let westward_left := westward_initial - westward_caught
  let north_left := north_initial
  eastward_left + westward_left + north_left = 2870 := 
by 
  sorry

end fish_left_in_sea_l61_61820


namespace laura_owes_amount_l61_61478

def principal : ℝ := 35
def rate : ℝ := 0.05
def time : ℝ := 1
def interest (P R T : ℝ) := P * R * T
def totalAmountOwed (P I : ℝ) := P + I

theorem laura_owes_amount : totalAmountOwed principal (interest principal rate time) = 36.75 :=
by
  sorry

end laura_owes_amount_l61_61478


namespace arccos_sin_2_equals_l61_61997

theorem arccos_sin_2_equals : Real.arccos (Real.sin 2) = 2 - Real.pi / 2 := by
  sorry

end arccos_sin_2_equals_l61_61997


namespace solve_in_primes_l61_61948

theorem solve_in_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) := 
sorry

end solve_in_primes_l61_61948


namespace find_largest_element_l61_61699

noncomputable def increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ i j, 1 ≤ i → i < j → j ≤ 8 → a i < a j

noncomputable def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) (i : ℕ) : Prop :=
a (i+1) - a i = d ∧ a (i+2) - a (i+1) = d ∧ a (i+3) - a (i+2) = d

noncomputable def geometric_progression (a : ℕ → ℝ) (i : ℕ) : Prop :=
a (i+1) / a i = a (i+2) / a (i+1) ∧ a (i+2) / a (i+1) = a (i+3) / a (i+2)

theorem find_largest_element
  (a : ℕ → ℝ)
  (h_inc : increasing_sequence a)
  (h_ap1 : ∃ i, 1 ≤ i ∧ i ≤ 5 ∧ arithmetic_progression a 4 i)
  (h_ap2 : ∃ j, 1 ≤ j ∧ j ≤ 5 ∧ arithmetic_progression a 36 j)
  (h_gp : ∃ k, 1 ≤ k ∧ k ≤ 5 ∧ geometric_progression a k) :
  a 8 = 126 :=
sorry

end find_largest_element_l61_61699


namespace a_seq_correct_b_seq_max_m_l61_61341

noncomputable def a_seq (n : ℕ) : ℕ :=
if n = 0 then 3 else (n + 1)^2 + 2

-- Verification that the sequence follows the provided conditions.
theorem a_seq_correct (n : ℕ) : 
  (a_seq 0 = 3) ∧
  (a_seq 1 = 6) ∧
  (a_seq 2 = 11) ∧
  (∀ m : ℕ, m ≥ 1 → a_seq (m + 1) - a_seq m = 2 * m + 1) := sorry

noncomputable def b_seq (n : ℕ) : ℝ := 
(a_seq n : ℝ) / (3 ^ (Real.sqrt (a_seq n - 2)))

theorem b_seq_max_m (m : ℝ) : 
  (∀ n : ℕ, b_seq n ≤ m) ↔ (1 ≤ m) := sorry

end a_seq_correct_b_seq_max_m_l61_61341


namespace find_a_b_c_l61_61719

theorem find_a_b_c :
  ∃ a b c : ℕ, a = 1 ∧ b = 17 ∧ c = 2 ∧ (Nat.gcd a c = 1) ∧ a + b + c = 20 :=
by {
  -- the proof would go here
  sorry
}

end find_a_b_c_l61_61719


namespace max_knights_is_seven_l61_61807

-- Definitions of conditions
def students : ℕ := 11
def total_statements : ℕ := students * (students - 1)
def liar_statements : ℕ := 56

-- Definition translating the problem statement
theorem max_knights_is_seven : ∃ (k li : ℕ), 
  (k + li = students) ∧ 
  (k * li = liar_statements) ∧ 
  (k = 7) := 
by
  sorry

end max_knights_is_seven_l61_61807


namespace machine_work_rate_l61_61494

theorem machine_work_rate (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -6 ∧ x ≠ -1) : 
  (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x) = 1 / x) → x = 2 / 3 :=
by
  sorry

end machine_work_rate_l61_61494


namespace find_k_l61_61518

theorem find_k (k : ℝ) :
  (∀ x, x^2 + k*x + 10 = 0 → (∃ r s : ℝ, x = r ∨ x = s) ∧ r + s = -k ∧ r * s = 10) ∧
  (∀ x, x^2 - k*x + 10 = 0 → (∃ r s : ℝ, x = r + 4 ∨ x = s + 4) ∧ (r + 4) + (s + 4) = k) → 
  k = 4 :=
by
  sorry

end find_k_l61_61518


namespace value_of_y_l61_61753

theorem value_of_y (x y : ℝ) (h1 : 3 * (x - y) = 18) (h2 : x + y = 20) : y = 7 := by
  sorry

end value_of_y_l61_61753


namespace average_of_first_40_results_l61_61623

theorem average_of_first_40_results 
  (A : ℝ)
  (avg_other_30 : ℝ := 40)
  (avg_all_70 : ℝ := 34.285714285714285) : A = 30 :=
by 
  let sum1 := A * 40
  let sum2 := avg_other_30 * 30
  let combined_sum := sum1 + sum2
  let combined_avg := combined_sum / 70
  have h1 : combined_avg = avg_all_70 := by sorry
  have h2 : combined_avg = 34.285714285714285 := by sorry
  have h3 : combined_sum = (A * 40) + (40 * 30) := by sorry
  have h4 : (A * 40) + 1200 = 2400 := by sorry
  have h5 : A * 40 = 1200 := by sorry
  have h6 : A = 1200 / 40 := by sorry
  have h7 : A = 30 := by sorry
  exact h7

end average_of_first_40_results_l61_61623


namespace sum_of_ages_l61_61550

variables (S F : ℕ)

theorem sum_of_ages
  (h1 : F - 18 = 3 * (S - 18))
  (h2 : F = 2 * S) :
  S + F = 108 :=
by
  sorry

end sum_of_ages_l61_61550


namespace digits_difference_l61_61468

theorem digits_difference (X Y : ℕ) (h : 10 * X + Y - (10 * Y + X) = 90) : X - Y = 10 :=
by
  sorry

end digits_difference_l61_61468


namespace men_absent_l61_61833

/-- 
A group of men decided to do a work in 20 days, but some of them became absent. 
The rest of the group did the work in 40 days. The original number of men was 20. 
Prove that 10 men became absent. 
--/
theorem men_absent 
    (original_men : ℕ) (absent_men : ℕ) (planned_days : ℕ) (actual_days : ℕ)
    (h1 : original_men = 20) (h2 : planned_days = 20) (h3 : actual_days = 40)
    (h_work : original_men * planned_days = (original_men - absent_men) * actual_days) : 
    absent_men = 10 :=
    by 
    rw [h1, h2, h3] at h_work
    -- Proceed to manually solve the equation, but here we add sorry
    sorry

end men_absent_l61_61833


namespace not_divisible_by_10100_l61_61876

theorem not_divisible_by_10100 (n : ℕ) : (3^n + 1) % 10100 ≠ 0 := 
by 
  sorry

end not_divisible_by_10100_l61_61876


namespace option_c_correct_l61_61784

theorem option_c_correct (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end option_c_correct_l61_61784


namespace nested_fraction_value_l61_61122

theorem nested_fraction_value : 
  let expr := 1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))))
  expr = 21 / 55 :=
by 
  sorry

end nested_fraction_value_l61_61122


namespace only_solution_is_two_l61_61187

theorem only_solution_is_two :
  ∀ n : ℕ, (Nat.Prime (n^n + 1) ∧ Nat.Prime ((2*n)^(2*n) + 1)) → n = 2 :=
by
  sorry

end only_solution_is_two_l61_61187


namespace simplify_expression_l61_61798

theorem simplify_expression (x : ℝ) : 
  (3 * x^2 + 4 * x - 5) * (x - 2) + (x - 2) * (2 * x^2 - 3 * x + 9) - (4 * x - 7) * (x - 2) * (x - 3) 
  = x^3 + x^2 + 12 * x - 36 := 
by
  sorry

end simplify_expression_l61_61798


namespace inequality_always_holds_l61_61498

theorem inequality_always_holds (x b : ℝ) (h : ∀ x : ℝ, x^2 + b * x + b > 0) : 0 < b ∧ b < 4 :=
sorry

end inequality_always_holds_l61_61498


namespace can_form_triangle_l61_61296

theorem can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

example : can_form_triangle 8 6 3 := by
  sorry

end can_form_triangle_l61_61296


namespace never_attains_95_l61_61020

def dihedral_angle_condition (α β : ℝ) : Prop :=
  0 < α ∧ 0 < β ∧ α + β < 90

theorem never_attains_95 (α β : ℝ) (h : dihedral_angle_condition α β) :
  α + β ≠ 95 :=
by
  sorry

end never_attains_95_l61_61020


namespace parabola_fixed_point_thm_l61_61379

-- Define the parabola condition
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

-- Define the focus condition
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define the slope product condition
def slope_product (A B : ℝ × ℝ) : Prop :=
  (A.1 ≠ 0 ∧ B.1 ≠ 0) → ((A.2 / A.1) * (B.2 / B.1) = -1 / 3)

-- Define the fixed point condition
def fixed_point (A B : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, A ≠ B ∧ (x = 12) ∧ ((A.2 - B.2) / (A.1 - B.1)) * 12 = A.2

-- Problem statement in Lean
theorem parabola_fixed_point_thm (A B : ℝ × ℝ) (p : ℝ) :
  (∃ O : ℝ × ℝ, O = (0, 0)) →
  (∃ C : ℝ → ℝ → ℝ → Prop, C = parabola) →
  (∃ F : ℝ × ℝ, focus F) →
  parabola A.2 A.1 p →
  parabola B.2 B.1 p →
  slope_product A B →
  fixed_point A B :=
by 
-- Sorry is used to skip the proof
sorry

end parabola_fixed_point_thm_l61_61379


namespace total_amount_correct_l61_61279

noncomputable def total_amount (p_a r_a t_a p_b r_b t_b p_c r_c t_c : ℚ) : ℚ :=
  let final_price (p r t : ℚ) := p - (p * r / 100) + ((p - (p * r / 100)) * t / 100)
  final_price p_a r_a t_a + final_price p_b r_b t_b + final_price p_c r_c t_c

theorem total_amount_correct :
  total_amount 2500 6 10 3150 8 12 1000 5 7 = 6847.26 :=
by
  sorry

end total_amount_correct_l61_61279


namespace find_f_and_min_g_l61_61601

theorem find_f_and_min_g (f g : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x : ℝ, f (2 * x - 3) = 4 * x^2 + 2 * x + 1)
  (h2 : ∀ x : ℝ, g x = f (x + a) - 7 * x):
  
  (∀ x : ℝ, f x = x^2 + 7 * x + 13) ∧
  
  (∀ a : ℝ, 
    ∀ x : ℝ, 
      (x = 1 → (a ≥ -1 → g x = a^2 + 9 * a + 14)) ∧
      (-3 < a ∧ a < -1 → g (-a) = 7 * a + 13) ∧
      (x = 3 → (a ≤ -3 → g x = a^2 + 13 * a + 22))) :=
by
  sorry

end find_f_and_min_g_l61_61601


namespace fourth_intersection_point_l61_61806

noncomputable def fourth_point_of_intersection : Prop :=
  let hyperbola (x y : ℝ) := x * y = 1
  let circle (x y : ℝ) := (x - 1)^2 + (y + 1)^2 = 10
  let known_points : List (ℝ × ℝ) := [(3, 1/3), (-4, -1/4), (1/2, 2)]
  let fourth_point := (-1/6, -6)
  (hyperbola 3 (1/3)) ∧ (hyperbola (-4) (-1/4)) ∧ (hyperbola (1/2) 2) ∧
  (circle 3 (1/3)) ∧ (circle (-4) (-1/4)) ∧ (circle (1/2) 2) ∧ 
  (hyperbola (-1/6) (-6)) ∧ (circle (-1/6) (-6)) ∧ 
  ∀ (x y : ℝ), (hyperbola x y) → (circle x y) → ((x, y) = fourth_point ∨ (x, y) ∈ known_points)
  
theorem fourth_intersection_point :
  fourth_point_of_intersection :=
sorry

end fourth_intersection_point_l61_61806


namespace area_conversion_correct_l61_61766

-- Define the legs of the right triangle
def leg1 : ℕ := 60
def leg2 : ℕ := 80

-- Define the conversion factor
def square_feet_in_square_yard : ℕ := 9

-- Calculate the area of the triangle in square feet
def area_in_square_feet : ℕ := (leg1 * leg2) / 2

-- Calculate the area of the triangle in square yards
def area_in_square_yards : ℚ := area_in_square_feet / square_feet_in_square_yard

-- The theorem stating the problem
theorem area_conversion_correct : area_in_square_yards = 266 + 2 / 3 := by
  sorry

end area_conversion_correct_l61_61766


namespace balance_after_6_months_l61_61592

noncomputable def final_balance : ℝ :=
  let balance_m1 := 5000 * (1 + 0.04 / 12)
  let balance_m2 := (balance_m1 + 1000) * (1 + 0.042 / 12)
  let balance_m3 := balance_m2 * (1 + 0.038 / 12)
  let balance_m4 := (balance_m3 - 1500) * (1 + 0.05 / 12)
  let balance_m5 := (balance_m4 + 750) * (1 + 0.052 / 12)
  let balance_m6 := (balance_m5 - 1000) * (1 + 0.045 / 12)
  balance_m6

theorem balance_after_6_months : final_balance = 4371.51 := sorry

end balance_after_6_months_l61_61592


namespace slope_y_intercept_sum_l61_61962

theorem slope_y_intercept_sum 
  (m b : ℝ) 
  (h1 : (2 : ℝ) * m + b = -1) 
  (h2 : (5 : ℝ) * m + b = 2) : 
  m + b = -2 := 
sorry

end slope_y_intercept_sum_l61_61962


namespace min_value_proof_l61_61599

noncomputable def min_value_condition (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 / (a + 3) + 1 / (b + 3) = 1 / 4)

theorem min_value_proof : ∃ a b : ℝ, min_value_condition a b ∧ a + 3 * b = 4 + 8 * Real.sqrt 3 := by
  sorry

end min_value_proof_l61_61599


namespace bonus_tasks_l61_61895

-- Definition for earnings without bonus
def earnings_without_bonus (tasks : ℕ) : ℕ := tasks * 2

-- Definition for calculating the total bonus received
def total_bonus (tasks : ℕ) (earnings : ℕ) : ℕ := earnings - earnings_without_bonus tasks

-- Definition for the number of bonuses received given the total bonus and a single bonus amount
def number_of_bonuses (total_bonus : ℕ) (bonus_amount : ℕ) : ℕ := total_bonus / bonus_amount

-- The theorem we want to prove
theorem bonus_tasks (tasks : ℕ) (earnings : ℕ) (bonus_amount : ℕ) (bonus_tasks : ℕ) :
  earnings = 78 →
  tasks = 30 →
  bonus_amount = 6 →
  bonus_tasks = tasks / (number_of_bonuses (total_bonus tasks earnings) bonus_amount) →
  bonus_tasks = 10 :=
by
  intros h_earnings h_tasks h_bonus_amount h_bonus_tasks
  sorry

end bonus_tasks_l61_61895


namespace find_function_l61_61310

/-- A function f satisfies the equation f(x) + (x + 1/2) * f(1 - x) = 1. -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x + (x + 1 / 2) * f (1 - x) = 1

/-- We want to prove two things:
 1) f(0) = 2 and f(1) = -2
 2) f(x) =  2 / (1 - 2x) for x ≠ 1/2
 -/
theorem find_function (f : ℝ → ℝ) (h : satisfies_equation f) :
  (f 0 = 2 ∧ f 1 = -2) ∧ (∀ x : ℝ, x ≠ 1 / 2 → f x = 2 / (1 - 2 * x)) ∧ (f (1 / 2) = 1 / 2) :=
by
  sorry

end find_function_l61_61310


namespace percentage_refund_l61_61517

theorem percentage_refund
  (initial_amount : ℕ)
  (sweater_cost : ℕ)
  (tshirt_cost : ℕ)
  (shoes_cost : ℕ)
  (amount_left_after_refund : ℕ)
  (refund_percentage : ℕ) :
  initial_amount = 74 →
  sweater_cost = 9 →
  tshirt_cost = 11 →
  shoes_cost = 30 →
  amount_left_after_refund = 51 →
  refund_percentage = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_refund_l61_61517


namespace derivative_correct_l61_61055

noncomputable def f (x : ℝ) : ℝ := 
  (1 / (2 * Real.sqrt 2)) * (Real.sin (Real.log x) - (Real.sqrt 2 - 1) * Real.cos (Real.log x)) * x^(Real.sqrt 2 + 1)

noncomputable def df (x : ℝ) : ℝ := 
  (x^(Real.sqrt 2)) / (2 * Real.sqrt 2) * (2 * Real.cos (Real.log x) - Real.sqrt 2 * Real.cos (Real.log x) + 2 * Real.sqrt 2 * Real.sin (Real.log x))

theorem derivative_correct (x : ℝ) (hx : 0 < x) :
  deriv f x = df x := by
  sorry

end derivative_correct_l61_61055


namespace find_abc_l61_61393

open Real

theorem find_abc {a b c : ℝ}
  (h1 : b + c = 16)
  (h2 : c + a = 17)
  (h3 : a + b = 18) :
  a * b * c = 606.375 :=
sorry

end find_abc_l61_61393


namespace compare_x_y_l61_61309

theorem compare_x_y :
  let x := 123456789 * 123456786
  let y := 123456788 * 123456787
  x < y := sorry

end compare_x_y_l61_61309


namespace alice_journey_duration_l61_61917
noncomputable def journey_duration (start_hour start_minute end_hour end_minute : ℕ) : ℕ :=
  let start_in_minutes := start_hour * 60 + start_minute
  let end_in_minutes := end_hour * 60 + end_minute
  if end_in_minutes >= start_in_minutes then end_in_minutes - start_in_minutes
  else end_in_minutes + 24 * 60 - start_in_minutes
  
theorem alice_journey_duration :
  ∃ start_hour start_minute end_hour end_minute,
  (7 ≤ start_hour ∧ start_hour < 8 ∧ start_minute = 38) ∧
  (16 ≤ end_hour ∧ end_hour < 17 ∧ end_minute = 35) ∧
  journey_duration start_hour start_minute end_hour end_minute = 537 :=
by {
  sorry
}

end alice_journey_duration_l61_61917


namespace eval_expression_l61_61911

theorem eval_expression : -20 + 12 * ((5 + 15) / 4) = 40 :=
by
  sorry

end eval_expression_l61_61911


namespace jessica_coins_worth_l61_61920

theorem jessica_coins_worth :
  ∃ (n d : ℕ), n + d = 30 ∧ 5 * (30 - d) + 10 * d = 165 :=
by {
  sorry
}

end jessica_coins_worth_l61_61920


namespace cos_three_pi_over_four_l61_61746

theorem cos_three_pi_over_four :
  Real.cos (3 * Real.pi / 4) = -1 / Real.sqrt 2 :=
by
  sorry

end cos_three_pi_over_four_l61_61746


namespace moles_required_to_form_2_moles_H2O_l61_61445

def moles_of_NH4NO3_needed (moles_of_H2O : ℕ) : ℕ := moles_of_H2O

theorem moles_required_to_form_2_moles_H2O :
  moles_of_NH4NO3_needed 2 = 2 := 
by 
  -- From the balanced equation 1 mole of NH4NO3 produces 1 mole of H2O
  -- Therefore, 2 moles of NH4NO3 are needed to produce 2 moles of H2O
  sorry

end moles_required_to_form_2_moles_H2O_l61_61445


namespace smallest_divisible_12_13_14_l61_61226

theorem smallest_divisible_12_13_14 :
  ∃ n : ℕ, n > 0 ∧ (n % 12 = 0) ∧ (n % 13 = 0) ∧ (n % 14 = 0) ∧ n = 1092 := by
  sorry

end smallest_divisible_12_13_14_l61_61226


namespace find_D_l61_61386

-- Definitions from conditions
def is_different (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- The proof problem
theorem find_D (A B C D : ℕ) (h_diff: is_different A B C D) (h_eq : 700 + 10 * A + 5 + 100 * B + 70 + C = 100 * D + 38) : D = 9 :=
sorry

end find_D_l61_61386


namespace price_difference_eq_l61_61630

-- Define the problem conditions
variable (P : ℝ) -- Original price
variable (H1 : P - 0.15 * P = 61.2) -- Condition 1: 15% discount results in $61.2
variable (H2 : P * (1 - 0.15) = 61.2) -- Another way to represent Condition 1 (if needed)
variable (H3 : 61.2 * 1.25 = 76.5) -- Condition 4: Price raises by 25% after the 15% discount
variable (H4 : 76.5 * 0.9 = 68.85) -- Condition 5: Additional 10% discount after raise
variable (H5 : P = 72) -- Calculated original price

-- Define the theorem to prove
theorem price_difference_eq :
  (P - 68.85 = 3.15) := 
by
  sorry

end price_difference_eq_l61_61630


namespace total_calories_in_jerrys_breakfast_l61_61084

theorem total_calories_in_jerrys_breakfast :
  let pancakes := 7 * 120
  let bacon := 3 * 100
  let orange_juice := 2 * 300
  let cereal := 1 * 200
  let chocolate_muffin := 1 * 350
  pancakes + bacon + orange_juice + cereal + chocolate_muffin = 2290 :=
by
  -- Proof omitted
  sorry

end total_calories_in_jerrys_breakfast_l61_61084


namespace find_largest_number_l61_61424

theorem find_largest_number 
  (a b c : ℕ) 
  (h1 : a + b = 16) 
  (h2 : a + c = 20) 
  (h3 : b + c = 23) : 
  c = 19 := 
sorry

end find_largest_number_l61_61424


namespace find_second_number_in_denominator_l61_61627

theorem find_second_number_in_denominator :
  (0.625 * 0.0729 * 28.9) / (0.0017 * x * 8.1) = 382.5 → x = 0.24847 :=
by
  intro h
  sorry

end find_second_number_in_denominator_l61_61627


namespace train_length_proper_l61_61372

noncomputable def train_length (speed distance_time pass_time : ℝ) : ℝ :=
  speed * pass_time

axiom speed_of_train : ∀ (distance_time : ℝ), 
  (10 * 1000 / (15 * 60)) = 11.11

theorem train_length_proper :
  train_length 11.11 900 10 = 111.1 := by
  sorry

end train_length_proper_l61_61372


namespace range_of_x_l61_61783

-- Defining the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2 * x - 3 > 0
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Theorem statement
theorem range_of_x (x : ℝ) : (¬ q x ∧ p x) → (x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3) :=
by
  sorry

end range_of_x_l61_61783


namespace sum_of_digits_l61_61919

variable (a b c d e f : ℕ)

theorem sum_of_digits :
  ∀ (a b c d e f : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧
    100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 →
    a + b + c + d + e + f = 28 := 
by
  intros a b c d e f h
  sorry

end sum_of_digits_l61_61919


namespace james_collected_on_first_day_l61_61747

-- Conditions
variables (x : ℕ) -- the number of tins collected on the first day
variable (h1 : 500 = x + 3 * x + (3 * x - 50) + 4 * 50) -- total number of tins collected

-- Theorem to be proved
theorem james_collected_on_first_day : x = 50 :=
by
  sorry

end james_collected_on_first_day_l61_61747


namespace f_99_eq_1_l61_61772

-- Define an even function on ℝ
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- The conditions to be satisfied by the function f
variables (f : ℝ → ℝ)
variable (h_even : is_even_function f)
variable (h_f1 : f 1 = 1)
variable (h_period : ∀ x, f (x + 4) = f x)

-- Prove that f(99) = 1
theorem f_99_eq_1 : f 99 = 1 :=
by
  sorry

end f_99_eq_1_l61_61772


namespace shaded_area_correct_l61_61622

def diameter := 3 -- inches
def pattern_length := 18 -- inches equivalent to 1.5 feet

def radius := diameter / 2 -- radius calculation

noncomputable def area_of_one_circle := Real.pi * (radius ^ 2)
def number_of_circles := pattern_length / diameter
noncomputable def total_shaded_area := number_of_circles * area_of_one_circle

theorem shaded_area_correct :
  total_shaded_area = 13.5 * Real.pi :=
  by
  sorry

end shaded_area_correct_l61_61622


namespace proof_f_1_add_g_2_l61_61540

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x + 1

theorem proof_f_1_add_g_2 : f (1 + g 2) = 8 := by
  sorry

end proof_f_1_add_g_2_l61_61540


namespace min_value_f_at_3_f_increasing_for_k_neg4_l61_61933

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x + k / (x - 1)

-- Problem (1): If k = 4, find the minimum value of f(x) and the corresponding value of x.
theorem min_value_f_at_3 : ∃ x > 1, @f x 4 = 5 ∧ x = 3 :=
  sorry

-- Problem (2): If k = -4, prove that f(x) is an increasing function for x > 1.
theorem f_increasing_for_k_neg4 : ∀ ⦃x y : ℝ⦄, 1 < x → x < y → f x (-4) < f y (-4) :=
  sorry

end min_value_f_at_3_f_increasing_for_k_neg4_l61_61933


namespace unique_primes_solution_l61_61032

theorem unique_primes_solution (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) : 
    p + q^2 = r^4 ↔ (p = 7 ∧ q = 3 ∧ r = 2) := 
by
  sorry

end unique_primes_solution_l61_61032


namespace number_of_students_l61_61099

theorem number_of_students (total_students : ℕ) :
  (total_students = 19 * 6 + 4) ∧ 
  (∃ (x y : ℕ), x + y = 22 ∧ x > 7 ∧ total_students = x * 6 + y * 5) →
  total_students = 118 :=
by
  sorry

end number_of_students_l61_61099


namespace trajectory_proof_l61_61394

noncomputable def trajectory_eqn (x y : ℝ) : Prop :=
  (y + Real.sqrt 2) * (y - Real.sqrt 2) / (x * x) = -2

theorem trajectory_proof :
  ∀ (x y : ℝ), x ≠ 0 → trajectory_eqn x y → (y*y / 2 + x*x = 1) :=
by
  intros x y hx htrajectory
  sorry

end trajectory_proof_l61_61394


namespace unit_digit_8_pow_1533_l61_61829

theorem unit_digit_8_pow_1533 : (8^1533 % 10) = 8 := by
  sorry

end unit_digit_8_pow_1533_l61_61829


namespace decreasing_power_function_l61_61818

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x ^ k

theorem decreasing_power_function (k : ℝ) : 
  (∀ x : ℝ, 0 < x → (f k x) ≤ 0) ↔ k < 0 ∧ k ≠ 0 := sorry

end decreasing_power_function_l61_61818


namespace last_three_digits_of_11_pow_210_l61_61639

theorem last_three_digits_of_11_pow_210 : (11 ^ 210) % 1000 = 601 :=
by sorry

end last_three_digits_of_11_pow_210_l61_61639


namespace no_integer_solutions_l61_61049

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^1988 + y^1988 + z^1988 = 7^1990 := by
  sorry

end no_integer_solutions_l61_61049


namespace least_positive_x_l61_61822

variable (a b : ℝ)

noncomputable def tan_inv (x : ℝ) : ℝ := Real.arctan x

theorem least_positive_x (x k : ℝ) 
  (h1 : Real.tan x = a / b)
  (h2 : Real.tan (2 * x) = b / (a + b))
  (h3 : Real.tan (3 * x) = (a - b) / (a + b))
  (h4 : x = tan_inv k)
  : k = 13 / 9 := sorry

end least_positive_x_l61_61822


namespace length_of_each_cut_section_xiao_hong_age_l61_61007

theorem length_of_each_cut_section (x : ℝ) (h : 60 - 2 * x = 10) : x = 25 := sorry

theorem xiao_hong_age (y : ℝ) (h : 2 * y + 10 = 30) : y = 10 := sorry

end length_of_each_cut_section_xiao_hong_age_l61_61007


namespace exists_multiple_of_n_with_ones_l61_61019

theorem exists_multiple_of_n_with_ones (n : ℤ) (hn1 : n ≥ 1) (hn2 : Int.gcd n 10 = 1) :
  ∃ k : ℕ, n ∣ (10^k - 1) / 9 :=
by sorry

end exists_multiple_of_n_with_ones_l61_61019


namespace largest_possible_A_l61_61569

theorem largest_possible_A (A B : ℕ) (h1 : A = 5 * 2 + B) (h2 : B < 5) : A ≤ 14 :=
by
  have h3 : A ≤ 10 + 4 := sorry
  exact h3

end largest_possible_A_l61_61569


namespace increased_sales_type_B_l61_61665

-- Definitions for sales equations
def store_A_sales (x y : ℝ) : Prop :=
  60 * x + 15 * y = 3600

def store_B_sales (x y : ℝ) : Prop :=
  40 * x + 60 * y = 4400

-- Definition for the price of clothing items
def price_A (x : ℝ) : Prop :=
  x = 50

def price_B (y : ℝ) : Prop :=
  y = 40

-- Definition for the increased sales in May for type A
def may_sales_A (x : ℝ) : Prop :=
  100 * x * 1.2 = 6000

-- Definition to prove percentage increase for type B sales in May
noncomputable def percentage_increase_B (x y : ℝ) : ℝ :=
  ((4500 - (100 * y * 0.4)) / (100 * y * 0.4)) * 100

theorem increased_sales_type_B (x y : ℝ)
  (h1 : store_A_sales x y)
  (h2 : store_B_sales x y)
  (hA : price_A x)
  (hB : price_B y)
  (hMayA : may_sales_A x) :
  percentage_increase_B x y = 50 :=
sorry

end increased_sales_type_B_l61_61665


namespace joan_writing_time_l61_61150

theorem joan_writing_time
  (total_time : ℕ)
  (time_piano : ℕ)
  (time_reading : ℕ)
  (time_exerciser : ℕ)
  (h1 : total_time = 120)
  (h2 : time_piano = 30)
  (h3 : time_reading = 38)
  (h4 : time_exerciser = 27) : 
  total_time - (time_piano + time_reading + time_exerciser) = 25 :=
by
  sorry

end joan_writing_time_l61_61150


namespace honey_production_l61_61660

-- Define the conditions:
def bees : ℕ := 60
def days : ℕ := 60
def honey_per_bee : ℕ := 1

-- Statement to prove:
theorem honey_production (bees_eq : 60 = bees) (days_eq : 60 = days) (honey_per_bee_eq : 1 = honey_per_bee) :
  bees * honey_per_bee = 60 := by
  sorry

end honey_production_l61_61660


namespace lloyd_total_hours_worked_l61_61197

-- Conditions
def regular_hours_per_day : ℝ := 7.5
def regular_rate : ℝ := 4.5
def overtime_multiplier : ℝ := 2.5
def total_earnings : ℝ := 67.5

-- Proof problem
theorem lloyd_total_hours_worked :
  let overtime_rate := overtime_multiplier * regular_rate
  let regular_earnings := regular_hours_per_day * regular_rate
  let earnings_from_overtime := total_earnings - regular_earnings
  let hours_of_overtime := earnings_from_overtime / overtime_rate
  let total_hours := regular_hours_per_day + hours_of_overtime
  total_hours = 10.5 :=
by
  sorry

end lloyd_total_hours_worked_l61_61197


namespace largest_positive_x_l61_61402

def largest_positive_solution : ℝ := 1

theorem largest_positive_x 
  (x : ℝ) 
  (h : (2 * x^3 - x^2 - x + 1) ^ (1 + 1 / (2 * x + 1)) = 1) : 
  x ≤ largest_positive_solution := 
sorry

end largest_positive_x_l61_61402


namespace average_salary_of_all_workers_l61_61472

-- Definitions of conditions
def T : ℕ := 7
def total_workers : ℕ := 56
def W : ℕ := total_workers - T
def A_T : ℕ := 12000
def A_W : ℕ := 6000

-- Definition of total salary and average salary
def total_salary : ℕ := (T * A_T) + (W * A_W)

theorem average_salary_of_all_workers : total_salary / total_workers = 6750 := 
  by sorry

end average_salary_of_all_workers_l61_61472


namespace find_x_plus_y_l61_61199

theorem find_x_plus_y
  (x y : ℝ)
  (hx : x^3 - 3 * x^2 + 5 * x - 17 = 0)
  (hy : y^3 - 3 * y^2 + 5 * y + 11 = 0) :
  x + y = 2 := 
sorry

end find_x_plus_y_l61_61199


namespace projection_cardinal_inequality_l61_61901

variables {Point : Type} [Fintype Point] [DecidableEq Point]

def projection_Oyz (S : Finset Point) : Finset Point := sorry
def projection_Ozx (S : Finset Point) : Finset Point := sorry
def projection_Oxy (S : Finset Point) : Finset Point := sorry

theorem projection_cardinal_inequality
  (S : Finset Point)
  (S_x := projection_Oyz S)
  (S_y := projection_Ozx S)
  (S_z := projection_Oxy S)
  : (Finset.card S)^2 ≤ (Finset.card S_x) * (Finset.card S_y) * (Finset.card S_z) :=
sorry

end projection_cardinal_inequality_l61_61901


namespace jane_payment_per_bulb_l61_61185

theorem jane_payment_per_bulb :
  let tulip_bulbs := 20
  let iris_bulbs := tulip_bulbs / 2
  let daffodil_bulbs := 30
  let crocus_bulbs := 3 * daffodil_bulbs
  let total_bulbs := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  let total_earned := 75
  let payment_per_bulb := total_earned / total_bulbs
  payment_per_bulb = 0.50 := 
by
  sorry

end jane_payment_per_bulb_l61_61185


namespace distinct_nonzero_reals_product_l61_61521

theorem distinct_nonzero_reals_product 
  (x y : ℝ) 
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy: x ≠ y)
  (h : x + 3 / x = y + 3 / y) :
  x * y = 3 :=
sorry

end distinct_nonzero_reals_product_l61_61521


namespace shape_is_spiral_l61_61697

-- Assume cylindrical coordinates and constants.
variables (c : ℝ)
-- Define cylindrical coordinate properties.
variables (r θ z : ℝ)

-- Define the equation rθ = c.
def cylindrical_equation : Prop := r * θ = c

theorem shape_is_spiral (h : cylindrical_equation c r θ):
  ∃ f : ℝ → ℝ, ∀ θ > 0, r = f θ ∧ (∀ θ₁ θ₂, θ₁ < θ₂ ↔ f θ₁ > f θ₂) :=
sorry

end shape_is_spiral_l61_61697


namespace equal_intercepts_lines_area_two_lines_l61_61651

-- Defining the general equation of the line l with parameter a
def line_eq (a : ℝ) (x y : ℝ) : Prop := y = -(a + 1) * x + 2 - a

-- Problem statement for equal intercepts condition
theorem equal_intercepts_lines (a : ℝ) : 
  (∃ x y : ℝ, line_eq a x y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (x = y ∨ x + y = 2*a + 2)) →
  (a = 2 ∨ a = 0) → 
  (line_eq a 1 (-3) ∨ line_eq a 1 1) :=
sorry

-- Problem statement for triangle area condition
theorem area_two_lines (a : ℝ) : 
  (∃ x y : ℝ, line_eq a x y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (1 / 2 * |x| * |y| = 2)) →
  (a = 8 ∨ a = 0) → 
  (line_eq a 1 (-9) ∨ line_eq a 1 1) :=
sorry

end equal_intercepts_lines_area_two_lines_l61_61651


namespace amount_paid_to_shopkeeper_l61_61945

theorem amount_paid_to_shopkeeper :
  let price_of_grapes := 8 * 70
  let price_of_mangoes := 9 * 55
  price_of_grapes + price_of_mangoes = 1055 :=
by
  sorry

end amount_paid_to_shopkeeper_l61_61945


namespace solution_x_y_zero_l61_61531

theorem solution_x_y_zero (x y : ℤ) (h : x^2 * y^2 = x^2 + y^2) : x = 0 ∧ y = 0 :=
by
sorry

end solution_x_y_zero_l61_61531


namespace rowing_speed_in_still_water_l61_61271

theorem rowing_speed_in_still_water (d t1 t2 : ℝ) 
  (h1 : d = 750) (h2 : t1 = 675) (h3 : t2 = 450) : 
  (d / t1 + (d / t2 - d / t1) / 2) = 1.389 := 
by
  sorry

end rowing_speed_in_still_water_l61_61271
