import Mathlib

namespace NUMINAMATH_GPT_right_triangle_exists_l399_39958

-- Define the setup: equilateral triangle ABC, point P, and angle condition
def Point (α : Type*) := α 
def inside {α : Type*} (p : Point α) (A B C : Point α) : Prop := sorry
def angle_at {α : Type*} (p q r : Point α) (θ : ℝ) : Prop := sorry
noncomputable def PA {α : Type*} (P A : Point α) : ℝ := sorry
noncomputable def PB {α : Type*} (P B : Point α) : ℝ := sorry
noncomputable def PC {α : Type*} (P C : Point α) : ℝ := sorry

-- Theorem we need to prove
theorem right_triangle_exists {α : Type*} 
  (A B C P : Point α)
  (h1 : inside P A B C)
  (h2 : angle_at P A B 150) :
  ∃ (Q : Point α), angle_at P Q B 90 :=
sorry

end NUMINAMATH_GPT_right_triangle_exists_l399_39958


namespace NUMINAMATH_GPT_part_a_sequence_l399_39911

def circle_sequence (n m : ℕ) : List ℕ :=
  List.replicate m 1 -- Placeholder: Define the sequence computation properly

theorem part_a_sequence :
  circle_sequence 5 12 = [1, 6, 11, 4, 9, 2, 7, 12, 5, 10, 3, 8, 1] := 
sorry

end NUMINAMATH_GPT_part_a_sequence_l399_39911


namespace NUMINAMATH_GPT_arithmetic_mean_l399_39926

theorem arithmetic_mean (a b c : ℚ) (h₁ : a = 8 / 12) (h₂ : b = 10 / 12) (h₃ : c = 9 / 12) :
  c = (a + b) / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_l399_39926


namespace NUMINAMATH_GPT_smallest_exponentiated_number_l399_39929

theorem smallest_exponentiated_number :
  127^8 < 63^10 ∧ 63^10 < 33^12 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_smallest_exponentiated_number_l399_39929


namespace NUMINAMATH_GPT_find_x_l399_39908

variables {x : ℝ}
def vector_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_x
  (h1 : (6, 1) = (6, 1))
  (h2 : (x, -3) = (x, -3))
  (h3 : vector_parallel (6, 1) (x, -3)) :
  x = -18 := by
  sorry

end NUMINAMATH_GPT_find_x_l399_39908


namespace NUMINAMATH_GPT_solve_x_in_equation_l399_39939

theorem solve_x_in_equation (x : ℕ) (h : x + (x + 1) + (x + 2) + (x + 3) = 18) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_in_equation_l399_39939


namespace NUMINAMATH_GPT_negation_of_proposition_l399_39992

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l399_39992


namespace NUMINAMATH_GPT_problem_statement_l399_39915
noncomputable def f (M : ℝ) (x : ℝ) : ℝ := M * Real.sin (2 * x + Real.pi / 6)
def is_symmetric (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def is_center_of_symmetry (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop := ∀ x, f (2 * c.1 - x) = 2 * c.2 - f x

theorem problem_statement (M : ℝ) (hM : M ≠ 0) : 
    is_symmetric (f M) (2 * Real.pi / 3) ∧ 
    is_periodic (f M) Real.pi ∧ 
    is_center_of_symmetry (f M) (5 * Real.pi / 12, 0) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l399_39915


namespace NUMINAMATH_GPT_min_cut_length_no_triangle_l399_39942

theorem min_cut_length_no_triangle (a b c x : ℝ) 
  (h_y : a = 7) 
  (h_z : b = 24) 
  (h_w : c = 25) 
  (h1 : a - x > 0)
  (h2 : b - x > 0)
  (h3 : c - x > 0)
  (h4 : (a - x) + (b - x) ≤ (c - x)) :
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_cut_length_no_triangle_l399_39942


namespace NUMINAMATH_GPT_rain_on_first_day_l399_39994

theorem rain_on_first_day (x : ℝ) (h1 : x >= 0)
  (h2 : (2 * x) + 50 / 100 * (2 * x) = 3 * x) 
  (h3 : 6 * 12 = 72)
  (h4 : 3 * 3 = 9)
  (h5 : x + 2 * x + 3 * x = 6 * x)
  (h6 : 6 * x + 21 - 9 = 72) : x = 10 :=
by 
  -- Proof would go here, but we skip it according to instructions
  sorry

end NUMINAMATH_GPT_rain_on_first_day_l399_39994


namespace NUMINAMATH_GPT_range_of_a_l399_39937

variable (a : ℝ)

def a_n (n : ℕ) : ℝ :=
if n = 1 then a else 4 * ↑n + (-1 : ℝ) ^ n * (8 - 2 * a)

theorem range_of_a (h : ∀ n : ℕ, n > 0 → a_n a n < a_n a (n + 1)) : 3 < a ∧ a < 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l399_39937


namespace NUMINAMATH_GPT_problem_l399_39951

noncomputable def f (x : ℝ) : ℝ := 3 / (x + 1)

theorem problem (x : ℝ) (h1 : 3 ≤ x) (h2 : x ≤ 5) :
  (∀ x₁ x₂, 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ > f x₁) ∧
  f 3 = 3/4 ∧
  f 5 = 1/2 :=
by sorry

end NUMINAMATH_GPT_problem_l399_39951


namespace NUMINAMATH_GPT_final_movie_length_l399_39943

-- Definitions based on conditions
def original_movie_length : ℕ := 60
def cut_scene_length : ℕ := 3

-- Theorem statement proving the final length of the movie
theorem final_movie_length : original_movie_length - cut_scene_length = 57 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_final_movie_length_l399_39943


namespace NUMINAMATH_GPT_number_of_pear_trees_l399_39967

theorem number_of_pear_trees (A P : ℕ) (h1 : A + P = 46)
  (h2 : ∀ (s : Finset (Fin 46)), s.card = 28 → ∃ (i : Fin 46), i ∈ s ∧ i < A)
  (h3 : ∀ (s : Finset (Fin 46)), s.card = 20 → ∃ (i : Fin 46), i ∈ s ∧ A ≤ i) :
  P = 27 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pear_trees_l399_39967


namespace NUMINAMATH_GPT_painted_cubes_on_two_faces_l399_39924

theorem painted_cubes_on_two_faces (n : ℕ) (painted_faces_all : Prop) (equal_smaller_cubes : n = 27) : ∃ k : ℕ, k = 12 :=
by
  -- We only need the statement, not the proof
  sorry

end NUMINAMATH_GPT_painted_cubes_on_two_faces_l399_39924


namespace NUMINAMATH_GPT_scientific_notation_101_49_billion_l399_39930

-- Define the term "one hundred and one point four nine billion"
def billion (n : ℝ) := n * 10^9

-- Axiomatization of the specific number in question
def hundredOnePointFourNineBillion := billion 101.49

-- Theorem stating that the scientific notation for 101.49 billion is 1.0149 × 10^10
theorem scientific_notation_101_49_billion : hundredOnePointFourNineBillion = 1.0149 * 10^10 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_101_49_billion_l399_39930


namespace NUMINAMATH_GPT_vendelin_pastels_l399_39997

theorem vendelin_pastels (M V W : ℕ) (h1 : M = 5) (h2 : V < 5) (h3 : W = M + V) (h4 : M + V + W = 7 * V) : W = 7 := 
sorry

end NUMINAMATH_GPT_vendelin_pastels_l399_39997


namespace NUMINAMATH_GPT_find_m_collinear_l399_39923

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

end NUMINAMATH_GPT_find_m_collinear_l399_39923


namespace NUMINAMATH_GPT_smallest_two_ks_l399_39987

theorem smallest_two_ks (k : ℕ) (h : ℕ → Prop) : 
  (∀ k, (k^2 + 36) % 180 = 0 → k = 12 ∨ k = 18) :=
by {
 sorry
}

end NUMINAMATH_GPT_smallest_two_ks_l399_39987


namespace NUMINAMATH_GPT_woodworker_tables_count_l399_39952

/-- A woodworker made a total of 40 furniture legs and has built 6 chairs.
    Each chair requires 4 legs. Prove that the number of tables made is 4,
    assuming each table also requires 4 legs. -/
theorem woodworker_tables_count (total_legs chairs tables : ℕ)
  (legs_per_chair legs_per_table : ℕ)
  (H1 : total_legs = 40)
  (H2 : chairs = 6)
  (H3 : legs_per_chair = 4)
  (H4 : legs_per_table = 4)
  (H5 : total_legs = chairs * legs_per_chair + tables * legs_per_table) :
  tables = 4 := 
  sorry

end NUMINAMATH_GPT_woodworker_tables_count_l399_39952


namespace NUMINAMATH_GPT_grape_juice_amount_l399_39999

theorem grape_juice_amount 
  (T : ℝ) -- total amount of the drink 
  (orange_juice_percentage watermelon_juice_percentage : ℝ) -- percentages 
  (combined_amount_of_oj_wj : ℝ) -- combined amount of orange and watermelon juice 
  (h1 : orange_juice_percentage = 0.15)
  (h2 : watermelon_juice_percentage = 0.60)
  (h3 : combined_amount_of_oj_wj = 120)
  (h4 : combined_amount_of_oj_wj = (orange_juice_percentage + watermelon_juice_percentage) * T) : 
  (T * (1 - (orange_juice_percentage + watermelon_juice_percentage)) = 40) := 
sorry

end NUMINAMATH_GPT_grape_juice_amount_l399_39999


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l399_39920

theorem quadratic_inequality_solution_set (p q : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - p * x - q < 0) →
  p = 5 ∧ q = -6 ∧
  (∀ x : ℝ, - (1 : ℝ) / 2 < x ∧ x < - (1 : ℝ) / 3 → 6 * x^2 + 5 * x + 1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l399_39920


namespace NUMINAMATH_GPT_smaller_octagon_half_area_l399_39979

-- Define what it means to be a regular octagon
def is_regular_octagon (O : Point) (ABCDEFGH : List Point) : Prop :=
  -- Definition capturing the properties of a regular octagon around center O
  sorry

-- Define the function that computes the area of an octagon
def area_of_octagon (ABCDEFGH : List Point) : Real :=
  sorry

-- Define the function to create the smaller octagon by joining midpoints
def smaller_octagon (ABCDEFGH : List Point) : List Point :=
  sorry

theorem smaller_octagon_half_area (O : Point) (ABCDEFGH : List Point) :
  is_regular_octagon O ABCDEFGH →
  area_of_octagon (smaller_octagon ABCDEFGH) = (1 / 2) * area_of_octagon ABCDEFGH :=
by
  sorry

end NUMINAMATH_GPT_smaller_octagon_half_area_l399_39979


namespace NUMINAMATH_GPT_sphere_surface_area_l399_39973

theorem sphere_surface_area (a : ℝ) (d : ℝ) (S : ℝ) : 
  a = 3 → d = Real.sqrt 7 → S = 40 * Real.pi := by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l399_39973


namespace NUMINAMATH_GPT_ticket_price_l399_39945

theorem ticket_price (regular_price : ℕ) (discount_percent : ℕ) (paid_price : ℕ) 
  (h1 : regular_price = 15) 
  (h2 : discount_percent = 40) 
  (h3 : paid_price = regular_price - (regular_price * discount_percent / 100)) : 
  paid_price = 9 :=
by
  sorry

end NUMINAMATH_GPT_ticket_price_l399_39945


namespace NUMINAMATH_GPT_stratified_sampling_correct_l399_39931

def total_employees : ℕ := 150
def senior_titles : ℕ := 15
def intermediate_titles : ℕ := 45
def general_staff : ℕ := 90
def sample_size : ℕ := 30

def stratified_sampling (total_employees senior_titles intermediate_titles general_staff sample_size : ℕ) : (ℕ × ℕ × ℕ) :=
  (senior_titles * sample_size / total_employees, 
   intermediate_titles * sample_size / total_employees, 
   general_staff * sample_size / total_employees)

theorem stratified_sampling_correct :
  stratified_sampling total_employees senior_titles intermediate_titles general_staff sample_size = (3, 9, 18) :=
  by sorry

end NUMINAMATH_GPT_stratified_sampling_correct_l399_39931


namespace NUMINAMATH_GPT_multiplication_of_powers_l399_39906

theorem multiplication_of_powers :
  2^4 * 3^2 * 5^2 * 11 = 39600 := by
  sorry

end NUMINAMATH_GPT_multiplication_of_powers_l399_39906


namespace NUMINAMATH_GPT_coaches_needed_l399_39974

theorem coaches_needed (x : ℕ) : 44 * x + 64 = 328 := by
  sorry

end NUMINAMATH_GPT_coaches_needed_l399_39974


namespace NUMINAMATH_GPT_average_honey_per_bee_per_day_l399_39955

-- Definitions based on conditions
def num_honey_bees : ℕ := 50
def honey_bee_days : ℕ := 35
def total_honey_produced : ℕ := 75
def expected_avg_honey_per_bee_per_day : ℝ := 2.14

-- Statement of the proof problem
theorem average_honey_per_bee_per_day :
  ((total_honey_produced : ℝ) / (num_honey_bees * honey_bee_days)) = expected_avg_honey_per_bee_per_day := by
  sorry

end NUMINAMATH_GPT_average_honey_per_bee_per_day_l399_39955


namespace NUMINAMATH_GPT_cos_alpha_plus_beta_l399_39953

variable {α β : ℝ}
variable (sin_alpha : Real.sin α = 3/5)
variable (cos_beta : Real.cos β = 4/5)
variable (α_interval : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
variable (β_interval : β ∈ Set.Ioo 0 (Real.pi / 2))

theorem cos_alpha_plus_beta: Real.cos (α + β) = -1 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_beta_l399_39953


namespace NUMINAMATH_GPT_solve_percentage_of_X_in_B_l399_39917

variable (P : ℝ)

def liquid_X_in_A_percentage : ℝ := 0.008
def mass_of_A : ℝ := 200
def mass_of_B : ℝ := 700
def mixed_solution_percentage_of_X : ℝ := 0.0142
def target_percentage_of_P_in_B : ℝ := 0.01597

theorem solve_percentage_of_X_in_B (P : ℝ) 
  (h1 : mass_of_A * liquid_X_in_A_percentage + mass_of_B * P = (mass_of_A + mass_of_B) * mixed_solution_percentage_of_X) :
  P = target_percentage_of_P_in_B :=
sorry

end NUMINAMATH_GPT_solve_percentage_of_X_in_B_l399_39917


namespace NUMINAMATH_GPT_even_numbers_set_l399_39965

-- Define the set of all even numbers in set-builder notation
def even_set : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

-- Theorem stating that this set is the set of all even numbers
theorem even_numbers_set :
  ∀ x : ℤ, (x ∈ even_set ↔ ∃ n : ℤ, x = 2 * n) := by
  sorry

end NUMINAMATH_GPT_even_numbers_set_l399_39965


namespace NUMINAMATH_GPT_find_integer_solutions_l399_39966

theorem find_integer_solutions :
  { (a, b, c, d) : ℤ × ℤ × ℤ × ℤ |
    (a * b - 2 * c * d = 3) ∧ (a * c + b * d = 1) } =
  { (1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1) } :=
by
  sorry

end NUMINAMATH_GPT_find_integer_solutions_l399_39966


namespace NUMINAMATH_GPT_polynomial_solution_l399_39980

variable (P : ℚ) -- Assuming P is a constant polynomial

theorem polynomial_solution (P : ℚ) 
  (condition : P + (2 : ℚ) * X^2 + (5 : ℚ) * X - (2 : ℚ) = (2 : ℚ) * X^2 + (5 : ℚ) * X + (4 : ℚ)): 
  P = 6 := 
  sorry

end NUMINAMATH_GPT_polynomial_solution_l399_39980


namespace NUMINAMATH_GPT_complete_square_equation_l399_39941

theorem complete_square_equation (b c : ℤ) (h : (x : ℝ) → x^2 - 6 * x + 5 = (x + b)^2 - c) : b + c = 1 :=
by
  sorry  -- This is where the proof would go

end NUMINAMATH_GPT_complete_square_equation_l399_39941


namespace NUMINAMATH_GPT_sum_of_values_not_satisfying_eq_l399_39960

variable {A B C x : ℝ}

theorem sum_of_values_not_satisfying_eq (h : (∀ x, ∃ C, ∃ B, A = 3 ∧ ((x + B) * (A * x + 36) = 3 * (x + C) * (x + 9)) ∧ (x ≠ -9))):
  ∃ y, y = -9 := sorry

end NUMINAMATH_GPT_sum_of_values_not_satisfying_eq_l399_39960


namespace NUMINAMATH_GPT_probability_of_picking_same_color_shoes_l399_39905

theorem probability_of_picking_same_color_shoes
  (n_pairs_black : ℕ) (n_pairs_brown : ℕ) (n_pairs_gray : ℕ)
  (h_black_pairs : n_pairs_black = 8)
  (h_brown_pairs : n_pairs_brown = 4)
  (h_gray_pairs : n_pairs_gray = 3)
  (total_shoes : ℕ := 2 * (n_pairs_black + n_pairs_brown + n_pairs_gray)) :
  (16 / total_shoes * 8 / (total_shoes - 1) + 
   8 / total_shoes * 4 / (total_shoes - 1) + 
   6 / total_shoes * 3 / (total_shoes - 1)) = 89 / 435 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_picking_same_color_shoes_l399_39905


namespace NUMINAMATH_GPT_frustum_volume_correct_l399_39947

-- Definitions of pyramids and their properties
structure Pyramid :=
  (base_edge : ℕ)
  (altitude : ℕ)
  (volume : ℚ)

-- Definition of the original pyramid and smaller pyramid
def original_pyramid : Pyramid := {
  base_edge := 20,
  altitude := 10,
  volume := (1 / 3 : ℚ) * (20 ^ 2) * 10
}

def smaller_pyramid : Pyramid := {
  base_edge := 8,
  altitude := 5,
  volume := (1 / 3 : ℚ) * (8 ^ 2) * 5
}

-- Definition and calculation of the volume of the frustum 
def volume_frustum (p1 p2 : Pyramid) : ℚ :=
  p1.volume - p2.volume

-- Main theorem to be proved
theorem frustum_volume_correct :
  volume_frustum original_pyramid smaller_pyramid = 992 := by
  sorry

end NUMINAMATH_GPT_frustum_volume_correct_l399_39947


namespace NUMINAMATH_GPT_segment_shadow_ratio_l399_39919

theorem segment_shadow_ratio (a b a' b' : ℝ) (h : a / b = a' / b') : a / a' = b / b' :=
sorry

end NUMINAMATH_GPT_segment_shadow_ratio_l399_39919


namespace NUMINAMATH_GPT_greater_number_l399_39984

theorem greater_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : x = 35 := 
by sorry

end NUMINAMATH_GPT_greater_number_l399_39984


namespace NUMINAMATH_GPT_tourist_total_value_l399_39964

theorem tourist_total_value
    (tax_rate : ℝ)
    (V : ℝ)
    (tax_paid : ℝ)
    (exempt_amount : ℝ) :
    exempt_amount = 600 ∧
    tax_rate = 0.07 ∧
    tax_paid = 78.4 →
    (tax_rate * (V - exempt_amount) = tax_paid) →
    V = 1720 :=
by
  intros h1 h2
  have h_exempt : exempt_amount = 600 := h1.left
  have h_tax_rate : tax_rate = 0.07 := h1.right.left
  have h_tax_paid : tax_paid = 78.4 := h1.right.right
  sorry

end NUMINAMATH_GPT_tourist_total_value_l399_39964


namespace NUMINAMATH_GPT_sticker_ratio_l399_39944

theorem sticker_ratio (gold : ℕ) (silver : ℕ) (bronze : ℕ)
  (students : ℕ) (stickers_per_student : ℕ)
  (h1 : gold = 50)
  (h2 : bronze = silver - 20)
  (h3 : students = 5)
  (h4 : stickers_per_student = 46)
  (h5 : gold + silver + bronze = students * stickers_per_student) :
  silver / gold = 2 / 1 :=
by
  sorry

end NUMINAMATH_GPT_sticker_ratio_l399_39944


namespace NUMINAMATH_GPT_min_shirts_to_save_money_l399_39972

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), 60 + 11 * x < 20 + 15 * x ∧ (∀ y : ℕ, 60 + 11 * y < 20 + 15 * y → y ≥ x) ∧ x = 11 :=
by
  sorry

end NUMINAMATH_GPT_min_shirts_to_save_money_l399_39972


namespace NUMINAMATH_GPT_nat_le_two_pow_million_l399_39970

theorem nat_le_two_pow_million (n : ℕ) (h : n ≤ 2^1000000) : 
  ∃ (x : ℕ → ℕ) (k : ℕ), k ≤ 1100000 ∧ x 0 = 1 ∧ x k = n ∧ 
  ∀ (i : ℕ), 1 ≤ i → i ≤ k → ∃ (r s : ℕ), 0 ≤ r ∧ r ≤ s ∧ s < i ∧ x i = x r + x s :=
sorry

end NUMINAMATH_GPT_nat_le_two_pow_million_l399_39970


namespace NUMINAMATH_GPT_find_D_c_l399_39904

-- Define the given conditions
def daily_wage_ratio (W_a W_b W_c : ℝ) : Prop :=
  W_a / W_b = 3 / 4 ∧ W_a / W_c = 3 / 5 ∧ W_b / W_c = 4 / 5

def total_earnings (W_a W_b W_c : ℝ) (D_a D_b D_c : ℕ) : ℝ :=
  W_a * D_a + W_b * D_b + W_c * D_c

variables {W_a W_b W_c : ℝ} 
variables {D_a D_b D_c : ℕ} 

-- Given values according to the problem
def W_c_value : ℝ := 110
def D_a_value : ℕ := 6
def D_b_value : ℕ := 9
def total_earnings_value : ℝ := 1628

-- The target proof statement
theorem find_D_c 
  (h_ratio : daily_wage_ratio W_a W_b W_c)
  (h_Wc : W_c = W_c_value)
  (h_earnings : total_earnings W_a W_b W_c D_a_value D_b_value D_c = total_earnings_value) 
  : D_c = 4 := 
sorry

end NUMINAMATH_GPT_find_D_c_l399_39904


namespace NUMINAMATH_GPT_arithmetic_mean_two_digit_multiples_of_8_l399_39969

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_two_digit_multiples_of_8_l399_39969


namespace NUMINAMATH_GPT_construct_circle_feasible_l399_39962

theorem construct_circle_feasible (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : b^2 > (a^2 + c^2) / 2) :
  ∃ x y d : ℝ, 
  d > 0 ∧ 
  (d / 2)^2 = y^2 + (a / 2)^2 ∧ 
  (d / 2)^2 = (y - x)^2 + (b / 2)^2 ∧ 
  (d / 2)^2 = (y - 2 * x)^2 + (c / 2)^2 :=
sorry

end NUMINAMATH_GPT_construct_circle_feasible_l399_39962


namespace NUMINAMATH_GPT_find_x_of_arithmetic_mean_l399_39903

theorem find_x_of_arithmetic_mean (x : ℝ) (h : (6 + 13 + 18 + 4 + x) / 5 = 10) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_x_of_arithmetic_mean_l399_39903


namespace NUMINAMATH_GPT_largest_common_term_l399_39978

-- Definitions for the first arithmetic sequence
def arithmetic_seq1 (n : ℕ) : ℕ := 2 + 5 * n

-- Definitions for the second arithmetic sequence
def arithmetic_seq2 (m : ℕ) : ℕ := 5 + 8 * m

-- Main statement of the problem
theorem largest_common_term (n m k : ℕ) (a : ℕ) :
  (a = arithmetic_seq1 n) ∧ (a = arithmetic_seq2 m) ∧ (1 ≤ a) ∧ (a ≤ 150) →
  a = 117 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_common_term_l399_39978


namespace NUMINAMATH_GPT_no_x_squared_term_l399_39913

theorem no_x_squared_term {m : ℚ} (h : (x+1) * (x^2 + 5*m*x + 3) = x^3 + (5*m + 1)*x^2 + (3 + 5*m)*x + 3) : 
  5*m + 1 = 0 → m = -1/5 := by sorry

end NUMINAMATH_GPT_no_x_squared_term_l399_39913


namespace NUMINAMATH_GPT_weight_of_white_ring_l399_39936

def weight_orange := 0.08333333333333333
def weight_purple := 0.3333333333333333
def total_weight := 0.8333333333

def weight_white := 0.41666666663333337

theorem weight_of_white_ring :
  weight_white + weight_orange + weight_purple = total_weight :=
by
  sorry

end NUMINAMATH_GPT_weight_of_white_ring_l399_39936


namespace NUMINAMATH_GPT_no_infinite_prime_sequence_l399_39914

theorem no_infinite_prime_sequence (p : ℕ) (h_prime : Nat.Prime p) :
  ¬(∃ (p_seq : ℕ → ℕ), (∀ n, Nat.Prime (p_seq n)) ∧ (∀ n, p_seq (n + 1) = 2 * p_seq n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_no_infinite_prime_sequence_l399_39914


namespace NUMINAMATH_GPT_find_x_l399_39986

-- Define the conditions according to the problem statement
variables {C x : ℝ} -- C is the cost per liter of pure spirit, x is the volume of water in the first solution

-- Condition 1: The cost for the first solution
def cost_first_solution (C : ℝ) (x : ℝ) : Prop := 0.50 = C * (1 / (1 + x))

-- Condition 2: The cost for the second solution (approximating 0.4999999999999999 as 0.50)
def cost_second_solution (C : ℝ) : Prop := 0.50 = C * (1 / 3)

-- The theorem to prove: x = 2 given the two conditions
theorem find_x (C : ℝ) (x : ℝ) (h1 : cost_first_solution C x) (h2 : cost_second_solution C) : x = 2 := 
sorry

end NUMINAMATH_GPT_find_x_l399_39986


namespace NUMINAMATH_GPT_son_work_rate_l399_39932

theorem son_work_rate (M S : ℝ) (hM : M = 1 / 5) (hMS : M + S = 1 / 4) : 1 / S = 20 :=
by
  sorry

end NUMINAMATH_GPT_son_work_rate_l399_39932


namespace NUMINAMATH_GPT_circle_radius_l399_39993

theorem circle_radius (A : ℝ) (r : ℝ) (h : A = 36 * Real.pi) (h2 : A = Real.pi * r ^ 2) : r = 6 :=
sorry

end NUMINAMATH_GPT_circle_radius_l399_39993


namespace NUMINAMATH_GPT_range_of_f_1_over_f_2_l399_39975

theorem range_of_f_1_over_f_2 {f : ℝ → ℝ} (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  1 / 8 < f 1 / f 2 ∧ f 1 / f 2 < 1 / 4 :=
by sorry

end NUMINAMATH_GPT_range_of_f_1_over_f_2_l399_39975


namespace NUMINAMATH_GPT_inverse_variation_solution_l399_39982

noncomputable def const_k (x y : ℝ) := (x^2) * (y^4)

theorem inverse_variation_solution (x y : ℝ) (k : ℝ) (h1 : x = 8) (h2 : y = 2) (h3 : k = const_k x y) :
  ∀ y' : ℝ, y' = 4 → const_k x y' = 1024 → x^2 = 4 := by
  intros
  sorry

end NUMINAMATH_GPT_inverse_variation_solution_l399_39982


namespace NUMINAMATH_GPT_intersection_complement_l399_39928

open Set

-- Defining sets A, B and universal set U
def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {x | 1 < x ∧ x ≤ 6}
def U : Set ℕ := A ∪ B

-- Statement of the proof problem
theorem intersection_complement :
  A ∩ (U \ B) = {1, 7} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l399_39928


namespace NUMINAMATH_GPT_point_B_coordinates_sum_l399_39995

theorem point_B_coordinates_sum (x : ℚ) (h1 : ∃ (B : ℚ × ℚ), B = (x, 5))
    (h2 : (5 - 0) / (x - 0) = 3/4) :
    x + 5 = 35/3 :=
by
  sorry

end NUMINAMATH_GPT_point_B_coordinates_sum_l399_39995


namespace NUMINAMATH_GPT_length_of_AB_l399_39963

theorem length_of_AB
  (P Q : ℝ) (AB : ℝ)
  (hP : P = 3 / 7 * AB)
  (hQ : Q = 4 / 9 * AB)
  (hPQ : abs (Q - P) = 3) :
  AB = 189 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l399_39963


namespace NUMINAMATH_GPT_find_required_school_year_hours_l399_39988

-- Define constants for the problem
def summer_hours_per_week : ℕ := 40
def summer_weeks : ℕ := 12
def summer_earnings : ℕ := 6000
def school_year_weeks : ℕ := 36
def school_year_earnings : ℕ := 9000

-- Calculate total summer hours, hourly rate, total school year hours, and required school year weekly hours
def total_summer_hours := summer_hours_per_week * summer_weeks
def hourly_rate := summer_earnings / total_summer_hours
def total_school_year_hours := school_year_earnings / hourly_rate
def required_school_year_hours_per_week := total_school_year_hours / school_year_weeks

-- Prove the required hours per week is 20
theorem find_required_school_year_hours : required_school_year_hours_per_week = 20 := by
  sorry

end NUMINAMATH_GPT_find_required_school_year_hours_l399_39988


namespace NUMINAMATH_GPT_children_more_than_adults_l399_39938

-- Definitions based on given conditions
def price_per_child : ℚ := 4.50
def price_per_adult : ℚ := 6.75
def total_receipts : ℚ := 405
def number_of_children : ℕ := 48

-- Goal: Prove the number of children is 20 more than the number of adults.
theorem children_more_than_adults :
  ∃ (A : ℕ), (number_of_children - A) = 20 ∧ (price_per_child * number_of_children) + (price_per_adult * A) = total_receipts := by
  sorry

end NUMINAMATH_GPT_children_more_than_adults_l399_39938


namespace NUMINAMATH_GPT_find_n_l399_39909

theorem find_n (a b c : ℤ) (m n p : ℕ)
  (h1 : a = 3)
  (h2 : b = -7)
  (h3 : c = -6)
  (h4 : m > 0)
  (h5 : n > 0)
  (h6 : p > 0)
  (h7 : Nat.gcd m p = 1)
  (h8 : Nat.gcd m n = 1)
  (h9 : Nat.gcd n p = 1)
  (h10 : ∃ x1 x2 : ℤ, x1 = (m + Int.sqrt n) / p ∧ x2 = (m - Int.sqrt n) / p)
  : n = 121 :=
sorry

end NUMINAMATH_GPT_find_n_l399_39909


namespace NUMINAMATH_GPT_find_a_l399_39910

theorem find_a (a x y : ℝ) (h1 : a^(3*x - 1) * 3^(4*y - 3) = 49^x * 27^y) (h2 : x + y = 4) : a = 7 := by
  sorry

end NUMINAMATH_GPT_find_a_l399_39910


namespace NUMINAMATH_GPT_number_of_solutions_l399_39949

theorem number_of_solutions :
  ∃ (s : Finset (ℤ × ℤ)), (∀ (a : ℤ × ℤ), a ∈ s ↔ (a.1^4 + a.2^4 = 4 * a.2)) ∧ s.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l399_39949


namespace NUMINAMATH_GPT_A_times_B_correct_l399_39950

noncomputable def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {y | y > 1}
noncomputable def A_times_B : Set ℝ := {x | (x ∈ A ∪ B) ∧ ¬(x ∈ A ∩ B)}

theorem A_times_B_correct : A_times_B = {x | (0 ≤ x ∧ x ≤ 1) ∨ x > 2} := 
sorry

end NUMINAMATH_GPT_A_times_B_correct_l399_39950


namespace NUMINAMATH_GPT_lisa_speed_l399_39968

-- Define conditions
def distance : ℕ := 256
def time : ℕ := 8

-- Define the speed calculation theorem
theorem lisa_speed : (distance / time) = 32 := 
by {
  sorry
}

end NUMINAMATH_GPT_lisa_speed_l399_39968


namespace NUMINAMATH_GPT_prove_q_ge_bd_and_p_eq_ac_l399_39907

-- Definitions for the problem
variables (a b c d p q : ℕ)

-- Conditions given in the problem
axiom h1: a * d - b * c = 1
axiom h2: (a : ℚ) / b > (p : ℚ) / q
axiom h3: (p : ℚ) / q > (c : ℚ) / d

-- The theorem to be proved
theorem prove_q_ge_bd_and_p_eq_ac (a b c d p q : ℕ) (h1 : a * d - b * c = 1) 
  (h2 : (a : ℚ) / b > (p : ℚ) / q) (h3 : (p : ℚ) / q > (c : ℚ) / d) :
  q ≥ b + d ∧ (q = b + d → p = a + c) :=
by
  sorry

end NUMINAMATH_GPT_prove_q_ge_bd_and_p_eq_ac_l399_39907


namespace NUMINAMATH_GPT_worker_new_wage_after_increase_l399_39946

theorem worker_new_wage_after_increase (initial_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) 
  (h1 : initial_wage = 34) (h2 : increase_percentage = 0.50) 
  (h3 : new_wage = initial_wage + (increase_percentage * initial_wage)) : new_wage = 51 := 
by
  sorry

end NUMINAMATH_GPT_worker_new_wage_after_increase_l399_39946


namespace NUMINAMATH_GPT_airplane_seats_theorem_l399_39989

def airplane_seats_proof : Prop :=
  ∀ (s : ℝ),
  (∃ (first_class business_class economy premium_economy : ℝ),
    first_class = 30 ∧
    business_class = 0.4 * s ∧
    economy = 0.6 * s ∧
    premium_economy = s - (first_class + business_class + economy)) →
  s = 150

theorem airplane_seats_theorem : airplane_seats_proof :=
sorry

end NUMINAMATH_GPT_airplane_seats_theorem_l399_39989


namespace NUMINAMATH_GPT_difference_of_squares_l399_39925

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 20) : a^2 - b^2 = 1200 := 
sorry

end NUMINAMATH_GPT_difference_of_squares_l399_39925


namespace NUMINAMATH_GPT_simplify_expression_l399_39954

theorem simplify_expression : (8 * (15 / 9) * (-45 / 40) = -(1 / 15)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l399_39954


namespace NUMINAMATH_GPT_number_of_coaches_l399_39902

theorem number_of_coaches (r : ℕ) (v : ℕ) (c : ℕ) (h1 : r = 60) (h2 : v = 3) (h3 : c * 5 = 60 * 3) : c = 36 :=
by
  -- We skip the proof as per instructions
  sorry

end NUMINAMATH_GPT_number_of_coaches_l399_39902


namespace NUMINAMATH_GPT_inequality_for_a_l399_39901

noncomputable def f (x : ℝ) : ℝ :=
  2^x + (Real.log x) / (Real.log 2)

theorem inequality_for_a (n : ℕ) (a : ℝ) (h₁ : 2 < n) (h₂ : 0 < a) (h₃ : 2^a + Real.log a / Real.log 2 = n^2) :
  2 * Real.log n / Real.log 2 > a ∧ a > 2 * Real.log n / Real.log 2 - 1 / n :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_a_l399_39901


namespace NUMINAMATH_GPT_simplify_sqrt_expression_l399_39956

theorem simplify_sqrt_expression :
  (Real.sqrt 726 / Real.sqrt 242) + (Real.sqrt 484 / Real.sqrt 121) = Real.sqrt 3 + 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_l399_39956


namespace NUMINAMATH_GPT_Deepak_and_Wife_meet_time_l399_39971

theorem Deepak_and_Wife_meet_time 
    (circumference : ℕ) 
    (Deepak_speed : ℕ)
    (wife_speed : ℕ) 
    (conversion_factor_km_hr_to_m_hr : ℕ) 
    (minutes_per_hour : ℕ) :
    circumference = 726 →
    Deepak_speed = 4500 →  -- speed in meters per hour
    wife_speed = 3750 →  -- speed in meters per hour
    conversion_factor_km_hr_to_m_hr = 1000 →
    minutes_per_hour = 60 →
    (726 / ((4500 + 3750) / 1000) * 60 = 5.28) :=
by 
    sorry

end NUMINAMATH_GPT_Deepak_and_Wife_meet_time_l399_39971


namespace NUMINAMATH_GPT_length_of_bridge_l399_39933

noncomputable def L_train : ℝ := 110
noncomputable def v_train : ℝ := 72 * (1000 / 3600)
noncomputable def t : ℝ := 12.099

theorem length_of_bridge : (v_train * t - L_train) = 131.98 :=
by
  -- The proof should come here
  sorry

end NUMINAMATH_GPT_length_of_bridge_l399_39933


namespace NUMINAMATH_GPT_tom_blue_marbles_l399_39927

-- Definitions based on conditions
def jason_blue_marbles : Nat := 44
def total_blue_marbles : Nat := 68

-- The problem statement to prove
theorem tom_blue_marbles : (total_blue_marbles - jason_blue_marbles) = 24 :=
by
  sorry

end NUMINAMATH_GPT_tom_blue_marbles_l399_39927


namespace NUMINAMATH_GPT_normal_price_of_article_l399_39918

theorem normal_price_of_article (P : ℝ) (h : 0.90 * 0.80 * P = 36) : P = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_normal_price_of_article_l399_39918


namespace NUMINAMATH_GPT_solve_for_a_l399_39900

theorem solve_for_a (a y x : ℝ)
  (h1 : y = 5 * a)
  (h2 : x = 2 * a - 2)
  (h3 : y + 3 = x) :
  a = -5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l399_39900


namespace NUMINAMATH_GPT_qualified_products_correct_l399_39990

def defect_rate : ℝ := 0.005
def total_produced : ℝ := 18000

theorem qualified_products_correct :
  total_produced * (1 - defect_rate) = 17910 := by
  sorry

end NUMINAMATH_GPT_qualified_products_correct_l399_39990


namespace NUMINAMATH_GPT_John_finishes_at_610PM_l399_39935

def TaskTime : Nat := 55
def StartTime : Nat := 14 * 60 + 30 -- 2:30 PM in minutes
def EndSecondTask : Nat := 16 * 60 + 20 -- 4:20 PM in minutes

theorem John_finishes_at_610PM (h1 : TaskTime * 2 = EndSecondTask - StartTime) : 
  (EndSecondTask + TaskTime * 2) = (18 * 60 + 10) :=
by
  sorry

end NUMINAMATH_GPT_John_finishes_at_610PM_l399_39935


namespace NUMINAMATH_GPT_inequality_correct_l399_39934

-- Theorem: For all real numbers x and y, if x ≥ y, then x² + y² ≥ 2xy.
theorem inequality_correct (x y : ℝ) (h : x ≥ y) : x^2 + y^2 ≥ 2 * x * y := 
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_inequality_correct_l399_39934


namespace NUMINAMATH_GPT_inscribed_rectangle_area_l399_39998

variable (a b h x : ℝ)
variable (h_pos : 0 < h) (a_b_pos : a > b) (b_pos : b > 0) (a_pos : a > 0) (x_pos : 0 < x) (hx : x < h)

theorem inscribed_rectangle_area (hb : b > 0) (ha : a > 0) (hx : 0 < x) (hxa : x < h) : 
  x * (a - b) * (h - x) / h = x * (a - b) * (h - x) / h := by
  sorry

end NUMINAMATH_GPT_inscribed_rectangle_area_l399_39998


namespace NUMINAMATH_GPT_mari_buttons_l399_39921

/-- 
Given that:
1. Sue made 6 buttons
2. Sue made half as many buttons as Kendra.
3. Mari made 4 more than five times as many buttons as Kendra.

We are to prove that Mari made 64 buttons.
-/
theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : sue_buttons = kendra_buttons / 2)
  (h3 : mari_buttons = 5 * kendra_buttons + 4) :
  mari_buttons = 64 :=
  sorry

end NUMINAMATH_GPT_mari_buttons_l399_39921


namespace NUMINAMATH_GPT_total_flowers_bouquets_l399_39976

-- Define the number of tulips Lana picked
def tulips : ℕ := 36

-- Define the number of roses Lana picked
def roses : ℕ := 37

-- Define the number of extra flowers Lana picked
def extra_flowers : ℕ := 3

-- Prove that the total number of flowers used by Lana for the bouquets is 76
theorem total_flowers_bouquets : (tulips + roses + extra_flowers) = 76 :=
by
  sorry

end NUMINAMATH_GPT_total_flowers_bouquets_l399_39976


namespace NUMINAMATH_GPT_graph_is_hyperbola_l399_39940

theorem graph_is_hyperbola : ∀ x y : ℝ, (x + y) ^ 2 = x ^ 2 + y ^ 2 + 2 * x + 2 * y ↔ (x - 1) * (y - 1) = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_graph_is_hyperbola_l399_39940


namespace NUMINAMATH_GPT_compute_expression_l399_39957

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l399_39957


namespace NUMINAMATH_GPT_thirtieth_triangular_number_is_465_l399_39981

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem thirtieth_triangular_number_is_465 : triangular_number 30 = 465 :=
by
  sorry

end NUMINAMATH_GPT_thirtieth_triangular_number_is_465_l399_39981


namespace NUMINAMATH_GPT_drew_got_wrong_19_l399_39922

theorem drew_got_wrong_19 :
  ∃ (D_wrong C_wrong : ℕ), 
    (20 + D_wrong = 52) ∧
    (14 + C_wrong = 52) ∧
    (C_wrong = 2 * D_wrong) ∧
    D_wrong = 19 :=
by
  sorry

end NUMINAMATH_GPT_drew_got_wrong_19_l399_39922


namespace NUMINAMATH_GPT_remainder_of_poly_div_l399_39977

theorem remainder_of_poly_div (x : ℤ) : 
  (x + 1)^2009 % (x^2 + x + 1) = x + 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_poly_div_l399_39977


namespace NUMINAMATH_GPT_mary_more_candy_initially_l399_39948

-- Definitions of the conditions
def Megan_initial_candy : ℕ := 5
def Mary_candy_after_addition : ℕ := 25
def additional_candy_Mary_adds : ℕ := 10

-- The proof problem statement
theorem mary_more_candy_initially :
  (Mary_candy_after_addition - additional_candy_Mary_adds) / Megan_initial_candy = 3 :=
by
  sorry

end NUMINAMATH_GPT_mary_more_candy_initially_l399_39948


namespace NUMINAMATH_GPT_empty_bidon_weight_l399_39959

theorem empty_bidon_weight (B M : ℝ) 
  (h1 : B + M = 34) 
  (h2 : B + M / 2 = 17.5) : 
  B = 1 := 
by {
  -- The proof steps would go here, but we just add sorry
  sorry
}

end NUMINAMATH_GPT_empty_bidon_weight_l399_39959


namespace NUMINAMATH_GPT_total_books_is_10_l399_39991

def total_books (B : ℕ) : Prop :=
  (2 / 5 : ℚ) * B + (3 / 10 : ℚ) * B + ((3 / 10 : ℚ) * B - 1) + 1 = B

theorem total_books_is_10 : total_books 10 := by
  sorry

end NUMINAMATH_GPT_total_books_is_10_l399_39991


namespace NUMINAMATH_GPT_condition_for_ellipse_l399_39961

-- Definition of the problem conditions
def is_ellipse (m : ℝ) : Prop :=
  (m - 2 > 0) ∧ (5 - m > 0) ∧ (m - 2 ≠ 5 - m)

noncomputable def necessary_not_sufficient_condition (m : ℝ) : Prop :=
  (2 < m) ∧ (m < 5)

-- The theorem to be proved
theorem condition_for_ellipse (m : ℝ) : 
  (necessary_not_sufficient_condition m) → (is_ellipse m) :=
by
  -- proof to be written here
  sorry

end NUMINAMATH_GPT_condition_for_ellipse_l399_39961


namespace NUMINAMATH_GPT_xyz_value_l399_39985

theorem xyz_value
  (x y z : ℝ)
  (h1 : (x + y + z) * (xy + xz + yz) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9)
  (h3 : x + y + z = 3)
  : xyz = 5 :=
by
  sorry

end NUMINAMATH_GPT_xyz_value_l399_39985


namespace NUMINAMATH_GPT_number_of_children_l399_39916

variables (n : ℕ) (y : ℕ) (d : ℕ)

def sum_of_ages (n : ℕ) (y : ℕ) (d : ℕ) : ℕ :=
  n * y + d * (n * (n - 1) / 2)

theorem number_of_children (H1 : sum_of_ages n 6 3 = 60) : n = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_children_l399_39916


namespace NUMINAMATH_GPT_product_of_numbers_l399_39996

variable {x y : ℝ}

theorem product_of_numbers (h1 : x - y = 1 * k) (h2 : x + y = 8 * k) (h3 : x * y = 40 * k) : 
  x * y = 6400 / 63 := by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l399_39996


namespace NUMINAMATH_GPT_range_of_m_l399_39983

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) ↔ -4 ≤ m ∧ m ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l399_39983


namespace NUMINAMATH_GPT_distinct_colorings_l399_39912

def sections : ℕ := 6
def red_count : ℕ := 3
def blue_count : ℕ := 1
def green_count : ℕ := 1
def yellow_count : ℕ := 1

def permutations_without_rotation : ℕ := Nat.factorial sections / 
  (Nat.factorial red_count * Nat.factorial blue_count * Nat.factorial green_count * Nat.factorial yellow_count)

def rotational_symmetry : ℕ := permutations_without_rotation / sections

theorem distinct_colorings (rotational_symmetry) : rotational_symmetry = 20 :=
  sorry

end NUMINAMATH_GPT_distinct_colorings_l399_39912
