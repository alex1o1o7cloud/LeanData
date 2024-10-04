import Mathlib

namespace functions_not_exist_l262_262885

theorem functions_not_exist :
  ¬ (∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), x ≠ y → |f x - f y| + |g x - g y| > 1) :=
by
  sorry

end functions_not_exist_l262_262885


namespace closest_square_to_350_l262_262269

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l262_262269


namespace A_not_on_transformed_plane_l262_262815

noncomputable def A : ℝ × ℝ × ℝ := (-3, -2, 4)
noncomputable def k : ℝ := -4/5
noncomputable def original_plane (x y z : ℝ) : Prop := 2 * x - 3 * y + z - 5 = 0

noncomputable def transformed_plane (x y z : ℝ) : Prop := 
  2 * x - 3 * y + z + (k * -5) = 0

theorem A_not_on_transformed_plane :
  ¬ transformed_plane (-3) (-2) 4 :=
by
  sorry

end A_not_on_transformed_plane_l262_262815


namespace right_triangle_hypotenuse_length_l262_262504

theorem right_triangle_hypotenuse_length 
    (AB AC x y : ℝ) 
    (P : AB = x) (Q : AC = y) 
    (ratio_AP_PB : AP / PB = 1 / 3) 
    (ratio_AQ_QC : AQ / QC = 2 / 1) 
    (BQ_length : BQ = 18) 
    (CP_length : CP = 24) : 
    BC = 24 := 
by 
  sorry

end right_triangle_hypotenuse_length_l262_262504


namespace cos_beta_value_l262_262996

noncomputable def cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) : Real :=
  Real.cos β

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) :
  Real.cos β = 56 / 65 :=
by
  sorry

end cos_beta_value_l262_262996


namespace closest_perfect_square_to_350_l262_262256

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l262_262256


namespace trace_bag_weight_l262_262750

-- Definitions for the given problem
def weight_gordon_bag1 := 3
def weight_gordon_bag2 := 7
def total_weight_gordon := weight_gordon_bag1 + weight_gordon_bag2

noncomputable def weight_trace_one_bag : ℕ :=
  sorry

-- Theorem for what we need to prove
theorem trace_bag_weight :
  total_weight_gordon = 10 ∧
  weight_trace_one_bag = total_weight_gordon / 5 :=
sorry

end trace_bag_weight_l262_262750


namespace unique_solution_of_quadratic_l262_262866

theorem unique_solution_of_quadratic (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a = 9 / 8) :=
by
  sorry

end unique_solution_of_quadratic_l262_262866


namespace glucose_solution_l262_262824

theorem glucose_solution (x : ℝ) (h : (15 / 100 : ℝ) = (6.75 / x)) : x = 45 :=
sorry

end glucose_solution_l262_262824


namespace necessary_but_not_sufficient_condition_for_a_lt_neg_one_l262_262817

theorem necessary_but_not_sufficient_condition_for_a_lt_neg_one (a : ℝ) : 
  (1 / a > -1) ↔ (a < -1) :=
by sorry

end necessary_but_not_sufficient_condition_for_a_lt_neg_one_l262_262817


namespace f_one_minus_a_l262_262019

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 2) = f x
axiom f_one_plus_a {a : ℝ} : f (1 + a) = 1

theorem f_one_minus_a (a : ℝ) : f (1 - a) = -1 :=
by
  sorry

end f_one_minus_a_l262_262019


namespace train_B_departure_time_l262_262922

def distance : ℕ := 65
def speed_A : ℕ := 20
def speed_B : ℕ := 25
def departure_A := 7
def meeting_time := 9

theorem train_B_departure_time : ∀ (d : ℕ) (vA : ℕ) (vB : ℕ) (tA : ℕ) (m : ℕ), 
  d = 65 → vA = 20 → vB = 25 → tA = 7 → m = 9 → ((9 - (m - tA + (d - (2 * vA)) / vB)) = 1) → 
  8 = ((9 - (meeting_time - departure_A + (distance - (2 * speed_A)) / speed_B))) := 
  by {
    sorry
  }

end train_B_departure_time_l262_262922


namespace vectors_parallel_x_eq_four_l262_262335

theorem vectors_parallel_x_eq_four (x : ℝ) :
  (x > 0) →
  (∃ k : ℝ, (8 + 1/2 * x, x) = k • (x + 1, 2)) →
  x = 4 :=
by
  intro h1 h2
  sorry

end vectors_parallel_x_eq_four_l262_262335


namespace cos_54_eq_3_sub_sqrt_5_div_8_l262_262457

theorem cos_54_eq_3_sub_sqrt_5_div_8 :
  let x := Real.cos (Real.pi / 10) in
  let y := Real.cos (3 * Real.pi / 10) in
  y = (3 - Real.sqrt 5) / 8 :=
by
  -- Proof of the statement is omitted.
  sorry

end cos_54_eq_3_sub_sqrt_5_div_8_l262_262457


namespace dodecahedron_interior_diagonals_eq_160_l262_262652

-- Definition of a dodecahedron for the context of the proof
structure Dodecahedron where
  num_faces : Nat
  num_vertices : Nat
  faces_meeting_at_vertex : Nat
  vertices_connected_by_edge : Nat

-- Given our specific dodecahedron properties
def dodecahedron : Dodecahedron := {
  num_faces := 12,
  num_vertices := 20,
  faces_meeting_at_vertex := 3,
  vertices_connected_by_edge := 3
}

-- The interior diagonal is defined as a segment connecting two vertices
-- which do not lie on the same face.
def is_interior_diagonal (d : Dodecahedron) (v1 v2 : Fin d.num_vertices) : Prop :=
  v1 ≠ v2 ∧ ¬∃ f, f < d.num_faces ∧ v1 ∈ f ∧ v2 ∈ f

-- Now, we need to calculate the total number of such interior diagonals
noncomputable def num_interior_diagonals (d : Dodecahedron) : Nat :=
  d.num_vertices * (d.num_vertices - d.vertices_connected_by_edge - 1) / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals_eq_160 :
  num_interior_diagonals dodecahedron = 160 := by
  sorry

end dodecahedron_interior_diagonals_eq_160_l262_262652


namespace average_speed_is_9_mph_l262_262941

-- Define the conditions
def distance_north_ft := 5280
def north_speed_min_per_mile := 3
def rest_time_min := 10
def south_speed_miles_per_min := 3

-- Define a function to convert feet to miles
def feet_to_miles (ft : ℕ) : ℕ := ft / 5280

-- Define the time calculation for north and south trips
def time_north_min (speed : ℕ) (distance_ft : ℕ) : ℕ :=
  speed * feet_to_miles distance_ft

def time_south_min (speed_miles_per_min : ℕ) (distance_ft : ℕ) : ℕ :=
  (feet_to_miles distance_ft) / speed_miles_per_min

def total_time_min (time_north rest_time time_south : ℕ) : Rat :=
  time_north + rest_time + time_south

-- Convert total time into hours
def total_time_hr (total_time_min : Rat) : Rat :=
  total_time_min / 60

-- Define the total distance in miles
def total_distance_miles (distance_ft : ℕ) : ℕ :=
  2 * feet_to_miles distance_ft

-- Calculate the average speed
def average_speed (total_distance : ℕ) (total_time_hr : Rat) : Rat :=
  total_distance / total_time_hr

-- Prove the average speed is 9 miles per hour
theorem average_speed_is_9_mph : 
  average_speed (total_distance_miles distance_north_ft)
                (total_time_hr (total_time_min (time_north_min north_speed_min_per_mile distance_north_ft)
                                              rest_time_min
                                              (time_south_min south_speed_miles_per_min distance_north_ft)))
    = 9 := by
  sorry

end average_speed_is_9_mph_l262_262941


namespace original_four_digit_number_l262_262937

theorem original_four_digit_number : 
  ∃ x y z: ℕ, (x = 1 ∧ y = 9 ∧ z = 7 ∧ 1000 * x + 100 * y + 10 * z + y = 1979) ∧ 
  (1000 * y + 100 * z + 10 * y + x - (1000 * x + 100 * y + 10 * z + y) = 7812) ∧ 
  (1000 * y + 100 * z + 10 * y + x < 10000 ∧ 1000 * x + 100 * y + 10 * z + y < 10000) := 
sorry

end original_four_digit_number_l262_262937


namespace carrots_per_bundle_l262_262429

theorem carrots_per_bundle (potatoes_total: ℕ) (potatoes_in_bundle: ℕ) (price_per_potato_bundle: ℝ) 
(carrot_total: ℕ) (price_per_carrot_bundle: ℝ) (total_revenue: ℝ) (carrots_per_bundle : ℕ) :
potatoes_total = 250 → potatoes_in_bundle = 25 → price_per_potato_bundle = 1.90 → 
carrot_total = 320 → price_per_carrot_bundle = 2 → total_revenue = 51 →
((carrots_per_bundle = carrot_total / ((total_revenue - (potatoes_total / potatoes_in_bundle) 
    * price_per_potato_bundle) / price_per_carrot_bundle))  ↔ carrots_per_bundle = 20) := by
  sorry

end carrots_per_bundle_l262_262429


namespace fraction_of_phones_l262_262450

-- The total number of valid 8-digit phone numbers (b)
def valid_phone_numbers_total : ℕ := 5 * 10^7

-- The number of valid phone numbers that begin with 5 and end with 2 (a)
def valid_phone_numbers_special : ℕ := 10^6

-- The fraction of phone numbers that begin with 5 and end with 2
def fraction_phone_numbers_special : ℚ := valid_phone_numbers_special / valid_phone_numbers_total

-- Prove that the fraction of such phone numbers is 1/50
theorem fraction_of_phones : fraction_phone_numbers_special = 1 / 50 := by
  sorry

end fraction_of_phones_l262_262450


namespace four_digit_sum_of_digits_divisible_by_101_l262_262134

theorem four_digit_sum_of_digits_divisible_by_101 (a b c d : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 1 ≤ b ∧ b ≤ 9)
  (h3 : 1 ≤ c ∧ c ≤ 9)
  (h4 : 1 ≤ d ∧ d ≤ 9)
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_div : (1001 * a + 110 * b + 110 * c + 1001 * d) % 101 = 0) :
  (a + d) % 101 = (b + c) % 101 :=
by
  sorry

end four_digit_sum_of_digits_divisible_by_101_l262_262134


namespace correct_operation_l262_262807

theorem correct_operation :
  (2 * a - a ≠ 2) ∧ ((a - 1) * (a - 1) ≠ a ^ 2 - 1) ∧ (a ^ 6 / a ^ 3 ≠ a ^ 2) ∧ ((-2 * a ^ 3) ^ 2 = 4 * a ^ 6) :=
by
  sorry

end correct_operation_l262_262807


namespace arccos_neg1_l262_262595

theorem arccos_neg1 : Real.arccos (-1) = Real.pi := 
sorry

end arccos_neg1_l262_262595


namespace distinct_angle_values_in_cube_l262_262338

-- Define the vertices and unit cube properties
def is_vertex (v : ℝ × ℝ × ℝ) : Prop := 
  v ∈ {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 
       (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)}

-- Condition for distinct vertices
def are_distinct (A B C : ℝ × ℝ × ℝ) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Main theorem statement
theorem distinct_angle_values_in_cube : 
  ∀ A B C : ℝ × ℝ × ℝ, is_vertex A → is_vertex B → is_vertex C → 
  are_distinct A B C → 
  ∃ unique_angles : finset ℝ, unique_angles.card = 5 ∧ 
  ∀ angle ∈ unique_angles, angle = ∠ABC :=
sorry

end distinct_angle_values_in_cube_l262_262338


namespace compare_decimal_to_fraction_l262_262222

theorem compare_decimal_to_fraction : (0.650 - (1 / 8) = 0.525) :=
by
  /- We need to prove that 0.650 - 1/8 = 0.525 -/
  sorry

end compare_decimal_to_fraction_l262_262222


namespace compute_3X4_l262_262165

def operation_X (a b : ℤ) : ℤ := b + 12 * a - a^2

theorem compute_3X4 : operation_X 3 4 = 31 := 
by
  sorry

end compute_3X4_l262_262165


namespace books_arrangement_count_l262_262868

theorem books_arrangement_count : 
  let totalBooks := 7
  let identicalMathBooks := 2
  let identicalScienceBooks := 2
  let differentBooks := totalBooks - identicalMathBooks - identicalScienceBooks
  (totalBooks.factorial / (identicalMathBooks.factorial * identicalScienceBooks.factorial) = 1260) := 
by
  sorry

end books_arrangement_count_l262_262868


namespace problem_proof_l262_262563

theorem problem_proof :
  1.25 * 67.875 + 125 * 6.7875 + 1250 * 0.053375 = 1000 :=
by
  sorry

end problem_proof_l262_262563


namespace sum_arith_seq_l262_262507

theorem sum_arith_seq (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h₁ : ∀ n, S n = n * a 1 + (n * (n - 1)) * d / 2)
    (h₂ : S 10 = S 20)
    (h₃ : d > 0) :
    a 10 + a 22 > 0 := 
sorry

end sum_arith_seq_l262_262507


namespace largest_number_l262_262550

theorem largest_number 
  (a b c : ℝ) (h1 : a = 0.8) (h2 : b = 1/2) (h3 : c = 0.9) (h4 : a ≤ 2) (h5 : b ≤ 2) (h6 : c ≤ 2) :
  max (max a b) c = 0.9 :=
by
  sorry

end largest_number_l262_262550


namespace base_conversion_min_sum_l262_262388

theorem base_conversion_min_sum : ∃ a b : ℕ, a > 6 ∧ b > 6 ∧ (6 * a + 3 = 3 * b + 6) ∧ (a + b = 20) :=
by
  sorry

end base_conversion_min_sum_l262_262388


namespace max_value_of_S_n_divided_l262_262148

noncomputable def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def S_n (a₁ d n : ℕ) : ℕ :=
  n * (n + 4)

theorem max_value_of_S_n_divided (a₁ d : ℕ) (h₁ : ∀ n, a₁ + (2 * n - 1) * d = 2 * (a₁ + (n - 1) * d) - 3)
  (h₂ : (a₁ + 5 * d)^2 = a₁ * (a₁ + 20 * d)) :
  ∃ n, 2 * S_n a₁ d n / 2^n = 6 := 
sorry

end max_value_of_S_n_divided_l262_262148


namespace largest_3_digit_sum_l262_262150

theorem largest_3_digit_sum : ∃ A B : ℕ, A ≠ B ∧ A < 10 ∧ B < 10 ∧ 100 ≤ 111 * A + 12 * B ∧ 111 * A + 12 * B = 996 := by
  sorry

end largest_3_digit_sum_l262_262150


namespace number_of_interior_diagonals_of_dodecahedron_l262_262648

-- Definitions based on conditions
def dodecahedron_vertices := 20
def faces_per_vertex := 3
def vertices_per_face := 5
def shared_edges_per_vertex := faces_per_vertex
def total_faces := 12
def total_vertices := 20

-- Property of the dodecahedron
def potential_diagonals_per_vertex := dodecahedron_vertices - 1 - shared_edges_per_vertex - (vertices_per_face - 1)
def total_potential_diagonals := potential_diagonals_per_vertex * total_vertices

-- Proof statement:
theorem number_of_interior_diagonals_of_dodecahedron :
  total_potential_diagonals / 2 = 90 :=
by
  -- This is where the proof would go.
  sorry

end number_of_interior_diagonals_of_dodecahedron_l262_262648


namespace regions_formula_l262_262176

-- Define the number of regions R(n) created by n lines
def regions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

-- Theorem statement: for n lines, no two parallel, no three concurrent, the regions are defined by the formula
theorem regions_formula (n : ℕ) : regions n = 1 + (n * (n + 1)) / 2 := 
by sorry

end regions_formula_l262_262176


namespace greatest_two_digit_prod_12_l262_262772

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l262_262772


namespace find_A_minus_B_l262_262179

def A : ℤ := 3^7 + Nat.choose 7 2 * 3^5 + Nat.choose 7 4 * 3^3 + Nat.choose 7 6 * 3
def B : ℤ := Nat.choose 7 1 * 3^6 + Nat.choose 7 3 * 3^4 + Nat.choose 7 5 * 3^2 + 1

theorem find_A_minus_B : A - B = 128 := 
by
  -- Proof goes here
  sorry

end find_A_minus_B_l262_262179


namespace tan_of_alpha_intersects_unit_circle_l262_262047

theorem tan_of_alpha_intersects_unit_circle (α : ℝ) (hα : ∃ P : ℝ × ℝ, P = (12 / 13, -5 / 13) ∧ ∀ x y : ℝ, P = (x, y) → x^2 + y^2 = 1) : 
  Real.tan α = -5 / 12 :=
by
  -- proof to be completed
  sorry

end tan_of_alpha_intersects_unit_circle_l262_262047


namespace function_monotonically_increasing_iff_range_of_a_l262_262694

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem function_monotonically_increasing_iff_range_of_a (a : ℝ) :
  (∀ x, (deriv (f a) x) ≥ 0) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) :=
by
  sorry

end function_monotonically_increasing_iff_range_of_a_l262_262694


namespace poodle_barks_count_l262_262117

-- Define the conditions as hypothesis
variables (poodle_barks terrier_barks terrier_hushes : ℕ)

-- Define the conditions
def condition1 : Prop :=
  poodle_barks = 2 * terrier_barks

def condition2 : Prop :=
  terrier_hushes = terrier_barks / 2

def condition3 : Prop :=
  terrier_hushes = 6

-- The theorem we need to prove
theorem poodle_barks_count (poodle_barks terrier_barks terrier_hushes : ℕ)
  (h1 : condition1 poodle_barks terrier_barks)
  (h2 : condition2 terrier_barks terrier_hushes)
  (h3 : condition3 terrier_hushes) :
  poodle_barks = 24 :=
by
  -- Proof is not required as per instructions
  sorry

end poodle_barks_count_l262_262117


namespace no_solution_values_l262_262043

theorem no_solution_values (m : ℝ) :
  (∀ x : ℝ, x ≠ 5 → x ≠ -5 → (1 / (x - 5) + m / (x + 5) ≠ (m + 5) / (x^2 - 25))) ↔
  m = -1 ∨ m = 5 ∨ m = -5 / 11 :=
by
  sorry

end no_solution_values_l262_262043


namespace max_area_triangle_l262_262164

/-- Given two fixed points A and B on the plane with distance 2 between them, 
and a point P moving such that the ratio of distances |PA| / |PB| = sqrt(2), 
prove that the maximum area of triangle PAB is 2 * sqrt(2). -/
theorem max_area_triangle 
  (A B P : EuclideanSpace ℝ (Fin 2)) 
  (hAB : dist A B = 2)
  (h_ratio : dist P A = Real.sqrt 2 * dist P B)
  (h_non_collinear : ¬ ∃ k : ℝ, ∃ l : ℝ, k ≠ l ∧ A = k • B ∧ P = l • B) 
  : ∃ S_max : ℝ, S_max = 2 * Real.sqrt 2 := 
sorry

end max_area_triangle_l262_262164


namespace sum_tens_units_digit_9_pow_1001_l262_262805

-- Define a function to extract the last two digits of a number
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Define a function to extract the tens digit
def tens_digit (n : ℕ) : ℕ := (last_two_digits n) / 10

-- Define a function to extract the units digit
def units_digit (n : ℕ) : ℕ := (last_two_digits n) % 10

-- The main theorem
theorem sum_tens_units_digit_9_pow_1001 :
  tens_digit (9 ^ 1001) + units_digit (9 ^ 1001) = 9 :=
by
  sorry

end sum_tens_units_digit_9_pow_1001_l262_262805


namespace max_num_triangles_for_right_triangle_l262_262727

-- Define a right triangle on graph paper
def right_triangle (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ n ∧ 0 ≤ b ∧ b ≤ n

-- Define maximum number of triangles that can be formed within the triangle
def max_triangles (n : ℕ) : ℕ :=
  if h : n = 7 then 28 else 0  -- Given n = 7, the max number is 28

-- Define the theorem to be proven
theorem max_num_triangles_for_right_triangle :
  right_triangle 7 → max_triangles 7 = 28 :=
by
  intro h
  -- Proof goes here
  sorry

end max_num_triangles_for_right_triangle_l262_262727


namespace sin_half_angle_correct_l262_262340

noncomputable def sin_half_angle (theta : ℝ) (h1 : Real.sin theta = 3 / 5) (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) : ℝ :=
  -3 * Real.sqrt 10 / 10

theorem sin_half_angle_correct (theta : ℝ) (h1 : Real.sin theta = 3 / 5) (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) :
  sin_half_angle theta h1 h2 = Real.sin (theta / 2) :=
by
  sorry

end sin_half_angle_correct_l262_262340


namespace maximum_value_fraction_sum_l262_262012

theorem maximum_value_fraction_sum (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : 0 < c) (hd : 0 < d) (h1 : a + c = 20) (h2 : (a : ℝ) / b + (c : ℝ) / d < 1) :
  (a : ℝ) / b + (c : ℝ) / d ≤ 1385 / 1386 :=
sorry

end maximum_value_fraction_sum_l262_262012


namespace closest_perfect_square_to_350_l262_262257

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l262_262257


namespace algebraic_expression_value_l262_262690

theorem algebraic_expression_value (a b : ℤ) (h : 2 * (-3) - a + 2 * b = 0) : 2 * a - 4 * b + 1 = -11 := 
by {
  sorry
}

end algebraic_expression_value_l262_262690


namespace find_prime_squares_l262_262977

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

theorem find_prime_squares :
  ∀ (p q : ℕ), is_prime p → is_prime q → is_square (p^(q+1) + q^(p+1)) → (p = 2 ∧ q = 2) :=
by 
  intros p q h_prime_p h_prime_q h_square
  sorry

end find_prime_squares_l262_262977


namespace greatest_area_difference_l262_262231

def first_rectangle_perimeter (l w : ℕ) : Prop :=
  2 * l + 2 * w = 156

def second_rectangle_perimeter (l w : ℕ) : Prop :=
  2 * l + 2 * w = 144

theorem greatest_area_difference : 
  ∃ (l1 w1 l2 w2 : ℕ), 
  first_rectangle_perimeter l1 w1 ∧ 
  second_rectangle_perimeter l2 w2 ∧ 
  (l1 * (78 - l1) - l2 * (72 - l2) = 225) := 
sorry

end greatest_area_difference_l262_262231


namespace students_like_burgers_l262_262878

theorem students_like_burgers (total_students : ℕ) (french_fries_likers : ℕ) (both_likers : ℕ) (neither_likers : ℕ) 
    (h1 : total_students = 25) (h2 : french_fries_likers = 15) (h3 : both_likers = 6) (h4 : neither_likers = 6) : 
    (total_students - neither_likers) - (french_fries_likers - both_likers) = 10 :=
by
  -- The proof will go here.
  sorry

end students_like_burgers_l262_262878


namespace arccos_neg1_l262_262596

theorem arccos_neg1 : Real.arccos (-1) = Real.pi := 
sorry

end arccos_neg1_l262_262596


namespace minimum_value_of_a_l262_262351

theorem minimum_value_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5 / 2 :=
sorry

end minimum_value_of_a_l262_262351


namespace base7_digits_of_143_l262_262706

theorem base7_digits_of_143 : ∃ d1 d2 d3 : ℕ, (d1 < 7 ∧ d2 < 7 ∧ d3 < 7) ∧ (143 = d1 * 49 + d2 * 7 + d3) ∧ (d1 = 2 ∧ d2 = 6 ∧ d3 = 3) :=
by
  sorry

end base7_digits_of_143_l262_262706


namespace no_four_consecutive_product_square_l262_262962

/-- Prove that there do not exist four consecutive positive integers whose product is a perfect square. -/
theorem no_four_consecutive_product_square :
  ¬ ∃ (x : ℕ), ∃ (n : ℕ), n * n = x * (x + 1) * (x + 2) * (x + 3) :=
sorry

end no_four_consecutive_product_square_l262_262962


namespace equal_real_roots_value_l262_262857

theorem equal_real_roots_value (a c : ℝ) (ha : a ≠ 0) (h : 4 - 4 * a * (2 - c) = 0) : (1 / a) + c = 2 := 
by
  sorry

end equal_real_roots_value_l262_262857


namespace range_of_f_l262_262859

noncomputable def f (x : ℝ) : ℝ := 2^x
def valid_range (S : Set ℝ) : Prop := ∃ x ∈ Set.Icc (0 : ℝ) (3 : ℝ), f x ∈ S

theorem range_of_f : valid_range (Set.Icc (1 : ℝ) (8 : ℝ)) :=
sorry

end range_of_f_l262_262859


namespace annual_rent_per_square_foot_l262_262283

theorem annual_rent_per_square_foot (length width : ℕ) (monthly_rent : ℕ)
  (h_length : length = 20) (h_width : width = 15) (h_monthly_rent : monthly_rent = 3600) :
  let area := length * width
  let annual_rent := monthly_rent * 12
  let annual_rent_per_sq_ft := annual_rent / area
  annual_rent_per_sq_ft = 144 := by
  sorry

end annual_rent_per_square_foot_l262_262283


namespace dodecahedron_interior_diagonals_l262_262681

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l262_262681


namespace greatest_two_digit_product_is_12_l262_262777

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l262_262777


namespace pete_books_ratio_l262_262527

theorem pete_books_ratio 
  (M_last : ℝ) (P_last P_this_year M_this_year : ℝ)
  (h1 : P_last = 2 * M_last)
  (h2 : M_this_year = 1.5 * M_last)
  (h3 : P_last + P_this_year = 300)
  (h4 : M_this_year = 75) :
  P_this_year / P_last = 2 :=
by
  sorry

end pete_books_ratio_l262_262527


namespace remainder_of_14_pow_53_mod_7_l262_262246

theorem remainder_of_14_pow_53_mod_7 : (14 ^ 53) % 7 = 0 := by
  sorry

end remainder_of_14_pow_53_mod_7_l262_262246


namespace perfect_square_factors_count_l262_262037

def perfectSquares := [4, 9, 16, 25, 36, 49, 64, 81]

def countNumbersWithPerfectSquareFactors : Nat :=
  List.length (List.filter (fun n => perfectSquares.any (fun p => n % p = 0)) [1..100])

theorem perfect_square_factors_count :
  countNumbersWithPerfectSquareFactors = 41 := sorry

end perfect_square_factors_count_l262_262037


namespace value_of_b_l262_262024

theorem value_of_b (a b c y1 y2 y3 : ℝ)
( h1 : y1 = a + b + c )
( h2 : y2 = a - b + c )
( h3 : y3 = 4 * a + 2 * b + c )
( h4 : y1 - y2 = 8 )
( h5 : y3 = y1 + 2 )
: b = 4 :=
sorry

end value_of_b_l262_262024


namespace even_factors_count_of_n_l262_262867

def n : ℕ := 2^3 * 3^2 * 7 * 5

theorem even_factors_count_of_n : ∃ k : ℕ, k = 36 ∧ ∀ (a b c d : ℕ), 
  1 ≤ a ∧ a ≤ 3 →
  b ≤ 2 →
  c ≤ 1 →
  d ≤ 1 →
  2^a * 3^b * 7^c * 5^d ∣ n :=
sorry

end even_factors_count_of_n_l262_262867


namespace nat_power_of_p_iff_only_prime_factor_l262_262508

theorem nat_power_of_p_iff_only_prime_factor (p n : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℕ, n = p^k) ↔ (∀ q : ℕ, Nat.Prime q → q ∣ n → q = p) := 
sorry

end nat_power_of_p_iff_only_prime_factor_l262_262508


namespace repeating_decimal_to_fraction_l262_262843

theorem repeating_decimal_to_fraction : ∀ (x : ℝ), x = 0.7 + 0.08 / (1-0.1) → x = 71 / 90 :=
by
  intros x hx
  sorry

end repeating_decimal_to_fraction_l262_262843


namespace inscribed_circle_radius_eq_four_l262_262703

theorem inscribed_circle_radius_eq_four
  (A p s r : ℝ)
  (hA : A = 2 * p)
  (hp : p = 2 * s)
  (hArea : A = r * s) :
  r = 4 :=
by
  -- Proof would go here.
  sorry

end inscribed_circle_radius_eq_four_l262_262703


namespace dodecahedron_interior_diagonals_l262_262659

theorem dodecahedron_interior_diagonals (vertices faces: ℕ) (pentagonal: ∀ face, face = 5) (vertices_per_face : ∀ face ∈ (range faces), 3) :
  vertices = 20 ∧ faces = 12 ∧ interior_diagonals = 160 :=
by
    -- Definitions and problem conditions
    let vertices := 20
    let faces := 12
    let pentagonal := (λ face, face = 5)
    let vertices_per_face := (λ face, 3)

    -- Assertion
    have interior_diagonals : ℕ := (vertices * 16) / 2
    rw Nat.div_eq_of_eq_mul_right _ _ _,
    have h: vertices = 20 ∧ faces = 12 ∧ interior_diagonals = 160 := by
        sorry

    exact h

end dodecahedron_interior_diagonals_l262_262659


namespace evaluate_expression_l262_262235

theorem evaluate_expression :
  (1 / (-8^2)^4) * (-8)^9 = -8 := by
  sorry

end evaluate_expression_l262_262235


namespace sum_of_roots_eq_zero_product_of_roots_eq_neg_twentyfive_l262_262078

theorem sum_of_roots_eq_zero (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  ∃ x1 x2 : ℝ, (|x1| = 5) ∧ (|x2| = 5) ∧ x1 + x2 = 0 :=
by
  sorry

theorem product_of_roots_eq_neg_twentyfive (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  ∃ x1 x2 : ℝ, (|x1| = 5) ∧ (|x2| = 5) ∧ x1 * x2 = -25 :=
by
  sorry

end sum_of_roots_eq_zero_product_of_roots_eq_neg_twentyfive_l262_262078


namespace last_three_digits_of_power_l262_262926

theorem last_three_digits_of_power (h : 3^400 ≡ 1 [MOD 800]) : 3^8000 ≡ 1 [MOD 800] :=
by {
  sorry
}

end last_three_digits_of_power_l262_262926


namespace ratio_of_areas_l262_262704

noncomputable section

open Real

variables {A B C D E F : Point}
variables (AB AC AD CF : ℝ)
variables (AB_pos : AB = 115)
   (AC_pos : AC = 115)
   (AD_pos : AD = 38)
   (CF_pos : CF = 77)

theorem ratio_of_areas : 
  ∃ (r : ℝ), r = (19 / 96) ∧ 
  let BD := AB - AD in
  let CE := 192 in
  let BE := 38 in
  let EF := 115 in
  let DE := 115 in
  r = (EF / DE) * (CE / BE) * (sin(CEF_angle) / sin(BED_angle)) :=
by
  let BD := AB - AD
  let CE := 192
  let BE := 38
  let EF := 115
  let DE := 115
  sorry

end ratio_of_areas_l262_262704


namespace div_inside_parentheses_l262_262233

theorem div_inside_parentheses :
  100 / (6 / 2) = 100 / 3 :=
by
  sorry

end div_inside_parentheses_l262_262233


namespace probability_distribution_correct_l262_262291

noncomputable def numCombinations (n k : ℕ) : ℕ :=
  (Nat.choose n k)

theorem probability_distribution_correct :
  let totalCombinations := numCombinations 5 2
  let prob_two_red := (numCombinations 3 2 : ℚ) / totalCombinations
  let prob_two_white := (numCombinations 2 2 : ℚ) / totalCombinations
  let prob_one_red_one_white := ((numCombinations 3 1) * (numCombinations 2 1) : ℚ) / totalCombinations
  (prob_two_red, prob_one_red_one_white, prob_two_white) = (0.3, 0.6, 0.1) :=
by
  sorry

end probability_distribution_correct_l262_262291


namespace S9_is_45_l262_262632

-- Define the required sequence and conditions
variable {a : ℕ → ℝ} -- a function that gives us the arithmetic sequence
variable {S : ℕ → ℝ} -- a function that gives us the sum of the first n terms of the sequence

-- Define the condition that a_2 + a_8 = 10
axiom a2_a8_condition : a 2 + a 8 = 10

-- Define the arithmetic property of the sequence
axiom arithmetic_property (n m : ℕ) : a (n + m) = a n + a m

-- Define the sum formula for the first n terms of an arithmetic sequence
axiom sum_formula (n : ℕ) : S n = (n / 2) * (a 1 + a n)

-- The main theorem to prove
theorem S9_is_45 : S 9 = 45 :=
by
  -- Here would go the proof, but it is omitted
  sorry

end S9_is_45_l262_262632


namespace greatest_two_digit_product_12_l262_262781

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l262_262781


namespace f_of_1_l262_262219

theorem f_of_1 (f : ℕ+ → ℕ+) (h_mono : ∀ {a b : ℕ+}, a < b → f a < f b)
  (h_fn_prop : ∀ n : ℕ+, f (f n) = 3 * n) : f 1 = 2 :=
sorry

end f_of_1_l262_262219


namespace s_neq_t_if_Q_on_DE_l262_262510

-- Conditions and Definitions
noncomputable def DQ (x : ℝ) := x
noncomputable def QE (x : ℝ) := 10 - x
noncomputable def FQ := 5 * Real.sqrt 3
noncomputable def s (x : ℝ) := (DQ x) ^ 2 + (QE x) ^ 2
noncomputable def t := 2 * FQ ^ 2

-- Lean 4 Statement
theorem s_neq_t_if_Q_on_DE (x : ℝ) : s x ≠ t :=
by
  sorry -- Provided proof step to be filled in

end s_neq_t_if_Q_on_DE_l262_262510


namespace dodecahedron_interior_diagonals_l262_262660

theorem dodecahedron_interior_diagonals (vertices faces: ℕ) (pentagonal: ∀ face, face = 5) (vertices_per_face : ∀ face ∈ (range faces), 3) :
  vertices = 20 ∧ faces = 12 ∧ interior_diagonals = 160 :=
by
    -- Definitions and problem conditions
    let vertices := 20
    let faces := 12
    let pentagonal := (λ face, face = 5)
    let vertices_per_face := (λ face, 3)

    -- Assertion
    have interior_diagonals : ℕ := (vertices * 16) / 2
    rw Nat.div_eq_of_eq_mul_right _ _ _,
    have h: vertices = 20 ∧ faces = 12 ∧ interior_diagonals = 160 := by
        sorry

    exact h

end dodecahedron_interior_diagonals_l262_262660


namespace squirrels_cannot_divide_equally_l262_262141

theorem squirrels_cannot_divide_equally
    (n : ℕ) : ¬ (∃ k, 2022 + n * (n + 1) = 5 * k) :=
by
sorry

end squirrels_cannot_divide_equally_l262_262141


namespace sqrt_mixed_number_simplified_l262_262970

theorem sqrt_mixed_number_simplified :
  (sqrt (8 + 9 / 16) = sqrt 137 / 4) :=
begin
  sorry
end

end sqrt_mixed_number_simplified_l262_262970


namespace dodecahedron_interior_diagonals_eq_160_l262_262653

-- Definition of a dodecahedron for the context of the proof
structure Dodecahedron where
  num_faces : Nat
  num_vertices : Nat
  faces_meeting_at_vertex : Nat
  vertices_connected_by_edge : Nat

-- Given our specific dodecahedron properties
def dodecahedron : Dodecahedron := {
  num_faces := 12,
  num_vertices := 20,
  faces_meeting_at_vertex := 3,
  vertices_connected_by_edge := 3
}

-- The interior diagonal is defined as a segment connecting two vertices
-- which do not lie on the same face.
def is_interior_diagonal (d : Dodecahedron) (v1 v2 : Fin d.num_vertices) : Prop :=
  v1 ≠ v2 ∧ ¬∃ f, f < d.num_faces ∧ v1 ∈ f ∧ v2 ∈ f

-- Now, we need to calculate the total number of such interior diagonals
noncomputable def num_interior_diagonals (d : Dodecahedron) : Nat :=
  d.num_vertices * (d.num_vertices - d.vertices_connected_by_edge - 1) / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals_eq_160 :
  num_interior_diagonals dodecahedron = 160 := by
  sorry

end dodecahedron_interior_diagonals_eq_160_l262_262653


namespace transport_cost_in_euros_l262_262385

def cost_per_kg : ℝ := 18000
def weight_g : ℝ := 300
def exchange_rate : ℝ := 0.95

theorem transport_cost_in_euros :
  (cost_per_kg * (weight_g / 1000) * exchange_rate) = 5130 :=
by sorry

end transport_cost_in_euros_l262_262385


namespace number_of_trailing_zeros_l262_262624

def trailing_zeros (n : Nat) : Nat :=
  let powers_of_two := 2 * 52^5
  let powers_of_five := 2 * 25^2
  min powers_of_two powers_of_five

theorem number_of_trailing_zeros : trailing_zeros (525^(25^2) * 252^(52^5)) = 1250 := 
by sorry

end number_of_trailing_zeros_l262_262624


namespace find_a5_l262_262174

variable {a_n : ℕ → ℤ}
variable (d : ℤ)

def arithmetic_sequence (a_n : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  a_n 1 = a1 ∧ ∀ n, a_n (n + 1) = a_n n + d

theorem find_a5 (h_seq : arithmetic_sequence a_n 6 d) (h_a3 : a_n 3 = 2) : a_n 5 = -2 :=
by
  obtain ⟨h_a1, h_arith⟩ := h_seq
  sorry

end find_a5_l262_262174


namespace power_function_zeros_l262_262022

theorem power_function_zeros :
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = x ^ 3) ∧ (f 2 = 8) ∧ (∀ y : ℝ, (f y - y = 0) ↔ (y = 0 ∨ y = 1 ∨ y = -1)) := by
  sorry

end power_function_zeros_l262_262022


namespace problem1_problem2_l262_262572

open Real -- Open the Real namespace to use real number trigonometric functions

-- Problem 1
theorem problem1 (α : ℝ) (hα : tan α = 3) : 
  (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5/7 :=
sorry

-- Problem 2
theorem problem2 (θ : ℝ) (hθ : tan θ = -3/4) : 
  2 + sin θ * cos θ - cos θ ^ 2 = 22 / 25 :=
sorry

end problem1_problem2_l262_262572


namespace measure_angle_BRC_l262_262571

inductive Point : Type
| A 
| B 
| C 
| P 
| Q 
| R 

open Point

def is_inside_triangle (P : Point) (A B C : Point) : Prop := sorry

def intersection (a b c : Point) : Point := sorry

def length (a b : Point) : ℝ := sorry

def angle (a b c : Point) : ℝ := sorry

theorem measure_angle_BRC 
  (P : Point) (A B C : Point)
  (h_inside : is_inside_triangle P A B C)
  (hQ : Q = intersection A C P)
  (hR : R = intersection A B P)
  (h_lengths_equal : length A R = length R B ∧ length R B = length C P)
  (h_CQ_PQ : length C Q = length P Q) :
  angle B R C = 120 := 
sorry

end measure_angle_BRC_l262_262571


namespace fraction_of_fraction_of_fraction_l262_262243

theorem fraction_of_fraction_of_fraction (a b c d : ℝ) (h₁ : a = 1/5) (h₂ : b = 1/3) (h₃ : c = 1/6) (h₄ : d = 90) :
  (a * b * c * d) = 1 :=
by
  rw [h₁, h₂, h₃, h₄]
  simp
  sorry -- To indicate that the proof is missing

end fraction_of_fraction_of_fraction_l262_262243


namespace closest_perfect_square_to_350_l262_262251

theorem closest_perfect_square_to_350 : ∃ (n : ℕ), n^2 = 361 ∧ ∀ (m : ℕ), m^2 ≤ 350 ∨ 350 < m^2 → abs (361 - 350) ≤ abs (m^2 - 350) :=
by
  sorry

end closest_perfect_square_to_350_l262_262251


namespace eliana_total_steps_l262_262613

noncomputable def day1_steps : ℕ := 200 + 300
noncomputable def day2_steps : ℕ := 2 * day1_steps
noncomputable def day3_steps : ℕ := day1_steps + day2_steps + 100

theorem eliana_total_steps : day3_steps = 1600 := by
  sorry

end eliana_total_steps_l262_262613


namespace greatest_two_digit_product_12_l262_262788

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l262_262788


namespace max_S_n_l262_262474

/-- Arithmetic sequence proof problem -/
theorem max_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 + a 3 + a 5 = 15)
  (h2 : a 2 + a 4 + a 6 = 0)
  (d : ℝ) (h3 : ∀ n, a (n + 1) = a n + d) :
  (∃ n, S n = 30) :=
sorry

end max_S_n_l262_262474


namespace actual_average_height_l262_262420

theorem actual_average_height
  (incorrect_avg_height : ℝ)
  (n : ℕ)
  (incorrect_height : ℝ)
  (actual_height : ℝ)
  (h1 : incorrect_avg_height = 184)
  (h2 : n = 35)
  (h3 : incorrect_height = 166)
  (h4 : actual_height = 106) :
  let incorrect_total_height := incorrect_avg_height * n
  let difference := incorrect_height - actual_height
  let correct_total_height := incorrect_total_height - difference
  let correct_avg_height := correct_total_height / n
  correct_avg_height = 182.29 :=
by {
  sorry
}

end actual_average_height_l262_262420


namespace ordered_quadruple_ellipse_l262_262833

noncomputable def ellipse_quadruple := 
  let f₁ : (ℝ × ℝ) := (1, 1)
  let f₂ : (ℝ × ℝ) := (1, 7)
  let p : (ℝ × ℝ) := (12, -1)
  let a := (5 / 2) * (Real.sqrt 5 + Real.sqrt 37)
  let b := (1 / 2) * Real.sqrt (1014 + 50 * Real.sqrt 185)
  let h := 1
  let k := 4
  (a, b, h, k)

theorem ordered_quadruple_ellipse :
  let e : (ℝ × ℝ × ℝ × ℝ) := θse_quadruple
  e = ((5 / 2 * (Real.sqrt 5 + Real.sqrt 37)), (1 / 2 * Real.sqrt (1014 + 50 * Real.sqrt 185)), 1, 4) :=
by
  sorry

end ordered_quadruple_ellipse_l262_262833


namespace primes_square_condition_l262_262979

open Nat

theorem primes_square_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  ∃ n : ℕ, p^(q+1) + q^(p+1) = n^2 ↔ p = 2 ∧ q = 2 := by
  sorry

end primes_square_condition_l262_262979


namespace greatest_two_digit_prod_12_l262_262771

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l262_262771


namespace olivia_total_payment_l262_262203

theorem olivia_total_payment : 
  (4 / 4 + 12 / 4 = 4) :=
by
  sorry

end olivia_total_payment_l262_262203


namespace circles_intersect_iff_l262_262920

-- Definitions of the two circles and their parameters
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9

def circle2 (x y r : ℝ) : Prop := x^2 + y^2 + 8 * x - 6 * y + 25 - r^2 = 0

-- Lean statement to prove the range of r
theorem circles_intersect_iff (r : ℝ) (hr : 0 < r) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y r) ↔ (2 < r ∧ r < 8) :=
by
  sorry

end circles_intersect_iff_l262_262920


namespace sum_of_x_values_satisfying_eq_l262_262416

noncomputable def rational_eq_sum (x : ℝ) : Prop :=
3 = (x^3 - 3 * x^2 - 12 * x) / (x + 3)

theorem sum_of_x_values_satisfying_eq :
  (∃ (x : ℝ), rational_eq_sum x) ∧ (x ≠ -3 → (x_1 + x_2) = 6) :=
sorry

end sum_of_x_values_satisfying_eq_l262_262416


namespace libby_quarters_left_after_payment_l262_262518

noncomputable def quarters_needed (usd_target : ℝ) (usd_per_quarter : ℝ) : ℝ := 
  usd_target / usd_per_quarter

noncomputable def quarters_left (initial_quarters : ℝ) (used_quarters : ℝ) : ℝ := 
  initial_quarters - used_quarters

theorem libby_quarters_left_after_payment
  (initial_quarters : ℝ) (usd_target : ℝ) (usd_per_quarter : ℝ) 
  (h_initial : initial_quarters = 160) 
  (h_usd_target : usd_target = 35) 
  (h_usd_per_quarter : usd_per_quarter = 0.25) : 
  quarters_left initial_quarters (quarters_needed usd_target usd_per_quarter) = 20 := 
by
  sorry

end libby_quarters_left_after_payment_l262_262518


namespace no_real_roots_l262_262131

theorem no_real_roots 
    (h : ∀ x : ℝ, (3 * x^2 / (x - 2)) - (3 * x + 8) / 2 + (5 - 9 * x) / (x - 2) + 2 = 0) 
    : False := by
  sorry

end no_real_roots_l262_262131


namespace line_perpendicular_through_P_l262_262907

/-
  Given:
  1. The point P(-2, 2).
  2. The line 2x - y + 1 = 0.
  Prove:
  The equation of the line that passes through P and is perpendicular to the given line is x + 2y - 2 = 0.
-/

def P : ℝ × ℝ := (-2, 2)
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0

theorem line_perpendicular_through_P :
  ∃ (x y : ℝ) (m : ℝ), (x = -2) ∧ (y = 2) ∧ (m = -1/2) ∧ 
  (∀ (x₁ y₁ : ℝ), (y₁ - y) = m * (x₁ - x)) ∧ 
  (∀ (lx ly : ℝ), line1 lx ly → x + 2 * y - 2 = 0) := sorry

end line_perpendicular_through_P_l262_262907


namespace gcd_12345_6789_l262_262753

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l262_262753


namespace crayons_total_l262_262875

theorem crayons_total (blue_crayons : ℕ) (red_crayons : ℕ) 
  (H1 : red_crayons = 4 * blue_crayons) (H2 : blue_crayons = 3) : 
  blue_crayons + red_crayons = 15 := 
by
  sorry

end crayons_total_l262_262875


namespace field_area_is_13_point854_hectares_l262_262935

noncomputable def area_of_field_in_hectares (cost_fencing: ℝ) (rate_per_meter: ℝ): ℝ :=
  let length_of_fence := cost_fencing / rate_per_meter
  let radius := length_of_fence / (2 * Real.pi)
  let area_in_square_meters := Real.pi * (radius * radius)
  area_in_square_meters / 10000

theorem field_area_is_13_point854_hectares :
  area_of_field_in_hectares 6202.75 4.70 = 13.854 :=
by
  sorry

end field_area_is_13_point854_hectares_l262_262935


namespace scale_model_height_l262_262825

theorem scale_model_height (real_height : ℕ) (scale_ratio : ℕ) (h_real : real_height = 1454) (h_scale : scale_ratio = 50) : 
⌊(real_height : ℝ) / scale_ratio + 0.5⌋ = 29 :=
by
  rw [h_real, h_scale]
  norm_num
  sorry

end scale_model_height_l262_262825


namespace greatest_two_digit_with_product_12_l262_262763

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l262_262763


namespace comparison_1_comparison_2_l262_262959

noncomputable def expr1 := -(-((6: ℝ) / 7))
noncomputable def expr2 := -((abs (-((4: ℝ) / 5))))
noncomputable def expr3 := -((4: ℝ) / 5)
noncomputable def expr4 := -((2: ℝ) / 3)

theorem comparison_1 : expr1 > expr2 := sorry
theorem comparison_2 : expr3 < expr4 := sorry

end comparison_1_comparison_2_l262_262959


namespace number_of_interior_diagonals_of_dodecahedron_l262_262649

-- Definitions based on conditions
def dodecahedron_vertices := 20
def faces_per_vertex := 3
def vertices_per_face := 5
def shared_edges_per_vertex := faces_per_vertex
def total_faces := 12
def total_vertices := 20

-- Property of the dodecahedron
def potential_diagonals_per_vertex := dodecahedron_vertices - 1 - shared_edges_per_vertex - (vertices_per_face - 1)
def total_potential_diagonals := potential_diagonals_per_vertex * total_vertices

-- Proof statement:
theorem number_of_interior_diagonals_of_dodecahedron :
  total_potential_diagonals / 2 = 90 :=
by
  -- This is where the proof would go.
  sorry

end number_of_interior_diagonals_of_dodecahedron_l262_262649


namespace greatest_two_digit_prod_12_l262_262767

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l262_262767


namespace area_of_triangle_ABC_l262_262637

theorem area_of_triangle_ABC :
  let A'B' := 4
  let B'C' := 3
  let angle_A'B'C' := 60
  let area_A'B'C' := (1 / 2) * A'B' * B'C' * Real.sin (angle_A'B'C' * Real.pi / 180)
  let ratio := 2 * Real.sqrt 2
  let area_ABC := ratio * area_A'B'C'
  area_ABC = 6 * Real.sqrt 6 := 
by
  sorry

end area_of_triangle_ABC_l262_262637


namespace complementary_event_A_l262_262170

def EventA (n : ℕ) := n ≥ 2

def ComplementaryEventA (n : ℕ) := n ≤ 1

theorem complementary_event_A (n : ℕ) : ComplementaryEventA n ↔ ¬ EventA n := by
  sorry

end complementary_event_A_l262_262170


namespace inequalities_l262_262533

variable {a b c : ℝ}

theorem inequalities (ha : a < 0) (hab : a < b) (hbc : b < c) :
  a^2 * b < b^2 * c ∧ a^2 * c < b^2 * c ∧ a^2 * b < a^2 * c :=
by
  sorry

end inequalities_l262_262533


namespace gold_copper_ratio_l262_262106

theorem gold_copper_ratio (G C : ℕ) 
  (h1 : 19 * G + 9 * C = 18 * (G + C)) : 
  G = 9 * C :=
by
  sorry

end gold_copper_ratio_l262_262106


namespace john_remaining_income_l262_262191

/-- 
  Mr. John's monthly income is $2000, and he spends 5% of his income on public transport.
  Prove that after deducting his monthly transport fare, his remaining income is $1900.
-/
theorem john_remaining_income : 
  let income := 2000 
  let transport_percent := 5 
  let transport_fare := income * transport_percent / 100 
  income - transport_fare = 1900 := 
by 
  let income := 2000 
  let transport_percent := 5 
  let transport_fare := income * transport_percent / 100 
  have transport_fare_eq : transport_fare = 100 := by sorry
  have remaining_income_eq : income - transport_fare = 1900 := by sorry
  exact remaining_income_eq

end john_remaining_income_l262_262191


namespace length_of_base_of_isosceles_triangle_l262_262214

noncomputable def length_congruent_sides : ℝ := 8
noncomputable def perimeter_triangle : ℝ := 26

theorem length_of_base_of_isosceles_triangle : 
  ∀ (b : ℝ), 
  2 * length_congruent_sides + b = perimeter_triangle → 
  b = 10 :=
by
  intros b h
  -- The proof is omitted.
  sorry

end length_of_base_of_isosceles_triangle_l262_262214


namespace simplify_and_ratio_l262_262531

theorem simplify_and_ratio (k : ℤ) : 
  let a := 1
  let b := 2
  (∀ (k : ℤ), (6 * k + 12) / 6 = a * k + b) →
  (a / b = 1 / 2) :=
by
  intros
  sorry
  
end simplify_and_ratio_l262_262531


namespace required_CO2_l262_262686

noncomputable def moles_of_CO2_required (Mg CO2 MgO C : ℕ) (hMgO : MgO = 2) (hC : C = 1) : ℕ :=
  if Mg = 2 then 1 else 0

theorem required_CO2
  (Mg CO2 MgO C : ℕ)
  (hMgO : MgO = 2)
  (hC : C = 1)
  (hMg : Mg = 2)
  : moles_of_CO2_required Mg CO2 MgO C hMgO hC = 1 :=
  by simp [moles_of_CO2_required, hMg]

end required_CO2_l262_262686


namespace line_properties_l262_262984

theorem line_properties : ∃ m x_intercept, 
  (∀ (x y : ℝ), 4 * x + 7 * y = 28 → y = m * x + 4) ∧ 
  (∀ (x y : ℝ), y = 0 → 4 * x + 7 * y = 28 → x = x_intercept) ∧ 
  m = -4 / 7 ∧ 
  x_intercept = 7 :=
by 
  sorry

end line_properties_l262_262984


namespace tricycles_in_garage_l262_262228

theorem tricycles_in_garage 
    (T : ℕ) 
    (total_bicycles : ℕ := 3) 
    (total_unicycles : ℕ := 7) 
    (bicycle_wheels : ℕ := 2) 
    (tricycle_wheels : ℕ := 3) 
    (unicycle_wheels : ℕ := 1) 
    (total_wheels : ℕ := 25) 
    (eq_wheels : total_bicycles * bicycle_wheels + total_unicycles * unicycle_wheels + T * tricycle_wheels = total_wheels) :
    T = 4 :=
by {
  sorry
}

end tricycles_in_garage_l262_262228


namespace bookseller_loss_l262_262931

theorem bookseller_loss (C S : ℝ) (h : 20 * C = 25 * S) : (C - S) / C * 100 = 20 := by
  sorry

end bookseller_loss_l262_262931


namespace gcd_101_power_l262_262599

theorem gcd_101_power (a b : ℕ) (h1 : a = 101^6 + 1) (h2 : b = 3 * 101^6 + 101^3 + 1) (h_prime : Nat.Prime 101) : Nat.gcd a b = 1 :=
by
  -- proof goes here
  sorry

end gcd_101_power_l262_262599


namespace maximum_ab_value_l262_262160

noncomputable def max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 6) : ℝ :=
  by
    apply sqrt (max (9 : ℝ) 0)

theorem maximum_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 6) :
  ab = 9 :=
  sorry

end maximum_ab_value_l262_262160


namespace mass_percentage_O_in_CaO_l262_262001

theorem mass_percentage_O_in_CaO :
  (16.00 / (40.08 + 16.00)) * 100 = 28.53 :=
by
  sorry

end mass_percentage_O_in_CaO_l262_262001


namespace no_perfect_square_abc_sum_l262_262900

theorem no_perfect_square_abc_sum (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
  ¬ ∃ m : ℕ, m * m = (100 * a + 10 * b + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) :=
by
  sorry

end no_perfect_square_abc_sum_l262_262900


namespace number_of_interior_diagonals_of_dodecahedron_l262_262647

-- Definitions based on conditions
def dodecahedron_vertices := 20
def faces_per_vertex := 3
def vertices_per_face := 5
def shared_edges_per_vertex := faces_per_vertex
def total_faces := 12
def total_vertices := 20

-- Property of the dodecahedron
def potential_diagonals_per_vertex := dodecahedron_vertices - 1 - shared_edges_per_vertex - (vertices_per_face - 1)
def total_potential_diagonals := potential_diagonals_per_vertex * total_vertices

-- Proof statement:
theorem number_of_interior_diagonals_of_dodecahedron :
  total_potential_diagonals / 2 = 90 :=
by
  -- This is where the proof would go.
  sorry

end number_of_interior_diagonals_of_dodecahedron_l262_262647


namespace geometric_sequence_sum_ratio_l262_262990

theorem geometric_sequence_sum_ratio 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n+1) = a 0 * q ^ n)
  (h2 : ∀ n, S n = (a 0 * (q ^ n - 1)) / (q - 1))
  (h3 : 6 * a 3 = a 0 * q ^ 5 - a 0 * q ^ 4) :
  S 4 / S 2 = 10 := 
sorry

end geometric_sequence_sum_ratio_l262_262990


namespace probability_divisible_by_5_l262_262544

def is_three_digit_integer (M : ℕ) : Prop :=
  100 ≤ M ∧ M < 1000

def ones_digit_is_4 (M : ℕ) : Prop :=
  (M % 10) = 4

theorem probability_divisible_by_5 (M : ℕ) (h1 : is_three_digit_integer M) (h2 : ones_digit_is_4 M) :
  (∃ p : ℚ, p = 0) :=
by
  sorry

end probability_divisible_by_5_l262_262544


namespace cross_ratio_eq_one_implies_equal_points_l262_262374

-- Definitions corresponding to the points and hypothesis.
variable {A B C D : ℝ}
variable (h_line : collinear ℝ A B C D) (h_cross_ratio : cross_ratio A B C D = 1)

-- The theorem statement based on the given problem and solution.
theorem cross_ratio_eq_one_implies_equal_points :
  A = B ∨ C = D :=
sorry

end cross_ratio_eq_one_implies_equal_points_l262_262374


namespace dodecahedron_interior_diagonals_l262_262661

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l262_262661


namespace range_of_a_l262_262059

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ [-2, 0] then 2 - (1/2)^x else 2 - 2^x

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∃! x ∈ Ioc(-2, 6), f x - log a (x + 2) = 0) ↔ a ∈ Ioc(√2/4, 1/2) :=
sorry

end range_of_a_l262_262059


namespace range_of_x_coordinate_l262_262327

theorem range_of_x_coordinate (x : ℝ) : 
  (0 ≤ 2*x + 2 ∧ 2*x + 2 ≤ 1) ↔ (-1 ≤ x ∧ x ≤ -1/2) := 
sorry

end range_of_x_coordinate_l262_262327


namespace nala_seashells_l262_262196

theorem nala_seashells (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 2 * (a + b)) : a + b + c = 36 :=
by {
  sorry
}

end nala_seashells_l262_262196


namespace trevor_quarters_counted_l262_262098

-- Define the conditions from the problem
variable (Q D : ℕ) 
variable (total_coins : ℕ := 77)
variable (excess : ℕ := 48)

-- Use the conditions to assert the existence of quarters and dimes such that the totals align with the given constraints
theorem trevor_quarters_counted : (Q + D = total_coins) ∧ (D = Q + excess) → Q = 29 :=
by
  -- Add sorry to skip the actual proof, as we are only writing the statement
  sorry

end trevor_quarters_counted_l262_262098


namespace daisy_lunch_vs_breakfast_spending_l262_262006

noncomputable def breakfast_cost : ℝ := 2 + 3 + 4 + 3.5 + 1.5
noncomputable def lunch_base_cost : ℝ := 3.5 + 4 + 5.25 + 6 + 1 + 3
noncomputable def service_charge : ℝ := 0.10 * lunch_base_cost
noncomputable def lunch_cost_with_service_charge : ℝ := lunch_base_cost + service_charge
noncomputable def food_tax : ℝ := 0.05 * lunch_cost_with_service_charge
noncomputable def total_lunch_cost : ℝ := lunch_cost_with_service_charge + food_tax
noncomputable def difference : ℝ := total_lunch_cost - breakfast_cost

theorem daisy_lunch_vs_breakfast_spending :
  difference = 12.28 :=
by 
  sorry

end daisy_lunch_vs_breakfast_spending_l262_262006


namespace isosceles_triangle_base_length_l262_262090

theorem isosceles_triangle_base_length (b : ℕ) (h₁ : 6 + 6 + b = 20) : b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l262_262090


namespace xy_uv_zero_l262_262494

theorem xy_uv_zero (x y u v : ℝ) (h1 : x^2 + y^2 = 1) (h2 : u^2 + v^2 = 1) (h3 : x * u + y * v = 0) : x * y + u * v = 0 :=
by
  sorry

end xy_uv_zero_l262_262494


namespace valid_passcodes_count_l262_262901

-- Define a predicate that checks whether a list of digits forms a valid passcode.
def is_valid_passcode (lst : List ℕ) : Prop :=
  lst.length = 4 ∧ lst.prod = 18 ∧ ∀ x ∈ lst, 1 ≤ x ∧ x ≤ 9

-- Count the number of valid passcodes
def count_valid_passcodes : ℕ :=
  (List.range' 1 9).product (fun _ => List.range' 1 9).filter is_valid_passcode).length

theorem valid_passcodes_count : count_valid_passcodes = 36 :=
  sorry

end valid_passcodes_count_l262_262901


namespace marbles_total_l262_262425

-- Conditions
variables (T : ℕ) -- Total number of marbles
variables (h_red : T ≥ 12) -- At least 12 red marbles
variables (h_blue : T ≥ 8) -- At least 8 blue marbles
variables (h_prob : (T - 12 : ℚ) / T = (3 / 4 : ℚ)) -- Probability condition

-- Proof statement
theorem marbles_total : T = 48 :=
by
  -- Proof here
  sorry

end marbles_total_l262_262425


namespace multiplicative_inverse_modulo_l262_262892

noncomputable def A := 123456
noncomputable def B := 153846
noncomputable def N := 500000

theorem multiplicative_inverse_modulo :
  (A * B * N) % 1000000 = 1 % 1000000 :=
by
  sorry

end multiplicative_inverse_modulo_l262_262892


namespace f_six_equals_twenty_two_l262_262701

-- Definitions as per conditions
variable (n : ℕ) (f : ℕ → ℕ)

-- Conditions of the problem
-- n is a natural number greater than or equal to 3
-- f(n) satisfies the properties defined in the given solution
axiom f_base : f 1 = 2
axiom f_recursion {k : ℕ} (hk : k ≥ 1) : f (k + 1) = f k + (k + 1)

-- Goal to prove
theorem f_six_equals_twenty_two : f 6 = 22 := sorry

end f_six_equals_twenty_two_l262_262701


namespace sqrt_mixed_number_eq_l262_262972

def improper_fraction (a b c : ℕ) (d : ℕ) : ℚ :=
  a + b / d

theorem sqrt_mixed_number_eq (a b c d : ℕ) (h : d ≠ 0) :
  (d * a + b) ^ 2 = c * d^2 → 
  sqrt (improper_fraction a b c d) = (sqrt (d * a + b)) / (sqrt d) :=
by sorry

example : sqrt (improper_fraction 8 9 0 16) = (sqrt 137) / 4 := 
  sqrt_mixed_number_eq 8 9 0 16 sorry sorry

end sqrt_mixed_number_eq_l262_262972


namespace value_of_first_equation_l262_262483

theorem value_of_first_equation (x y : ℚ) 
  (h1 : 5 * x + 6 * y = 7) 
  (h2 : 3 * x + 5 * y = 6) : 
  x + 4 * y = 5 :=
sorry

end value_of_first_equation_l262_262483


namespace inequality_may_not_hold_l262_262687

theorem inequality_may_not_hold (a b c : ℝ) (h : a > b) : (c < 0) → ¬ (a/c > b/c) := 
sorry

end inequality_may_not_hold_l262_262687


namespace find_deductive_reasoning_l262_262950

noncomputable def is_deductive_reasoning (reasoning : String) : Prop :=
  match reasoning with
  | "B" => true
  | _ => false

theorem find_deductive_reasoning : is_deductive_reasoning "B" = true :=
  sorry

end find_deductive_reasoning_l262_262950


namespace sum_of_cubes_l262_262352

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) : x^3 + y^3 = -10 :=
by
  sorry

end sum_of_cubes_l262_262352


namespace teacher_selection_l262_262946

/-- A school has 150 teachers, including 15 senior teachers, 45 intermediate teachers, 
and 90 junior teachers. By stratified sampling, 30 teachers are selected to 
participate in the teachers' representative conference. 
--/

def total_teachers : ℕ := 150
def senior_teachers : ℕ := 15
def intermediate_teachers : ℕ := 45
def junior_teachers : ℕ := 90

def total_selected_teachers : ℕ := 30
def selected_senior_teachers : ℕ := 3
def selected_intermediate_teachers : ℕ := 9
def selected_junior_teachers : ℕ := 18

def ratio (a b : ℕ) : ℕ × ℕ := (a / (gcd a b), b / (gcd a b))

theorem teacher_selection :
  ratio senior_teachers (gcd senior_teachers total_teachers) = ratio intermediate_teachers (gcd intermediate_teachers total_teachers) ∧
  ratio intermediate_teachers (gcd intermediate_teachers total_teachers) = ratio junior_teachers (gcd junior_teachers total_teachers) →
  selected_senior_teachers / selected_intermediate_teachers / selected_junior_teachers = 1 / 3 / 6 → 
  selected_senior_teachers + selected_intermediate_teachers + selected_junior_teachers = 30 :=
sorry

end teacher_selection_l262_262946


namespace greatest_two_digit_product_12_l262_262785

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l262_262785


namespace circumcircle_radius_of_triangle_l262_262728

theorem circumcircle_radius_of_triangle
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (AB BC : ℝ)
  (angle_ABC : ℝ)
  (hAB : AB = 4)
  (hBC : BC = 4)
  (h_angle_ABC : angle_ABC = 120) :
  ∃ (R : ℝ), R = 4 := by
  sorry

end circumcircle_radius_of_triangle_l262_262728


namespace arith_seq_sum_correct_l262_262225

-- Define the arithmetic sequence given the first term and common difference
def arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def arith_seq_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

-- Given Problem Conditions
def a₁ := -5
def d := 3
def n := 20

-- Theorem: Sum of the first 20 terms of the arithmetic sequence is 470
theorem arith_seq_sum_correct : arith_seq_sum a₁ d n = 470 :=
  sorry

end arith_seq_sum_correct_l262_262225


namespace greatest_two_digit_product_is_12_l262_262778

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l262_262778


namespace lcm_23_46_827_l262_262140

theorem lcm_23_46_827 :
  (23 * 46 * 827) / gcd (23 * 2) 827 = 38042 := by
  sorry

end lcm_23_46_827_l262_262140


namespace gain_is_rs_150_l262_262296

noncomputable def P : ℝ := 5000
noncomputable def R_borrow : ℝ := 4
noncomputable def R_lend : ℝ := 7
noncomputable def T : ℝ := 2

noncomputable def SI (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (P * R * T) / 100

noncomputable def interest_paid := SI P R_borrow T
noncomputable def interest_earned := SI P R_lend T

noncomputable def gain_per_year : ℝ :=
  (interest_earned / T) - (interest_paid / T)

theorem gain_is_rs_150 : gain_per_year = 150 :=
by
  sorry

end gain_is_rs_150_l262_262296


namespace sqrt_of_mixed_number_as_fraction_l262_262973

def mixed_number_to_improper_fraction (a : ℚ) : ℚ :=
  8 + 9 / 16

theorem sqrt_of_mixed_number_as_fraction :
  (√ (mixed_number_to_improper_fraction 8) : ℚ) = (√137) / 4 :=
by
  sorry

end sqrt_of_mixed_number_as_fraction_l262_262973


namespace greatest_two_digit_with_product_12_l262_262798

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l262_262798


namespace range_of_k_l262_262044

noncomputable def f (k x : ℝ) := k * x - Real.exp x
noncomputable def g (x : ℝ) := Real.exp x / x

theorem range_of_k (k : ℝ) (h : ∃ x : ℝ, x ≠ 0 ∧ f k x = 0) :
  k < 0 ∨ k ≥ Real.exp 1 := sorry

end range_of_k_l262_262044


namespace slope_of_line_l262_262628

noncomputable def slope_range : Set ℝ :=
  {α | (5 * Real.pi / 6) ≤ α ∧ α < Real.pi}

theorem slope_of_line (x a : ℝ) :
  let k := -1 / (a^2 + Real.sqrt 3)
  ∃ α ∈ slope_range, k = Real.tan α :=
sorry

end slope_of_line_l262_262628


namespace sum_and_product_of_roots_cube_l262_262346

theorem sum_and_product_of_roots_cube (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by {
  sorry
}

end sum_and_product_of_roots_cube_l262_262346


namespace count_numbers_with_perfect_square_factors_l262_262033

open Set

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≥ 2 ∧ m * m ∣ n

theorem count_numbers_with_perfect_square_factors (s : Finset ℕ) (hs : s = Finset.range 101) :
  (Finset.filter has_perfect_square_factor_other_than_one s).card = 41 :=
by {
  sorry
}

end count_numbers_with_perfect_square_factors_l262_262033


namespace sqrt_mixed_number_simplified_l262_262974

theorem sqrt_mixed_number_simplified :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 := sorry

end sqrt_mixed_number_simplified_l262_262974


namespace range_of_a_l262_262161

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x + 1| ≤ a) → a ≥ 2 :=
by 
  intro h
  sorry

end range_of_a_l262_262161


namespace greatest_two_digit_with_product_12_l262_262797

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l262_262797


namespace fraction_of_seats_taken_l262_262358

theorem fraction_of_seats_taken : 
  ∀ (total_seats broken_fraction available_seats : ℕ), 
    total_seats = 500 → 
    broken_fraction = 1 / 10 → 
    available_seats = 250 → 
    (total_seats - available_seats - total_seats * broken_fraction) / total_seats = 2 / 5 :=
by
  intro total_seats broken_fraction available_seats
  intro h1 h2 h3
  sorry

end fraction_of_seats_taken_l262_262358


namespace cost_price_of_article_l262_262421

theorem cost_price_of_article (x : ℝ) :
  (86 - x = x - 42) → x = 64 :=
by
  intro h
  sorry

end cost_price_of_article_l262_262421


namespace intersection_A_B_l262_262865

-- Define the set A
def A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}

-- Define the set B
def B := {x : ℝ | x^2 - x < 0}

-- The proof problem statement in Lean 4
theorem intersection_A_B : A ∩ B = {y : ℝ | 0 < y ∧ y < 1} :=
by
  sorry

end intersection_A_B_l262_262865


namespace max_ab_at_extremum_l262_262159

noncomputable def f (a b x : ℝ) : ℝ := 4*x^3 - a*x^2 - 2*b*x + 2

theorem max_ab_at_extremum (a b : ℝ) (h0: a > 0) (h1 : b > 0) (h2 : ∃ x, f a b x = 4*x^3 - a*x^2 - 2*b*x + 2 ∧ x = 1 ∧ 12*x^2 - 2*a*x - 2*b = 0) :
  ab ≤ 9 := 
sorry  -- proof not required

end max_ab_at_extremum_l262_262159


namespace dodecahedron_interior_diagonals_l262_262682

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l262_262682


namespace closest_perfect_square_to_350_l262_262253

theorem closest_perfect_square_to_350 : ∃ (n : ℕ), n^2 = 361 ∧ ∀ (m : ℕ), m^2 ≤ 350 ∨ 350 < m^2 → abs (361 - 350) ≤ abs (m^2 - 350) :=
by
  sorry

end closest_perfect_square_to_350_l262_262253


namespace thirty_two_not_sum_consecutive_natural_l262_262206

theorem thirty_two_not_sum_consecutive_natural (n k : ℕ) : 
  (n > 0) → (32 ≠ (n * (2 * k + n - 1)) / 2) :=
by
  sorry

end thirty_two_not_sum_consecutive_natural_l262_262206


namespace tan_monotone_increasing_interval_l262_262909

theorem tan_monotone_increasing_interval :
  ∀ k : ℤ, ∀ x : ℝ, 
  (-π / 2 + k * π < x + π / 4 ∧ x + π / 4 < π / 2 + k * π) ↔
  (k * π - 3 * π / 4 < x ∧ x < k * π + π / 4) :=
by sorry

end tan_monotone_increasing_interval_l262_262909


namespace periodic_function_l262_262479

variable {α : Type*} [AddGroup α] {f : α → α} {a b : α}

def symmetric_around (c : α) (f : α → α) : Prop := ∀ x, f (c - x) = f (c + x)

theorem periodic_function (h1 : symmetric_around a f) (h2 : symmetric_around b f) (h_ab : a ≠ b) : ∃ T, (∀ x, f (x + T) = f x) := 
sorry

end periodic_function_l262_262479


namespace total_blue_marbles_l262_262721

noncomputable def total_blue_marbles_collected_by_friends : ℕ := 
  let jenny_red := 30 in
  let jenny_blue := 25 in
  let mary_red := 2 * jenny_red in
  let anie_red := mary_red + 20 in
  let anie_blue := 2 * jenny_blue in
  let mary_blue := anie_blue / 2 in
  jenny_blue + mary_blue + anie_blue

theorem total_blue_marbles (jenny_red jenny_blue mary_red anie_red mary_blue anie_blue : ℕ) :
  jenny_red = 30 → 
  jenny_blue = 25 → 
  mary_red = 2 * jenny_red → 
  anie_red = mary_red + 20 → 
  anie_blue = 2 * jenny_blue → 
  mary_blue = anie_blue / 2 → 
  jenny_blue + mary_blue + anie_blue = 100 :=
begin
  intros h1 h2 h3 h4 h5 h6,
  rw [h1, h2, h3, h4, h5, h6],
  norm_num,
end

end total_blue_marbles_l262_262721


namespace polynomial_eval_at_3_is_290_l262_262557

noncomputable def polynomial_eval : Polynomial ℤ :=
  Polynomial.C 2 * Polynomial.X ^ 4 +
  Polynomial.C 3 * Polynomial.X ^ 3 +
  Polynomial.C 4 * Polynomial.X ^ 2 +
  Polynomial.C 5 * Polynomial.X - Polynomial.C 4

theorem polynomial_eval_at_3_is_290 :
  polynomial_eval.eval 3 = 290 :=
by
  sorry

end polynomial_eval_at_3_is_290_l262_262557


namespace number_of_interior_diagonals_of_dodecahedron_l262_262646

-- Definitions based on conditions
def dodecahedron_vertices := 20
def faces_per_vertex := 3
def vertices_per_face := 5
def shared_edges_per_vertex := faces_per_vertex
def total_faces := 12
def total_vertices := 20

-- Property of the dodecahedron
def potential_diagonals_per_vertex := dodecahedron_vertices - 1 - shared_edges_per_vertex - (vertices_per_face - 1)
def total_potential_diagonals := potential_diagonals_per_vertex * total_vertices

-- Proof statement:
theorem number_of_interior_diagonals_of_dodecahedron :
  total_potential_diagonals / 2 = 90 :=
by
  -- This is where the proof would go.
  sorry

end number_of_interior_diagonals_of_dodecahedron_l262_262646


namespace probability_of_adjacent_vertices_in_dodecagon_l262_262818

def probability_at_least_two_adjacent_vertices (n : ℕ) : ℚ :=
  if n = 12 then 24 / 55 else 0  -- Only considering the dodecagon case

theorem probability_of_adjacent_vertices_in_dodecagon :
  probability_at_least_two_adjacent_vertices 12 = 24 / 55 :=
by
  sorry

end probability_of_adjacent_vertices_in_dodecagon_l262_262818


namespace exists_root_in_interval_l262_262444

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

-- Conditions given in the problem
variables {a b c : ℝ}
variable  (h_a_nonzero : a ≠ 0)
variable  (h_neg_value : quadratic a b c 3.24 = -0.02)
variable  (h_pos_value : quadratic a b c 3.25 = 0.01)

-- Problem statement to be proved
theorem exists_root_in_interval : ∃ x : ℝ, 3.24 < x ∧ x < 3.25 ∧ quadratic a b c x = 0 :=
sorry

end exists_root_in_interval_l262_262444


namespace zeros_in_square_of_nines_l262_262157

theorem zeros_in_square_of_nines : 
  let n := 10
  in (10^n - 1)^2 = 10^20 - 2*10^n + 1 →
     (∃ z : ℕ, z = 10) :=
by 
  sorry

end zeros_in_square_of_nines_l262_262157


namespace speed_limit_correct_l262_262436

def speed_limit_statement (v : ℝ) : Prop :=
  v ≤ 70

theorem speed_limit_correct (v : ℝ) (h : v ≤ 70) : speed_limit_statement v :=
by
  exact h

#print axioms speed_limit_correct

end speed_limit_correct_l262_262436


namespace jeans_to_tshirt_ratio_l262_262337

noncomputable def socks_price := 5
noncomputable def tshirt_price := socks_price + 10
noncomputable def jeans_price := 30

theorem jeans_to_tshirt_ratio :
  jeans_price / tshirt_price = (2 : ℝ) :=
by sorry

end jeans_to_tshirt_ratio_l262_262337


namespace greatest_two_digit_with_product_12_l262_262760

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l262_262760


namespace find_certain_number_l262_262290

theorem find_certain_number (x : ℕ) (h: x - 82 = 17) : x = 99 :=
by
  sorry

end find_certain_number_l262_262290


namespace problem_l262_262130

-- Define sets A and B
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y <= -1 }

-- Define set C as a function of a
def C (a : ℝ) : Set ℝ := { x | x < -a / 2 }

-- The statement of the problem: if B ⊆ C, then a < 2
theorem problem (a : ℝ) : (B ⊆ C a) → a < 2 :=
by sorry

end problem_l262_262130


namespace is_correct_functional_expression_l262_262389

variable (x : ℝ)

def is_isosceles_triangle (x : ℝ) (y : ℝ) : Prop :=
  2*x + y = 20

theorem is_correct_functional_expression (h1 : 5 < x) (h2 : x < 10) : 
  ∃ y, y = 20 - 2*x :=
by
  sorry

end is_correct_functional_expression_l262_262389


namespace solve_for_y_l262_262566

theorem solve_for_y (x y : ℝ) (h1 : x * y = 25) (h2 : x / y = 36) (h3 : x > 0) (h4 : y > 0) : y = 5 / 6 := 
by
  sorry

end solve_for_y_l262_262566


namespace no_such_abc_exists_l262_262965

theorem no_such_abc_exists : ¬ ∃ (a b c : ℝ), ∀ (x y : ℝ),
  |x + a| + |x + y + b| + |y + c| > |x| + |x + y| + |y| :=
by
  sorry

end no_such_abc_exists_l262_262965


namespace complement_U_A_correct_l262_262156

-- Define the universal set U and set A
def U : Set Int := {-1, 0, 2}
def A : Set Int := {-1, 0}

-- Define the complement of A in U
def complement_U_A : Set Int := {x | x ∈ U ∧ x ∉ A}

-- Theorem stating the required proof
theorem complement_U_A_correct : complement_U_A = {2} :=
by
  sorry -- Proof will be filled in

end complement_U_A_correct_l262_262156


namespace y_order_of_quadratic_l262_262046

theorem y_order_of_quadratic (k : ℝ) (y1 y2 y3 : ℝ) :
  (y1 = (-4)^2 + 4 * (-4) + k) → 
  (y2 = (-1)^2 + 4 * (-1) + k) → 
  (y3 = (1)^2 + 4 * (1) + k) → 
  y2 < y1 ∧ y1 < y3 :=
by
  intro hy1 hy2 hy3
  sorry

end y_order_of_quadratic_l262_262046


namespace total_lucky_stars_l262_262735

theorem total_lucky_stars : 
  (∃ n : ℕ, 10 * n + 6 = 116 ∧ 4 * 8 + (n - 4) * 12 = 116) → 
  116 = 116 := 
by
  intro h
  obtain ⟨n, h1, h2⟩ := h
  sorry

end total_lucky_stars_l262_262735


namespace centripetal_accel_v_r_centripetal_accel_omega_r_centripetal_accel_T_r_l262_262620

-- Defining the variables involved
variables {a v r ω T : ℝ}

-- Main theorem statements representing the problem
theorem centripetal_accel_v_r (v r : ℝ) (h₁ : 0 < r) : a = v^2 / r :=
sorry

theorem centripetal_accel_omega_r (ω r : ℝ) (h₁ : 0 < r) : a = r * ω^2 :=
sorry

theorem centripetal_accel_T_r (T r : ℝ) (h₁ : 0 < r) (h₂ : 0 < T) : a = 4 * π^2 * r / T^2 :=
sorry

end centripetal_accel_v_r_centripetal_accel_omega_r_centripetal_accel_T_r_l262_262620


namespace minimum_boxes_to_eliminate_50_percent_chance_l262_262357

def total_boxes : Nat := 30
def high_value_boxes : Nat := 6
def minimum_boxes_to_eliminate (total_boxes high_value_boxes : Nat) : Nat :=
  total_boxes - high_value_boxes - high_value_boxes

theorem minimum_boxes_to_eliminate_50_percent_chance :
  minimum_boxes_to_eliminate total_boxes high_value_boxes = 18 :=
by
  sorry

end minimum_boxes_to_eliminate_50_percent_chance_l262_262357


namespace polynomial_coefficient_sum_equality_l262_262719

theorem polynomial_coefficient_sum_equality :
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ),
    (∀ x : ℝ, (2 * x + 1)^4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
    (a₀ - a₁ + a₂ - a₃ + a₄ = 1) :=
by
  intros
  sorry

end polynomial_coefficient_sum_equality_l262_262719


namespace sum_inf_series_l262_262850

noncomputable def H (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (1 : ℝ) / (i + 1)

open Real

theorem sum_inf_series : ∑' (n : ℕ in Finset.univ), (n : ℝ) / ((n + 1) * H n * H (n + 1)) = 1 / 2 := 
by
  sorry

end sum_inf_series_l262_262850


namespace direction_vector_of_line_m_l262_262991

noncomputable def projectionMatrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![ 5 / 21, -2 / 21, -2 / 7 ],
    ![ -2 / 21, 1 / 42, 1 / 14 ],
    ![ -2 / 7,  1 / 14, 4 / 7 ]
  ]

noncomputable def vectorI : Fin 3 → ℚ
  | 0 => 1
  | _ => 0

noncomputable def projectedVector : Fin 3 → ℚ :=
  fun i => (projectionMatrix.mulVec vectorI) i

theorem direction_vector_of_line_m :
  (projectedVector 0 = 5 / 21) ∧ 
  (projectedVector 1 = -2 / 21) ∧
  (projectedVector 2 = -6 / 21) ∧
  Nat.gcd (Nat.gcd 5 2) 6 = 1 :=
by
  sorry

end direction_vector_of_line_m_l262_262991


namespace binar_operation_correct_l262_262839

theorem binar_operation_correct : 
  let a := 13  -- 1101_2 in decimal
  let b := 15  -- 1111_2 in decimal
  let c := 9   -- 1001_2 in decimal
  let d := 2   -- 10_2 in decimal
  a + b - c * d = 10 ↔ "1010" = "1010" := 
by 
  intros
  simp
  sorry

end binar_operation_correct_l262_262839


namespace find_radius_l262_262331

theorem find_radius (a : ℝ) :
  (∃ (x y : ℝ), (x + 2) ^ 2 + (y - 2) ^ 2 = a ∧ x + y + 2 = 0) ∧
  (∃ (l : ℝ), l = 6 ∧ 2 * Real.sqrt (a - 2) = l) →
  a = 11 :=
by
  sorry

end find_radius_l262_262331


namespace final_amount_correct_l262_262838

def wallet_cost : ℝ := 22
def purse_cost : ℝ := 4 * wallet_cost - 3
def shoes_cost : ℝ := wallet_cost + purse_cost + 7
def total_cost_before_discount : ℝ := wallet_cost + purse_cost + shoes_cost
def discount_rate : ℝ := 0.10
def discounted_amount : ℝ := total_cost_before_discount * discount_rate
def final_amount : ℝ := total_cost_before_discount - discounted_amount

theorem final_amount_correct :
  final_amount = 198.90 := by
  -- Here we would provide the proof of the theorem
  sorry

end final_amount_correct_l262_262838


namespace possible_point_counts_l262_262227

theorem possible_point_counts (r b g : ℕ) (d_RB d_RG d_BG : ℕ) :
    r + b + g = 15 →
    r * b * d_RB = 51 →
    r * g * d_RG = 39 →
    b * g * d_BG = 1 →
    (r = 13 ∧ b = 1 ∧ g = 1) ∨ (r = 8 ∧ b = 4 ∧ g = 3) :=
by {
    sorry
}

end possible_point_counts_l262_262227


namespace range_of_a_l262_262486

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x - a) * (x + 1 - a) >= 0 → x ≠ 1) ↔ (1 < a ∧ a < 2) := 
sorry

end range_of_a_l262_262486


namespace Olivia_pays_4_dollars_l262_262201

-- Definitions based on the conditions
def quarters_chips : ℕ := 4
def quarters_soda : ℕ := 12
def conversion_rate : ℕ := 4

-- Prove that the total dollars Olivia pays is 4
theorem Olivia_pays_4_dollars (h1 : quarters_chips = 4) (h2 : quarters_soda = 12) (h3 : conversion_rate = 4) : 
  (quarters_chips + quarters_soda) / conversion_rate = 4 :=
by
  -- skipping the proof
  sorry

end Olivia_pays_4_dollars_l262_262201


namespace total_seashells_l262_262193

theorem total_seashells (a b : Nat) (h1 : a = 5) (h2 : b = 7) : 
  let total_first_two_days := a + b
  let third_day := 2 * total_first_two_days
  let total := total_first_two_days + third_day
  total = 36 := 
by
  sorry

end total_seashells_l262_262193


namespace x_plus_y_equals_six_l262_262715

theorem x_plus_y_equals_six (x y : ℝ) (h₁ : y - x = 1) (h₂ : y^2 = x^2 + 6) : x + y = 6 :=
by
  sorry

end x_plus_y_equals_six_l262_262715


namespace S2_side_length_656_l262_262209

noncomputable def S1_S2_S3_side_lengths (l1 l2 a b c : ℕ) (total_length : ℕ) : Prop :=
  l1 + l2 + a + b + c = total_length

theorem S2_side_length_656 :
  ∃ (l1 l2 a c : ℕ), S1_S2_S3_side_lengths l1 l2 a 656 c 3322 :=
by
  sorry

end S2_side_length_656_l262_262209


namespace gift_cost_l262_262619

def ErikaSavings : ℕ := 155
def CakeCost : ℕ := 25
def LeftOver : ℕ := 5

noncomputable def CostOfGift (RickSavings : ℕ) : ℕ :=
  2 * RickSavings

theorem gift_cost (RickSavings : ℕ)
  (hRick : RickSavings = CostOfGift RickSavings / 2)
  (hTotal : ErikaSavings + RickSavings = CostOfGift RickSavings + CakeCost + LeftOver) :
  CostOfGift RickSavings = 250 :=
by
  sorry

end gift_cost_l262_262619


namespace real_distance_between_cities_l262_262736

-- Condition: the map distance between Goteborg and Jonkoping
def map_distance_cm : ℝ := 88

-- Condition: the map scale
def map_scale_km_per_cm : ℝ := 15

-- The real distance to be proven
theorem real_distance_between_cities :
  (map_distance_cm * map_scale_km_per_cm) = 1320 := by
  sorry

end real_distance_between_cities_l262_262736


namespace olivia_pays_in_dollars_l262_262197

theorem olivia_pays_in_dollars (q_chips q_soda : ℕ) 
  (h_chips : q_chips = 4) (h_soda : q_soda = 12) : (q_chips + q_soda) / 4 = 4 := by
  sorry

end olivia_pays_in_dollars_l262_262197


namespace range_a_l262_262152

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then Real.log x / Real.log a else -2 * x + 8

theorem range_a (a : ℝ) (hf : ∀ x, f a x ≤ f a 2) :
  1 < a ∧ a ≤ Real.sqrt 3 := by
  sorry

end range_a_l262_262152


namespace cars_people_count_l262_262423

-- Define the problem conditions
def cars_people_conditions (x y : ℕ) : Prop :=
  y = 3 * (x - 2) ∧ y = 2 * x + 9

-- Define the theorem stating that there exist numbers of cars and people that satisfy the conditions
theorem cars_people_count (x y : ℕ) : cars_people_conditions x y ↔ (y = 3 * (x - 2) ∧ y = 2 * x + 9) := by
  -- skip the proof
  sorry

end cars_people_count_l262_262423


namespace dodecahedron_interior_diagonals_l262_262667

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l262_262667


namespace dodecahedron_interior_diagonals_l262_262677

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l262_262677


namespace harmony_numbers_with_first_digit_2_count_l262_262100

def is_harmony_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (1000 ≤ n ∧ n < 10000) ∧ (a + b + c + d = 6)

noncomputable def count_harmony_numbers_with_first_digit_2 : ℕ :=
  Nat.card { n : ℕ // is_harmony_number n ∧ n / 1000 = 2 }

theorem harmony_numbers_with_first_digit_2_count :
  count_harmony_numbers_with_first_digit_2 = 15 :=
sorry

end harmony_numbers_with_first_digit_2_count_l262_262100


namespace sum_diff_square_cube_l262_262167

/-- If the sum of two numbers is 25 and the difference between them is 15,
    then the difference between the square of the larger number and the cube of the smaller number is 275. -/
theorem sum_diff_square_cube (x y : ℝ) 
  (h1 : x + y = 25)
  (h2 : x - y = 15) :
  x^2 - y^3 = 275 :=
sorry

end sum_diff_square_cube_l262_262167


namespace min_value_of_squares_l262_262185

theorem min_value_of_squares (a b c : ℝ) (h : a^3 + b^3 + c^3 - 3 * a * b * c = 8) : 
  ∃ m, m ≥ 4 ∧ ∀ a b c, a^3 + b^3 + c^3 - 3 * a * b * c = 8 → a^2 + b^2 + c^2 ≥ m :=
sorry

end min_value_of_squares_l262_262185


namespace arithmetic_series_sum_l262_262129

theorem arithmetic_series_sum : 
  ∀ (a d a_n : ℤ), 
  a = -48 → d = 2 → a_n = 0 → 
  ∃ n S : ℤ, 
  a + (n - 1) * d = a_n ∧ 
  S = n * (a + a_n) / 2 ∧ 
  S = -600 :=
by
  intros a d a_n ha hd han
  have h₁ : a = -48 := ha
  have h₂ : d = 2 := hd
  have h₃ : a_n = 0 := han
  sorry

end arithmetic_series_sum_l262_262129


namespace closest_perfect_square_to_350_l262_262273

theorem closest_perfect_square_to_350 :
  let n := 19 in
  let m := 18 in
  18^2 = 324 ∧ 19^2 = 361 ∧ 324 < 350 ∧ 350 < 361 ∧ 
  abs (350 - 324) = 26 ∧ abs (361 - 350) = 11 ∧ 11 < 26 → 
  n^2 = 361 :=
by
  intro n m h
  sorry

end closest_perfect_square_to_350_l262_262273


namespace libby_quarters_left_after_payment_l262_262519

noncomputable def quarters_needed (usd_target : ℝ) (usd_per_quarter : ℝ) : ℝ := 
  usd_target / usd_per_quarter

noncomputable def quarters_left (initial_quarters : ℝ) (used_quarters : ℝ) : ℝ := 
  initial_quarters - used_quarters

theorem libby_quarters_left_after_payment
  (initial_quarters : ℝ) (usd_target : ℝ) (usd_per_quarter : ℝ) 
  (h_initial : initial_quarters = 160) 
  (h_usd_target : usd_target = 35) 
  (h_usd_per_quarter : usd_per_quarter = 0.25) : 
  quarters_left initial_quarters (quarters_needed usd_target usd_per_quarter) = 20 := 
by
  sorry

end libby_quarters_left_after_payment_l262_262519


namespace sum_and_product_of_roots_cube_l262_262345

theorem sum_and_product_of_roots_cube (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by {
  sorry
}

end sum_and_product_of_roots_cube_l262_262345


namespace spaceship_distance_traveled_l262_262582

theorem spaceship_distance_traveled (d_ex : ℝ) (d_xy : ℝ) (d_total : ℝ) :
  d_ex = 0.5 → d_xy = 0.1 → d_total = 0.7 → (d_total - (d_ex + d_xy)) = 0.1 :=
by
  intros h1 h2 h3
  sorry

end spaceship_distance_traveled_l262_262582


namespace fraction_product_eq_l262_262836

theorem fraction_product_eq :
  (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) = 4 / 9 :=
by
  sorry

end fraction_product_eq_l262_262836


namespace find_a_l262_262693

variables (x y : ℝ) (a : ℝ)

-- Condition 1: Original profit equation
def original_profit := y - x = x * (a / 100)

-- Condition 2: New profit equation with 5% cost decrease
def new_profit := y - 0.95 * x = 0.95 * x * ((a + 15) / 100)

theorem find_a (h1 : original_profit x y a) (h2 : new_profit x y a) : a = 185 :=
sorry

end find_a_l262_262693


namespace greatest_two_digit_product_is_12_l262_262776

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l262_262776


namespace value_of_box_l262_262025

theorem value_of_box (a b c : ℕ) (h1 : a + b = c) (h2 : a + b + c = 100) : c = 50 :=
sorry

end value_of_box_l262_262025


namespace range_of_m_l262_262013

def P (m : ℝ) : Prop := m^2 - 4 > 0
def Q (m : ℝ) : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m (m : ℝ) : ¬(P m ∧ Q m) ∧ (P m ∨ Q m) ↔ (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l262_262013


namespace min_value_of_expr_min_value_at_specific_points_l262_262888

noncomputable def min_value_expr (p q r : ℝ) : ℝ := 8 * p^4 + 18 * q^4 + 50 * r^4 + 1 / (8 * p * q * r)

theorem min_value_of_expr : ∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → min_value_expr p q r ≥ 6 :=
by
  intro p q r hp hq hr
  sorry

theorem min_value_at_specific_points : min_value_expr (1 / (8 : ℝ)^(1 / 4)) (1 / (18 : ℝ)^(1 / 4)) (1 / (50 : ℝ)^(1 / 4)) = 6 :=
by
  sorry

end min_value_of_expr_min_value_at_specific_points_l262_262888


namespace eliana_total_steps_l262_262611

-- Define the conditions given in the problem
def steps_first_day_exercise : Nat := 200
def steps_first_day_additional : Nat := 300
def steps_first_day : Nat := steps_first_day_exercise + steps_first_day_additional

def steps_second_day : Nat := 2 * steps_first_day
def steps_additional_on_third_day : Nat := 100
def steps_third_day : Nat := steps_second_day + steps_additional_on_third_day

-- Mathematical proof problem proving that the total number of steps is 2600
theorem eliana_total_steps : steps_first_day + steps_second_day + steps_third_day = 2600 := 
by
  sorry

end eliana_total_steps_l262_262611


namespace find_k_values_l262_262466

theorem find_k_values :
    ∀ (k : ℚ),
    (∀ (a b : ℚ), (5 * a^2 + 7 * a + k = 0) ∧ (5 * b^2 + 7 * b + k = 0) ∧ |a - b| = a^2 + b^2 → k = 21 / 25 ∨ k = -21 / 25) :=
by
  sorry

end find_k_values_l262_262466


namespace expression_equals_neg_one_l262_262870

theorem expression_equals_neg_one (a b c : ℝ) (h : a + b + c = 0) :
  (|a| / a) + (|b| / b) + (|c| / c) + (|a * b| / (a * b)) + (|a * c| / (a * c)) + (|b * c| / (b * c)) + (|a * b * c| / (a * b * c)) = -1 :=
  sorry

end expression_equals_neg_one_l262_262870


namespace dodecahedron_interior_diagonals_l262_262672

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l262_262672


namespace symmetric_points_a_minus_b_l262_262017

theorem symmetric_points_a_minus_b (a b : ℝ) 
  (h1 : a = -5) 
  (h2 : b = -1) :
  a - b = -4 := 
sorry

end symmetric_points_a_minus_b_l262_262017


namespace factorization_l262_262975
-- Import the necessary library

-- Define the expression
def expr (x : ℝ) : ℝ := 75 * x^2 + 50 * x

-- Define the factored form
def factored_form (x : ℝ) : ℝ := 25 * x * (3 * x + 2)

-- Statement of the equality to be proved
theorem factorization (x : ℝ) : expr x = factored_form x :=
by {
  sorry
}

end factorization_l262_262975


namespace division_theorem_l262_262318

theorem division_theorem (k : ℕ) (h : k = 6) : 24 / k = 4 := by
  sorry

end division_theorem_l262_262318


namespace coeff_matrix_correct_l262_262031

-- Define the system of linear equations as given conditions
def eq1 (x y : ℝ) : Prop := 2 * x + 3 * y = 1
def eq2 (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the coefficient matrix
def coeffMatrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 3],
  ![1, -2]
]

-- The theorem stating that the coefficient matrix of the system is as defined
theorem coeff_matrix_correct (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) : 
  coeffMatrix = ![
    ![2, 3],
    ![1, -2]
  ] :=
sorry

end coeff_matrix_correct_l262_262031


namespace dodecahedron_interior_diagonals_l262_262675

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l262_262675


namespace greatest_two_digit_with_product_12_l262_262795

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l262_262795


namespace speed_of_man_l262_262927

theorem speed_of_man (v_m v_s : ℝ) 
    (h1 : (v_m + v_s) * 4 = 32) 
    (h2 : (v_m - v_s) * 4 = 24) : v_m = 7 := 
by
  sorry

end speed_of_man_l262_262927


namespace calc_f_five_times_l262_262187

def f (x : ℕ) : ℕ := if x % 2 = 0 then x / 2 else 5 * x + 1

theorem calc_f_five_times : f (f (f (f (f 5)))) = 166 :=
by 
  sorry

end calc_f_five_times_l262_262187


namespace problem_a_problem_b_l262_262718

-- Define the polynomial P(x) = ax^3 + bx
def P (a b x : ℤ) : ℤ := a * x^3 + b * x

-- Define when a pair (a, b) is n-good
def is_ngood (n a b : ℤ) : Prop :=
  ∀ m k : ℤ, n ∣ P a b m - P a b k → n ∣ m - k

-- Define when a pair (a, b) is very good
def is_verygood (a b : ℤ) : Prop :=
  ∀ n : ℤ, ∃ (infinitely_many_n : ℕ), is_ngood n a b

-- Problem (a): Find a pair (1, -51^2) which is 51-good but not very good
theorem problem_a : ∃ a b : ℤ, a = 1 ∧ b = -(51^2) ∧ is_ngood 51 a b ∧ ¬is_verygood a b := 
by {
  sorry
}

-- Problem (b): Show that all 2010-good pairs are very good
theorem problem_b : ∀ a b : ℤ, is_ngood 2010 a b → is_verygood a b := 
by {
  sorry
}

end problem_a_problem_b_l262_262718


namespace MrJohnMonthlySavings_l262_262190

theorem MrJohnMonthlySavings
  (monthly_income : ℝ := 2000)
  (percent_spent_on_transport : ℝ := 5 / 100)
  (transport_fare : ℝ := percent_spent_on_transport * monthly_income) :
  (monthly_income - transport_fare) = 1900 :=
begin
  sorry
end

end MrJohnMonthlySavings_l262_262190


namespace two_a_sq_minus_six_b_plus_one_l262_262689

theorem two_a_sq_minus_six_b_plus_one (a b : ℝ) (h : a^2 - 3 * b = 5) : 2 * a^2 - 6 * b + 1 = 11 := by
  sorry

end two_a_sq_minus_six_b_plus_one_l262_262689


namespace power_function_evaluation_l262_262030

theorem power_function_evaluation (f : ℝ → ℝ) (a : ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = (Real.sqrt 2) / 2) :
  f 4 = 1 / 2 := by
  sorry

end power_function_evaluation_l262_262030


namespace find_xy_solution_l262_262469

theorem find_xy_solution (x y : ℕ) (hx : x > 0) (hy : y > 0) 
    (h : 3^x + x^4 = y.factorial + 2019) : 
    (x = 6 ∧ y = 3) :=
by {
  sorry
}

end find_xy_solution_l262_262469


namespace find_quadratic_eq_l262_262847

theorem find_quadratic_eq (x y : ℝ) 
  (h₁ : x + y = 10)
  (h₂ : |x - y| = 6) :
  x^2 - 10 * x + 16 = 0 :=
sorry

end find_quadratic_eq_l262_262847


namespace retailer_discount_percentage_l262_262297

noncomputable def market_price (P : ℝ) : ℝ := 36 * P
noncomputable def profit (CP : ℝ) : ℝ := CP * 0.1
noncomputable def selling_price (P : ℝ) : ℝ := 40 * P
noncomputable def total_revenue (CP Profit : ℝ) : ℝ := CP + Profit
noncomputable def discount (P S : ℝ) : ℝ := P - S
noncomputable def discount_percentage (D P : ℝ) : ℝ := (D / P) * 100

theorem retailer_discount_percentage (P CP Profit TR S D : ℝ) (h1 : CP = market_price P)
  (h2 : Profit = profit CP) (h3 : TR = total_revenue CP Profit)
  (h4 : TR = selling_price S) (h5 : S = TR / 40) (h6 : D = discount P S) :
  discount_percentage D P = 1 :=
by
  sorry

end retailer_discount_percentage_l262_262297


namespace math_problem_l262_262899

variable (x y : ℝ)

theorem math_problem (h1 : x^2 - 3 * x * y + 2 * y^2 + x - y = 0) (h2 : x^2 - 2 * x * y + y^2 - 5 * x + 7 * y = 0) :
  x * y - 12 * x + 15 * y = 0 :=
  sorry

end math_problem_l262_262899


namespace uncovered_side_length_l262_262580

theorem uncovered_side_length (L W : ℝ) (h1 : L * W = 120) (h2 : L + 2 * W = 32) : L = 20 :=
sorry

end uncovered_side_length_l262_262580


namespace more_than_1000_triplets_l262_262073

theorem more_than_1000_triplets :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 1000 < S.card ∧ 
  ∀ (a b c : ℕ), (a, b, c) ∈ S → a^15 + b^15 = c^16 :=
by sorry

end more_than_1000_triplets_l262_262073


namespace greatest_two_digit_product_is_12_l262_262774

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l262_262774


namespace total_cost_accurate_l262_262607

def price_iphone: ℝ := 800
def price_iwatch: ℝ := 300
def price_ipad: ℝ := 500

def discount_iphone: ℝ := 0.15
def discount_iwatch: ℝ := 0.10
def discount_ipad: ℝ := 0.05

def tax_iphone: ℝ := 0.07
def tax_iwatch: ℝ := 0.05
def tax_ipad: ℝ := 0.06

def cashback: ℝ := 0.02

theorem total_cost_accurate:
  let discounted_auction (price: ℝ) (discount: ℝ) := price * (1 - discount)
  let taxed_auction (price: ℝ) (tax: ℝ) := price * (1 + tax)
  let total_cost :=
    let discount_iphone_cost := discounted_auction price_iphone discount_iphone
    let discount_iwatch_cost := discounted_auction price_iwatch discount_iwatch
    let discount_ipad_cost := discounted_auction price_ipad discount_ipad
    
    let tax_iphone_cost := taxed_auction discount_iphone_cost tax_iphone
    let tax_iwatch_cost := taxed_auction discount_iwatch_cost tax_iwatch
    let tax_ipad_cost := taxed_auction discount_ipad_cost tax_ipad
    
    let total_price := tax_iphone_cost + tax_iwatch_cost + tax_ipad_cost
    total_price * (1 - cashback)
  total_cost = 1484.31 := 
  by sorry

end total_cost_accurate_l262_262607


namespace max_omega_satisfying_conditions_l262_262153

theorem max_omega_satisfying_conditions : 
  ∃ ω > 0, (∀ φ, (|φ| ≤ π / 2) ∧ 
  (∀ x, (x = -π / 4 → sin (ω * x + φ) = 0)) ∧ 
  (∀ x, (x = π / 4 → sin (ω * x + φ) = sin (ω * x + φ))) ∧ 
  (∀ x, (π / 18 < x ∧ x < 5 * π / 36 → 
         (∀ y z, (π / 18 < y ∧ y < 5 * π / 36 ∧ y < z ∧ z < 5 * π / 36 → (sin (ω * y + φ) < sin (ω * z + φ))))))) → 
  ω = 9 := sorry

end max_omega_satisfying_conditions_l262_262153


namespace complement_intersection_l262_262333

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def set_B : Set ℕ := {2, 3, 6}

theorem complement_intersection :
  ((universal_set \ set_A) ∩ set_B) = {2, 6} :=
by
  sorry

end complement_intersection_l262_262333


namespace probability_of_woman_lawyer_is_54_percent_l262_262930

variable (total_members : ℕ) (women_percentage lawyers_percentage : ℕ)
variable (H_total_members_pos : total_members > 0) 
variable (H_women_percentage : women_percentage = 90)
variable (H_lawyers_percentage : lawyers_percentage = 60)

def probability_woman_lawyer : ℕ :=
  (women_percentage * lawyers_percentage * total_members) / (100 * 100)

theorem probability_of_woman_lawyer_is_54_percent (H_total_members_pos : total_members > 0)
  (H_women_percentage : women_percentage = 90)
  (H_lawyers_percentage : lawyers_percentage = 60) :
  probability_woman_lawyer total_members women_percentage lawyers_percentage = 54 :=
by
  sorry

end probability_of_woman_lawyer_is_54_percent_l262_262930


namespace system_nonzero_solution_l262_262551

-- Definition of the game setup and conditions
def initial_equations (a b c : ℤ) (x y z : ℤ) : Prop :=
  (a * x + b * y + c * z = 0) ∧
  (a * x + b * y + c * z = 0) ∧
  (a * x + b * y + c * z = 0)

-- The main proposition statement in Lean
theorem system_nonzero_solution :
  ∀ (a b c : ℤ), ∃ (x y z : ℤ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∧ initial_equations a b c x y z :=
by
  sorry

end system_nonzero_solution_l262_262551


namespace calc_pow_l262_262592

-- Definitions used in the conditions
def base := 2
def exp := 10
def power := 2 / 5

-- Given condition
def given_identity : Pow.pow base exp = 1024 := by sorry

-- Statement to be proved
theorem calc_pow : Pow.pow 1024 power = 16 := by
  -- Use the given identity and known exponentiation rules to derive the result
  sorry

end calc_pow_l262_262592


namespace AM_GM_inequality_example_l262_262379

theorem AM_GM_inequality_example (a b c d : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_prod : a * b * c * d = 1) :
  a^3 + b^3 + c^3 + d^3 ≥ max (a + b + c + d) (1 / a + 1 / b + 1 / c + 1 / d) :=
by
  sorry

end AM_GM_inequality_example_l262_262379


namespace which_set_can_form_triangle_l262_262586

-- Definition of the triangle inequality theorem
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions for each set of line segments
def setA := (2, 6, 8)
def setB := (4, 6, 7)
def setC := (5, 6, 12)
def setD := (2, 3, 6)

-- Proof problem statement
theorem which_set_can_form_triangle : 
  triangle_inequality 2 6 8 = false ∧
  triangle_inequality 4 6 7 = true ∧
  triangle_inequality 5 6 12 = false ∧
  triangle_inequality 2 3 6 = false := 
by
  sorry -- Proof omitted

end which_set_can_form_triangle_l262_262586


namespace sqrt_four_eq_pm_two_l262_262402

theorem sqrt_four_eq_pm_two : ∀ (x : ℝ), x * x = 4 ↔ x = 2 ∨ x = -2 := 
by
  sorry

end sqrt_four_eq_pm_two_l262_262402


namespace sum_of_coordinates_of_X_l262_262712

theorem sum_of_coordinates_of_X 
  (X Y Z : ℝ × ℝ)
  (h1 : dist X Z / dist X Y = 1 / 2)
  (h2 : dist Z Y / dist X Y = 1 / 2)
  (hY : Y = (1, 7))
  (hZ : Z = (-1, -7)) :
  (X.1 + X.2) = -24 :=
sorry

end sum_of_coordinates_of_X_l262_262712


namespace dodecahedron_interior_diagonals_l262_262685

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l262_262685


namespace third_consecutive_even_sum_52_l262_262568

theorem third_consecutive_even_sum_52
  (x : ℤ)
  (h : x + (x + 2) + (x + 4) + (x + 6) = 52) :
  x + 4 = 14 :=
by
  sorry

end third_consecutive_even_sum_52_l262_262568


namespace distance_incenters_ACD_BCD_l262_262709

noncomputable def distance_between_incenters (AC : ℝ) (angle_ABC : ℝ) (angle_BAC : ℝ) : ℝ :=
  -- Use the given conditions to derive the distance value
  -- Skipping the detailed calculations, denoted by "sorry"
  sorry

theorem distance_incenters_ACD_BCD :
  distance_between_incenters 1 (30 : ℝ) (60 : ℝ) = 0.5177 := sorry

end distance_incenters_ACD_BCD_l262_262709


namespace expand_expression_l262_262967

open Nat

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 :=
by
  sorry

end expand_expression_l262_262967


namespace triangle_side_relation_triangle_perimeter_l262_262367

theorem triangle_side_relation (a b c : ℝ) (A B C : ℝ)
  (h1 : a / (Real.sin A) = b / (Real.sin B)) (h2 : a / (Real.sin A) = c / (Real.sin C))
  (h3 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 := sorry

theorem triangle_perimeter (a b c : ℝ) (A : ℝ) (hcosA : Real.cos A = 25 / 31)
  (h1 : a / (Real.sin A) = b / (Real.sin B)) (h2 : a / (Real.sin A) = c / (Real.sin C))
  (h3 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) (ha : a = 5) :
  a + b + c = 14 := sorry

end triangle_side_relation_triangle_perimeter_l262_262367


namespace arccos_neg_one_eq_pi_l262_262598

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l262_262598


namespace flower_shop_sold_bouquets_l262_262936

theorem flower_shop_sold_bouquets (roses_per_bouquet : ℕ) (daisies_per_bouquet : ℕ) 
  (rose_bouquets_sold : ℕ) (daisy_bouquets_sold : ℕ) (total_flowers_sold : ℕ)
  (h1 : roses_per_bouquet = 12) (h2 : rose_bouquets_sold = 10) 
  (h3 : daisy_bouquets_sold = 10) (h4 : total_flowers_sold = 190) : 
  (rose_bouquets_sold + daisy_bouquets_sold) = 20 :=
by sorry

end flower_shop_sold_bouquets_l262_262936


namespace number_divided_by_21_l262_262819

theorem number_divided_by_21 (x : ℝ) (h : 6000 - (x / 21.0) = 5995) : x = 105 :=
by
  sorry

end number_divided_by_21_l262_262819


namespace problem1_l262_262573

theorem problem1 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
sorry

end problem1_l262_262573


namespace greatest_two_digit_with_product_12_l262_262762

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l262_262762


namespace product_mod_eq_l262_262803

theorem product_mod_eq :
  (1497 * 2003) % 600 = 291 := 
sorry

end product_mod_eq_l262_262803


namespace dodecahedron_interior_diagonals_l262_262671

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l262_262671


namespace find_circle_diameter_l262_262216

noncomputable def circle_diameter (AB CD : ℝ) (h_AB : AB = 16) (h_CD : CD = 4)
  (h_perp : ∃ M : ℝ → ℝ → Prop, M AB CD) : ℝ :=
  2 * 10

theorem find_circle_diameter (AB CD : ℝ)
  (h_AB : AB = 16)
  (h_CD : CD = 4)
  (h_perp : ∃ M : ℝ → ℝ → Prop, M AB CD) :
  circle_diameter AB CD h_AB h_CD h_perp = 20 := 
  by sorry

end find_circle_diameter_l262_262216


namespace max_popsicles_with_10_dollars_l262_262070

def price (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 3 then 2
  else if n = 5 then 3
  else if n = 7 then 4
  else 0

theorem max_popsicles_with_10_dollars : ∀ (a b c d : ℕ),
  a * price 1 + b * price 3 + c * price 5 + d * price 7 = 10 →
  a + 3 * b + 5 * c + 7 * d ≤ 17 :=
sorry

end max_popsicles_with_10_dollars_l262_262070


namespace quadratic_vertex_properties_l262_262370

theorem quadratic_vertex_properties (a : ℝ) (x1 x2 y1 y2 : ℝ) (h_ax : a ≠ 0) (h_sum : x1 + x2 = 2) (h_order : x1 < x2) (h_value : y1 > y2) :
  a < -2 / 5 :=
sorry

end quadratic_vertex_properties_l262_262370


namespace distinct_sums_count_l262_262324

theorem distinct_sums_count {n : ℕ} (hpos : 0 < n) (a : Fin n → ℕ)
  (h : ∀ i j, a i < a j → i < j) :
  ∃ S : Finset ℕ, S.card ≥ n * (n + 1) / 2 ∧
  (∀ k : Fin n, a k ∈ S) ∧ 
  (∀ (m : Fin n) (i : Fin n) (j : Fin n), i ≠ j → a i + a j ∈ S → ∃ (p : Fin n) (q : Fin n), p < q ∧ a p + a q = a i + a j) :=
by
  sorry

end distinct_sums_count_l262_262324


namespace mark_owes_joanna_l262_262505

def dollars_per_room : ℚ := 12 / 3
def rooms_cleaned : ℚ := 9 / 4
def total_amount_owed : ℚ := 9

theorem mark_owes_joanna :
  dollars_per_room * rooms_cleaned = total_amount_owed :=
by
  sorry

end mark_owes_joanna_l262_262505


namespace evaluate_expression_l262_262234

theorem evaluate_expression :
  (1 / (-8^2)^4) * (-8)^9 = -8 := by
  sorry

end evaluate_expression_l262_262234


namespace total_turtles_rabbits_l262_262548

-- Number of turtles and rabbits on Happy Island
def turtles_happy : ℕ := 120
def rabbits_happy : ℕ := 80

-- Number of turtles and rabbits on Lonely Island
def turtles_lonely : ℕ := turtles_happy / 3
def rabbits_lonely : ℕ := turtles_lonely

-- Number of turtles and rabbits on Serene Island
def rabbits_serene : ℕ := 2 * rabbits_lonely
def turtles_serene : ℕ := (3 * rabbits_lonely) / 4

-- Number of turtles and rabbits on Tranquil Island
def turtles_tranquil : ℕ := (turtles_happy - turtles_serene) + 5
def rabbits_tranquil : ℕ := turtles_tranquil

-- Proving the total numbers
theorem total_turtles_rabbits :
    turtles_happy = 120 ∧ rabbits_happy = 80 ∧
    turtles_lonely = 40 ∧ rabbits_lonely = 40 ∧
    turtles_serene = 30 ∧ rabbits_serene = 80 ∧
    turtles_tranquil = 95 ∧ rabbits_tranquil = 95 ∧
    (turtles_happy + turtles_lonely + turtles_serene + turtles_tranquil = 285) ∧
    (rabbits_happy + rabbits_lonely + rabbits_serene + rabbits_tranquil = 295) := 
    by
        -- Here we prove each part step by step using the definitions and conditions provided above
        sorry

end total_turtles_rabbits_l262_262548


namespace minimum_value_l262_262411

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 1487

theorem minimum_value : ∃ x : ℝ, f x = 1484 := 
sorry

end minimum_value_l262_262411


namespace discount_percent_l262_262115

theorem discount_percent (CP MP SP : ℝ) (markup profit: ℝ) (h1 : CP = 100) (h2 : MP = CP + (markup * CP))
  (h3 : SP = CP + (profit * CP)) (h4 : markup = 0.75) (h5 : profit = 0.225) : 
  (MP - SP) / MP * 100 = 30 :=
by
  sorry

end discount_percent_l262_262115


namespace range_y_eq_2cosx_minus_1_range_y_eq_sq_2sinx_minus_1_plus_3_l262_262393

open Real

theorem range_y_eq_2cosx_minus_1 : 
  (∀ x : ℝ, -1 ≤ cos x ∧ cos x ≤ 1) →
  (∀ y : ℝ, y = 2 * (cos x) - 1 → -3 ≤ y ∧ y ≤ 1) :=
by
  intros h1 y h2
  sorry

theorem range_y_eq_sq_2sinx_minus_1_plus_3 : 
  (∀ x : ℝ, -1 ≤ sin x ∧ sin x ≤ 1) →
  (∀ y : ℝ, y = (2 * (sin x) - 1)^2 + 3 → 3 ≤ y ∧ y ≤ 12) :=
by
  intros h1 y h2
  sorry

end range_y_eq_2cosx_minus_1_range_y_eq_sq_2sinx_minus_1_plus_3_l262_262393


namespace paint_remaining_fraction_l262_262578

theorem paint_remaining_fraction :
  ∃ (gallons_initial used_first_day used_second_day remaining_second_day : ℚ),
    gallons_initial = 1 ∧
    used_first_day = 1 / 4 ∧
    used_second_day = 1 / 2 * (gallons_initial - used_first_day) ∧
    remaining_second_day = gallons_initial - used_first_day - used_second_day ∧
    remaining_second_day = 3 / 8 :=
by
  -- Define the initial amount of paint
  let gallons_initial : ℚ := 1
  -- Calculate the amount of paint used on the first day
  let used_first_day : ℚ := 1 / 4
  -- Calculate the damount of paint remaining after the first day
  let remaining_after_first_day := gallons_initial - used_first_day
  -- Calculate the amount of paint used on the second day
  let used_second_day : ℚ := 1 / 2 * remaining_after_first_day
  -- Calculate the amount of paint remaining after the second day
  let remaining_second_day : ℚ := remaining_after_first_day - used_second_day
  -- Assert that the remaining paint on the third day is the expected value
  use [gallons_initial, used_first_day, used_second_day, remaining_second_day]
  split
  sorry

end paint_remaining_fraction_l262_262578


namespace people_sitting_between_same_l262_262302

theorem people_sitting_between_same 
  (n : ℕ) (h_even : n % 2 = 0) 
  (f : Fin (2 * n) → Fin (2 * n)) :
  ∃ (a b : Fin (2 * n)), 
  ∃ (k k' : ℕ), k < 2 * n ∧ k' < 2 * n ∧ (a : ℕ) < (b : ℕ) ∧ 
  ((b - a = k) ∧ (f b - f a = k)) ∨ ((a - b + 2*n = k') ∧ ((f a - f b + 2 * n) % (2 * n) = k')) :=
by
  sorry

end people_sitting_between_same_l262_262302


namespace age_of_twin_brothers_l262_262447

theorem age_of_twin_brothers (x : Nat) : (x + 1) * (x + 1) = x * x + 11 ↔ x = 5 :=
by
  sorry  -- Proof omitted.

end age_of_twin_brothers_l262_262447


namespace product_of_five_consecutive_integers_not_square_l262_262730

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (h : a > 0) :
  ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l262_262730


namespace power_of_m_l262_262496

theorem power_of_m (m : ℕ) (h₁ : ∀ k : ℕ, m^k % 24 = 0) (h₂ : ∀ d : ℕ, d ∣ m → d ≤ 8) : ∃ k : ℕ, m^k = 24 :=
sorry

end power_of_m_l262_262496


namespace subset_of_difference_empty_l262_262625

theorem subset_of_difference_empty {α : Type*} (A B : Set α) :
  (A \ B = ∅) → (A ⊆ B) :=
by
  sorry

end subset_of_difference_empty_l262_262625


namespace greatest_two_digit_prod_12_l262_262769

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l262_262769


namespace fred_balloons_l262_262007

variable (initial_balloons : ℕ := 709)
variable (balloons_given : ℕ := 221)
variable (remaining_balloons : ℕ := 488)

theorem fred_balloons :
  initial_balloons - balloons_given = remaining_balloons :=
  by
    sorry

end fred_balloons_l262_262007


namespace eric_time_ratio_l262_262618

-- Defining the problem context
def eric_runs : ℕ := 20
def eric_jogs : ℕ := 10
def eric_return_time : ℕ := 90

-- The ratio is represented as a fraction
def ratio (a b : ℕ) := a / b

-- Stating the theorem
theorem eric_time_ratio :
  ratio eric_return_time (eric_runs + eric_jogs) = 3 :=
by
  sorry

end eric_time_ratio_l262_262618


namespace queenie_total_earnings_l262_262528

-- Define the conditions
def daily_wage : ℕ := 150
def overtime_wage_per_hour : ℕ := 5
def days_worked : ℕ := 5
def overtime_hours : ℕ := 4

-- Define the main problem
theorem queenie_total_earnings : 
  (daily_wage * days_worked + overtime_wage_per_hour * overtime_hours) = 770 :=
by
  sorry

end queenie_total_earnings_l262_262528


namespace no_A_with_integer_roots_l262_262463

theorem no_A_with_integer_roots 
  (A : ℕ) 
  (h1 : A > 0) 
  (h2 : A < 10) 
  : ¬ ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ p + q = 10 + A ∧ p * q = 10 * A + A :=
by sorry

end no_A_with_integer_roots_l262_262463


namespace odd_function_f_l262_262585

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f (x: ℝ) := Real.log ((1 - x) / (1 + x))

theorem odd_function_f :
  odd_function f :=
sorry

end odd_function_f_l262_262585


namespace count_triples_satisfying_conditions_l262_262002

open Nat

theorem count_triples_satisfying_conditions :
  let gcd15 (a b c : ℕ) := gcd (gcd a b) c = 15
  let lcm_comp (a b c : ℕ) := lcm (lcm a b) c = 3^15 * 5^18 
  Finset.card { (a, b, c) : ℕ × ℕ × ℕ | gcd15 a b c ∧ lcm_comp a b c } = 8568 := by
  sorry

end count_triples_satisfying_conditions_l262_262002


namespace find_N_value_l262_262482

variable (a b N : ℚ)
variable (h1 : a + 2 * b = N)
variable (h2 : a * b = 4)
variable (h3 : 2 / a + 1 / b = 1.5)

theorem find_N_value : N = 6 :=
by
  sorry

end find_N_value_l262_262482


namespace interval_of_monotonic_increase_sum_greater_than_2e_l262_262861

noncomputable def f (a x : ℝ) : ℝ := a * x / (Real.log x)

theorem interval_of_monotonic_increase :
  ∀ (x : ℝ), (e < x → f 1 x > f 1 e) := 
sorry

theorem sum_greater_than_2e (x1 x2 : ℝ) (a : ℝ) (h1 : x1 ≠ x2) (hx1 : f 1 x1 = 1) (hx2 : f 1 x2 = 1) :
  x1 + x2 > 2 * Real.exp 1 :=
sorry

end interval_of_monotonic_increase_sum_greater_than_2e_l262_262861


namespace alice_paper_cranes_l262_262445

theorem alice_paper_cranes : 
  ∀ (total : ℕ) (half : ℕ) (one_fifth : ℕ) (thirty_percent : ℕ),
    total = 1000 →
    half = total / 2 →
    one_fifth = (total - half) / 5 →
    thirty_percent = ((total - half) - one_fifth) * 3 / 10 →
    total - (half + one_fifth + thirty_percent) = 280 :=
by
  intros total half one_fifth thirty_percent h_total h_half h_one_fifth h_thirty_percent
  sorry

end alice_paper_cranes_l262_262445


namespace square_side_length_l262_262820

theorem square_side_length :
  ∀ (w l : ℕ) (area : ℕ),
  w = 9 → l = 27 → area = w * l →
  ∃ s : ℝ, s^2 = area ∧ s = 9 * Real.sqrt 3 :=
by
  intros w l area hw hl harea
  sorry

end square_side_length_l262_262820


namespace parabola_equation_l262_262317

theorem parabola_equation (P : ℝ × ℝ) (hP : P = (-4, -2)) :
  (∃ p : ℝ, P.1^2 = -2 * p * P.2 ∧ p = -4 ∧ x^2 = -8*y) ∨ 
  (∃ p : ℝ, P.2^2 = -2 * p * P.1 ∧ p = -1/2 ∧ y^2 = -x) :=
by
  sorry

end parabola_equation_l262_262317


namespace lines_intersect_l262_262579

noncomputable def line1 (t : ℚ) : ℚ × ℚ :=
  (2 + 3 * t, 2 - 4 * t)

noncomputable def line2 (u : ℚ) : ℚ × ℚ :=
  (4 + 5 * u, -6 + 3 * u)

theorem lines_intersect :
  ∃ (t u : ℚ), line1 t = line2 u ∧ line1 t = (160 / 29, -160 / 29) :=
by
  sorry

end lines_intersect_l262_262579


namespace find_unit_price_B_l262_262523

/-- Definitions based on the conditions --/
def total_cost_A := 7500
def total_cost_B := 4800
def quantity_difference := 30
def price_ratio : ℝ := 2.5

/-- Define the variable x as the unit price of B type soccer balls --/
def unit_price_B (x : ℝ) : Prop :=
  (total_cost_A / (price_ratio * x)) + 30 = (total_cost_B / x) ∧
  total_cost_A > 0 ∧ total_cost_B > 0 ∧ x > 0

/-- The main statement to prove --/
theorem find_unit_price_B (x : ℝ) : unit_price_B x ↔ x = 60 :=
by
  sorry

end find_unit_price_B_l262_262523


namespace greatest_two_digit_prod_12_l262_262770

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l262_262770


namespace line_equation_l262_262110

theorem line_equation (a T : ℝ) (h : 0 < a ∧ 0 < T) :
  ∃ (x y : ℝ), (2 * T * x - a^2 * y + 2 * a * T = 0) :=
by
  sorry

end line_equation_l262_262110


namespace total_money_raised_for_charity_l262_262906

theorem total_money_raised_for_charity:
    let price_small := 2
    let price_medium := 3
    let price_large := 5
    let num_small := 150
    let num_medium := 221
    let num_large := 185
    num_small * price_small + num_medium * price_medium + num_large * price_large = 1888 := by
  sorry

end total_money_raised_for_charity_l262_262906


namespace proof_goal_l262_262390

noncomputable def proof_problem (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a^2 + b^2 + c^2 = 1) : Prop :=
  (1 / a) + (1 / b) + (1 / c) > 4

theorem proof_goal (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a^2 + b^2 + c^2 = 1) : 
  (1 / a) + (1 / b) + (1 / c) > 4 :=
sorry

end proof_goal_l262_262390


namespace greatest_two_digit_with_product_12_l262_262766

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l262_262766


namespace olivia_pays_in_dollars_l262_262198

theorem olivia_pays_in_dollars (q_chips q_soda : ℕ) 
  (h_chips : q_chips = 4) (h_soda : q_soda = 12) : (q_chips + q_soda) / 4 = 4 := by
  sorry

end olivia_pays_in_dollars_l262_262198


namespace points_on_line_l262_262873

theorem points_on_line (y1 y2 : ℝ) 
  (hA : y1 = - (1 / 2 : ℝ) * 1 - 1) 
  (hB : y2 = - (1 / 2 : ℝ) * 3 - 1) :
  y1 > y2 := 
by
  sorry

end points_on_line_l262_262873


namespace dice_probability_sum_17_l262_262697

theorem dice_probability_sum_17 :
  let s : Finset (ℕ × ℕ × ℕ) := 
    (Finset.range 6).image (λ x, (x + 1, x + 1, x + 1))
  ∀ (d1 d2 d3 : ℕ), 
  d1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d3 ∈ {1, 2, 3, 4, 5, 6} → 
  (d1 + d2 + d3 = 17 ↔ (d1, d2, d3) ∈ s) → 
  s.card = 1 / 72 := 
begin
  sorry
end

end dice_probability_sum_17_l262_262697


namespace cos_beta_value_l262_262997

noncomputable def cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) : Real :=
  Real.cos β

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) :
  Real.cos β = 56 / 65 :=
by
  sorry

end cos_beta_value_l262_262997


namespace position_after_2010_transformations_l262_262536

-- Define the initial position of the square
def init_position := "ABCD"

-- Define the transformation function
def transform (position : String) (steps : Nat) : String :=
  match steps % 8 with
  | 0 => "ABCD"
  | 1 => "CABD"
  | 2 => "DACB"
  | 3 => "BCAD"
  | 4 => "ADCB"
  | 5 => "CBDA"
  | 6 => "BADC"
  | 7 => "CDAB"
  | _ => "ABCD"  -- Default case, should never happen

-- The theorem to prove the correct position after 2010 transformations
theorem position_after_2010_transformations : transform init_position 2010 = "CABD" := 
by
  sorry

end position_after_2010_transformations_l262_262536


namespace gcd_12345_6789_l262_262754

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l262_262754


namespace quadrilateral_perpendicular_diagonals_l262_262589

theorem quadrilateral_perpendicular_diagonals
  (AB BC CD DA : ℝ)
  (m n : ℝ)
  (hAB : AB = 6)
  (hBC : BC = m)
  (hCD : CD = 8)
  (hDA : DA = n)
  (h_diagonals_perpendicular : true)
  : m^2 + n^2 = 100 := 
by
  sorry

end quadrilateral_perpendicular_diagonals_l262_262589


namespace f_decreasing_in_interval_l262_262529

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

noncomputable def shifted_g (x : ℝ) : ℝ := g (x + Real.pi / 6)

noncomputable def f (x : ℝ) : ℝ := shifted_g (2 * x)

theorem f_decreasing_in_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 4 → f y < f x :=
by
  sorry

end f_decreasing_in_interval_l262_262529


namespace value_of_3_over_x_l262_262638

theorem value_of_3_over_x (x : ℝ) (hx : 1 - 6 / x + 9 / x^2 - 4 / x^3 = 0) : 
  (3 / x = 3 ∨ 3 / x = 3 / 4) :=
  sorry

end value_of_3_over_x_l262_262638


namespace probability_not_pass_fourth_quadrant_l262_262882

-- The set of possible numbers on the balls
def ball_numbers : Set ℚ := {-1, 0, 1/3}

-- The quadratic function with given m and n
def quadratic (m n : ℚ) (x : ℚ) : ℚ := x^2 + m * x + n

-- The condition for the quadratic function not to pass through the fourth quadrant
def does_not_pass_fourth_quadrant (m n : ℚ) : Prop :=
  (m ≥ 0 ∧ n ≥ 0) ∨ (m ≥ 0 ∧ n * 4 ≥ m^2)

-- The predicate that for a given (m, n) the quadratic function does not pass through the fourth quadrant
def valid_pair (m n : ℚ) : Prop :=
  does_not_pass_fourth_quadrant m n

-- All possible (m, n) pairs
def all_pairs : List (ℚ × ℚ) :=
  [( -1, -1), ( -1,  0), ( -1,  1/3),
   (  0, -1), (  0,  0), (  0,  1/3),
   (1/3, -1), (1/3,  0), (1/3, 1/3)]

-- Valid pairs count
def valid_pair_count : ℕ :=
  ((all_pairs.filter $ λ pair, valid_pair pair.fst pair.snd).length : ℕ)

-- Total pairs count
def total_pair_count : ℕ := (all_pairs.length : ℕ)

-- The probability the quadratic function does not pass through the fourth quadrant
def probability_does_not_pass_fourth_quadrant : ℚ :=
  (valid_pair_count : ℚ) / (total_pair_count : ℚ)

theorem probability_not_pass_fourth_quadrant :
  probability_does_not_pass_fourth_quadrant = 5 / 9 :=
by
  sorry

end probability_not_pass_fourth_quadrant_l262_262882


namespace series_sum_equals_one_l262_262467

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (2 : ℝ)^(2 * (k + 1)) / ((3 : ℝ)^(2 * (k + 1)) - 1)

theorem series_sum_equals_one :
  series_sum = 1 :=
sorry

end series_sum_equals_one_l262_262467


namespace dog_bones_initial_count_l262_262192

theorem dog_bones_initial_count (buried : ℝ) (final : ℝ) : buried = 367.5 → final = -860 → (buried + (final + 367.5) + 860) = 367.5 :=
by
  intros h1 h2
  sorry

end dog_bones_initial_count_l262_262192


namespace angle_Y_measure_l262_262883

def hexagon_interior_angle_sum (n : ℕ) : ℕ :=
  180 * (n - 2)

def supplementary (α β : ℕ) : Prop :=
  α + β = 180

def equal_angles (α β γ δ : ℕ) : Prop :=
  α = β ∧ β = γ ∧ γ = δ

theorem angle_Y_measure :
  ∀ (C H E S1 S2 Y : ℕ),
    C = E ∧ E = S1 ∧ S1 = Y →
    supplementary H S2 →
    hexagon_interior_angle_sum 6 = C + H + E + S1 + S2 + Y →
    Y = 135 :=
by
  intros C H E S1 S2 Y h1 h2 h3
  sorry

end angle_Y_measure_l262_262883


namespace closest_perfect_square_to_350_l262_262264

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l262_262264


namespace find_a_b_l262_262029

theorem find_a_b (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a * x^2 + b) 
  (h2 : ∀ x, f' x = 3 * x^2 + 2 * a * x) 
  (h3 : f' 1 = -3) 
  (h4 : f 1 = 0) : 
  a = -3 ∧ b = 2 := 
by
  sorry

end find_a_b_l262_262029


namespace exp_add_l262_262375

theorem exp_add (z w : Complex) : Complex.exp z * Complex.exp w = Complex.exp (z + w) := 
by 
  sorry

end exp_add_l262_262375


namespace greatest_two_digit_with_product_12_l262_262800

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l262_262800


namespace log15_12_eq_l262_262477

-- Goal: Define the constants and statement per the identified conditions and goal
variable (a b : ℝ)
#check Real.log
#check Real.logb

-- Math conditions
def lg2_eq_a := Real.log 2 = a
def lg3_eq_b := Real.log 3 = b

-- Math proof problem statement
theorem log15_12_eq : lg2_eq_a a → lg3_eq_b b → Real.logb 15 12 = (2 * a + b) / (1 - a + b) :=
by intros h1 h2; sorry

end log15_12_eq_l262_262477


namespace karlson_max_eat_chocolates_l262_262069

noncomputable def maximum_chocolates_eaten : ℕ :=
  34 * (34 - 1) / 2

theorem karlson_max_eat_chocolates : maximum_chocolates_eaten = 561 := by
  sorry

end karlson_max_eat_chocolates_l262_262069


namespace arrangement_of_students_l262_262849

theorem arrangement_of_students :
  let total_students := 5
  let total_communities := 2
  (2 ^ total_students - 2) = 30 :=
by
  let total_students := 5
  let total_communities := 2
  sorry

end arrangement_of_students_l262_262849


namespace cos_beta_l262_262999

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = 3 / 5)
variable (h2 : Real.cos (α + β) = 5 / 13)

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 3 / 5) (h2 : Real.cos (α + β) = 5 / 13) : 
  Real.cos β = 56 / 65 := by
  sorry

end cos_beta_l262_262999


namespace simplify_expression_l262_262238

theorem simplify_expression : (1 / (-8^2)^4) * (-8)^9 = -8 := 
by
  sorry

end simplify_expression_l262_262238


namespace evaluate_expression_l262_262236

theorem evaluate_expression :
  (1 / (-8^2)^4) * (-8)^9 = -8 := by
  sorry

end evaluate_expression_l262_262236


namespace olivia_total_payment_l262_262205

theorem olivia_total_payment : 
  (4 / 4 + 12 / 4 = 4) :=
by
  sorry

end olivia_total_payment_l262_262205


namespace cubic_sum_l262_262349

theorem cubic_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end cubic_sum_l262_262349


namespace intersection_complement_l262_262188

def set_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def set_B := {x : ℝ | x < 1}
def complement_B := {x : ℝ | x ≥ 1}

theorem intersection_complement :
  (set_A ∩ complement_B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_complement_l262_262188


namespace find_w_l262_262417

theorem find_w {w : ℝ} : (3, w^3) ∈ {p : ℝ × ℝ | ∃ x, p = (x, x^2 - 1)} → w = 2 :=
by
  sorry

end find_w_l262_262417


namespace train_leave_tunnel_l262_262439

noncomputable def train_leave_time 
  (train_speed : ℝ) 
  (tunnel_length : ℝ) 
  (train_length : ℝ) 
  (enter_time : ℝ × ℝ) : ℝ × ℝ :=
  let speed_km_min := train_speed / 60
  let total_distance := train_length + tunnel_length
  let time_to_pass := total_distance / speed_km_min
  let enter_minutes := enter_time.1 * 60 + enter_time.2
  let leave_minutes := enter_minutes + time_to_pass
  let leave_hours := leave_minutes / 60
  let leave_remainder_minutes := leave_minutes % 60
  (leave_hours, leave_remainder_minutes)

theorem train_leave_tunnel : 
  train_leave_time 80 70 1 (5, 12) = (6, 5.25) := 
sorry

end train_leave_tunnel_l262_262439


namespace other_candidate_votes_l262_262282

theorem other_candidate_votes (h1 : one_candidate_votes / valid_votes = 0.6)
    (h2 : 0.3 * total_votes = invalid_votes)
    (h3 : total_votes = 9000)
    (h4 : valid_votes + invalid_votes = total_votes) :
    valid_votes - one_candidate_votes = 2520 :=
by
  sorry

end other_candidate_votes_l262_262282


namespace percent_carnations_l262_262938

theorem percent_carnations (F : ℕ) (H1 : 3 / 5 * F = pink) (H2 : 1 / 5 * F = white) 
(H3 : F - pink - white = red) (H4 : 1 / 2 * pink = pink_roses)
(H5 : pink - pink_roses = pink_carnations) (H6 : 1 / 2 * red = red_carnations)
(H7 : white = white_carnations) : 
100 * (pink_carnations + red_carnations + white_carnations) / F = 60 :=
sorry

end percent_carnations_l262_262938


namespace count_perfect_square_factors_l262_262034

open Finset

noncomputable def count_divisible_by (n : ℕ) (s : Finset ℕ) : ℕ :=
s.filter (λ x, x % n = 0).card

theorem count_perfect_square_factors :
  let S := (range 100).map (λ n, n + 1)
  let perfect_squares := [4, 9, 16, 25, 36, 49, 64, 81, 100] 
  let total := S.card 
  let count_4 := count_divisible_by 4 S
  let count_9 := count_divisible_by 9 S
  let count_16 := count_divisible_by 16 S
  let count_25 := count_divisible_by 25 S
  let count_36 := count_divisible_by 36 S
  let count_49 := count_divisible_by 49 S
  let count_64 := count_divisible_by 64 S
  let count_81 := count_divisible_by 81 S
  let count_100 := count_divisible_by 100 S
  count_4 + (count_9 - count_divisible_by (Nat.lcm 4 9) S) +
    (count_16 - count_divisible_by 4 S) +
    (count_25 - count_divisible_by (Nat.lcm 4 25) S) +
    (count_36 - count_divisible_by 4 S) +
    count_49 + (count_64 - count_divisible_by 4 S) +
    (count_81 - count_divisible_by (Nat.lcm 9 81) S) +
    (count_100 - count_divisible_by 4 S)
    = 40 := 
by
  sorry

end count_perfect_square_factors_l262_262034


namespace sqrt_of_4_l262_262397

theorem sqrt_of_4 : {x : ℤ | x^2 = 4} = {2, -2} :=
by
  sorry

end sqrt_of_4_l262_262397


namespace problem_lean_statement_l262_262515

open Real

noncomputable def f (ω ϕ x : ℝ) : ℝ := 2 * sin (ω * x + ϕ)

theorem problem_lean_statement (ω ϕ : ℝ) (hω : 0 < ω) (hϕ : abs ϕ < π)
  (h1 : f ω ϕ (5 * π / 8) = 2)
  (h2 : f ω ϕ (11 * π / 8) = 0)
  (h3 : 2 * π / ω > 2 * π) : ω = 2 / 3 ∧ ϕ = π / 12 :=
by
  sorry

end problem_lean_statement_l262_262515


namespace solve_for_x_l262_262212

theorem solve_for_x
  (n m x : ℕ)
  (h1 : 7 / 8 = n / 96)
  (h2 : 7 / 8 = (m + n) / 112)
  (h3 : 7 / 8 = (x - m) / 144) :
  x = 140 :=
by
  sorry

end solve_for_x_l262_262212


namespace min_value_sin_cos_l262_262605

open Real

theorem min_value_sin_cos : ∃ x : ℝ, sin x * cos x = -1 / 2 := by
  sorry

end min_value_sin_cos_l262_262605


namespace rank_from_left_l262_262300

theorem rank_from_left (total_students : ℕ) (rank_from_right : ℕ) (h1 : total_students = 20) (h2 : rank_from_right = 13) : 
  (total_students - rank_from_right + 1 = 8) :=
by
  sorry

end rank_from_left_l262_262300


namespace compound_interest_rate_l262_262292

-- Defining the principal amount and total repayment
def P : ℝ := 200
def A : ℝ := 220

-- The annual compound interest rate
noncomputable def annual_compound_interest_rate (P A : ℝ) (n : ℕ) : ℝ :=
  (A / P)^(1 / n) - 1

-- Introducing the conditions
axiom compounded_annually : ∀ (P A : ℝ), annual_compound_interest_rate P A 1 = 0.1

-- Stating the theorem
theorem compound_interest_rate :
  annual_compound_interest_rate P A 1 = 0.1 :=
by {
  exact compounded_annually P A
}

end compound_interest_rate_l262_262292


namespace mike_total_spending_l262_262852

def mike_spent_on_speakers : ℝ := 235.87
def mike_spent_on_tires : ℝ := 281.45
def mike_spent_on_steering_wheel_cover : ℝ := 179.99
def mike_spent_on_seat_covers : ℝ := 122.31
def mike_spent_on_headlights : ℝ := 98.63

theorem mike_total_spending :
  mike_spent_on_speakers + mike_spent_on_tires + mike_spent_on_steering_wheel_cover + mike_spent_on_seat_covers + mike_spent_on_headlights = 918.25 :=
  sorry

end mike_total_spending_l262_262852


namespace at_least_one_ge_two_l262_262713

theorem at_least_one_ge_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 :=
sorry

end at_least_one_ge_two_l262_262713


namespace increasing_function_implies_a_nonpositive_l262_262478

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem increasing_function_implies_a_nonpositive (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) → a ≤ 0 :=
by
  sorry

end increasing_function_implies_a_nonpositive_l262_262478


namespace sqrt_of_4_l262_262398

theorem sqrt_of_4 : {x : ℤ | x^2 = 4} = {2, -2} :=
by
  sorry

end sqrt_of_4_l262_262398


namespace passing_grade_fraction_l262_262048

variables (students : ℕ) -- total number of students in Mrs. Susna's class

-- Conditions
def fraction_A : ℚ := 1/4
def fraction_B : ℚ := 1/2
def fraction_C : ℚ := 1/8
def fraction_D : ℚ := 1/12
def fraction_F : ℚ := 1/24

-- Prove the fraction of students getting a passing grade (C or higher) is 7/8
theorem passing_grade_fraction : 
  fraction_A + fraction_B + fraction_C = 7/8 :=
by
  sorry

end passing_grade_fraction_l262_262048


namespace ratio_of_width_to_length_is_correct_l262_262169

-- Define the given conditions
def length := 10
def perimeter := 36

-- Define the width and the expected ratio
def width (l P : Nat) : Nat := (P - 2 * l) / 2
def ratio (w l : Nat) := w / l

-- Statement to prove that given the conditions, the ratio of width to length is 4/5
theorem ratio_of_width_to_length_is_correct :
  ratio (width length perimeter) length = 4 / 5 :=
by
  sorry

end ratio_of_width_to_length_is_correct_l262_262169


namespace number_of_real_roots_l262_262603

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then 2010 * x + Real.log x / Real.log 2010
  else if x < 0 then - (2010 * (-x) + Real.log (-x) / Real.log 2010)
  else 0

theorem number_of_real_roots : 
  (∃ x1 x2 x3 : ℝ, 
    f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) ∧ 
    ∀ x y z : ℝ, 
    (f x = 0 ∧ f y = 0 ∧ f z = 0 → 
    (x = y ∨ x = z ∨ y = z)) 
  :=
by
  sorry

end number_of_real_roots_l262_262603


namespace xuzhou_test_2014_l262_262173

variables (A B C D : ℝ) -- Assume A, B, C, D are real numbers.

theorem xuzhou_test_2014 :
  (C < D) → (A > B) :=
sorry

end xuzhou_test_2014_l262_262173


namespace min_value_x2_y2_z2_l262_262369

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^2 + y^2 + z^2 - 3 * x * y * z = 1) :
  x^2 + y^2 + z^2 ≥ 1 := 
sorry

end min_value_x2_y2_z2_l262_262369


namespace abs_k_eq_sqrt_19_div_4_l262_262313

theorem abs_k_eq_sqrt_19_div_4
  (k : ℝ)
  (h : ∀ x : ℝ, x^2 - 4 * k * x + 1 = 0 → (x = r ∨ x = s))
  (h₁ : r + s = 4 * k)
  (h₂ : r * s = 1)
  (h₃ : r^2 + s^2 = 17) :
  |k| = (Real.sqrt 19) / 4 := by
sorry

end abs_k_eq_sqrt_19_div_4_l262_262313


namespace monotonically_increasing_intervals_sin_value_l262_262028

noncomputable def f (x : Real) : Real := 2 * Real.cos (x - Real.pi / 3) * Real.cos x + 1

theorem monotonically_increasing_intervals :
  ∀ (k : Int), ∃ (a b : Real), a = k * Real.pi - Real.pi / 3 ∧ b = k * Real.pi + Real.pi / 6 ∧
                 ∀ (x y : Real), a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y :=
sorry

theorem sin_value 
  (α : Real) (hα : 0 < α ∧ α < Real.pi / 2) 
  (h : f (α + Real.pi / 12) = 7 / 6) : 
  Real.sin (7 * Real.pi / 6 - 2 * α) = 2 * Real.sqrt 2 / 3 :=
sorry

end monotonically_increasing_intervals_sin_value_l262_262028


namespace y_value_when_x_is_20_l262_262049

theorem y_value_when_x_is_20 :
  ∀ (x : ℝ), (∀ m c : ℝ, m = 2.5 → c = 3 → (y = m * x + c) → x = 20 → y = 53) :=
by
  sorry

end y_value_when_x_is_20_l262_262049


namespace repeating_decimal_to_fraction_l262_262842

theorem repeating_decimal_to_fraction :
  let x := (0.7 : ℝ) + Real.repeat' 8 9 in
  x = 781 / 990 :=
by
  sorry

end repeating_decimal_to_fraction_l262_262842


namespace closest_perfect_square_to_350_l262_262259

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l262_262259


namespace janet_saving_l262_262055

def tile_cost_difference_saving : ℕ :=
  let turquoise_cost_per_tile := 13
  let purple_cost_per_tile := 11
  let area_wall1 := 5 * 8
  let area_wall2 := 7 * 8
  let total_area := area_wall1 + area_wall2
  let tiles_per_square_foot := 4
  let number_of_tiles := total_area * tiles_per_square_foot
  let cost_difference_per_tile := turquoise_cost_per_tile - purple_cost_per_tile
  number_of_tiles * cost_difference_per_tile

theorem janet_saving : tile_cost_difference_saving = 768 := by
  sorry

end janet_saving_l262_262055


namespace Olivia_pays_4_dollars_l262_262202

-- Definitions based on the conditions
def quarters_chips : ℕ := 4
def quarters_soda : ℕ := 12
def conversion_rate : ℕ := 4

-- Prove that the total dollars Olivia pays is 4
theorem Olivia_pays_4_dollars (h1 : quarters_chips = 4) (h2 : quarters_soda = 12) (h3 : conversion_rate = 4) : 
  (quarters_chips + quarters_soda) / conversion_rate = 4 :=
by
  -- skipping the proof
  sorry

end Olivia_pays_4_dollars_l262_262202


namespace carrie_spent_l262_262958

-- Define the cost of one t-shirt
def cost_per_tshirt : ℝ := 9.65

-- Define the number of t-shirts bought
def num_tshirts : ℝ := 12

-- Define the total cost function
def total_cost (cost_per_tshirt : ℝ) (num_tshirts : ℝ) : ℝ := cost_per_tshirt * num_tshirts

-- State the theorem which we need to prove
theorem carrie_spent :
  total_cost cost_per_tshirt num_tshirts = 115.80 :=
by
  sorry

end carrie_spent_l262_262958


namespace symmetric_points_origin_l262_262015

theorem symmetric_points_origin (a b : ℤ) (h1 : a = -5) (h2 : b = -1) : a - b = -4 :=
by
  sorry

end symmetric_points_origin_l262_262015


namespace valid_votes_other_candidate_l262_262281

theorem valid_votes_other_candidate (total_votes : ℕ)
  (invalid_percentage valid_percentage candidate1_percentage candidate2_percentage : ℕ)
  (h_invalid_valid_sum : invalid_percentage + valid_percentage = 100)
  (h_candidates_sum : candidate1_percentage + candidate2_percentage = 100)
  (h_invalid_percentage : invalid_percentage = 20)
  (h_candidate1_percentage : candidate1_percentage = 55)
  (h_total_votes : total_votes = 7500)
  (h_valid_percentage_eq : valid_percentage = 100 - invalid_percentage)
  (h_candidate2_percentage_eq : candidate2_percentage = 100 - candidate1_percentage) :
  ( ( candidate2_percentage * ( valid_percentage * total_votes / 100) ) / 100 ) = 2700 :=
  sorry

end valid_votes_other_candidate_l262_262281


namespace isosceles_triangle_base_length_l262_262088

theorem isosceles_triangle_base_length (b : ℕ) (h₁ : 6 + 6 + b = 20) : b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l262_262088


namespace greatest_two_digit_product_is_12_l262_262779

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l262_262779


namespace barry_pretzels_l262_262953

theorem barry_pretzels (A S B : ℕ) (h1 : A = 3 * S) (h2 : S = B / 2) (h3 : A = 18) : B = 12 :=
  by
  sorry

end barry_pretzels_l262_262953


namespace length_of_shortest_side_30_60_90_l262_262220

theorem length_of_shortest_side_30_60_90 (x : ℝ) : 
  (∃ x : ℝ, (2 * x = 15)) → x = 15 / 2 :=
by
  sorry

end length_of_shortest_side_30_60_90_l262_262220


namespace closest_square_to_350_l262_262266

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l262_262266


namespace relationship_a_b_l262_262119

theorem relationship_a_b (a b : ℝ) :
  (∃ (P : ℝ × ℝ), P ∈ {Q : ℝ × ℝ | Q.snd = -3 * Q.fst + b} ∧
                   ∃ (R : ℝ × ℝ), R ∈ {S : ℝ × ℝ | S.snd = -a * S.fst + 3} ∧
                   R = (-P.snd, -P.fst)) →
  a = 1 / 3 ∧ b = -9 :=
by
  intro h
  sorry

end relationship_a_b_l262_262119


namespace sqrt_4_eq_pm2_l262_262404

theorem sqrt_4_eq_pm2 : {y : ℝ | y^2 = 4} = {2, -2} :=
by
  sorry

end sqrt_4_eq_pm2_l262_262404


namespace max_area_rectangle_l262_262433

theorem max_area_rectangle (l w : ℕ) (h : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
sorry

end max_area_rectangle_l262_262433


namespace dice_probability_sum_17_l262_262698

-- Problem: Prove the probability that the sum of the face-up integers is 17 when three standard 6-faced dice are rolled is 1/24.

def probability_sum_17 (dice_rolls : ℕ → ℕ) (n : ℕ) : ℝ :=
  let probability_6 := 1 / 6  in
  let probability_case_A := (6 * (probability_6^3))  -- Case where one die shows 6 and other two sum to 11
  let probability_case_B := (3 * (probability_6^3))  -- Case where two dice show 6 and third shows 5
  probability_case_A + probability_case_B

theorem dice_probability_sum_17 : probability_sum_17 = 1 / 24 :=
by
  sorry

end dice_probability_sum_17_l262_262698


namespace sin_half_alpha_plus_beta_eq_sqrt2_div_2_l262_262635

open Real

theorem sin_half_alpha_plus_beta_eq_sqrt2_div_2
  (α β : ℝ)
  (hα : α ∈ Set.Icc (π / 2) (3 * π / 2))
  (hβ : β ∈ Set.Icc (-π / 2) 0)
  (h1 : (α - π / 2)^3 - sin α - 2 = 0)
  (h2 : 8 * β^3 + 2 * (cos β)^2 + 1 = 0) :
  sin (α / 2 + β) = sqrt 2 / 2 := 
sorry

end sin_half_alpha_plus_beta_eq_sqrt2_div_2_l262_262635


namespace compute_special_op_l262_262986

-- Define the operation ※
def special_op (m n : ℚ) := (3 * m + n) * (3 * m - n) + n

-- Hypothesis for specific m and n
def m := (1 : ℚ) / 6
def n := (-1 : ℚ)

-- Proof goal
theorem compute_special_op : special_op m n = -7 / 4 := by
  sorry

end compute_special_op_l262_262986


namespace greatest_two_digit_product_12_l262_262791

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l262_262791


namespace sum_of_first_twelve_multiples_of_18_l262_262804

-- Given conditions
def sum_of_first_n_positives (n : ℕ) : ℕ := n * (n + 1) / 2

def first_twelve_multiples_sum (k : ℕ) : ℕ := k * (sum_of_first_n_positives 12)

-- The question to prove
theorem sum_of_first_twelve_multiples_of_18 : first_twelve_multiples_sum 18 = 1404 :=
by
  sorry

end sum_of_first_twelve_multiples_of_18_l262_262804


namespace total_bill_calculation_l262_262577

theorem total_bill_calculation (n : ℕ) (amount_per_person : ℝ) (total_amount : ℝ) :
  n = 9 → amount_per_person = 514.19 → total_amount = 4627.71 → 
  n * amount_per_person = total_amount :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_bill_calculation_l262_262577


namespace cosine_54_deg_l262_262456

theorem cosine_54_deg : ∃ c : ℝ, c = cos (54 : ℝ) ∧ c = 1 / 2 :=
  by 
    let c := cos (54 : ℝ)
    let d := cos (108 : ℝ)
    have h1 : d = 2 * c^2 - 1 := sorry
    have h2 : d = -c := sorry
    have h3 : 2 * c^2 + c - 1 = 0 := sorry
    use 1 / 2 
    have h4 : c = 1 / 2 := sorry
    exact ⟨cos_eq_cos_of_eq_rad 54 1, h4⟩

end cosine_54_deg_l262_262456


namespace parallelogram_side_length_l262_262050

theorem parallelogram_side_length (x y : ℚ) (h1 : 3 * x + 2 = 12) (h2 : 5 * y - 3 = 9) : x + y = 86 / 15 :=
by 
  sorry

end parallelogram_side_length_l262_262050


namespace problem_statement_l262_262840

def star (x y : Nat) : Nat :=
  match x, y with
  | 1, 1 => 4 | 1, 2 => 3 | 1, 3 => 2 | 1, 4 => 1
  | 2, 1 => 1 | 2, 2 => 4 | 2, 3 => 3 | 2, 4 => 2
  | 3, 1 => 2 | 3, 2 => 1 | 3, 3 => 4 | 3, 4 => 3
  | 4, 1 => 3 | 4, 2 => 2 | 4, 3 => 1 | 4, 4 => 4
  | _, _ => 0  -- This line handles unexpected inputs.

theorem problem_statement : star (star 3 2) (star 2 1) = 4 := by
  sorry

end problem_statement_l262_262840


namespace inequality_holds_l262_262004

theorem inequality_holds (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x * y + y * z) := 
by 
  sorry

end inequality_holds_l262_262004


namespace select_pencils_l262_262501

theorem select_pencils (boxes : Fin 10 → ℕ) (colors : ∀ (i : Fin 10), Fin (boxes i) → Fin 10) :
  (∀ i : Fin 10, 1 ≤ boxes i) → -- Each box is non-empty
  (∀ i j : Fin 10, i ≠ j → boxes i ≠ boxes j) → -- Different number of pencils in each box
  ∃ (selection : Fin 10 → Fin 10), -- Function to select a pencil color from each box
  Function.Injective selection := -- All selected pencils have different colors
sorry

end select_pencils_l262_262501


namespace proof_problem_l262_262365

variable (a b c : ℝ)
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_prod : a * b * c = 1)
variable (h_ineq : a^2011 + b^2011 + c^2011 < (1 / a)^2011 + (1 / b)^2011 + (1 / c)^2011)

theorem proof_problem : a + b + c < 1 / a + 1 / b + 1 / c := 
  sorry

end proof_problem_l262_262365


namespace truncated_pyramid_smaller_base_area_l262_262913

noncomputable def smaller_base_area (a : ℝ) (α β : ℝ) : ℝ :=
  (a^2 * (Real.sin (α - β))^2) / (Real.sin (α + β))^2

theorem truncated_pyramid_smaller_base_area (a α β : ℝ) :
  smaller_base_area a α β = (a^2 * (Real.sin (α - β))^2) / (Real.sin (α + β))^2 :=
by
  unfold smaller_base_area
  sorry

end truncated_pyramid_smaller_base_area_l262_262913


namespace hyuksu_total_meat_l262_262038

/-- 
Given that Hyuksu ate 2.6 kilograms (kg) of meat yesterday and 5.98 kilograms (kg) of meat today,
prove that the total kilograms (kg) of meat he ate in two days is 8.58 kg.
-/
theorem hyuksu_total_meat (yesterday today : ℝ) (hy1 : yesterday = 2.6) (hy2 : today = 5.98) :
  yesterday + today = 8.58 := 
by
  rw [hy1, hy2]
  norm_num

end hyuksu_total_meat_l262_262038


namespace probability_meeting_is_approx_0_02_l262_262897
  
theorem probability_meeting_is_approx_0_02 :
  let A_paths := 2^6 in
  let B_paths := 2^6 in
  let meet_prob :=
    (∑ i in Finset.range 11, (Nat.choose 6 i) * (Nat.choose 6 (10 - i))) / (A_paths * B_paths)
  abs (meet_prob - 0.02) < 0.01 :=
by
  sorry  

end probability_meeting_is_approx_0_02_l262_262897


namespace a37_b37_sum_l262_262032

-- Declare the sequences as functions from natural numbers to real numbers
variables {a b : ℕ → ℝ}

-- State the hypotheses based on the conditions
variables (h1 : ∀ n, a (n + 1) = a n + a 2 - a 1)
variables (h2 : ∀ n, b (n + 1) = b n + b 2 - b 1)
variables (h3 : a 1 = 25)
variables (h4 : b 1 = 75)
variables (h5 : a 2 + b 2 = 100)

-- State the theorem to be proved
theorem a37_b37_sum : a 37 + b 37 = 100 := 
by 
  sorry

end a37_b37_sum_l262_262032


namespace vasya_cuts_larger_area_l262_262451

noncomputable def E_Vasya_square_area : ℝ :=
  (1/6) * (1^2) + (1/6) * (2^2) + (1/6) * (3^2) + (1/6) * (4^2) + (1/6) * (5^2) + (1/6) * (6^2)

noncomputable def E_Asya_rectangle_area : ℝ :=
  (3.5 * 3.5)

theorem vasya_cuts_larger_area :
  E_Vasya_square_area > E_Asya_rectangle_area :=
  by
    sorry

end vasya_cuts_larger_area_l262_262451


namespace total_right_handed_players_is_correct_l262_262522

variable (total_players : ℕ)
variable (throwers : ℕ)
variable (left_handed_non_throwers_ratio : ℕ)
variable (total_right_handed_players : ℕ)

theorem total_right_handed_players_is_correct
  (h1 : total_players = 61)
  (h2 : throwers = 37)
  (h3 : left_handed_non_throwers_ratio = 1 / 3)
  (h4 : total_right_handed_players = 53) :
  total_right_handed_players = throwers + (total_players - throwers) -
    left_handed_non_throwers_ratio * (total_players - throwers) :=
by
  sorry

end total_right_handed_players_is_correct_l262_262522


namespace orchestra_members_l262_262540

theorem orchestra_members (n : ℕ) (h₀ : 100 ≤ n) (h₁ : n ≤ 300)
    (h₂ : n % 4 = 3) (h₃ : n % 5 = 1) (h₄ : n % 7 = 5) : n = 231 := by
  sorry

end orchestra_members_l262_262540


namespace gnollish_valid_sentences_l262_262080

def valid_sentences_count : ℕ :=
  let words := ["splargh", "glumph", "amr", "krack"]
  let total_words := 4
  let total_sentences := total_words ^ 3
  let invalid_splargh_glumph := 2 * total_words
  let invalid_amr_krack := 2 * total_words
  let total_invalid := invalid_splargh_glumph + invalid_amr_krack
  total_sentences - total_invalid

theorem gnollish_valid_sentences : valid_sentences_count = 48 :=
by
  sorry

end gnollish_valid_sentences_l262_262080


namespace product_ge_half_l262_262226

theorem product_ge_half (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3) (h_sum : x1 + x2 + x3 ≤ 1/2) :
  (1 - x1) * (1 - x2) * (1 - x3) ≥ 1/2 :=
by
  sorry

end product_ge_half_l262_262226


namespace proportion_decrease_l262_262383

open Real

/-- 
Given \(x\) and \(y\) are directly proportional and positive,
if \(x\) decreases by \(q\%\), then \(y\) decreases by \(q\%\).
-/
theorem proportion_decrease (c x q : ℝ) (h_pos : x > 0) (h_q_pos : q > 0)
    (h_direct : ∀ x y, y = c * x) :
    ((x * (1 - q / 100)) = y) → ((y * (1 - q / 100)) = (c * x * (1 - q / 100))) := by
  sorry

end proportion_decrease_l262_262383


namespace triangle_angle_inequality_l262_262207

open Real

theorem triangle_angle_inequality (α β γ α₁ β₁ γ₁ : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : α₁ + β₁ + γ₁ = π) :
  (cos α₁ / sin α) + (cos β₁ / sin β) + (cos γ₁ / sin γ) 
  ≤ (cos α / sin α) + (cos β / sin β) + (cos γ / sin γ) :=
sorry

end triangle_angle_inequality_l262_262207


namespace janice_purchase_l262_262177

theorem janice_purchase : 
  ∃ (a b c : ℕ), a + b + c = 50 ∧ 50 * a + 400 * b + 500 * c = 10000 ∧ a = 23 :=
by
  sorry

end janice_purchase_l262_262177


namespace base_length_of_isosceles_triangle_l262_262086

-- Definitions based on given conditions
def is_isosceles (a b : ℕ) (c : ℕ) :=
a = b ∧ c = c

def side_length : ℕ := 6
def perimeter : ℕ := 20

-- Theorem to prove the base length
theorem base_length_of_isosceles_triangle (b : ℕ) (h1 : 2 * side_length + b = perimeter) :
  b = 8 :=
sorry

end base_length_of_isosceles_triangle_l262_262086


namespace problem_l262_262851

-- Define the polynomial g(x) with given coefficients
def g (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x^2 + x + 8

-- Define the polynomial f(x) with given coefficients
def f (x : ℝ) (a b c : ℝ) : ℝ :=
  x^4 + x^3 + b * x^2 + 50 * x + c

-- Define the conditions
def conditions (a b c r : ℝ) : Prop :=
  ∃ roots : Finset ℝ, (∀ x ∈ roots, g x a = 0) ∧ (∀ x ∈ roots, f x a b c = 0) ∧ (roots.card = 3) ∧
  (8 - r = 50) ∧ (a - r = 1) ∧ (1 - a * r = b) ∧ (-8 * r = c)

-- Define the theorem to be proved
theorem problem (a b c r : ℝ) (h : conditions a b c r) : f 1 a b c = -1333 :=
by sorry

end problem_l262_262851


namespace dodecahedron_interior_diagonals_l262_262666

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l262_262666


namespace sum_of_first_60_digits_l262_262925

noncomputable def decimal_expansion_period : List ℕ := [0, 0, 0, 8, 1, 0, 3, 7, 2, 7, 7, 1, 4, 7, 4, 8, 7, 8, 4, 4, 4, 0, 8, 4, 2, 7, 8, 7, 6, 8]

def sum_of_list (l : List ℕ) : ℕ := l.foldl (· + ·) 0

theorem sum_of_first_60_digits : sum_of_list (decimal_expansion_period ++ decimal_expansion_period) = 282 := 
by
  simp [decimal_expansion_period, sum_of_list]
  sorry

end sum_of_first_60_digits_l262_262925


namespace part1_part2_l262_262382

variable {a b c : ℝ}

theorem part1 (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) : 
  a * b + b * c + a * c ≤ 1 / 3 := 
sorry 

theorem part2 (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) : 
  a^2 / b + b^2 / c + c^2 / a ≥ 1 := 
sorry

end part1_part2_l262_262382


namespace S6_value_l262_262929

noncomputable def S_m (x : ℝ) (m : ℕ) : ℝ := x^m + (1/x)^m

theorem S6_value (x : ℝ) (h : x + 1/x = 4) : S_m x 6 = 2700 :=
by
  -- Skipping proof
  sorry

end S6_value_l262_262929


namespace jose_profit_share_l262_262284

theorem jose_profit_share (investment_tom : ℕ) (months_tom : ℕ) 
                         (investment_jose : ℕ) (months_jose : ℕ) 
                         (total_profit : ℕ) :
                         investment_tom = 30000 →
                         months_tom = 12 →
                         investment_jose = 45000 →
                         months_jose = 10 →
                         total_profit = 63000 →
                         (investment_jose * months_jose / 
                         (investment_tom * months_tom + investment_jose * months_jose)) * total_profit = 35000 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  norm_num
  sorry

end jose_profit_share_l262_262284


namespace population_reaches_target_l262_262452

def initial_year : ℕ := 2020
def initial_population : ℕ := 450
def growth_period : ℕ := 25
def growth_factor : ℕ := 3
def target_population : ℕ := 10800

theorem population_reaches_target : ∃ (year : ℕ), year - initial_year = 3 * growth_period ∧ (initial_population * growth_factor ^ 3) >= target_population := by
  sorry

end population_reaches_target_l262_262452


namespace simplify_complex_expression_l262_262241

theorem simplify_complex_expression :
  (1 / (-8 ^ 2) ^ 4 * (-8) ^ 9) = -8 := by
  sorry

end simplify_complex_expression_l262_262241


namespace smallest_solution_l262_262848

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l262_262848


namespace no_four_consecutive_perf_square_l262_262963

theorem no_four_consecutive_perf_square :
  ¬ ∃ (x : ℕ), x > 0 ∧ ∃ (k : ℕ), x * (x + 1) * (x + 2) * (x + 3) = k^2 :=
by
  sorry

end no_four_consecutive_perf_square_l262_262963


namespace no_solution_fraction_eq_l262_262042

theorem no_solution_fraction_eq (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (a * x / (x - 1) + 3 / (1 - x) = 2) → false) ↔ a = 2 :=
by
  sorry

end no_solution_fraction_eq_l262_262042


namespace final_net_earnings_l262_262211

-- Declare constants representing the problem conditions
def connor_hourly_rate : ℝ := 7.20
def connor_hours_worked : ℝ := 8.0
def emily_hourly_rate : ℝ := 2 * connor_hourly_rate
def sarah_hourly_rate : ℝ := 5 * connor_hourly_rate
def emily_hours_worked : ℝ := 10.0
def connor_deduction_rate : ℝ := 0.05
def emily_deduction_rate : ℝ := 0.08
def sarah_deduction_rate : ℝ := 0.10

-- Combined final net earnings for the day
def combined_final_net_earnings (connor_hourly_rate emily_hourly_rate sarah_hourly_rate
                                  connor_hours_worked emily_hours_worked
                                  connor_deduction_rate emily_deduction_rate sarah_deduction_rate : ℝ) : ℝ :=
  let connor_gross := connor_hourly_rate * connor_hours_worked
  let emily_gross := emily_hourly_rate * emily_hours_worked
  let sarah_gross := sarah_hourly_rate * connor_hours_worked
  let connor_net := connor_gross * (1 - connor_deduction_rate)
  let emily_net := emily_gross * (1 - emily_deduction_rate)
  let sarah_net := sarah_gross * (1 - sarah_deduction_rate)
  connor_net + emily_net + sarah_net

-- The theorem statement proving their combined final net earnings
theorem final_net_earnings : 
  combined_final_net_earnings 7.20 14.40 36.00 8.0 10.0 0.05 0.08 0.10 = 498.24 :=
by sorry

end final_net_earnings_l262_262211


namespace intersection_point_l262_262802

noncomputable def line1 (x : ℝ) : ℝ := 3 * x + 10

noncomputable def slope_perp : ℝ := -1/3

noncomputable def line_perp (x : ℝ) : ℝ := slope_perp * x + (2 - slope_perp * 3)

theorem intersection_point : 
  ∃ (x y : ℝ), y = line1 x ∧ y = line_perp x ∧ x = -21 / 10 ∧ y = 37 / 10 :=
by
  sorry

end intersection_point_l262_262802


namespace cone_prism_volume_ratio_correct_l262_262826

noncomputable def cone_prism_volume_ratio (π : ℝ) : ℝ :=
  let r := 1.5
  let h := 5
  let V_cone := (1 / 3) * π * r^2 * h
  let V_prism := 3 * 4 * h
  V_cone / V_prism

theorem cone_prism_volume_ratio_correct (π : ℝ) : 
  cone_prism_volume_ratio π = π / 4.8 :=
sorry

end cone_prism_volume_ratio_correct_l262_262826


namespace dodecahedron_interior_diagonals_l262_262665

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l262_262665


namespace count_numbers_1000_to_5000_l262_262488

def countFourDigitNumbersInRange (lower upper : ℕ) : ℕ :=
  if lower <= upper then upper - lower + 1 else 0

theorem count_numbers_1000_to_5000 : countFourDigitNumbersInRange 1000 5000 = 4001 :=
by
  sorry

end count_numbers_1000_to_5000_l262_262488


namespace B_is_Brownian_bridge_l262_262061

noncomputable def B (X : ℝ → ℝ) (t : ℝ) : ℝ :=
if (0 < t ∧ t < 1) then (Real.sqrt (t * (1 - t)) * X (1 / 2) * Real.log (t / (1 - t))) else 0

theorem B_is_Brownian_bridge (X : ℝ → ℝ) (hOU : is_Ornstein_Uhlenbeck X) :
  is_Brownian_bridge (B X) := sorry

end B_is_Brownian_bridge_l262_262061


namespace dodecahedron_interior_diagonals_eq_160_l262_262654

-- Definition of a dodecahedron for the context of the proof
structure Dodecahedron where
  num_faces : Nat
  num_vertices : Nat
  faces_meeting_at_vertex : Nat
  vertices_connected_by_edge : Nat

-- Given our specific dodecahedron properties
def dodecahedron : Dodecahedron := {
  num_faces := 12,
  num_vertices := 20,
  faces_meeting_at_vertex := 3,
  vertices_connected_by_edge := 3
}

-- The interior diagonal is defined as a segment connecting two vertices
-- which do not lie on the same face.
def is_interior_diagonal (d : Dodecahedron) (v1 v2 : Fin d.num_vertices) : Prop :=
  v1 ≠ v2 ∧ ¬∃ f, f < d.num_faces ∧ v1 ∈ f ∧ v2 ∈ f

-- Now, we need to calculate the total number of such interior diagonals
noncomputable def num_interior_diagonals (d : Dodecahedron) : Nat :=
  d.num_vertices * (d.num_vertices - d.vertices_connected_by_edge - 1) / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals_eq_160 :
  num_interior_diagonals dodecahedron = 160 := by
  sorry

end dodecahedron_interior_diagonals_eq_160_l262_262654


namespace average_speed_difference_l262_262742

noncomputable def v_R : Float := 56.44102863722254
noncomputable def distance : Float := 750
noncomputable def t_R : Float := distance / v_R
noncomputable def t_P : Float := t_R - 2
noncomputable def v_P : Float := distance / t_P

theorem average_speed_difference : v_P - v_R = 10 := by
  sorry

end average_speed_difference_l262_262742


namespace bill_bought_60_rats_l262_262126

def chihuahuas_and_rats (C R : ℕ) : Prop :=
  C + R = 70 ∧ R = 6 * C

theorem bill_bought_60_rats (C R : ℕ) (h : chihuahuas_and_rats C R) : R = 60 :=
by
  sorry

end bill_bought_60_rats_l262_262126


namespace closest_perfect_square_to_350_l262_262252

theorem closest_perfect_square_to_350 : ∃ (n : ℕ), n^2 = 361 ∧ ∀ (m : ℕ), m^2 ≤ 350 ∨ 350 < m^2 → abs (361 - 350) ≤ abs (m^2 - 350) :=
by
  sorry

end closest_perfect_square_to_350_l262_262252


namespace simplify_f_value_of_f_l262_262323

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - (5 * Real.pi) / 2) * Real.cos ((3 * Real.pi) / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (Real.pi - α))

theorem simplify_f (α : ℝ) : f α = -Real.cos α := by
  sorry

theorem value_of_f (α : ℝ)
  (h : Real.cos (α + (3 * Real.pi) / 2) = 1 / 5)
  (h2 : α > Real.pi / 2 ∧ α < Real.pi ) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end simplify_f_value_of_f_l262_262323


namespace product_of_five_consecutive_integers_not_square_l262_262729

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (h : 0 < a) :
  ¬ is_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) := by
sorry

end product_of_five_consecutive_integers_not_square_l262_262729


namespace cubed_sum_identity_l262_262343

theorem cubed_sum_identity {x y : ℝ} (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubed_sum_identity_l262_262343


namespace probability_of_selection_l262_262470

theorem probability_of_selection (total_students : ℕ) (eliminated_students : ℕ) (groups : ℕ) (selected_students : ℕ)
(h1 : total_students = 1003) 
(h2 : eliminated_students = 3)
(h3 : groups = 20)
(h4 : selected_students = 50) : 
(selected_students : ℝ) / (total_students : ℝ) = 50 / 1003 :=
by
  sorry

end probability_of_selection_l262_262470


namespace square_pyramid_intersection_area_l262_262437

theorem square_pyramid_intersection_area (a b c d e : ℝ) (h_midpoints : a = 2 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ e = 4) : 
  ∃ p : ℝ, (p = 80) :=
by
  sorry

end square_pyramid_intersection_area_l262_262437


namespace arccos_neg_one_eq_pi_l262_262594

theorem arccos_neg_one_eq_pi : arccos (-1) = π := 
by
  sorry

end arccos_neg_one_eq_pi_l262_262594


namespace find_k_l262_262874

theorem find_k (k : ℝ) (h : ∀ x: ℝ, (x = -2) → (1 + k / (x - 1) = 0)) : k = 3 :=
by
  sorry

end find_k_l262_262874


namespace sin_2x_value_l262_262322

theorem sin_2x_value (x : ℝ) (h : Real.sin (π / 4 - x) = 1 / 3) : Real.sin (2 * x) = 7 / 9 := by
  sorry

end sin_2x_value_l262_262322


namespace probability_function_has_zero_point_l262_262491

noncomputable def probability_of_zero_point : ℚ :=
by
  let S := ({-1, 1, 2} : Finset ℤ).product ({-1, 1, 2} : Finset ℤ)
  let zero_point_pairs := S.filter (λ p => (p.1 * p.2 ≤ 1))
  let favorable_outcomes := zero_point_pairs.card
  let total_outcomes := S.card
  exact favorable_outcomes / total_outcomes

theorem probability_function_has_zero_point :
  probability_of_zero_point = (2 / 3 : ℚ) :=
  sorry

end probability_function_has_zero_point_l262_262491


namespace knights_can_attack_on_3x3_board_l262_262726

noncomputable def prob_knights_attacking (n : ℕ): ℚ :=
  if n = 3 then 209 / 256 else 0

theorem knights_can_attack_on_3x3_board :
  prob_knights_attacking 3 = 209 / 256 :=
by sorry

end knights_can_attack_on_3x3_board_l262_262726


namespace no_integer_in_interval_l262_262530

theorem no_integer_in_interval (n : ℕ) : ¬ ∃ k : ℤ, 
  (n ≠ 0 ∧ (n * Real.sqrt 2 - 1 / (3 * n) < k) ∧ (k < n * Real.sqrt 2 + 1 / (3 * n))) := 
sorry

end no_integer_in_interval_l262_262530


namespace distinct_triangles_in_regular_ngon_l262_262208

theorem distinct_triangles_in_regular_ngon (n : ℕ) (h : n ≥ 3) :
  ∃ t : ℕ, t = n * (n-1) * (n-2) / 6 := 
sorry

end distinct_triangles_in_regular_ngon_l262_262208


namespace ratio_female_to_male_members_l262_262835

theorem ratio_female_to_male_members (f m : ℕ)
  (h1 : 35 * f = SumAgesFemales)
  (h2 : 20 * m = SumAgesMales)
  (h3 : (35 * f + 20 * m) / (f + m) = 25) :
  f / m = 1 / 2 := by
  sorry

end ratio_female_to_male_members_l262_262835


namespace trace_bag_weight_l262_262748

-- Define the weights of Gordon's bags
def gordon_bag1_weight : ℕ := 3
def gordon_bag2_weight : ℕ := 7

-- Define the number of Trace's bags
def trace_num_bags : ℕ := 5

-- Define what we are trying to prove: the weight of one of Trace's shopping bags
theorem trace_bag_weight :
  (gordon_bag1_weight + gordon_bag2_weight) = (trace_num_bags * 2) :=
by
  sorry

end trace_bag_weight_l262_262748


namespace kitchen_supplies_sharon_wants_l262_262377

theorem kitchen_supplies_sharon_wants (P : ℕ) (plates_angela cutlery_angela pots_sharon plates_sharon cutlery_sharon : ℕ) 
  (h1 : plates_angela = 3 * P + 6) 
  (h2 : cutlery_angela = (3 * P + 6) / 2) 
  (h3 : pots_sharon = P / 2) 
  (h4 : plates_sharon = 3 * (3 * P + 6) - 20) 
  (h5 : cutlery_sharon = 2 * (3 * P + 6) / 2) 
  (h_total : pots_sharon + plates_sharon + cutlery_sharon = 254) : 
  P = 20 :=
sorry

end kitchen_supplies_sharon_wants_l262_262377


namespace roots_of_quadratic_l262_262911

theorem roots_of_quadratic (x : ℝ) : x^2 + x = 0 ↔ (x = 0 ∨ x = -1) :=
by sorry

end roots_of_quadratic_l262_262911


namespace dodecahedron_interior_diagonals_l262_262673

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l262_262673


namespace seven_digit_number_l262_262121

theorem seven_digit_number (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
(h1 : a_1 + a_2 = 9)
(h2 : a_2 + a_3 = 7)
(h3 : a_3 + a_4 = 9)
(h4 : a_4 + a_5 = 2)
(h5 : a_5 + a_6 = 8)
(h6 : a_6 + a_7 = 11)
(h_digits : ∀ (i : ℕ), i ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] → i < 10) :
a_1 = 9 ∧ a_2 = 0 ∧ a_3 = 7 ∧ a_4 = 2 ∧ a_5 = 0 ∧ a_6 = 8 ∧ a_7 = 3 :=
by sorry

end seven_digit_number_l262_262121


namespace zero_extreme_points_l262_262539

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x

theorem zero_extreme_points : ∀ x : ℝ, 
  ∃! (y : ℝ), deriv f y = 0 → y = x :=
by
  sorry

end zero_extreme_points_l262_262539


namespace initial_apples_l262_262075

theorem initial_apples (picked: ℕ) (newly_grown: ℕ) (still_on_tree: ℕ) (initial: ℕ):
  (picked = 7) →
  (newly_grown = 2) →
  (still_on_tree = 6) →
  (still_on_tree + picked - newly_grown = initial) →
  initial = 11 :=
by
  intros hpicked hnewly_grown hstill_on_tree hcalculation
  sorry

end initial_apples_l262_262075


namespace eliana_total_steps_l262_262609

-- Define the conditions given in the problem
def steps_first_day_exercise : Nat := 200
def steps_first_day_additional : Nat := 300
def steps_first_day : Nat := steps_first_day_exercise + steps_first_day_additional

def steps_second_day : Nat := 2 * steps_first_day
def steps_additional_on_third_day : Nat := 100
def steps_third_day : Nat := steps_second_day + steps_additional_on_third_day

-- Mathematical proof problem proving that the total number of steps is 2600
theorem eliana_total_steps : steps_first_day + steps_second_day + steps_third_day = 2600 := 
by
  sorry

end eliana_total_steps_l262_262609


namespace polar_to_rectangular_l262_262600

theorem polar_to_rectangular :
  ∀ (r θ : ℝ), r = 3 * Real.sqrt 2 → θ = (3 * Real.pi) / 4 → 
  (r * Real.cos θ, r * Real.sin θ) = (-3, 3) :=
by
  intro r θ hr hθ
  rw [hr, hθ]
  sorry

end polar_to_rectangular_l262_262600


namespace lattice_points_non_visible_square_l262_262320

theorem lattice_points_non_visible_square (n : ℕ) (h : n > 0) : 
  ∃ (a b : ℤ), ∀ (x y : ℤ), a < x ∧ x < a + n ∧ b < y ∧ y < b + n → Int.gcd x y > 1 :=
sorry

end lattice_points_non_visible_square_l262_262320


namespace no_four_consecutive_perf_square_l262_262964

theorem no_four_consecutive_perf_square :
  ¬ ∃ (x : ℕ), x > 0 ∧ ∃ (k : ℕ), x * (x + 1) * (x + 2) * (x + 3) = k^2 :=
by
  sorry

end no_four_consecutive_perf_square_l262_262964


namespace dodecahedron_interior_diagonals_l262_262658

theorem dodecahedron_interior_diagonals (vertices faces: ℕ) (pentagonal: ∀ face, face = 5) (vertices_per_face : ∀ face ∈ (range faces), 3) :
  vertices = 20 ∧ faces = 12 ∧ interior_diagonals = 160 :=
by
    -- Definitions and problem conditions
    let vertices := 20
    let faces := 12
    let pentagonal := (λ face, face = 5)
    let vertices_per_face := (λ face, 3)

    -- Assertion
    have interior_diagonals : ℕ := (vertices * 16) / 2
    rw Nat.div_eq_of_eq_mul_right _ _ _,
    have h: vertices = 20 ∧ faces = 12 ∧ interior_diagonals = 160 := by
        sorry

    exact h

end dodecahedron_interior_diagonals_l262_262658


namespace initial_pennies_in_each_compartment_l262_262076

theorem initial_pennies_in_each_compartment (x : ℕ) (h : 12 * (x + 6) = 96) : x = 2 :=
by sorry

end initial_pennies_in_each_compartment_l262_262076


namespace symmetric_points_origin_l262_262014

theorem symmetric_points_origin (a b : ℤ) (h1 : a = -5) (h2 : b = -1) : a - b = -4 :=
by
  sorry

end symmetric_points_origin_l262_262014


namespace wage_difference_l262_262955

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.2

theorem wage_difference : manager_wage - chef_wage = 3.40 := 
by
  sorry

end wage_difference_l262_262955


namespace total_jokes_sum_l262_262898

theorem total_jokes_sum :
  let jessy_week1 := 11
  let alan_week1 := 7
  let tom_week1 := 5
  let emily_week1 := 3
  let jessy_week4 := 11 * 3 ^ 3
  let alan_week4 := 7 * 2 ^ 3
  let tom_week4 := 5 * 4 ^ 3
  let emily_week4 := 3 * 4 ^ 3
  let jessy_total := 11 + 11 * 3 + 11 * 3 ^ 2 + jessy_week4
  let alan_total := 7 + 7 * 2 + 7 * 2 ^ 2 + alan_week4
  let tom_total := 5 + 5 * 4 + 5 * 4 ^ 2 + tom_week4
  let emily_total := 3 + 3 * 4 + 3 * 4 ^ 2 + emily_week4
  jessy_total + alan_total + tom_total + emily_total = 1225 :=
by 
  sorry

end total_jokes_sum_l262_262898


namespace dodecahedron_interior_diagonals_l262_262683

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l262_262683


namespace car_production_l262_262113

theorem car_production (mp : ℕ) (h1 : 1800 = (mp + 50) * 12) : mp = 100 :=
by
  sorry

end car_production_l262_262113


namespace imo_hosting_arrangements_l262_262380

structure IMOCompetition where
  countries : Finset String
  continents : Finset String
  assignments : Finset (String × String)
  constraints : String → String
  assignments_must_be_unique : ∀ {c1 c2 : String} {cnt1 cnt2 : String},
                                 (c1, cnt1) ∈ assignments → (c2, cnt2) ∈ assignments → 
                                 constraints c1 ≠ constraints c2 → c1 ≠ c2
  no_consecutive_same_continent : ∀ {c1 c2 : String} {cnt1 cnt2 : String},
                                   (c1, cnt1) ∈ assignments → (c2, cnt2) ∈ assignments → 
                                   (c1, cnt1) ≠ (c2, cnt2) →
                                   constraints c1 ≠ constraints c2

def number_of_valid_arrangements (comp: IMOCompetition) : Nat := 240

theorem imo_hosting_arrangements (comp : IMOCompetition) :
  number_of_valid_arrangements comp = 240 := by
  sorry

end imo_hosting_arrangements_l262_262380


namespace min_difference_xue_jie_ti_neng_li_l262_262052

theorem min_difference_xue_jie_ti_neng_li : 
  ∀ (shu hsue jie ti neng li zhan shi : ℕ), 
  shu = 8 ∧ hsue = 1 ∧ jie = 4 ∧ ti = 3 ∧ neng = 9 ∧ li = 5 ∧ zhan = 7 ∧ shi = 2 →
  (shu * 1000 + hsue * 100 + jie * 10 + ti) = 1842 →
  (neng * 10 + li) = 95 →
  1842 - 95 = 1747 := 
by
  intros shu hsue jie ti neng li zhan shi h_digits h_xue_jie_ti h_neng_li
  sorry

end min_difference_xue_jie_ti_neng_li_l262_262052


namespace initial_ratio_of_liquids_l262_262426

theorem initial_ratio_of_liquids (A B : ℕ) (H1 : A = 21)
  (H2 : 9 * A = 7 * (B + 9)) :
  A / B = 7 / 6 :=
sorry

end initial_ratio_of_liquids_l262_262426


namespace geometric_sequence_sum_l262_262054

variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, (r > 0) ∧ (∀ n : ℕ, a (n + 1) = a n * r)

theorem geometric_sequence_sum
  (a_seq_geometric : is_geometric_sequence a)
  (a_pos : ∀ n : ℕ, a n > 0)
  (eqn : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) :
  a 4 + a 6 = 10 :=
by
  sorry

end geometric_sequence_sum_l262_262054


namespace problem1_problem2_problem3_l262_262513

-- Definitions of sets A, B, and C as per given conditions
def set_A (a : ℝ) : Set ℝ :=
  {x | x^2 - a * x + a^2 - 19 = 0}

def set_B : Set ℝ :=
  {x | x^2 - 5 * x + 6 = 0}

def set_C : Set ℝ :=
  {x | x^2 + 2 * x - 8 = 0}

-- Questions reformulated as proof problems
theorem problem1 (a : ℝ) (h : set_A a = set_B) : a = 5 :=
sorry

theorem problem2 (a : ℝ) (h1 : ∃ x, x ∈ set_A a ∧ x ∈ set_B) (h2 : ∀ x, x ∈ set_A a → x ∉ set_C) : a = -2 :=
sorry

theorem problem3 (a : ℝ) (h1 : ∃ x, x ∈ set_A a ∧ x ∈ set_B) (h2 : set_A a ∩ set_B = set_A a ∩ set_C) : a = -3 :=
sorry

end problem1_problem2_problem3_l262_262513


namespace evaluate_fraction_l262_262492

theorem evaluate_fraction (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x - 1 / y ≠ 0) :
  (y - 1 / x) / (x - 1 / y) + y / x = 2 * y / x :=
by sorry

end evaluate_fraction_l262_262492


namespace problem_part_1_problem_part_2_problem_part_3_l262_262064

open Set

-- Definitions for the given problem conditions
def U : Set ℕ := { x | x > 0 ∧ x < 10 }
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6}
def D : Set ℕ := B ∩ C

-- Prove each part of the problem
theorem problem_part_1 :
  U = {1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

theorem problem_part_2 :
  D = {3, 4} ∧
  (∀ (s : Set ℕ), s ⊆ D ↔ s = ∅ ∨ s = {3} ∨ s = {4} ∨ s = {3, 4}) := by
  sorry

theorem problem_part_3 :
  (U \ D) = {1, 2, 5, 6, 7, 8, 9} := by
  sorry

end problem_part_1_problem_part_2_problem_part_3_l262_262064


namespace cylinder_height_l262_262218

theorem cylinder_height (r h : ℝ) (SA : ℝ) (h_cond : SA = 2 * π * r^2 + 2 * π * r * h) 
  (r_eq : r = 3) (SA_eq : SA = 27 * π) : h = 3 / 2 :=
by
  sorry

end cylinder_height_l262_262218


namespace total_number_of_orders_l262_262940

-- Define the conditions
def num_original_programs : Nat := 6
def num_added_programs : Nat := 3

-- State the theorem
theorem total_number_of_orders : ∃ n : ℕ, n = 210 :=
by
  -- This is where the proof would go
  sorry

end total_number_of_orders_l262_262940


namespace problem_arithmetic_sequence_l262_262053

-- Definitions based on given conditions
def a1 : ℕ := 2
def d := (13 - 2 * a1) / 3

-- Definition of the nth term in the arithmetic sequence
def a (n : ℕ) : ℕ := a1 + (n - 1) * d

-- The required proof problem statement
theorem problem_arithmetic_sequence : a 4 + a 5 + a 6 = 42 := 
by
  -- placeholders for the actual proof
  sorry

end problem_arithmetic_sequence_l262_262053


namespace closest_perfect_square_to_350_l262_262270

theorem closest_perfect_square_to_350 :
  let n := 19 in
  let m := 18 in
  18^2 = 324 ∧ 19^2 = 361 ∧ 324 < 350 ∧ 350 < 361 ∧ 
  abs (350 - 324) = 26 ∧ abs (361 - 350) = 11 ∧ 11 < 26 → 
  n^2 = 361 :=
by
  intro n m h
  sorry

end closest_perfect_square_to_350_l262_262270


namespace closest_perfect_square_to_350_l262_262272

theorem closest_perfect_square_to_350 :
  let n := 19 in
  let m := 18 in
  18^2 = 324 ∧ 19^2 = 361 ∧ 324 < 350 ∧ 350 < 361 ∧ 
  abs (350 - 324) = 26 ∧ abs (361 - 350) = 11 ∧ 11 < 26 → 
  n^2 = 361 :=
by
  intro n m h
  sorry

end closest_perfect_square_to_350_l262_262272


namespace base_7_sum_of_product_l262_262093

-- Definitions of the numbers in base-10 for base-7 numbers
def base_7_to_base_10 (d1 d0 : ℕ) : ℕ := d1 * 7 + d0

def sum_digits_base_7 (n : ℕ) : ℕ := 
  let d2 := n / 343
  let r2 := n % 343
  let d1 := r2 / 49
  let r1 := r2 % 49
  let d0 := r1 / 7 + r1 % 7
  d2 + d1 + d0

def convert_10_to_7 (n : ℕ) : ℕ := 
  let d1 := n / 7
  let r1 := n % 7
  d1 * 10 + r1

theorem base_7_sum_of_product : 
  let n36  := base_7_to_base_10 3 6
  let n52  := base_7_to_base_10 5 2
  let nadd := base_7_to_base_10 2 0
  let prod := n36 * n52
  let suma := prod + nadd
  convert_10_to_7 (sum_digits_base_7 suma) = 23 :=
by
  sorry

end base_7_sum_of_product_l262_262093


namespace unit_digit_of_fourth_number_l262_262545

theorem unit_digit_of_fourth_number
  (n1 n2 n3 n4 : ℕ)
  (h1 : n1 % 10 = 4)
  (h2 : n2 % 10 = 8)
  (h3 : n3 % 10 = 3)
  (h4 : (n1 * n2 * n3 * n4) % 10 = 8) : 
  n4 % 10 = 3 :=
sorry

end unit_digit_of_fourth_number_l262_262545


namespace minimum_perimeter_l262_262099

-- Define the area condition
def area_condition (l w : ℝ) : Prop := l * w = 64

-- Define the perimeter function
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

-- The theorem statement based on the conditions and the correct answer
theorem minimum_perimeter (l w : ℝ) (h : area_condition l w) : 
  perimeter l w ≥ 32 := by
sorry

end minimum_perimeter_l262_262099


namespace find_a_l262_262386

noncomputable def parabola_eq (a b c : ℤ) (x : ℤ) : ℤ :=
  a * x^2 + b * x + c

theorem find_a (a b c : ℤ)
  (h_vertex : ∀ x, parabola_eq a b c x = a * (x - 2)^2 + 5) 
  (h_point : parabola_eq a b c 1 = 6) :
  a = 1 := 
by 
  sorry

end find_a_l262_262386


namespace red_pens_count_l262_262549

theorem red_pens_count (R : ℕ) : 
  (∃ (black_pens blue_pens : ℕ), 
  black_pens = R + 10 ∧ 
  blue_pens = R + 7 ∧ 
  R + black_pens + blue_pens = 41) → 
  R = 8 := by
  sorry

end red_pens_count_l262_262549


namespace increase_expenditure_by_10_percent_l262_262524

variable (I : ℝ) (P : ℝ)
def E := 0.75 * I
def I_new := 1.20 * I
def S_new := 1.50 * (I - E)
def E_new := E * (1 + P / 100)

theorem increase_expenditure_by_10_percent :
  (E_new = 0.75 * I * (1 + P / 100)) → P = 10 :=
by
  sorry

end increase_expenditure_by_10_percent_l262_262524


namespace base_length_of_isosceles_triangle_l262_262084

theorem base_length_of_isosceles_triangle 
  (a b : ℕ) 
  (h1 : a = 6) 
  (h2 : b = 6) 
  (perimeter : ℕ) 
  (h3 : 2*a + b = perimeter)
  (h4 : perimeter = 20) 
  : b = 8 := 
by
  sorry

end base_length_of_isosceles_triangle_l262_262084


namespace greatest_two_digit_product_is_12_l262_262780

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l262_262780


namespace max_students_divide_equal_pen_pencil_l262_262813

theorem max_students_divide_equal_pen_pencil : Nat.gcd 2500 1575 = 25 := 
by
  sorry

end max_students_divide_equal_pen_pencil_l262_262813


namespace total_steps_eliana_walked_l262_262616

-- Define the conditions of the problem.
def first_day_exercise_steps : Nat := 200
def first_day_additional_steps : Nat := 300
def second_day_multiplier : Nat := 2
def third_day_additional_steps : Nat := 100

-- Define the steps calculation for each day.
def first_day_total_steps : Nat := first_day_exercise_steps + first_day_additional_steps
def second_day_total_steps : Nat := second_day_multiplier * first_day_total_steps
def third_day_total_steps : Nat := second_day_total_steps + third_day_additional_steps

-- Prove that the total number of steps Eliana walked during these three days is 1600.
theorem total_steps_eliana_walked :
  first_day_total_steps + second_day_total_steps + third_day_additional_steps = 1600 :=
by
  -- Conditional values are constants. We can use Lean's deterministic evaluator here.
  -- Hence, there's no need to write out full proof for now. Using sorry to bypass actual proof.
  sorry

end total_steps_eliana_walked_l262_262616


namespace planes_parallel_or_coincide_l262_262636

-- Define normal vectors
def normal_vector_u : ℝ × ℝ × ℝ := (1, 2, -2)
def normal_vector_v : ℝ × ℝ × ℝ := (-3, -6, 6)

-- The theorem states that planes defined by these normal vectors are either 
-- parallel or coincide if their normal vectors are collinear.
theorem planes_parallel_or_coincide (u v : ℝ × ℝ × ℝ) 
  (h_u : u = normal_vector_u) 
  (h_v : v = normal_vector_v) 
  (h_collinear : v = (-3) • u) : 
    ∃ k : ℝ, v = k • u := 
by
  sorry

end planes_parallel_or_coincide_l262_262636


namespace relationship_of_squares_and_products_l262_262734

theorem relationship_of_squares_and_products (a b x : ℝ) (h1 : b < x) (h2 : x < a) (h3 : a < 0) : 
  x^2 > ax ∧ ax > b^2 :=
by
  sorry

end relationship_of_squares_and_products_l262_262734


namespace bill_bought_60_rats_l262_262127

def chihuahuas_and_rats (C R : ℕ) : Prop :=
  C + R = 70 ∧ R = 6 * C

theorem bill_bought_60_rats (C R : ℕ) (h : chihuahuas_and_rats C R) : R = 60 :=
by
  sorry

end bill_bought_60_rats_l262_262127


namespace repeating_decimal_sum_to_fraction_l262_262844

def repeating_decimal_123 : ℚ := 123 / 999
def repeating_decimal_0045 : ℚ := 45 / 9999
def repeating_decimal_000678 : ℚ := 678 / 999999

theorem repeating_decimal_sum_to_fraction :
  repeating_decimal_123 + repeating_decimal_0045 + repeating_decimal_000678 = 128178 / 998001000 :=
by
  sorry

end repeating_decimal_sum_to_fraction_l262_262844


namespace max_product_l262_262325

noncomputable def max_of_product (x y : ℝ) : ℝ := x * y

theorem max_product (x y : ℝ) (h1 : x ∈ Set.Ioi 0) (h2 : y ∈ Set.Ioi 0) (h3 : x + 4 * y = 1) :
  max_of_product x y ≤ 1 / 16 := sorry

end max_product_l262_262325


namespace max_soap_boxes_in_carton_l262_262564

theorem max_soap_boxes_in_carton
  (L_carton W_carton H_carton : ℕ)
  (L_soap_box W_soap_box H_soap_box : ℕ)
  (vol_carton := L_carton * W_carton * H_carton)
  (vol_soap_box := L_soap_box * W_soap_box * H_soap_box)
  (max_soap_boxes := vol_carton / vol_soap_box) :
  L_carton = 25 → W_carton = 42 → H_carton = 60 →
  L_soap_box = 7 → W_soap_box = 6 → H_soap_box = 5 →
  max_soap_boxes = 300 :=
by
  intros hL hW hH hLs hWs hHs
  sorry

end max_soap_boxes_in_carton_l262_262564


namespace geometric_series_sum_l262_262924

theorem geometric_series_sum :
  (1 / 3 - 1 / 6 + 1 / 12 - 1 / 24 + 1 / 48 - 1 / 96) = 7 / 32 :=
by
  sorry

end geometric_series_sum_l262_262924


namespace sqrt_4_eq_pm2_l262_262405

theorem sqrt_4_eq_pm2 : {y : ℝ | y^2 = 4} = {2, -2} :=
by
  sorry

end sqrt_4_eq_pm2_l262_262405


namespace find_abc_l262_262629

theorem find_abc
  (a b c : ℝ)
  (h1 : a^2 * (b + c) = 2011)
  (h2 : b^2 * (a + c) = 2011)
  (h3 : a ≠ b) : 
  a * b * c = -2011 := 
by 
sorry

end find_abc_l262_262629


namespace dodecahedron_interior_diagonals_l262_262645

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l262_262645


namespace quadratic_has_unique_solution_l262_262987

theorem quadratic_has_unique_solution (k : ℝ) :
  (∀ x : ℝ, (x + 6) * (x + 3) = k + 3 * x) → k = 9 :=
by
  intro h
  sorry

end quadratic_has_unique_solution_l262_262987


namespace xy_positive_l262_262692

theorem xy_positive (x y : ℝ) (h1 : x - y < x) (h2 : x + y > y) : x > 0 ∧ y > 0 :=
sorry

end xy_positive_l262_262692


namespace pq_plus_qr_plus_rp_cubic_1_l262_262714

theorem pq_plus_qr_plus_rp_cubic_1 (p q r : ℝ) 
  (h1 : p + q + r = 0)
  (h2 : p * q + p * r + q * r = -2)
  (h3 : p * q * r = 2) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -6 :=
by
  sorry

end pq_plus_qr_plus_rp_cubic_1_l262_262714


namespace average_speed_l262_262724

theorem average_speed (D : ℝ) :
  let time_by_bus := D / 80
  let time_walking := D / 16
  let time_cycling := D / 120
  let total_time := time_by_bus + time_walking + time_cycling
  let total_distance := 2 * D
  total_distance / total_time = 24 := by
  sorry

end average_speed_l262_262724


namespace blue_balloons_l262_262951

theorem blue_balloons (total_balloons red_balloons green_balloons purple_balloons : ℕ)
  (h1 : total_balloons = 135)
  (h2 : red_balloons = 45)
  (h3 : green_balloons = 27)
  (h4 : purple_balloons = 32) :
  total_balloons - (red_balloons + green_balloons + purple_balloons) = 31 :=
by
  sorry

end blue_balloons_l262_262951


namespace final_amount_H2O_l262_262314

theorem final_amount_H2O (main_reaction : ∀ (Li3N H2O LiOH NH3 : ℕ), Li3N + 3 * H2O = 3 * LiOH + NH3)
  (side_reaction : ∀ (Li3N LiOH Li2O NH4OH : ℕ), Li3N + LiOH = Li2O + NH4OH)
  (temperature : ℕ) (pressure : ℕ)
  (percentage : ℝ) (init_moles_LiOH : ℕ) (init_moles_Li3N : ℕ)
  (H2O_req_main : ℝ) (H2O_req_side : ℝ) :
  400 = temperature →
  2 = pressure →
  0.05 = percentage →
  9 = init_moles_LiOH →
  3 = init_moles_Li3N →
  H2O_req_main = init_moles_Li3N * 3 →
  H2O_req_side = init_moles_LiOH * percentage →
  H2O_req_main + H2O_req_side = 9.45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end final_amount_H2O_l262_262314


namespace vasya_correct_l262_262428

-- Define the condition of a convex quadrilateral
def convex_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a < 180 ∧ b < 180 ∧ c < 180 ∧ d < 180

-- Define the properties of forming two types of triangles from a quadrilateral
def can_form_two_acute_triangles (a b c d : ℝ) : Prop :=
  a < 90 ∧ b < 90 ∧ c < 90 ∧ d < 90

def can_form_two_right_triangles (a b c d : ℝ) : Prop :=
  (a = 90 ∧ b = 90) ∨ (b = 90 ∧ c = 90) ∨ (c = 90 ∧ d = 90) ∨ (d = 90 ∧ a = 90)

def can_form_two_obtuse_triangles (a b c d : ℝ) : Prop :=
  ∃ x y z w, (x > 90 ∧ y < 90 ∧ z < 90 ∧ w < 90 ∧ (x + y + z + w = 360)) ∧
             (x > 90 ∨ y > 90 ∨ z > 90 ∨ w > 90)

-- Prove that Vasya's claim is definitively correct
theorem vasya_correct (a b c d : ℝ) (h : convex_quadrilateral a b c d) :
  can_form_two_obtuse_triangles a b c d ∧
  ¬(can_form_two_acute_triangles a b c d) ∧
  ¬(can_form_two_right_triangles a b c d) ∨
  can_form_two_right_triangles a b c d ∧
  can_form_two_obtuse_triangles a b c d := sorry

end vasya_correct_l262_262428


namespace solution_set_of_inequality_l262_262914

theorem solution_set_of_inequality : 
  {x : ℝ | (x - 1) * (2 - x) > 0} = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l262_262914


namespace greatest_two_digit_product_12_l262_262786

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l262_262786


namespace symmetry_axis_l262_262462

noncomputable def y_func (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

theorem symmetry_axis : ∃ a : ℝ, (∀ x : ℝ, y_func (a - x) = y_func (a + x)) ∧ a = Real.pi / 8 :=
by
  sorry

end symmetry_axis_l262_262462


namespace average_price_of_cow_l262_262575

variable (price_cow price_goat : ℝ)

theorem average_price_of_cow (h1 : 2 * price_cow + 8 * price_goat = 1400)
                             (h2 : price_goat = 60) :
                             price_cow = 460 := 
by
  -- The following line allows the Lean code to compile successfully without providing a proof.
  sorry

end average_price_of_cow_l262_262575


namespace dodecahedron_interior_diagonals_l262_262669

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l262_262669


namespace base_length_of_isosceles_triangle_l262_262082

theorem base_length_of_isosceles_triangle 
  (a b : ℕ) 
  (h1 : a = 6) 
  (h2 : b = 6) 
  (perimeter : ℕ) 
  (h3 : 2*a + b = perimeter)
  (h4 : perimeter = 20) 
  : b = 8 := 
by
  sorry

end base_length_of_isosceles_triangle_l262_262082


namespace inequality_proof_l262_262162

theorem inequality_proof (x y z : ℝ) (hx : x < 0) (hy : y < 0) (hz : z < 0) :
    (x * y * z) / ((1 + 5 * x) * (4 * x + 3 * y) * (5 * y + 6 * z) * (z + 18)) ≤ (1 : ℝ) / 5120 := 
by
  sorry

end inequality_proof_l262_262162


namespace isosceles_triangle_base_length_l262_262089

theorem isosceles_triangle_base_length (b : ℕ) (h₁ : 6 + 6 + b = 20) : b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l262_262089


namespace simplify_expression_l262_262239

theorem simplify_expression : (1 / (-8^2)^4) * (-8)^9 = -8 := 
by
  sorry

end simplify_expression_l262_262239


namespace number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l262_262944

theorem number_of_positive_integers_with_erased_digit_decreased_by_nine_times : 
  ∃ n : ℕ, 
  ∀ (m a k : ℕ),
  (m + 10^k * a + 10^(k + 1) * n = 9 * (m + 10^k * n)) → 
  m < 10^k ∧ n > 0 ∧ n < m ∧  m ≠ 0 → 
  (m + 10^k * n  = 9 * (m - a) ) ∧ 
  (m % 10 = 5 ∨ m % 10 = 0) → 
  n = 28 :=
by
  sorry

end number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l262_262944


namespace day_of_week_100_days_from_wednesday_l262_262918

theorem day_of_week_100_days_from_wednesday (today_is_wed : ∃ i : ℕ, i % 7 = 3) : 
  (100 % 7 + 3) % 7 = 5 := 
by
  sorry

end day_of_week_100_days_from_wednesday_l262_262918


namespace olivia_total_payment_l262_262204

theorem olivia_total_payment : 
  (4 / 4 + 12 / 4 = 4) :=
by
  sorry

end olivia_total_payment_l262_262204


namespace parabola_vertex_range_l262_262695

def parabola_vertex_in_first_quadrant (m : ℝ) : Prop :=
  ∃ v : ℝ × ℝ, v = (m, m - 1) ∧ 0 < m ∧ 0 < (m - 1)

theorem parabola_vertex_range (m : ℝ) (h_vertex : parabola_vertex_in_first_quadrant m) :
  1 < m :=
by
  sorry

end parabola_vertex_range_l262_262695


namespace day_100_days_from_wednesday_l262_262919

-- Definitions for the conditions
def today_is_wednesday := "Wednesday"

def days_in_week := 7

def day_of_the_week (n : Nat) : String := 
  let days := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
  days[n % days.length]

-- Theorem to prove
theorem day_100_days_from_wednesday : day_of_the_week ((4 + 100) % days_in_week) = "Friday" :=
  sorry

end day_100_days_from_wednesday_l262_262919


namespace cars_on_river_road_l262_262814

theorem cars_on_river_road (B C : ℕ) (h1 : B = C - 40) (h2 : B * 3 = C) : C = 60 := 
sorry

end cars_on_river_road_l262_262814


namespace card_sequence_probability_l262_262408

noncomputable def prob_first_card_club : ℚ := 13 / 52
noncomputable def prob_second_card_heart (first_card_is_club : bool) : ℚ := if first_card_is_club then 13 / 51 else 0
noncomputable def prob_third_card_king (first_card_is_club : bool) (second_card_is_heart : bool) : ℚ := if first_card_is_club && second_card_is_heart then 4 / 50 else 0

theorem card_sequence_probability :
  prob_first_card_club * (prob_second_card_heart true) * (prob_third_card_king true true) = 13 / 2550 :=
by
  field_simp
  norm_num
  exact rfl

end card_sequence_probability_l262_262408


namespace trapezoid_perimeter_l262_262440

theorem trapezoid_perimeter (AB CD AD BC h : ℝ)
  (AB_eq : AB = 40)
  (CD_eq : CD = 70)
  (AD_eq_BC : AD = BC)
  (h_eq : h = 24)
  : AB + BC + CD + AD = 110 + 2 * Real.sqrt 801 :=
by
  -- Proof goes here, you can replace this comment with actual proof.
  sorry

end trapezoid_perimeter_l262_262440


namespace statements_correct_l262_262832

theorem statements_correct :
  (∀ x : ℝ, (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (x^2 - 3*x + 2 = 0 → x = 1)) ∧
  (∀ x : ℝ, (∀ x, x^2 + x + 1 ≠ 0) ↔ (∃ x, x^2 + x + 1 = 0)) ∧
  (∀ p q : Prop, (p ∧ q) ↔ p ∧ q) ∧
  (∀ x : ℝ, (x > 2 → x^2 - 3*x + 2 > 0) ∧ (¬ (x^2 - 3*x + 2 > 0) → x ≤ 2)) :=
by
  sorry

end statements_correct_l262_262832


namespace square_side_length_l262_262210

-- Definitions based on problem conditions
def right_triangle : Prop := ∃ (A B C : Point) (AB AC BC : ℝ), AB = 9 ∧ AC = 12 ∧ right_angle A B C ∧ hypotenuse B C

def square_on_hypotenuse : Prop :=
  right_triangle ∧ ∃ (S : Point → Point → Point → Point → Prop)
  (s : ℝ), ∃ h : S(D E F G) (s_on_hypotenuse h s), vertex_on_legs S D E

theorem square_side_length (s : ℝ) : square_on_hypotenuse → s = 180 / 37 :=
by
  intro h
  sorry

end square_side_length_l262_262210


namespace range_of_a_l262_262910

theorem range_of_a (a : ℝ) : ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) := 
sorry

end range_of_a_l262_262910


namespace part1_part2_l262_262023

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then -3 * x + (1/2)^x - 1 else sorry -- Placeholder: function definition incomplete for x ≤ 0

def odd (f : ℝ → ℝ) :=
∀ x, f (-x) = - f x

def monotonic_decreasing (f : ℝ → ℝ) :=
∀ x y, x < y → f x > f y

axiom f_conditions :
  monotonic_decreasing f ∧
  odd f ∧
  (∀ x, x > 0 → f x = -3 * x + (1/2)^x - 1)

theorem part1 : f (-1) = 3.5 :=
by
  sorry

theorem part2 (t : ℝ) (k : ℝ) :
  (∀ t, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) ↔ k < -1/3 :=
by
  sorry

end part1_part2_l262_262023


namespace arithmetic_sequence_general_formula_l262_262994

variable (a : ℤ) 

def is_arithmetic_sequence (a1 a2 a3 : ℤ) : Prop :=
  2 * a2 = a1 + a3

theorem arithmetic_sequence_general_formula :
  ∀ {a1 a2 a3 : ℤ}, is_arithmetic_sequence a1 a2 a3 → a1 = a - 1 ∧ a2 = a + 1 ∧ a3 = 2 * a + 3 → 
  ∀ n : ℕ, a_n = 2 * n - 3
:= by
  sorry

end arithmetic_sequence_general_formula_l262_262994


namespace three_point_one_two_six_as_fraction_l262_262809

theorem three_point_one_two_six_as_fraction : (3126 / 1000 : ℚ) = 1563 / 500 := 
by 
  sorry

end three_point_one_two_six_as_fraction_l262_262809


namespace total_steps_eliana_walked_l262_262615

-- Define the conditions of the problem.
def first_day_exercise_steps : Nat := 200
def first_day_additional_steps : Nat := 300
def second_day_multiplier : Nat := 2
def third_day_additional_steps : Nat := 100

-- Define the steps calculation for each day.
def first_day_total_steps : Nat := first_day_exercise_steps + first_day_additional_steps
def second_day_total_steps : Nat := second_day_multiplier * first_day_total_steps
def third_day_total_steps : Nat := second_day_total_steps + third_day_additional_steps

-- Prove that the total number of steps Eliana walked during these three days is 1600.
theorem total_steps_eliana_walked :
  first_day_total_steps + second_day_total_steps + third_day_additional_steps = 1600 :=
by
  -- Conditional values are constants. We can use Lean's deterministic evaluator here.
  -- Hence, there's no need to write out full proof for now. Using sorry to bypass actual proof.
  sorry

end total_steps_eliana_walked_l262_262615


namespace eliana_total_steps_l262_262610

-- Define the conditions given in the problem
def steps_first_day_exercise : Nat := 200
def steps_first_day_additional : Nat := 300
def steps_first_day : Nat := steps_first_day_exercise + steps_first_day_additional

def steps_second_day : Nat := 2 * steps_first_day
def steps_additional_on_third_day : Nat := 100
def steps_third_day : Nat := steps_second_day + steps_additional_on_third_day

-- Mathematical proof problem proving that the total number of steps is 2600
theorem eliana_total_steps : steps_first_day + steps_second_day + steps_third_day = 2600 := 
by
  sorry

end eliana_total_steps_l262_262610


namespace range_of_p_l262_262954

noncomputable def success_prob_4_engine (p : ℝ) : ℝ :=
  4 * p^3 * (1 - p) + p^4

noncomputable def success_prob_2_engine (p : ℝ) : ℝ :=
  p^2

theorem range_of_p (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  success_prob_4_engine p > success_prob_2_engine p ↔ (1/3 < p ∧ p < 1) :=
by
  sorry

end range_of_p_l262_262954


namespace children_l262_262741

variable (C : ℝ) -- Define the weight of a children's book

theorem children's_book_weight :
  (9 * 0.8 + 7 * C = 10.98) → C = 0.54 :=
by  
sorry

end children_l262_262741


namespace how_many_one_halves_in_two_sevenths_l262_262339

theorem how_many_one_halves_in_two_sevenths : (2 / 7) / (1 / 2) = 4 / 7 := by 
  sorry

end how_many_one_halves_in_two_sevenths_l262_262339


namespace maximize_det_l262_262561

theorem maximize_det (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  (Matrix.det ![
    ![a, 1],
    ![1, b]
  ]) ≤ 0 :=
sorry

end maximize_det_l262_262561


namespace problem_statement_l262_262512

variables (u v w : ℝ)

theorem problem_statement (h₁: u + v + w = 3) : 
  (1 / (u^2 + 7) + 1 / (v^2 + 7) + 1 / (w^2 + 7) ≤ 3 / 8) :=
sorry

end problem_statement_l262_262512


namespace stable_k_digit_number_l262_262503

def is_stable (a k : ℕ) : Prop :=
  ∀ m n : ℕ, (10^k ∣ ((m * 10^k + a) * (n * 10^k + a) - a))

theorem stable_k_digit_number (k : ℕ) (h_pos : k > 0) : ∃ (a : ℕ) (h : ∀ m n : ℕ, 10^k ∣ ((m * 10^k + a) * (n * 10^k + a) - a)), (10^(k-1)) ≤ a ∧ a < 10^k ∧ ∀ b : ℕ, (∀ m n : ℕ, 10^k ∣ ((m * 10^k + b) * (n * 10^k + b) - b)) → (10^(k-1)) ≤ b ∧ b < 10^k → a = b :=
by
  sorry

end stable_k_digit_number_l262_262503


namespace greatest_two_digit_product_12_l262_262787

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l262_262787


namespace range_of_k_l262_262864

theorem range_of_k {k : ℝ} :
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
by sorry

end range_of_k_l262_262864


namespace dot_product_equilateral_l262_262475

-- Define the conditions for the equilateral triangle ABC
variable {A B C : ℝ}

noncomputable def equilateral_triangle (A B C : ℝ) := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ |A - B| = 1 ∧ |B - C| = 1 ∧ |C - A| = 1

-- Define the dot product of the vectors AB and BC
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved
theorem dot_product_equilateral (A B C : ℝ) (h : equilateral_triangle A B C) : 
  dot_product (B - A, 0) (C - B, 0) = -1 / 2 :=
sorry

end dot_product_equilateral_l262_262475


namespace percentage_reduction_l262_262740

theorem percentage_reduction (S P : ℝ) (h : S - (P / 100) * S = S / 2) : P = 50 :=
by
  sorry

end percentage_reduction_l262_262740


namespace calculate_value_l262_262306

theorem calculate_value : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end calculate_value_l262_262306


namespace tennis_balls_ordered_originally_l262_262122

-- Definitions according to the conditions in a)
def retailer_ordered_equal_white_yellow_balls (W Y : ℕ) : Prop :=
  W = Y

def dispatch_error (Y : ℕ) : ℕ :=
  Y + 90

def ratio_white_to_yellow (W Y : ℕ) : Prop :=
  W / dispatch_error Y = 8 / 13

-- Main statement
theorem tennis_balls_ordered_originally (W Y : ℕ) (h1 : retailer_ordered_equal_white_yellow_balls W Y)
  (h2 : ratio_white_to_yellow W Y) : W + Y = 288 :=
by
  sorry    -- Placeholder for the actual proof

end tennis_balls_ordered_originally_l262_262122


namespace stationery_sales_calculation_l262_262114

-- Definitions
def total_sales : ℕ := 120
def fabric_percentage : ℝ := 0.30
def jewelry_percentage : ℝ := 0.20
def knitting_percentage : ℝ := 0.15
def home_decor_percentage : ℝ := 0.10
def stationery_percentage := 1 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage)
def stationery_sales := stationery_percentage * total_sales

-- Statement to prove
theorem stationery_sales_calculation : stationery_sales = 30 := by
  -- Providing the initial values and assumptions to the context
  have h1 : total_sales = 120 := rfl
  have h2 : fabric_percentage = 0.30 := rfl
  have h3 : jewelry_percentage = 0.20 := rfl
  have h4 : knitting_percentage = 0.15 := rfl
  have h5 : home_decor_percentage = 0.10 := rfl
  
  -- Calculating the stationery percentage and sales
  have h_stationery_percentage : stationery_percentage = 1 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage) := rfl
  have h_stationery_sales : stationery_sales = stationery_percentage * total_sales := rfl

  -- The calculated value should match the proof's requirement
  sorry

end stationery_sales_calculation_l262_262114


namespace find_f_l262_262816

theorem find_f :
  ∀ (f : ℕ → ℕ),   
    (∀ a b : ℕ, f (a * b) = f a + f b - f (Nat.gcd a b)) →
    (∀ (p a : ℕ), Nat.Prime p → (f a ≥ f (a * p) → f a + f p ≥ f a * f p + 1)) →
    (∀ n : ℕ, f n = n ∨ f n = 1) :=
by
  intros f h1 h2
  sorry

end find_f_l262_262816


namespace tribe_leadership_choices_l262_262834

theorem tribe_leadership_choices :
  let members := 15
  let ways_to_choose_chief := members
  let remaining_after_chief := members - 1
  let ways_to_choose_supporting_chiefs := Nat.choose remaining_after_chief 2
  let remaining_after_supporting_chiefs := remaining_after_chief - 2
  let ways_to_choose_officers_A := Nat.choose remaining_after_supporting_chiefs 2
  let remaining_for_assistants_A := remaining_after_supporting_chiefs - 2
  let ways_to_choose_assistants_A := Nat.choose remaining_for_assistants_A 2 * Nat.choose (remaining_for_assistants_A - 2) 2
  let remaining_after_A := remaining_for_assistants_A - 2
  let ways_to_choose_officers_B := Nat.choose remaining_after_A 2
  let remaining_for_assistants_B := remaining_after_A - 2
  let ways_to_choose_assistants_B := Nat.choose remaining_for_assistants_B 2 * Nat.choose (remaining_for_assistants_B - 2) 2
  (ways_to_choose_chief * ways_to_choose_supporting_chiefs *
  ways_to_choose_officers_A * ways_to_choose_assistants_A *
  ways_to_choose_officers_B * ways_to_choose_assistants_B = 400762320000) := by
  sorry

end tribe_leadership_choices_l262_262834


namespace intersection_point_l262_262311

theorem intersection_point (x y : ℚ) (h1 : 8 * x - 5 * y = 40) (h2 : 6 * x + 2 * y = 14) :
  x = 75 / 23 ∧ y = -64 / 23 :=
by
  -- Proof not needed, so we finish with sorry
  sorry

end intersection_point_l262_262311


namespace range_of_a_l262_262005

theorem range_of_a (a : ℝ) (h : ∀ t : ℝ, 0 < t → t ≤ 2 → (t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2)) : 
  (2 / 13) ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l262_262005


namespace route_down_distance_l262_262431

-- Definitions
def rate_up : ℝ := 7
def time_up : ℝ := 2
def distance_up : ℝ := rate_up * time_up
def rate_down : ℝ := 1.5 * rate_up
def time_down : ℝ := time_up
def distance_down : ℝ := rate_down * time_down

-- Theorem
theorem route_down_distance : distance_down = 21 := by
  sorry

end route_down_distance_l262_262431


namespace commutative_otimes_l262_262707

def otimes (a b : ℝ) : ℝ := a * b + a + b

theorem commutative_otimes (a b : ℝ) : otimes a b = otimes b a :=
by
  /- The proof will go here, but we omit it and use sorry. -/
  sorry

end commutative_otimes_l262_262707


namespace dodecahedron_interior_diagonals_l262_262674

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l262_262674


namespace male_students_count_l262_262071

theorem male_students_count :
  ∃ (N M : ℕ), 
  (N % 4 = 2) ∧ 
  (N % 5 = 1) ∧ 
  (N = M + 15) ∧ 
  (15 > M) ∧ 
  (M = 11) :=
sorry

end male_students_count_l262_262071


namespace dodecahedron_interior_diagonals_eq_160_l262_262651

-- Definition of a dodecahedron for the context of the proof
structure Dodecahedron where
  num_faces : Nat
  num_vertices : Nat
  faces_meeting_at_vertex : Nat
  vertices_connected_by_edge : Nat

-- Given our specific dodecahedron properties
def dodecahedron : Dodecahedron := {
  num_faces := 12,
  num_vertices := 20,
  faces_meeting_at_vertex := 3,
  vertices_connected_by_edge := 3
}

-- The interior diagonal is defined as a segment connecting two vertices
-- which do not lie on the same face.
def is_interior_diagonal (d : Dodecahedron) (v1 v2 : Fin d.num_vertices) : Prop :=
  v1 ≠ v2 ∧ ¬∃ f, f < d.num_faces ∧ v1 ∈ f ∧ v2 ∈ f

-- Now, we need to calculate the total number of such interior diagonals
noncomputable def num_interior_diagonals (d : Dodecahedron) : Nat :=
  d.num_vertices * (d.num_vertices - d.vertices_connected_by_edge - 1) / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals_eq_160 :
  num_interior_diagonals dodecahedron = 160 := by
  sorry

end dodecahedron_interior_diagonals_eq_160_l262_262651


namespace payment_to_C_l262_262811

-- Work rates definition
def work_rate_A : ℚ := 1 / 6
def work_rate_B : ℚ := 1 / 8
def combined_work_rate_A_B : ℚ := work_rate_A + work_rate_B
def combined_work_rate_A_B_C : ℚ := 1 / 3

-- C's work rate calculation
def work_rate_C : ℚ := combined_work_rate_A_B_C - combined_work_rate_A_B

-- Payment calculation
def total_payment : ℚ := 3200
def C_payment_ratio : ℚ := work_rate_C / combined_work_rate_A_B_C
def C_payment : ℚ := total_payment * C_payment_ratio

-- Theorem stating the result
theorem payment_to_C : C_payment = 400 := by
  sorry

end payment_to_C_l262_262811


namespace horizontal_asymptote_is_3_l262_262956

-- Definitions of the polynomials
noncomputable def p (x : ℝ) : ℝ := 15 * x^5 + 10 * x^4 + 5 * x^3 + 7 * x^2 + 6 * x + 2
noncomputable def q (x : ℝ) : ℝ := 5 * x^5 + 3 * x^4 + 9 * x^3 + 4 * x^2 + 2 * x + 1

-- Statement that we need to prove
theorem horizontal_asymptote_is_3 : 
  (∃ (y : ℝ), (∀ x : ℝ, x ≠ 0 → (p x / q x) = y) ∧ y = 3) :=
  sorry -- The proof is left as an exercise.

end horizontal_asymptote_is_3_l262_262956


namespace other_root_of_quadratic_l262_262330

theorem other_root_of_quadratic (k : ℝ) (h : -2 * 1 = -2) (h_eq : x^2 + k * x - 2 = 0) :
  1 * -2 = -2 :=
by
  sorry

end other_root_of_quadratic_l262_262330


namespace number_of_students_l262_262125

theorem number_of_students 
  (P S : ℝ)
  (total_cost : ℝ) 
  (percent_free : ℝ) 
  (lunch_cost : ℝ)
  (h1 : percent_free = 0.40)
  (h2 : total_cost = 210)
  (h3 : lunch_cost = 7)
  (h4 : P = 0.60 * S)
  (h5 : P * lunch_cost = total_cost) :
  S = 50 :=
by
  sorry

end number_of_students_l262_262125


namespace polynomial_value_l262_262039

theorem polynomial_value (x : ℝ) (h : 3 * x^2 - x = 1) : 6 * x^3 + 7 * x^2 - 5 * x + 2008 = 2011 :=
by
  sorry

end polynomial_value_l262_262039


namespace find_c_for_Q_l262_262249

noncomputable def Q (c : ℚ) (x : ℚ) : ℚ := x^3 + 3*x^2 + c*x + 8

theorem find_c_for_Q (c : ℚ) : 
  (Q c 3 = 0) ↔ (c = -62 / 3) := by
  sorry

end find_c_for_Q_l262_262249


namespace highest_possible_characteristic_l262_262223

theorem highest_possible_characteristic (n : ℕ) (hn : n ≥ 2) (grid : Fin n → Fin n → ℕ) (hgrid : ∀ (i j : Fin n), 1 ≤ grid i j ∧ grid i j ≤ n^2 ∧ ∀ (i₁ i₂ j₁ j₂ : Fin n), (i₁ ≠ i₂ ∨ j₁ ≠ j₂) → grid i₁ j₁ ≠ grid i₂ j₂) :
  ∃ (char : ℚ), (∀ (i j : Fin n), ∀ (k₁ k₂ : Fin n), k₁ ≠ k₂ → (grid i k₁ / grid i k₂ < char) ∧ (grid k₁ j / grid k₂ j < char)) ∧ char = (n + 1) / n := 
sorry

end highest_possible_characteristic_l262_262223


namespace investment_amount_l262_262139

noncomputable def annual_income (investment : ℝ) (percent_stock : ℝ) (market_price : ℝ) : ℝ :=
  (investment * percent_stock / 100) / market_price * market_price

theorem investment_amount (annual_income_value : ℝ) (percent_stock : ℝ) (market_price : ℝ) (investment : ℝ) :
  annual_income investment percent_stock market_price = annual_income_value →
  investment = 6800 :=
by
  intros
  sorry

end investment_amount_l262_262139


namespace medical_bills_value_l262_262454

variable (M : ℝ)
variable (property_damage : ℝ := 40000)
variable (insurance_coverage : ℝ := 0.80)
variable (carl_coverage : ℝ := 0.20)
variable (carl_owes : ℝ := 22000)

theorem medical_bills_value : 0.20 * (property_damage + M) = carl_owes → M = 70000 := 
by
  intro h
  sorry

end medical_bills_value_l262_262454


namespace find_B_value_l262_262104

theorem find_B_value (A B : ℕ) : (A * 100 + B * 10 + 2) - 41 = 591 → B = 3 :=
by
  sorry

end find_B_value_l262_262104


namespace no_four_consecutive_product_square_l262_262961

/-- Prove that there do not exist four consecutive positive integers whose product is a perfect square. -/
theorem no_four_consecutive_product_square :
  ¬ ∃ (x : ℕ), ∃ (n : ℕ), n * n = x * (x + 1) * (x + 2) * (x + 3) :=
sorry

end no_four_consecutive_product_square_l262_262961


namespace number_of_tables_large_meeting_l262_262506

-- Conditions
def table_length : ℕ := 2
def table_width : ℕ := 1
def side_length_large_meeting : ℕ := 7

-- To be proved: number of tables needed for a large meeting is 12.
theorem number_of_tables_large_meeting : 
  let tables_per_side := side_length_large_meeting / (table_length + table_width)
  ∃ total_tables, total_tables = 4 * tables_per_side ∧ total_tables = 12 :=
by
  sorry

end number_of_tables_large_meeting_l262_262506


namespace cookies_per_batch_l262_262178

def family_size := 4
def chips_per_person := 18
def chips_per_cookie := 2
def batches := 3

theorem cookies_per_batch : (family_size * chips_per_person) / chips_per_cookie / batches = 12 := 
by
  -- Proof will go here
  sorry

end cookies_per_batch_l262_262178


namespace sequence_part1_sequence_part2_l262_262010

theorem sequence_part1 :
  ∃ a1 a2 a3 : ℝ, a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0 ∧
  (a1 + a2 + a3)^2 = a1^3 + a2^3 + a3^3 ∧
  ((a1 = 1 ∧ a2 = 2 ∧ a3 = 3) ∨ 
   (a1 = 1 ∧ a2 = 2 ∧ a3 = -2) ∨ 
   (a1 = 1 ∧ a2 = -1 ∧ a3 = 1)) :=
by sorry

theorem sequence_part2 :
  ∃ (a : ℕ → ℝ), (∀ n, a n ≠ 0) ∧ 
  (∀ n, (finset.range n).sum (λ k, a k) ^ 2 = (finset.range n).sum (λ k, a k ^ 3)) ∧
  a 2013 = -2012 ∧ 
  (∀ n, n ≤ 2012 → a n = n) ∧ 
  (∀ n, n > 2012 → a n = 2012 * (-1)^(n+1)) :=
by sorry

end sequence_part1_sequence_part2_l262_262010


namespace smallest_prime_with_digit_sum_23_l262_262412

-- Definition for the conditions
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The theorem stating the proof problem
theorem smallest_prime_with_digit_sum_23 : ∃ p : ℕ, Prime p ∧ sum_of_digits p = 23 ∧ p = 1993 := 
by {
 sorry
}

end smallest_prime_with_digit_sum_23_l262_262412


namespace ny_mets_fans_count_l262_262700

-- Define the known ratios and total fans
def ratio_Y_to_M (Y M : ℕ) : Prop := 3 * M = 2 * Y
def ratio_M_to_R (M R : ℕ) : Prop := 4 * R = 5 * M
def total_fans (Y M R : ℕ) : Prop := Y + M + R = 330

-- Define what we want to prove
theorem ny_mets_fans_count (Y M R : ℕ) (h1 : ratio_Y_to_M Y M) (h2 : ratio_M_to_R M R) (h3 : total_fans Y M R) : M = 88 :=
sorry

end ny_mets_fans_count_l262_262700


namespace number_count_l262_262943

open Nat

theorem number_count (n a k m : ℕ) (n_pos : n > 0) (m_bound : m < 10^k)
    (key_eqn : 8 * m = 10^k * (a + n)) : 
    (number_of_combinations (λ m a k n, 8 * m = 10^k * (a + n) ∧ n > 0 ∧ m < 10^k) = 28) :=
sorry

end number_count_l262_262943


namespace probability_ratio_l262_262321

theorem probability_ratio :
  let draws := 4
  let total_slips := 40
  let numbers := 10
  let slips_per_number := 4
  let p := 10 / (Nat.choose total_slips draws)
  let q := (Nat.choose numbers 2) * (Nat.choose slips_per_number 2) * (Nat.choose slips_per_number 2) / (Nat.choose total_slips draws)
  p ≠ 0 →
  (q / p) = 162 :=
by
  sorry

end probability_ratio_l262_262321


namespace minimum_value_l262_262640

def f (x a : ℝ) : ℝ := x^3 - a*x^2 - a^2*x
def f_prime (x a : ℝ) : ℝ := 3*x^2 - 2*a*x - a^2

theorem minimum_value (a : ℝ) (hf_prime : f_prime 1 a = 0) (ha : a = -3) : ∃ x : ℝ, f x a = -5 := 
sorry

end minimum_value_l262_262640


namespace greatest_two_digit_product_12_l262_262782

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l262_262782


namespace problem1_problem2_l262_262308

variable {a b x : ℝ}

theorem problem1 (h₀ : a ≠ b) (h₁ : a ≠ -b) :
  (a / (a - b)) - (b / (a + b)) = (a^2 + b^2) / (a^2 - b^2) :=
sorry

theorem problem2 (h₀ : x ≠ 2) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  ((x - 2) / (x - 1)) / ((x^2 - 4 * x + 4) / (x^2 - 1)) + ((1 - x) / (x - 2)) = 2 / (x - 2) :=
sorry

end problem1_problem2_l262_262308


namespace find_sixth_number_l262_262905

theorem find_sixth_number (A : ℕ → ℤ) 
  (h1 : (1 / 11 : ℚ) * (A 1 + A 2 + A 3 + A 4 + A 5 + A 6 + A 7 + A 8 + A 9 + A 10 + A 11) = 60)
  (h2 : (1 / 6 : ℚ) * (A 1 + A 2 + A 3 + A 4 + A 5 + A 6) = 88)
  (h3 : (1 / 6 : ℚ) * (A 6 + A 7 + A 8 + A 9 + A 10 + A 11) = 65) :
  A 6 = 258 :=
sorry

end find_sixth_number_l262_262905


namespace min_m_plus_inv_m_min_frac_expr_l262_262985

-- Sub-problem (1): Minimum value of m + 1/m for m > 0.
theorem min_m_plus_inv_m (m : ℝ) (h : m > 0) : m + 1/m = 2 :=
sorry

-- Sub-problem (2): Minimum value of (x^2 + x - 5)/(x - 2) for x > 2.
theorem min_frac_expr (x : ℝ) (h : x > 2) : (x^2 + x - 5)/(x - 2) = 7 :=
sorry

end min_m_plus_inv_m_min_frac_expr_l262_262985


namespace car_travel_distance_l262_262410

-- Define the conditions: speed and time
def speed : ℝ := 160 -- in km/h
def time : ℝ := 5 -- in hours

-- Define the calculation for distance
def distance (s t : ℝ) : ℝ := s * t

-- Prove that given the conditions, the distance is 800 km
theorem car_travel_distance : distance speed time = 800 := by
  sorry

end car_travel_distance_l262_262410


namespace rajan_income_l262_262394

theorem rajan_income (x y : ℝ) 
  (h₁ : 7 * x - 6 * y = 1000) 
  (h₂ : 6 * x - 5 * y = 1000) : 
  7 * x = 7000 :=
by 
  sorry

end rajan_income_l262_262394


namespace calc_expression_l262_262128

theorem calc_expression (a : ℝ) : 4 * a * a^3 - a^4 = 3 * a^4 := by
  sorry

end calc_expression_l262_262128


namespace car_rental_cost_l262_262720

theorem car_rental_cost
  (rent_per_day : ℝ) (cost_per_mile : ℝ) (days_rented : ℕ) (miles_driven : ℝ)
  (h1 : rent_per_day = 30)
  (h2 : cost_per_mile = 0.25)
  (h3 : days_rented = 5)
  (h4 : miles_driven = 500) :
  rent_per_day * days_rented + cost_per_mile * miles_driven = 275 := 
  by
  sorry

end car_rental_cost_l262_262720


namespace problem_solution_l262_262111

theorem problem_solution : (90 + 5) * (12 / (180 / (3^2))) = 57 :=
by
  sorry

end problem_solution_l262_262111


namespace dodecahedron_interior_diagonals_l262_262644

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l262_262644


namespace tenth_term_arithmetic_sequence_l262_262384

theorem tenth_term_arithmetic_sequence (a d : ℤ)
  (h1 : a + 3 * d = 23)
  (h2 : a + 7 * d = 55) :
  a + 9 * d = 71 :=
sorry

end tenth_term_arithmetic_sequence_l262_262384


namespace greatest_two_digit_with_product_12_l262_262796

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l262_262796


namespace cylinder_lateral_surface_area_l262_262983

theorem cylinder_lateral_surface_area :
  let side := 20
  let radius := side / 2
  let height := side
  2 * Real.pi * radius * height = 400 * Real.pi :=
by
  let side := 20
  let radius := side / 2
  let height := side
  sorry

end cylinder_lateral_surface_area_l262_262983


namespace problem_statement_l262_262011

-- Given that f(x) is an even function.
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Definition of the main condition f(x) + f(2 - x) = 0.
def special_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (2 - x) = 0

-- Theorem: Given the conditions, show that f(x) has a period of 4 and f(x-1) is odd.
theorem problem_statement {f : ℝ → ℝ} (h_even : is_even f) (h_cond : special_condition f) :
  (∀ x, f (4 + x) = f x) ∧ (∀ x, f (-x - 1) = -f (x - 1)) :=
by
  sorry

end problem_statement_l262_262011


namespace cos_sum_of_angles_l262_262471

theorem cos_sum_of_angles (α β : Real) (h1 : Real.sin α = 4/5) (h2 : (π/2) < α ∧ α < π) 
(h3 : Real.cos β = -5/13) (h4 : 0 < β ∧ β < π/2) : 
  Real.cos (α + β) = -33/65 := 
by
  sorry

end cos_sum_of_angles_l262_262471


namespace classroom_student_count_l262_262229

theorem classroom_student_count (n : ℕ) (students_avg : ℕ) (teacher_age : ℕ) (combined_avg : ℕ) 
  (h1 : students_avg = 8) (h2 : teacher_age = 32) (h3 : combined_avg = 11) 
  (h4 : (8 * n + 32) / (n + 1) = 11) : n + 1 = 8 :=
by
  sorry

end classroom_student_count_l262_262229


namespace max_volume_of_acetic_acid_solution_l262_262916

theorem max_volume_of_acetic_acid_solution :
  (∀ (V : ℝ), 0 ≤ V ∧ (V * 0.09) = (25 * 0.7 + (V - 25) * 0.05)) →
  V = 406.25 :=
by
  sorry

end max_volume_of_acetic_acid_solution_l262_262916


namespace gcd_7_nplus2_8_2nplus1_l262_262758

theorem gcd_7_nplus2_8_2nplus1 : 
  ∃ d : ℕ, (∀ n : ℕ, d ∣ (7^(n+2) + 8^(2*n+1))) ∧ (∀ n : ℕ, d = 57) :=
sorry

end gcd_7_nplus2_8_2nplus1_l262_262758


namespace total_oak_trees_after_planting_l262_262546

-- Definitions based on conditions
def initial_oak_trees : ℕ := 5
def new_oak_trees : ℕ := 4

-- Statement of the problem and solution
theorem total_oak_trees_after_planting : initial_oak_trees + new_oak_trees = 9 := by
  sorry

end total_oak_trees_after_planting_l262_262546


namespace part_a_part_b_l262_262286

def initial_rubles : ℕ := 12000
def exchange_rate_initial : ℚ := 60
def guaranteed_return_rate : ℚ := 0.12
def exchange_rate_final : ℚ := 80
def currency_conversion_fee : ℚ := 0.04
def broker_commission_rate : ℚ := 0.25

theorem part_a 
  (initial_rubles = 12000)
  (exchange_rate_initial = 60)
  (guaranteed_return_rate = 0.12)
  (exchange_rate_final = 80)
  (currency_conversion_fee = 0.04)
  (broker_commission_rate = 0.25) :
  let initial_dollars := initial_rubles / exchange_rate_initial
  let profit_dollars := initial_dollars * guaranteed_return_rate
  let total_dollars := initial_dollars + profit_dollars
  let broker_commission := profit_dollars * broker_commission_rate
  let dollars_after_commission := total_dollars - broker_commission
  let final_rubles := dollars_after_commission * exchange_rate_final
  let conversion_fee := final_rubles * currency_conversion_fee
  in final_rubles - conversion_fee = 16742.4 := by
  sorry

theorem part_b 
  (initial_rubles = 12000)
  (final_rubles = 16742.4) :
  let rate_of_return := (final_rubles / initial_rubles) - 1
  in rate_of_return * 100 = 39.52 := by
  sorry

end part_a_part_b_l262_262286


namespace dodecahedron_interior_diagonals_eq_160_l262_262655

-- Definition of a dodecahedron for the context of the proof
structure Dodecahedron where
  num_faces : Nat
  num_vertices : Nat
  faces_meeting_at_vertex : Nat
  vertices_connected_by_edge : Nat

-- Given our specific dodecahedron properties
def dodecahedron : Dodecahedron := {
  num_faces := 12,
  num_vertices := 20,
  faces_meeting_at_vertex := 3,
  vertices_connected_by_edge := 3
}

-- The interior diagonal is defined as a segment connecting two vertices
-- which do not lie on the same face.
def is_interior_diagonal (d : Dodecahedron) (v1 v2 : Fin d.num_vertices) : Prop :=
  v1 ≠ v2 ∧ ¬∃ f, f < d.num_faces ∧ v1 ∈ f ∧ v2 ∈ f

-- Now, we need to calculate the total number of such interior diagonals
noncomputable def num_interior_diagonals (d : Dodecahedron) : Nat :=
  d.num_vertices * (d.num_vertices - d.vertices_connected_by_edge - 1) / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals_eq_160 :
  num_interior_diagonals dodecahedron = 160 := by
  sorry

end dodecahedron_interior_diagonals_eq_160_l262_262655


namespace greatest_two_digit_product_12_l262_262793

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l262_262793


namespace closest_square_to_350_l262_262267

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l262_262267


namespace dodecahedron_interior_diagonals_l262_262643

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l262_262643


namespace f_triple_application_l262_262350

-- Define the function f : ℕ → ℕ such that f(x) = 3x + 2
def f (x : ℕ) : ℕ := 3 * x + 2

-- Theorem statement to prove f(f(f(1))) = 53
theorem f_triple_application : f (f (f 1)) = 53 := 
by 
  sorry

end f_triple_application_l262_262350


namespace dodecahedron_interior_diagonals_l262_262657

theorem dodecahedron_interior_diagonals (vertices faces: ℕ) (pentagonal: ∀ face, face = 5) (vertices_per_face : ∀ face ∈ (range faces), 3) :
  vertices = 20 ∧ faces = 12 ∧ interior_diagonals = 160 :=
by
    -- Definitions and problem conditions
    let vertices := 20
    let faces := 12
    let pentagonal := (λ face, face = 5)
    let vertices_per_face := (λ face, 3)

    -- Assertion
    have interior_diagonals : ℕ := (vertices * 16) / 2
    rw Nat.div_eq_of_eq_mul_right _ _ _,
    have h: vertices = 20 ∧ faces = 12 ∧ interior_diagonals = 160 := by
        sorry

    exact h

end dodecahedron_interior_diagonals_l262_262657


namespace mean_of_all_students_is_79_l262_262896

def mean_score_all_students (F S : ℕ) (f s : ℕ) (hf : f = 2/5 * s) : ℕ :=
  (36 * s + 75 * s) / ((2/5 * s) + s)

theorem mean_of_all_students_is_79 (F S : ℕ) (f s : ℕ) (hf : f = 2/5 * s) (hF : F = 90) (hS : S = 75) : 
  mean_score_all_students F S f s hf = 79 := by
  sorry

end mean_of_all_students_is_79_l262_262896


namespace find_A_minus_B_l262_262295

def A : ℕ := (55 * 100) + (19 * 10)
def B : ℕ := 173 + (5 * 224)

theorem find_A_minus_B : A - B = 4397 := by
  sorry

end find_A_minus_B_l262_262295


namespace probability_of_sum_17_l262_262696

noncomputable def prob_sum_dice_is_seventeen : ℚ :=
1 / 72

theorem probability_of_sum_17 :
  let dice := finset.product (finset.product finset.univ finset.univ) finset.univ in
  let event := dice.filter (λ (x : ℕ × (ℕ × ℕ)), x.1 + x.2.1 + x.2.2 = 17) in
  (event.card : ℚ) / (dice.card : ℚ) = prob_sum_dice_is_seventeen :=
by
  sorry

end probability_of_sum_17_l262_262696


namespace a_pow_5_mod_11_l262_262511

theorem a_pow_5_mod_11 (a : ℕ) : (a^5) % 11 = 0 ∨ (a^5) % 11 = 1 ∨ (a^5) % 11 = 10 :=
sorry

end a_pow_5_mod_11_l262_262511


namespace fair_tickets_sold_l262_262538

theorem fair_tickets_sold (F : ℕ) (number_of_baseball_game_tickets : ℕ) 
  (h1 : F = 2 * number_of_baseball_game_tickets + 6) (h2 : number_of_baseball_game_tickets = 56) :
  F = 118 :=
by
  sorry

end fair_tickets_sold_l262_262538


namespace total_goals_is_15_l262_262841

-- Define the conditions as variables
def KickersFirstPeriodGoals : ℕ := 2
def KickersSecondPeriodGoals : ℕ := 2 * KickersFirstPeriodGoals
def SpidersFirstPeriodGoals : ℕ := KickersFirstPeriodGoals / 2
def SpidersSecondPeriodGoals : ℕ := 2 * KickersSecondPeriodGoals

-- Define total goals by each team
def TotalKickersGoals : ℕ := KickersFirstPeriodGoals + KickersSecondPeriodGoals
def TotalSpidersGoals : ℕ := SpidersFirstPeriodGoals + SpidersSecondPeriodGoals

-- Define total goals by both teams
def TotalGoals : ℕ := TotalKickersGoals + TotalSpidersGoals

-- Prove the statement
theorem total_goals_is_15 : TotalGoals = 15 :=
by
  sorry

end total_goals_is_15_l262_262841


namespace calculate_expression_l262_262837

theorem calculate_expression :
  18 - ((-16) / (2 ^ 3)) = 20 :=
by
  sorry

end calculate_expression_l262_262837


namespace escalator_time_l262_262587

theorem escalator_time (speed_escalator: ℝ) (length_escalator: ℝ) (speed_person: ℝ) (combined_speed: ℝ)
  (h1: speed_escalator = 20) (h2: length_escalator = 250) (h3: speed_person = 5) (h4: combined_speed = speed_escalator + speed_person) :
  length_escalator / combined_speed = 10 := by
  sorry

end escalator_time_l262_262587


namespace imaginary_part_of_complex_l262_262737

theorem imaginary_part_of_complex :
  let i := Complex.I
  let z := 10 * i / (3 + i)
  z.im = 3 :=
by
  sorry

end imaginary_part_of_complex_l262_262737


namespace sin_double_angle_l262_262472

theorem sin_double_angle (x : ℝ) (h : Real.sin (x - π / 4) = 3 / 5) : Real.sin (2 * x) = 7 / 25 :=
by
  sorry

end sin_double_angle_l262_262472


namespace min_value_of_f_l262_262862

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 3 - 1) * (Real.log x / Real.log 3 - 3)

theorem min_value_of_f (x1 x2 : ℝ) (hx1_pos : 0 < x1) (hx1_distinct : x1 ≠ x2) (hx2_pos : 0 < x2)
  (h_f_eq : f x1 = f x2) : (1 / x1 + 9 / x2) = 2 / 3 :=
by
  sorry

end min_value_of_f_l262_262862


namespace area_ratio_proof_l262_262108

noncomputable def area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) : ℝ := 
  (a * b) / (c * d)

theorem area_ratio_proof (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  area_ratio a b c d h1 h2 = 4 / 9 := by
  sorry

end area_ratio_proof_l262_262108


namespace find_n_l262_262890

-- Define the arithmetic series sums
def s1 (n : ℕ) : ℕ := (5 * n^2 + 5 * n) / 2
def s2 (n : ℕ) : ℕ := n^2 + n

-- The theorem to be proved
theorem find_n : ∃ n : ℕ, s1 n + s2 n = 156 ∧ n = 7 :=
by
  sorry

end find_n_l262_262890


namespace dodecahedron_interior_diagonals_l262_262680

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l262_262680


namespace negation_example_l262_262476

theorem negation_example :
  (¬ (∃ n : ℕ, n^2 ≥ 2^n)) → (∀ n : ℕ, n^2 < 2^n) :=
by
  sorry

end negation_example_l262_262476


namespace pencil_cost_l262_262942

theorem pencil_cost (p e : ℝ) (h1 : p + e = 3.40) (h2 : p = 3 + e) : p = 3.20 :=
by
  sorry

end pencil_cost_l262_262942


namespace factor_expression_l262_262136

theorem factor_expression (x : ℝ) : 45 * x^3 + 135 * x^2 = 45 * x^2 * (x + 3) :=
  by
    sorry

end factor_expression_l262_262136


namespace total_blue_marbles_l262_262722

theorem total_blue_marbles (red_Jenny blue_Jenny red_Mary blue_Mary red_Anie blue_Anie : ℕ)
  (h1: red_Jenny = 30)
  (h2: blue_Jenny = 25)
  (h3: red_Mary = 2 * red_Jenny)
  (h4: blue_Mary = blue_Anie / 2)
  (h5: red_Anie = red_Mary + 20)
  (h6: blue_Anie = 2 * blue_Jenny) :
  blue_Mary + blue_Jenny + blue_Anie = 100 :=
by
  sorry

end total_blue_marbles_l262_262722


namespace some_number_is_105_l262_262163

def find_some_number (a : ℕ) (num : ℕ) : Prop :=
  a ^ 3 = 21 * 25 * num * 7

theorem some_number_is_105 (a : ℕ) (num : ℕ) (h : a = 105) (h_eq : find_some_number a num) : num = 105 :=
by
  sorry

end some_number_is_105_l262_262163


namespace find_D_l262_262584

variables (A B C D : ℤ)
axiom h1 : A + C = 15
axiom h2 : A - B = 1
axiom h3 : C + C = A
axiom h4 : B - D = 2
axiom h5 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem find_D : D = 7 :=
by sorry

end find_D_l262_262584


namespace problem_l262_262181

-- Definitions for conditions
def countMultiplesOf (n upperLimit : ℕ) : ℕ :=
  (upperLimit - 1) / n

def a : ℕ := countMultiplesOf 4 40
def b : ℕ := countMultiplesOf 4 40

-- Statement to prove
theorem problem : (a + b)^2 = 324 := by
  sorry

end problem_l262_262181


namespace expand_polynomial_expression_l262_262969

theorem expand_polynomial_expression (x : ℝ) : 
  (x + 6) * (x + 8) * (x - 3) = x^3 + 11 * x^2 + 6 * x - 144 :=
by
  sorry

end expand_polynomial_expression_l262_262969


namespace cows_in_group_l262_262879

theorem cows_in_group (D C : ℕ) 
  (h : 2 * D + 4 * C = 2 * (D + C) + 36) : 
  C = 18 :=
by
  sorry

end cows_in_group_l262_262879


namespace find_x_l262_262889

variable {a b x : ℝ}
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(2 * b) = a^b * x^b)

theorem find_x (h₀ : b ≠ 0) (h₁ : (3 * a)^(2 * b) = a^b * x^b) : x = 9 * a :=
by
  sorry

end find_x_l262_262889


namespace isosceles_right_triangle_area_l262_262224

noncomputable def triangle_area (p : ℝ) : ℝ :=
  (1 / 8) * ((p + p * Real.sqrt 2 + 2) * (2 - Real.sqrt 2)) ^ 2

theorem isosceles_right_triangle_area (p : ℝ) :
  let perimeter := p + p * Real.sqrt 2 + 2
  let x := (p + p * Real.sqrt 2 + 2) * (2 - Real.sqrt 2) / 2
  let area := 1 / 2 * x ^ 2
  area = triangle_area p :=
by
  sorry

end isosceles_right_triangle_area_l262_262224


namespace Walter_allocates_for_school_l262_262558

open Nat

def Walter_works_5_days_a_week := 5
def Walter_earns_per_hour := 5
def Walter_works_per_day := 4
def Proportion_for_school := 3/4

theorem Walter_allocates_for_school :
  let daily_earnings := Walter_works_per_day * Walter_earns_per_hour
  let weekly_earnings := daily_earnings * Walter_works_5_days_a_week
  let school_allocation := weekly_earnings * Proportion_for_school
  school_allocation = 75 := by
  sorry

end Walter_allocates_for_school_l262_262558


namespace common_ratio_of_geometric_sequence_l262_262294

-- Define the problem conditions and goal
theorem common_ratio_of_geometric_sequence 
  (a1 : ℝ)  -- nonzero first term
  (h₁ : a1 ≠ 0) -- first term is nonzero
  (r : ℝ)  -- common ratio
  (h₂ : r > 0) -- ratio is positive
  (h₃ : ∀ n m : ℕ, n ≠ m → a1 * r^n ≠ a1 * r^m) -- distinct terms in sequence
  (h₄ : a1 * r * r * r = (a1 * r) * (a1 * r^3) ∧ a1 * r ≠ (a1 * r^4)) -- arithmetic sequence condition
  : r = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l262_262294


namespace athlete_total_heartbeats_l262_262952

/-
  An athlete's heart rate starts at 140 beats per minute at the beginning of a race
  and increases by 5 beats per minute for each subsequent mile. How many times does
  the athlete's heart beat during a 10-mile race if the athlete runs at a pace of
  6 minutes per mile?
-/

def athlete_heartbeats (initial_rate : ℕ) (increase_rate : ℕ) (miles : ℕ) (minutes_per_mile : ℕ) : ℕ :=
  let n := miles
  let a := initial_rate
  let l := initial_rate + (increase_rate * (miles - 1))
  let S := (n * (a + l)) / 2
  S * minutes_per_mile

theorem athlete_total_heartbeats :
  athlete_heartbeats 140 5 10 6 = 9750 :=
sorry

end athlete_total_heartbeats_l262_262952


namespace find_z_l262_262855

open Complex

theorem find_z (z : ℂ) (h : ((1 - I) ^ 2) / z = 1 + I) : z = -1 - I :=
sorry

end find_z_l262_262855


namespace relationship_between_a_b_l262_262060

theorem relationship_between_a_b (a b x : ℝ) (h1 : 2 * x = a + b) (h2 : 2 * x^2 = a^2 - b^2) : 
  a = -b ∨ a = 3 * b :=
  sorry

end relationship_between_a_b_l262_262060


namespace distinct_roots_of_quadratic_l262_262186

variable {a b : ℝ}
-- condition: a and b are distinct
variable (h_distinct: a ≠ b)

theorem distinct_roots_of_quadratic (a b : ℝ) (h_distinct : a ≠ b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x + a)*(x + b) = 2*x + a + b :=
by
  sorry

end distinct_roots_of_quadratic_l262_262186


namespace probability_palindrome_divisible_by_11_is_zero_l262_262118

def is_palindrome (n : ℕ) :=
  3000 ≤ n ∧ n < 8000 ∧ ∃ (a b : ℕ), 3 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem probability_palindrome_divisible_by_11_is_zero :
  (∃ (n : ℕ), is_palindrome n ∧ n % 11 = 0) → false := by sorry

end probability_palindrome_divisible_by_11_is_zero_l262_262118


namespace asbestos_tiles_width_l262_262948

theorem asbestos_tiles_width (n : ℕ) (h : 0 < n) :
  let width_per_tile := 60
  let overlap := 10
  let effective_width := width_per_tile - overlap
  width_per_tile + (n - 1) * effective_width = 50 * n + 10 := by
sorry

end asbestos_tiles_width_l262_262948


namespace f_inv_f_inv_14_l262_262902

noncomputable def f (x : ℝ) : ℝ := 3 * x + 7

noncomputable def f_inv (x : ℝ) : ℝ := (x - 7) / 3

theorem f_inv_f_inv_14 : f_inv (f_inv 14) = -14 / 9 :=
by {
  sorry
}

end f_inv_f_inv_14_l262_262902


namespace base_length_of_isosceles_triangle_l262_262085

-- Definitions based on given conditions
def is_isosceles (a b : ℕ) (c : ℕ) :=
a = b ∧ c = c

def side_length : ℕ := 6
def perimeter : ℕ := 20

-- Theorem to prove the base length
theorem base_length_of_isosceles_triangle (b : ℕ) (h1 : 2 * side_length + b = perimeter) :
  b = 8 :=
sorry

end base_length_of_isosceles_triangle_l262_262085


namespace white_pieces_remaining_after_process_l262_262407

-- Definition to describe the removal process
def remove_every_second (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

-- Recursive function to model the process of removing pieces
def remaining_white_pieces (initial_white : ℕ) (rounds : ℕ) : ℕ :=
  match rounds with
  | 0     => initial_white
  | n + 1 => remaining_white_pieces (remove_every_second initial_white) n

-- Main theorem statement
theorem white_pieces_remaining_after_process :
  remaining_white_pieces 1990 4 = 124 :=
by
  sorry

end white_pieces_remaining_after_process_l262_262407


namespace remainder_of_division_l262_262535

theorem remainder_of_division : 
  ∀ (L x : ℕ), (L = 1430) → 
               (L - x = 1311) → 
               (L = 11 * x + (L % x)) → 
               (L % x = 121) :=
by
  intros L x L_value diff quotient
  sorry

end remainder_of_division_l262_262535


namespace new_rectangle_area_eq_a_squared_l262_262309

theorem new_rectangle_area_eq_a_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let d := Real.sqrt (a^2 + b^2)
  let base := 2 * (d + b)
  let height := (d - b) / 2
  base * height = a^2 := by
  sorry

end new_rectangle_area_eq_a_squared_l262_262309


namespace nell_gave_cards_l262_262373

theorem nell_gave_cards (c_original : ℕ) (c_left : ℕ) (cards_given : ℕ) :
  c_original = 528 → c_left = 252 → cards_given = c_original - c_left → cards_given = 276 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end nell_gave_cards_l262_262373


namespace emily_olivia_books_l262_262465

theorem emily_olivia_books (shared_books total_books_emily books_olivia_not_in_emily : ℕ)
  (h1 : shared_books = 15)
  (h2 : total_books_emily = 23)
  (h3 : books_olivia_not_in_emily = 8) : (total_books_emily - shared_books + books_olivia_not_in_emily = 16) :=
by
  sorry

end emily_olivia_books_l262_262465


namespace quotient_change_l262_262392

variables {a b : ℝ} (h : a / b = 0.78)

theorem quotient_change (a b : ℝ) (h : a / b = 0.78) : (10 * a) / (b / 10) = 78 :=
by
  sorry

end quotient_change_l262_262392


namespace greatest_two_digit_with_product_12_l262_262764

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l262_262764


namespace sqrt_four_eq_pm_two_l262_262401

theorem sqrt_four_eq_pm_two : ∀ (x : ℝ), x * x = 4 ↔ x = 2 ∨ x = -2 := 
by
  sorry

end sqrt_four_eq_pm_two_l262_262401


namespace sqrt_4_eq_pm2_l262_262403

theorem sqrt_4_eq_pm2 : {y : ℝ | y^2 = 4} = {2, -2} :=
by
  sorry

end sqrt_4_eq_pm2_l262_262403


namespace pyramid_levels_l262_262066

theorem pyramid_levels (n : ℕ) (h : (n * (n + 1) * (2 * n + 1)) / 6 = 225) : n = 6 :=
by
  sorry

end pyramid_levels_l262_262066


namespace nala_seashells_l262_262195

theorem nala_seashells (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 2 * (a + b)) : a + b + c = 36 :=
by {
  sorry
}

end nala_seashells_l262_262195


namespace sum_zero_quotient_l262_262891

   theorem sum_zero_quotient (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum_zero : x + y + z = 0) :
     (xy + yz + zx) / (x^2 + y^2 + z^2) = -1 / 2 :=
   by
     sorry
   
end sum_zero_quotient_l262_262891


namespace cos_beta_l262_262998

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = 3 / 5)
variable (h2 : Real.cos (α + β) = 5 / 13)

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 3 / 5) (h2 : Real.cos (α + β) = 5 / 13) : 
  Real.cos β = 56 / 65 := by
  sorry

end cos_beta_l262_262998


namespace sum_of_solutions_l262_262413

-- Define the polynomial equation and the condition
def equation (x : ℝ) : Prop := 3 = (x^3 - 3 * x^2 - 12 * x) / (x + 3)

-- Sum of solutions for the given polynomial equation under the constraint
theorem sum_of_solutions :
  (∀ x : ℝ, equation x → x ≠ -3) →
  ∃ (a b : ℝ), equation a ∧ equation b ∧ a + b = 4 := 
by
  intros h
  sorry

end sum_of_solutions_l262_262413


namespace sum_of_sequence_l262_262630

noncomputable def f (n x : ℝ) : ℝ := (1 / (8 * n)) * x^2 + 2 * n * x

theorem sum_of_sequence (n : ℕ) (hn : n > 0) :
  let a : ℝ := 1 / (8 * n)
  let b : ℝ := 2 * n
  let f' := 2 * a * ((-n : ℝ )) + b 
  ∃ S : ℝ, S = (n - 1) * 2^(n + 1) + 2 := 
sorry

end sum_of_sequence_l262_262630


namespace max_cells_cut_diagonals_l262_262101

theorem max_cells_cut_diagonals (board_size : ℕ) (k : ℕ) (internal_cells : ℕ) :
  board_size = 9 →
  internal_cells = (board_size - 2) ^ 2 →
  64 = internal_cells →
  V = internal_cells + k →
  E = 4 * k →
  k ≤ 21 :=
by
  sorry

end max_cells_cut_diagonals_l262_262101


namespace simplification_problem_l262_262806

theorem simplification_problem :
  (3^2015 - 3^2013 + 3^2011) / (3^2015 + 3^2013 - 3^2011) = 73 / 89 :=
  sorry

end simplification_problem_l262_262806


namespace complex_addition_zero_l262_262854

theorem complex_addition_zero (a b : ℝ) (i : ℂ) (h1 : (1 + i) * i = a + b * i) (h2 : i * i = -1) : a + b = 0 :=
sorry

end complex_addition_zero_l262_262854


namespace price_of_33_kgs_l262_262588

theorem price_of_33_kgs (l q : ℝ) 
  (h1 : l * 20 = 100) 
  (h2 : l * 30 + q * 6 = 186) : 
  l * 30 + q * 3 = 168 := 
by
  sorry

end price_of_33_kgs_l262_262588


namespace combination_square_octagon_tiles_l262_262275

-- Define the internal angles of the polygons
def internal_angle (shape : String) : Float :=
  match shape with
  | "Square"   => 90.0
  | "Pentagon" => 108.0
  | "Hexagon"  => 120.0
  | "Octagon"  => 135.0
  | _          => 0.0

-- Define the condition for the combination of two regular polygons to tile seamlessly
def can_tile (shape1 shape2 : String) : Bool :=
  let angle1 := internal_angle shape1
  let angle2 := internal_angle shape2
  angle1 + 2 * angle2 == 360.0

-- Define the tiling problem
theorem combination_square_octagon_tiles : can_tile "Square" "Octagon" = true :=
by {
  -- The proof of this theorem should show that Square and Octagon can indeed tile seamlessly
  sorry
}

end combination_square_octagon_tiles_l262_262275


namespace count_numbers_with_square_factors_l262_262035

theorem count_numbers_with_square_factors :
  let squares := [4, 9, 16, 25, 36, 49, 64]
  let multiples (n : ℕ) := ∀ k ∈ squares, n % k = 0
  let count_multiples (n : ℕ) := (1..100).count multiples
  count_multiples squares = 48 :=
  sorry

end count_numbers_with_square_factors_l262_262035


namespace solve_for_s_l262_262493

theorem solve_for_s (s t : ℚ) (h1 : 15 * s + 7 * t = 210) (h2 : t = 3 * s) : s = 35 / 6 := 
by
  sorry

end solve_for_s_l262_262493


namespace simplify_complex_expression_l262_262242

theorem simplify_complex_expression :
  (1 / (-8 ^ 2) ^ 4 * (-8) ^ 9) = -8 := by
  sorry

end simplify_complex_expression_l262_262242


namespace closest_perfect_square_to_350_l262_262260

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l262_262260


namespace finite_decimal_fractions_l262_262846

theorem finite_decimal_fractions (a b c d : ℕ) (n : ℕ) 
  (h1 : n = 2^a * 5^b)
  (h2 : n + 1 = 2^c * 5^d) :
  n = 1 ∨ n = 4 :=
by
  sorry

end finite_decimal_fractions_l262_262846


namespace gcd_12345_6789_eq_3_l262_262755

theorem gcd_12345_6789_eq_3 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_eq_3_l262_262755


namespace libby_quarters_left_l262_262516

theorem libby_quarters_left (initial_quarters : ℕ) (dress_cost_dollars : ℕ) (quarters_per_dollar : ℕ) 
  (h1 : initial_quarters = 160) (h2 : dress_cost_dollars = 35) (h3 : quarters_per_dollar = 4) : 
  initial_quarters - (dress_cost_dollars * quarters_per_dollar) = 20 := by
  sorry

end libby_quarters_left_l262_262516


namespace smallest_possible_value_of_a_largest_possible_value_of_a_l262_262368

-- Define that a is a positive integer and there are exactly 10 perfect squares greater than a and less than 2a

variable (a : ℕ) (h1 : a > 0)
variable (h2 : ∃ (s : ℕ) (t : ℕ), s + 10 = t ∧ (s^2 > a) ∧ (s + 9)^2 < 2 * a ∧ (t^2 - 10) + 9 < 2 * a)

-- Prove the smallest value of a
theorem smallest_possible_value_of_a : a = 481 :=
by sorry

-- Prove the largest value of a
theorem largest_possible_value_of_a : a = 684 :=
by sorry

end smallest_possible_value_of_a_largest_possible_value_of_a_l262_262368


namespace limits_of_ratios_l262_262473

noncomputable def x_n (n : ℕ) : ℝ := (1 + Real.sqrt 2 + Real.sqrt 3) ^ n

noncomputable def q_n (n : ℕ) : ℤ := -- definition derived from x_n
noncomputable def r_n (n : ℕ) : ℤ := -- definition derived from x_n
noncomputable def s_n (n : ℕ) : ℤ := -- definition derived from x_n
noncomputable def t_n (n : ℕ) : ℤ := -- definition derived from x_n

theorem limits_of_ratios :
  (∀ n, x_n n = (q_n n : ℝ) + (r_n n : ℝ) * Real.sqrt 2 + (s_n n : ℝ) * Real.sqrt 3 + (t_n n : ℝ) * Real.sqrt 6) →
  (∀ k, 2 ≤ k → |1 - Real.sqrt 2 + Real.sqrt 3| < |1 + Real.sqrt 2 + Real.sqrt 3| ∧
               |1 + Real.sqrt 2 - Real.sqrt 3| < |1 + Real.sqrt 2 + Real.sqrt 3| ∧
               |1 - Real.sqrt 2 - Real.sqrt 3| < |1 + Real.sqrt 2 + Real.sqrt 3|) →
  (lim n → ∞, (r_n n : ℝ) / (q_n n : ℝ)) = 1 / Real.sqrt 2 ∧
  (lim n → ∞, (s_n n : ℝ) / (q_n n : ℝ)) = 1 / Real.sqrt 3 ∧
  (lim n → ∞, (t_n n : ℝ) / (q_n n : ℝ)) = 1 / Real.sqrt 6 :=
by
  intros; sorry

end limits_of_ratios_l262_262473


namespace inequality_not_always_true_l262_262018

variables {a b c d : ℝ}

theorem inequality_not_always_true
  (h1 : a > b) (h2 : b > 0) (h3 : c > 0) (h4 : d ≠ 0) :
  ¬ ∀ (a b d : ℝ), (a > b) → (d ≠ 0) → (a + d)^2 > (b + d)^2 :=
by
  intro H
  specialize H a b d h1 h4
  sorry

end inequality_not_always_true_l262_262018


namespace files_deleted_is_3_l262_262602

-- Define the initial number of files
def initial_files : Nat := 24

-- Define the remaining number of files
def remaining_files : Nat := 21

-- Define the number of files deleted
def files_deleted : Nat := initial_files - remaining_files

-- Prove that the number of files deleted is 3
theorem files_deleted_is_3 : files_deleted = 3 :=
by
  sorry

end files_deleted_is_3_l262_262602


namespace sum_of_special_integers_l262_262058

theorem sum_of_special_integers :
  let a := 0
  let b := 1
  let c := -1
  a + b + c = 0 := by
  sorry

end sum_of_special_integers_l262_262058


namespace inequality_proof_l262_262570

open Real

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h1 : a ^ x = b * c) 
  (h2 : b ^ y = c * a) 
  (h3 : c ^ z = a * b) :
  (1 / (2 + x) + 1 / (2 + y) + 1 / (2 + z)) ≤ 3 / 4 := 
sorry

end inequality_proof_l262_262570


namespace six_coins_not_sum_to_14_l262_262733

def coin_values : Set ℕ := {1, 5, 10, 25}

theorem six_coins_not_sum_to_14 (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 ∈ coin_values) (h2 : a2 ∈ coin_values) (h3 : a3 ∈ coin_values) (h4 : a4 ∈ coin_values) (h5 : a5 ∈ coin_values) (h6 : a6 ∈ coin_values) : a1 + a2 + a3 + a4 + a5 + a6 ≠ 14 := 
sorry

end six_coins_not_sum_to_14_l262_262733


namespace dodecagon_enclosure_l262_262120

theorem dodecagon_enclosure (m n : ℕ) (h1 : m = 12) 
  (h2 : ∀ (x : ℕ), x ∈ { k | ∃ p : ℕ, p = n ∧ 12 = k * p}) :
  n = 12 :=
by
  -- begin proof steps here
sorry

end dodecagon_enclosure_l262_262120


namespace Sabrina_pencils_l262_262057

variable (S : ℕ) (J : ℕ)

theorem Sabrina_pencils (h1 : S + J = 50) (h2 : J = 2 * S + 8) :
  S = 14 :=
by
  sorry

end Sabrina_pencils_l262_262057


namespace fixed_point_of_line_l262_262319

theorem fixed_point_of_line (a : ℝ) : 
  (a + 3) * (-2) + (2 * a - 1) * 1 + 7 = 0 := 
by 
  sorry

end fixed_point_of_line_l262_262319


namespace arc_length_of_circle_l262_262147

theorem arc_length_of_circle (r : ℝ) (alpha : ℝ) (h_r : r = 10) (h_alpha : alpha = (2 * Real.pi) / 6) : 
  (alpha * r) = (10 * Real.pi) / 3 :=
by
  rw [h_r, h_alpha]
  sorry

end arc_length_of_circle_l262_262147


namespace total_seashells_l262_262194

theorem total_seashells (a b : Nat) (h1 : a = 5) (h2 : b = 7) : 
  let total_first_two_days := a + b
  let third_day := 2 * total_first_two_days
  let total := total_first_two_days + third_day
  total = 36 := 
by
  sorry

end total_seashells_l262_262194


namespace like_terms_exponents_l262_262045

theorem like_terms_exponents {m n : ℕ} (h1 : 4 * a * b^n = 4 * (a^1) * (b^n)) (h2 : -2 * a^m * b^4 = -2 * (a^m) * (b^4)) :
  (m = 1 ∧ n = 4) :=
by sorry

end like_terms_exponents_l262_262045


namespace square_root_problem_l262_262328

theorem square_root_problem
  (x : ℤ) (y : ℤ)
  (hx : x = Nat.sqrt 16)
  (hy : y^2 = 9) :
  x^2 + y^2 + x - 2 = 27 := by
  sorry

end square_root_problem_l262_262328


namespace length_of_AB_l262_262008

theorem length_of_AB 
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 + A.2 ^ 2 = 8)
  (hB : B.1 ^ 2 + B.2 ^ 2 = 8)
  (lA : A.1 - 2 * A.2 + 5 = 0)
  (lB : B.1 - 2 * B.2 + 5 = 0) :
  dist A B = 2 * Real.sqrt 3 := by
  sorry

end length_of_AB_l262_262008


namespace range_of_m_l262_262960

noncomputable def f : ℝ → ℝ := sorry

lemma function_symmetric {x : ℝ} : f (2 + x) = f (-x) := sorry

lemma f_decreasing_on_pos_halfline {x y : ℝ} (hx : x ≥ 1) (hy : y ≥ 1) (hxy : x < y) : f x ≥ f y := sorry

theorem range_of_m {m : ℝ} (h : f (1 - m) < f m) : m > (1 / 2) := sorry

end range_of_m_l262_262960


namespace geometric_sequence_min_n_l262_262395

theorem geometric_sequence_min_n (n : ℕ) (h : 2^(n + 1) - 2 - n > 1020) : n ≥ 10 :=
sorry

end geometric_sequence_min_n_l262_262395


namespace books_into_bags_l262_262591

def books := Finset.range 5
def bags := Finset.range 4

noncomputable def arrangement_count : ℕ :=
  -- definition of arrangement_count can be derived from the solution logic
  sorry

theorem books_into_bags : arrangement_count = 51 := 
  sorry

end books_into_bags_l262_262591


namespace geometric_sequence_product_l262_262705

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h1 : a 1 * a 3 * a 11 = 8) :
  a 2 * a 8 = 4 :=
sorry

end geometric_sequence_product_l262_262705


namespace cubic_sum_l262_262348

theorem cubic_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end cubic_sum_l262_262348


namespace total_area_painted_is_correct_l262_262821

noncomputable def barn_area_painted (width length height : ℝ) : ℝ :=
  let walls_area := 2 * (width * height + length * height) * 2
  let ceiling_and_roof_area := 2 * (width * length)
  walls_area + ceiling_and_roof_area

theorem total_area_painted_is_correct 
  (width length height : ℝ) 
  (h_w : width = 12) 
  (h_l : length = 15) 
  (h_h : height = 6) 
  : barn_area_painted width length height = 1008 :=
  by
  rw [h_w, h_l, h_h]
  -- Simplify steps omitted
  sorry

end total_area_painted_is_correct_l262_262821


namespace temperature_difference_l262_262091

def highest_temperature : ℤ := 8
def lowest_temperature : ℤ := -2

theorem temperature_difference :
  highest_temperature - lowest_temperature = 10 := by
  sorry

end temperature_difference_l262_262091


namespace dodecahedron_interior_diagonals_l262_262642

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l262_262642


namespace part_I_part_II_l262_262027

-- Define the function f(x) as per the problem's conditions
def f (x a : ℝ) : ℝ := abs (x - 2 * a) + abs (x - a)

theorem part_I (x : ℝ) (h₁ : 1 ≠ 0) : 
  (f x 1 > 2) ↔ (x < 1 / 2 ∨ x > 5 / 2) :=
by
  sorry

theorem part_II (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  f b a ≥ f a a ∧ (f b a = f a a ↔ ((2 * a - b ≥ 0 ∧ b - a ≥ 0) ∨ (2 * a - b ≤ 0 ∧ b - a ≤ 0) ∨ (2 * a - b = 0) ∨ (b - a = 0))) :=
by
  sorry

end part_I_part_II_l262_262027


namespace sum_of_ages_in_three_years_l262_262362

theorem sum_of_ages_in_three_years (H : ℕ) (J : ℕ) (SumAges : ℕ) 
  (h1 : J = 3 * H) 
  (h2 : H = 15) 
  (h3 : SumAges = (H + 3) + (J + 3)) : 
  SumAges = 66 :=
by
  sorry

end sum_of_ages_in_three_years_l262_262362


namespace roots_inequality_l262_262184

noncomputable def a : ℝ := Real.sqrt 2020

theorem roots_inequality (x1 x2 x3 : ℝ) (h_roots : ∀ x, (a * x^3 - 4040 * x^2 + 4 = 0) ↔ (x = x1 ∨ x = x2 ∨ x = x3))
  (h_inequality: x1 < x2 ∧ x2 < x3) : x2 * (x1 + x3) = 2 :=
sorry

end roots_inequality_l262_262184


namespace closest_perfect_square_to_350_l262_262250

theorem closest_perfect_square_to_350 : ∃ (n : ℕ), n^2 = 361 ∧ ∀ (m : ℕ), m^2 ≤ 350 ∨ 350 < m^2 → abs (361 - 350) ≤ abs (m^2 - 350) :=
by
  sorry

end closest_perfect_square_to_350_l262_262250


namespace base_length_of_isosceles_triangle_l262_262083

theorem base_length_of_isosceles_triangle 
  (a b : ℕ) 
  (h1 : a = 6) 
  (h2 : b = 6) 
  (perimeter : ℕ) 
  (h3 : 2*a + b = perimeter)
  (h4 : perimeter = 20) 
  : b = 8 := 
by
  sorry

end base_length_of_isosceles_triangle_l262_262083


namespace greatest_two_digit_prod_12_l262_262773

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l262_262773


namespace perfect_squares_digits_l262_262495

theorem perfect_squares_digits 
  (a b : ℕ) 
  (ha : ∃ m : ℕ, a = m * m) 
  (hb : ∃ n : ℕ, b = n * n) 
  (a_units_digit_1 : a % 10 = 1) 
  (b_units_digit_6 : b % 10 = 6) 
  (a_tens_digit : ∃ x : ℕ, (a / 10) % 10 = x) 
  (b_tens_digit : ∃ y : ℕ, (b / 10) % 10 = y) : 
  ∃ x y : ℕ, (x % 2 = 0) ∧ (y % 2 = 1) := 
sorry

end perfect_squares_digits_l262_262495


namespace magnitude_z1_pure_imaginary_l262_262480

open Complex

theorem magnitude_z1_pure_imaginary 
  (a : ℝ)
  (z1 : ℂ := a + 2 * I)
  (z2 : ℂ := 3 - 4 * I)
  (h : (z1 / z2).re = 0) :
  Complex.abs z1 = 10 / 3 := 
sorry

end magnitude_z1_pure_imaginary_l262_262480


namespace cubed_sum_identity_l262_262341

theorem cubed_sum_identity {x y : ℝ} (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubed_sum_identity_l262_262341


namespace necessity_of_A_for_B_l262_262146

variables {a b h : ℝ}

def PropA (a b h : ℝ) : Prop := |a - b| < 2 * h
def PropB (a b h : ℝ) : Prop := |a - 1| < h ∧ |b - 1| < h

theorem necessity_of_A_for_B (h_pos : 0 < h) : 
  (∀ a b, PropB a b h → PropA a b h) ∧ ¬ (∀ a b, PropA a b h → PropB a b h) :=
by sorry

end necessity_of_A_for_B_l262_262146


namespace max_star_player_salary_l262_262432

-- Define the constants given in the problem
def num_players : Nat := 12
def min_salary : Nat := 20000
def total_salary_cap : Nat := 1000000

-- Define the statement we want to prove
theorem max_star_player_salary :
  (∃ star_player_salary : Nat, 
    star_player_salary ≤ total_salary_cap - (num_players - 1) * min_salary ∧
    star_player_salary = 780000) :=
sorry

end max_star_player_salary_l262_262432


namespace last_two_digits_l262_262981

def x := Real.sqrt 29 + Real.sqrt 21
def y := Real.sqrt 29 - Real.sqrt 21
def a := x^2 = 50 + 2 * Real.sqrt 609
def b := y^2 = 50 - 2 * Real.sqrt 609
def S : ℕ → ℝ :=
  λ n => a^n + b^n

theorem last_two_digits (n : ℕ) :
  ((x : ℝ) ^ 2)^(n : ℕ) + ((y : ℝ)(^2))^(n : ℕ) = 71 := 
sorry

end last_two_digits_l262_262981


namespace energy_loss_per_bounce_l262_262112

theorem energy_loss_per_bounce
  (h : ℝ) (t : ℝ) (g : ℝ) (y : ℝ)
  (h_conds : h = 0.2)
  (t_conds : t = 18)
  (g_conds : g = 10)
  (model : t = Real.sqrt (2 * h / g) + 2 * (Real.sqrt (2 * h * y / g)) / (1 - Real.sqrt y)) :
  1 - y = 0.36 :=
by
  sorry

end energy_loss_per_bounce_l262_262112


namespace weekly_salary_correct_l262_262823

-- Define the daily salaries for each type of worker
def salary_A : ℝ := 200
def salary_B : ℝ := 250
def salary_C : ℝ := 300
def salary_D : ℝ := 350

-- Define the number of each type of worker
def num_A : ℕ := 3
def num_B : ℕ := 2
def num_C : ℕ := 3
def num_D : ℕ := 1

-- Define the total hours worked per day and the number of working days in a week
def hours_per_day : ℕ := 6
def working_days : ℕ := 7

-- Calculate the total daily salary for the team
def daily_salary_team : ℝ :=
  (num_A * salary_A) + (num_B * salary_B) + (num_C * salary_C) + (num_D * salary_D)

-- Calculate the total weekly salary for the team
def weekly_salary_team : ℝ := daily_salary_team * working_days

-- Problem: Prove that the total weekly salary for the team is Rs. 16,450
theorem weekly_salary_correct : weekly_salary_team = 16450 := by
  sorry

end weekly_salary_correct_l262_262823


namespace inequality_ay_bz_cx_lt_k_squared_l262_262995

theorem inequality_ay_bz_cx_lt_k_squared
  (a b c x y z k : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hk1 : a + x = k) (hk2 : b + y = k) (hk3 : c + z = k) :
  (a * y + b * z + c * x) < k^2 :=
sorry

end inequality_ay_bz_cx_lt_k_squared_l262_262995


namespace total_oil_leaked_correct_l262_262446

-- Definitions of given conditions.
def initial_leak_A : ℕ := 6522
def leak_rate_A : ℕ := 257
def time_A : ℕ := 20

def initial_leak_B : ℕ := 3894
def leak_rate_B : ℕ := 182
def time_B : ℕ := 15

def initial_leak_C : ℕ := 1421
def leak_rate_C : ℕ := 97
def time_C : ℕ := 12

-- Total additional leaks calculation.
def additional_leak (rate time : ℕ) : ℕ := rate * time
def additional_leak_A : ℕ := additional_leak leak_rate_A time_A
def additional_leak_B : ℕ := additional_leak leak_rate_B time_B
def additional_leak_C : ℕ := additional_leak leak_rate_C time_C

-- Total leaks from each pipe.
def total_leak_A : ℕ := initial_leak_A + additional_leak_A
def total_leak_B : ℕ := initial_leak_B + additional_leak_B
def total_leak_C : ℕ := initial_leak_C + additional_leak_C

-- Total oil leaked.
def total_oil_leaked : ℕ := total_leak_A + total_leak_B + total_leak_C

-- The proof problem statement.
theorem total_oil_leaked_correct : total_oil_leaked = 20871 := by
  sorry

end total_oil_leaked_correct_l262_262446


namespace fair_collection_l262_262534

theorem fair_collection 
  (children : ℕ) (fee_child : ℝ) (adults : ℕ) (fee_adult : ℝ) 
  (total_people : ℕ) (count_children : ℕ) (count_adults : ℕ)
  (total_collected: ℝ) :
  children = 700 →
  fee_child = 1.5 →
  adults = 1500 →
  fee_adult = 4.0 →
  total_people = children + adults →
  count_children = 700 →
  count_adults = 1500 →
  total_collected = (count_children * fee_child) + (count_adults * fee_adult) →
  total_collected = 7050 :=
by
  intros
  sorry

end fair_collection_l262_262534


namespace isosceles_base_lines_l262_262738
open Real

theorem isosceles_base_lines {x y : ℝ} (h1 : 7 * x - y - 9 = 0) (h2 : x + y - 7 = 0) (hx : x = 3) (hy : y = -8) :
  (x - 3 * y - 27 = 0) ∨ (3 * x + y - 1 = 0) :=
sorry

end isosceles_base_lines_l262_262738


namespace circle_inscribed_isosceles_trapezoid_l262_262448

theorem circle_inscribed_isosceles_trapezoid (r a c : ℝ) : 
  (∃ base1 base2 : ℝ,  2 * a = base1 ∧ 2 * c = base2) →
  (∃ O : ℝ, O = r) →
  r^2 = a * c :=
by
  sorry

end circle_inscribed_isosceles_trapezoid_l262_262448


namespace value_of_a_squared_plus_b_squared_l262_262180

theorem value_of_a_squared_plus_b_squared (a b : ℝ) (h1 : a * b = 16) (h2 : a + b = 10) :
  a^2 + b^2 = 68 :=
sorry

end value_of_a_squared_plus_b_squared_l262_262180


namespace exists_a_b_k_l262_262366

theorem exists_a_b_k (m : ℕ) (hm : 0 < m) : 
  ∃ a b k : ℤ, 
    (a % 2 = 1) ∧ 
    (b % 2 = 1) ∧ 
    (0 ≤ k) ∧ 
    (2 * m = a^19 + b^99 + k * 2^1999) :=
sorry

end exists_a_b_k_l262_262366


namespace question1_question2_l262_262334

def A (x : ℝ) : Prop := x^2 - 2*x - 3 ≤ 0
def B (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + m^2 - 4 ≤ 0

-- Question 1: If A ∩ B = [1, 3], then m = 3
theorem question1 (m : ℝ) : (∀ x, A x ∧ B m x ↔ (1 ≤ x ∧ x ≤ 3)) → m = 3 :=
sorry

-- Question 2: If A is a subset of the complement of B in ℝ, then m > 5 or m < -3
theorem question2 (m : ℝ) : (∀ x, A x → ¬ B m x) → (m > 5 ∨ m < -3) :=
sorry

end question1_question2_l262_262334


namespace Cody_total_bill_l262_262455

-- Definitions for the problem
def cost_per_child : ℝ := 7.5
def cost_per_adult : ℝ := 12.0

variables (A C : ℕ)

-- Conditions
def condition1 : Prop := C = A + 8
def condition2 : Prop := A + C = 12

-- Total bill
def total_cost := (A * cost_per_adult) + (C * cost_per_child)

-- The proof statement
theorem Cody_total_bill (h1 : condition1 A C) (h2 : condition2 A C) : total_cost A C = 99.0 := by
  sorry

end Cody_total_bill_l262_262455


namespace greatest_two_digit_with_product_12_l262_262761

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l262_262761


namespace percentage_increase_expenditure_l262_262526

variable (I : ℝ) -- original income
variable (E : ℝ) -- original expenditure
variable (I_new : ℝ) -- new income
variable (S : ℝ) -- original savings
variable (S_new : ℝ) -- new savings

-- a) Conditions
def initial_spend (I : ℝ) : ℝ := 0.75 * I
def income_increased (I : ℝ) : ℝ := 1.20 * I
def savings_increased (S : ℝ) : ℝ := 1.4999999999999996 * S

-- b) Definitions relating formulated conditions
def new_expenditure (I : ℝ) : ℝ := 1.20 * I - 0.3749999999999999 * I
def original_expenditure (I : ℝ) : ℝ := 0.75 * I

-- c) Proof statement
theorem percentage_increase_expenditure :
  initial_spend I = E →
  income_increased I = I_new →
  savings_increased (0.25 * I) = S_new →
  ((new_expenditure I - original_expenditure I) / original_expenditure I) * 100 = 10 := 
by 
  intros h1 h2 h3
  sorry

end percentage_increase_expenditure_l262_262526


namespace extremum_point_of_f_l262_262217

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem extremum_point_of_f : ∃ x, x = 1 ∧ (∀ y ≠ 1, f y ≥ f x) := 
sorry

end extremum_point_of_f_l262_262217


namespace proof_problem_l262_262422

-- Define the rates of P and Q
def P_rate : ℚ := 1/3
def Q_rate : ℚ := 1/18

-- Define the time they work together
def combined_time : ℚ := 2

-- Define the job completion rates
def combined_rate (P_rate Q_rate : ℚ) : ℚ := P_rate + Q_rate

-- Define the job completed together in given time
def job_completed_together (rate time : ℚ) : ℚ := rate * time

-- Define the remaining job
def remaining_job (total_job completed_job : ℚ) : ℚ := total_job - completed_job

-- Define the time required for P to complete the remaining job
def time_for_P (P_rate remaining_job : ℚ) : ℚ := remaining_job / P_rate

-- Define the total job as 1
def total_job : ℚ := 1

-- Correct answer in minutes
def correct_answer_in_minutes (time_in_hours : ℚ) : ℚ := time_in_hours * 60

-- Problem statement
theorem proof_problem : 
  correct_answer_in_minutes (time_for_P P_rate (remaining_job total_job 
    (job_completed_together (combined_rate P_rate Q_rate) combined_time))) = 40 := 
by
  sorry

end proof_problem_l262_262422


namespace greatest_two_digit_product_12_l262_262794

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l262_262794


namespace find_constant_l262_262537

noncomputable def f (x : ℝ) : ℝ := x + 4

theorem find_constant : ∃ c : ℝ, (∀ x : ℝ, x = 0.4 → (3 * f (x - c)) / f 0 + 4 = f (2 * x + 1)) ∧ c = 2 :=
by
  sorry

end find_constant_l262_262537


namespace number_of_correct_conclusions_l262_262725

theorem number_of_correct_conclusions
  (a b c : ℕ)
  (h1 : (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713) 
  (conclusion1 : (a^b - b^c) % 2 = 1 ∧ (b^c - c^a) % 2 = 1 ∧ (c^a - a^b) % 2 = 1)
  (conclusion4 : ¬ ∃ a b c : ℕ, (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end number_of_correct_conclusions_l262_262725


namespace student_needs_33_percent_to_pass_l262_262299

-- Define the conditions
def obtained_marks : ℕ := 125
def failed_by : ℕ := 40
def max_marks : ℕ := 500

-- The Lean statement to prove the required percentage
theorem student_needs_33_percent_to_pass : (obtained_marks + failed_by) * 100 / max_marks = 33 := by
  sorry

end student_needs_33_percent_to_pass_l262_262299


namespace area_of_triangle_CM_N_l262_262989

noncomputable def triangle_area (a : ℝ) : ℝ :=
  let M := (a / 2, a, a)
  let N := (a, a / 2, a)
  let MN := Real.sqrt ((a - a / 2) ^ 2 + (a / 2 - a) ^ 2)
  let CK := Real.sqrt (a ^ 2 + (a * Real.sqrt 2 / 4) ^ 2)
  (1/2) * MN * CK

theorem area_of_triangle_CM_N 
  (a : ℝ) :
  (a > 0) →
  triangle_area a = (3 * a^2) / 8 :=
by
  intro h
  -- Proof will go here.
  sorry

end area_of_triangle_CM_N_l262_262989


namespace range_of_a_l262_262863

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log (x + 1) - x^2

theorem range_of_a (a : ℝ) :
    (∀ (p q : ℝ), 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a (p + 1) - f a (q + 1)) / (p - q) > 2) →
    a ≥ 18 := sorry

end range_of_a_l262_262863


namespace arccos_neg_one_eq_pi_l262_262597

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l262_262597


namespace rectangle_side_lengths_l262_262992

theorem rectangle_side_lengths (x y : ℝ) (h1 : 2 * x + 4 = 10) (h2 : 8 * y - 2 = 10) : x + y = 4.5 := by
  sorry

end rectangle_side_lengths_l262_262992


namespace prism_cut_out_l262_262449

theorem prism_cut_out (x y : ℕ)
  (H1 : 15 * 5 * 4 - y * 5 * x = 120)
  (H2 : x < 4) :
  x = 3 ∧ y = 12 :=
sorry

end prism_cut_out_l262_262449


namespace total_students_l262_262829

-- Define the conditions
def rank_from_right := 17
def rank_from_left := 5

-- The proof statement
theorem total_students : rank_from_right + rank_from_left - 1 = 21 := 
by 
  -- Assuming the conditions represented by the definitions
  -- Without loss of generality the proof would be derived from these, but it is skipped
  sorry

end total_students_l262_262829


namespace completing_the_square_x_squared_minus_4x_plus_1_eq_0_l262_262077

theorem completing_the_square_x_squared_minus_4x_plus_1_eq_0 :
  ∀ x : ℝ, (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x
  intros h
  sorry

end completing_the_square_x_squared_minus_4x_plus_1_eq_0_l262_262077


namespace largest_value_x_l262_262316

theorem largest_value_x (x a b c d : ℝ) (h_eq : 7 * x ^ 2 + 15 * x - 20 = 0) (h_form : x = (a + b * Real.sqrt c) / d) (ha : a = -15) (hb : b = 1) (hc : c = 785) (hd : d = 14) : (a * c * d) / b = -164850 := 
sorry

end largest_value_x_l262_262316


namespace ratio_of_sums_l262_262305

/-- Define the relevant arithmetic sequences and sums -/

-- Sequence 1: 3, 6, 9, ..., 45
def seq1 : ℕ → ℕ
| n => 3 * n + 3

-- Sequence 2: 4, 8, 12, ..., 64
def seq2 : ℕ → ℕ
| n => 4 * n + 4

-- Sum function for arithmetic sequences
def sum_arith_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n-1) * d) / 2

noncomputable def sum_seq1 : ℕ := sum_arith_seq 3 3 15 -- 3 + 6 + ... + 45
noncomputable def sum_seq2 : ℕ := sum_arith_seq 4 4 16 -- 4 + 8 + ... + 64

-- Prove that the ratio of sums is 45/68
theorem ratio_of_sums : (sum_seq1 : ℚ) / sum_seq2 = 45 / 68 :=
  sorry

end ratio_of_sums_l262_262305


namespace Walter_allocates_75_for_school_l262_262559

/-- Walter's conditions -/
variables (days_per_week hours_per_day earnings_per_hour : ℕ) (allocation_fraction : ℝ)

/-- Given values from the problem -/
def Walter_conditions : Prop :=
  days_per_week = 5 ∧
  hours_per_day = 4 ∧
  earnings_per_hour = 5 ∧
  allocation_fraction = 3 / 4

/-- Walter's weekly earnings calculation -/
def weekly_earnings (days_per_week hours_per_day earnings_per_hour : ℕ) : ℕ :=
  days_per_week * hours_per_day * earnings_per_hour

/-- Amount allocated for school -/
def allocated_for_school (weekly_earnings : ℕ) (allocation_fraction : ℝ) : ℝ :=
  weekly_earnings * allocation_fraction

/-- Main Theorem: Walter allocates $75 for his school --/
theorem Walter_allocates_75_for_school :
  Walter_conditions days_per_week hours_per_day earnings_per_hour allocation_fraction →
  allocated_for_school (weekly_earnings days_per_week hours_per_day earnings_per_hour) allocation_fraction = 75 :=
begin
  sorry
end

end Walter_allocates_75_for_school_l262_262559


namespace glove_selection_correct_l262_262144

-- Define the total number of different pairs of gloves
def num_pairs : Nat := 6

-- Define the required number of gloves to select
def num_gloves_to_select : Nat := 4

-- Define the function to calculate the number of ways to select 4 gloves with exactly one matching pair
noncomputable def count_ways_to_select_gloves (num_pairs : Nat) : Nat :=
  let select_pair := Nat.choose num_pairs 1
  let remaining_gloves := 2 * (num_pairs - 1)
  let select_two_from_remaining := Nat.choose remaining_gloves 2
  let subtract_unwanted_pairs := num_pairs - 1
  select_pair * (select_two_from_remaining - subtract_unwanted_pairs)

-- The correct answer we need to prove
def expected_result : Nat := 240

-- The theorem to prove the number of ways to select the gloves
theorem glove_selection_correct : count_ways_to_select_gloves num_pairs = expected_result :=
  by
    sorry

end glove_selection_correct_l262_262144


namespace cube_modulo_9_l262_262872

theorem cube_modulo_9 (N : ℤ) (h : N % 9 = 2 ∨ N % 9 = 5 ∨ N % 9 = 8) : 
  (N^3) % 9 = 8 :=
by sorry

end cube_modulo_9_l262_262872


namespace root_condition_l262_262288

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m * x + m

theorem root_condition (m l : ℝ) (h : m < l) : 
  (∀ x : ℝ, f x m = 0 → x ≠ x) ∨ (∃ x : ℝ, f x m = 0) :=
sorry

end root_condition_l262_262288


namespace dodecahedron_interior_diagonals_l262_262678

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l262_262678


namespace greatest_two_digit_product_12_l262_262792

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l262_262792


namespace no_rectangle_from_five_distinct_squares_l262_262376

theorem no_rectangle_from_five_distinct_squares (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : q1 < q2) 
  (h2 : q2 < q3) 
  (h3 : q3 < q4) 
  (h4 : q4 < q5) : 
  ¬∃(a b: ℝ), a * b = 5 ∧ a = q1 + q2 + q3 + q4 + q5 := sorry

end no_rectangle_from_five_distinct_squares_l262_262376


namespace similar_triangle_legs_l262_262581

theorem similar_triangle_legs (y : ℝ) 
  (h1 : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 15 ∧ b = 12)
  (h2 : ∃ u v w : ℝ, u^2 + v^2 = w^2 ∧ u = y ∧ v = 9) 
  (h3 : ∀ (a b c u v w : ℝ), (a^2 + b^2 = c^2 ∧ u^2 + v^2 = w^2 ∧ a/u = b/v) → (a = b → u = v)) 
  : y = 11.25 := 
  by 
    sorry

end similar_triangle_legs_l262_262581


namespace cubed_sum_identity_l262_262342

theorem cubed_sum_identity {x y : ℝ} (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubed_sum_identity_l262_262342


namespace martha_to_doris_ratio_l262_262097

-- Define the amounts involved
def initial_amount : ℕ := 21
def doris_spent : ℕ := 6
def remaining_after_doris : ℕ := initial_amount - doris_spent
def final_amount : ℕ := 12
def martha_spent : ℕ := remaining_after_doris - final_amount

-- State the theorem about the ratio
theorem martha_to_doris_ratio : martha_spent * 2 = doris_spent :=
by
  -- Detailed proof is skipped
  sorry

end martha_to_doris_ratio_l262_262097


namespace round_robin_tournament_teams_l262_262702

theorem round_robin_tournament_teams (n : ℕ) (h_match_count : 34) (h_withdraw: 2 * 3) (h_play: (n - 2) * (n - 3) / 2 = 28) : n = 10 := 
by
  have h1 : (n - 2) * (n - 3) = 56 := by
    rw [← nat.choose_two_eq h_play]
    exact (mul_comm (n - 3) (n - 2)).symm
  have h2 : (n - 2) * (n - 3) = 56 := by sorry
  have h3 : nat.main_eq (8 * (n - 10 ) = 80) := by sorry
  sorry

end round_robin_tournament_teams_l262_262702


namespace expand_expression_l262_262968

open Nat

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 :=
by
  sorry

end expand_expression_l262_262968


namespace base_length_of_isosceles_triangle_l262_262087

-- Definitions based on given conditions
def is_isosceles (a b : ℕ) (c : ℕ) :=
a = b ∧ c = c

def side_length : ℕ := 6
def perimeter : ℕ := 20

-- Theorem to prove the base length
theorem base_length_of_isosceles_triangle (b : ℕ) (h1 : 2 * side_length + b = perimeter) :
  b = 8 :=
sorry

end base_length_of_isosceles_triangle_l262_262087


namespace coefficient_of_x_in_first_equation_is_one_l262_262484

theorem coefficient_of_x_in_first_equation_is_one
  (x y z : ℝ)
  (h1 : x - 5 * y + 3 * z = 22 / 6)
  (h2 : 4 * x + 8 * y - 11 * z = 7)
  (h3 : 5 * x - 6 * y + 2 * z = 12)
  (h4 : x + y + z = 10) :
  (1 : ℝ) = 1 := 
by 
  sorry

end coefficient_of_x_in_first_equation_is_one_l262_262484


namespace cost_of_jam_l262_262135

theorem cost_of_jam (N B J : ℕ) (hN : N > 1) (h_total_cost : N * (5 * B + 6 * J) = 348) :
    6 * N * J = 348 := by
  sorry

end cost_of_jam_l262_262135


namespace intersection_M_N_l262_262332

-- Define the sets M and N based on given conditions
def M : Set ℝ := { x : ℝ | x^2 < 4 }
def N : Set ℝ := { x : ℝ | x < 1 }

-- State the theorem to prove the intersection of M and N
theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l262_262332


namespace mrs_hilt_total_payment_l262_262372

noncomputable def total_hotdogs : ℕ := 12
noncomputable def cost_first_4 : ℝ := 4 * 0.60
noncomputable def cost_next_5 : ℝ := 5 * 0.75
noncomputable def cost_last_3 : ℝ := 3 * 0.90
noncomputable def total_cost : ℝ := cost_first_4 + cost_next_5 + cost_last_3

theorem mrs_hilt_total_payment : total_cost = 8.85 := by
  -- proof goes here
  sorry

end mrs_hilt_total_payment_l262_262372


namespace number_of_valid_integers_l262_262489

theorem number_of_valid_integers :
  let valid_values := List.filter (λ c, (10 * c + 3) % 7 = 0) (List.range' 10 100)
  valid_values.length = 12 :=
by
  let valid_values : List Nat := List.filter (λ c, (10 * c + 3) % 7 = 0) (List.range' 10 90)
  have : valid_values.length = 12 := sorry

end number_of_valid_integers_l262_262489


namespace total_cost_for_seven_hard_drives_l262_262409

-- Condition: Two identical hard drives cost $50.
def cost_of_two_hard_drives : ℝ := 50

-- Condition: There is a 10% discount if you buy more than four hard drives.
def discount_rate : ℝ := 0.10

-- Question: What is the total cost in dollars for buying seven of these hard drives?
theorem total_cost_for_seven_hard_drives : (7 * (cost_of_two_hard_drives / 2)) * (1 - discount_rate) = 157.5 := 
by 
  -- def cost_of_one_hard_drive
  let cost_of_one_hard_drive := cost_of_two_hard_drives / 2
  -- def cost_of_seven_hard_drives
  let cost_of_seven_hard_drives := 7 * cost_of_one_hard_drive
  have h₁ : 7 * (cost_of_two_hard_drives / 2) = cost_of_seven_hard_drives := by sorry
  have h₂ : cost_of_seven_hard_drives * (1 - discount_rate) = 157.5 := by sorry
  exact h₂

end total_cost_for_seven_hard_drives_l262_262409


namespace greatest_two_digit_product_12_l262_262789

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l262_262789


namespace number_of_interior_diagonals_of_dodecahedron_l262_262650

-- Definitions based on conditions
def dodecahedron_vertices := 20
def faces_per_vertex := 3
def vertices_per_face := 5
def shared_edges_per_vertex := faces_per_vertex
def total_faces := 12
def total_vertices := 20

-- Property of the dodecahedron
def potential_diagonals_per_vertex := dodecahedron_vertices - 1 - shared_edges_per_vertex - (vertices_per_face - 1)
def total_potential_diagonals := potential_diagonals_per_vertex * total_vertices

-- Proof statement:
theorem number_of_interior_diagonals_of_dodecahedron :
  total_potential_diagonals / 2 = 90 :=
by
  -- This is where the proof would go.
  sorry

end number_of_interior_diagonals_of_dodecahedron_l262_262650


namespace smallest_m_l262_262461

theorem smallest_m (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  ∃ m, (∀ (a b c : ℝ), a + b + c = 1 → 0 < a → 0 < b → 0 < c → m * (a ^ 3 + b ^ 3 + c ^ 3) ≥ 6 * (a ^ 2 + b ^ 2 + c ^ 2) + 1) ↔ m = 27 :=
by
  sorry

end smallest_m_l262_262461


namespace expression_value_l262_262957

def a : ℕ := 1000
def b1 : ℕ := 15
def b2 : ℕ := 314
def c1 : ℕ := 201
def c2 : ℕ := 360
def c3 : ℕ := 110
def d1 : ℕ := 201
def d2 : ℕ := 360
def d3 : ℕ := 110
def e1 : ℕ := 15
def e2 : ℕ := 314

theorem expression_value :
  (a + b1 + b2) * (c1 + c2 + c3) + (a - d1 - d2 - d3) * (e1 + e2) = 1000000 :=
by
  sorry

end expression_value_l262_262957


namespace fish_lifespan_proof_l262_262747

def hamster_lifespan : ℝ := 2.5

def dog_lifespan : ℝ := 4 * hamster_lifespan

def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_proof :
  fish_lifespan = 12 := 
  by
  sorry

end fish_lifespan_proof_l262_262747


namespace dodecahedron_interior_diagonals_l262_262668

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l262_262668


namespace primes_square_condition_l262_262978

open Nat

theorem primes_square_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  ∃ n : ℕ, p^(q+1) + q^(p+1) = n^2 ↔ p = 2 ∧ q = 2 := by
  sorry

end primes_square_condition_l262_262978


namespace xyz_value_l262_262858

theorem xyz_value
  (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + y * z + z * x) = 24) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 14 / 3 :=
  sorry

end xyz_value_l262_262858


namespace value_of_b_l262_262248

theorem value_of_b (b : ℝ) : 
  (∀ x : ℝ, -x ^ 2 + b * x + 7 < 0 ↔ x < -2 ∨ x > 3) → b = 1 :=
by
  sorry

end value_of_b_l262_262248


namespace minimize_expression_l262_262406

theorem minimize_expression (n : ℕ) (h : n > 0) : (n = 10) ↔ (∀ m : ℕ, m > 0 → (n / 2 + 50 / n: ℝ) ≤ (m / 2 + 50 / m: ℝ)) :=
sorry

end minimize_expression_l262_262406


namespace inequality_holds_l262_262003

theorem inequality_holds (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x * y + y * z) := 
by 
  sorry

end inequality_holds_l262_262003


namespace sum_of_coefficients_l262_262869

theorem sum_of_coefficients (a a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) :
  (∀ x : ℤ, (1 + x)^6 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  a = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 63 :=
by
  intros h ha
  sorry

end sum_of_coefficients_l262_262869


namespace lcm_1230_924_l262_262460

theorem lcm_1230_924 : Nat.lcm 1230 924 = 189420 :=
by
  /- Proof steps skipped -/
  sorry

end lcm_1230_924_l262_262460


namespace div_by_eight_l262_262072

theorem div_by_eight (n : ℕ) : (5^n + 2 * 3^(n-1) + 1) % 8 = 0 :=
by
  sorry

end div_by_eight_l262_262072


namespace A_share_is_9000_l262_262443

noncomputable def A_share_in_gain (x : ℝ) : ℝ :=
  let total_gain := 27000
  let A_investment_time := 12 * x
  let B_investment_time := 6 * 2 * x
  let C_investment_time := 4 * 3 * x
  let total_investment_time := A_investment_time + B_investment_time + C_investment_time
  total_gain * A_investment_time / total_investment_time

theorem A_share_is_9000 (x : ℝ) : A_share_in_gain x = 27000 / 3 :=
by
  sorry

end A_share_is_9000_l262_262443


namespace total_trees_after_planting_l262_262744

theorem total_trees_after_planting
  (initial_walnut_trees : ℕ) (initial_oak_trees : ℕ) (initial_maple_trees : ℕ)
  (plant_walnut_trees : ℕ) (plant_oak_trees : ℕ) (plant_maple_trees : ℕ) :
  (initial_walnut_trees = 107) →
  (initial_oak_trees = 65) →
  (initial_maple_trees = 32) →
  (plant_walnut_trees = 104) →
  (plant_oak_trees = 79) →
  (plant_maple_trees = 46) →
  initial_walnut_trees + plant_walnut_trees +
  initial_oak_trees + plant_oak_trees +
  initial_maple_trees + plant_maple_trees = 433 :=
by
  intros
  sorry

end total_trees_after_planting_l262_262744


namespace polynomial_simplification_l262_262731

theorem polynomial_simplification (p : ℝ) :
  (4 * p^4 + 2 * p^3 - 7 * p + 3) + (5 * p^3 - 8 * p^2 + 3 * p + 2) = 
  4 * p^4 + 7 * p^3 - 8 * p^2 - 4 * p + 5 :=
by
  sorry

end polynomial_simplification_l262_262731


namespace union_of_M_N_is_real_set_l262_262063

-- Define the set M
def M : Set ℝ := { x | x^2 + 3 * x + 2 > 0 }

-- Define the set N
def N : Set ℝ := { x | (1 / 2 : ℝ) ^ x ≤ 4 }

-- The goal is to prove that the union of M and N is the set of all real numbers
theorem union_of_M_N_is_real_set : M ∪ N = Set.univ :=
by
  sorry

end union_of_M_N_is_real_set_l262_262063


namespace probability_xy_even_l262_262921

open Finset

def set := ∅.insert 1 |>.insert 2 |>.insert 3 |>.insert 4 |>.insert 5
           |>.insert 6 |>.insert 7 |>.insert 8 |>.insert 9 |>.insert 10
           |>.insert 11 |>.insert 12 |>.insert 13 |>.insert 14 |>.insert 15

def primes_in_set := {2, 3, 5, 7, 11, 13}

noncomputable def probability_event := 5 / 12

theorem probability_xy_even :
  ∀ (x y : ℕ), x ∈ set → y ∈ set → y ∈ primes_in_set → x ≠ y →
  (∃ (p : ℚ), p = probability_event ∧ 
  ((Finset.card ((set.product primes_in_set).filter (λ (t : ℕ × ℕ),
     let (x, y) := t in
     x ≠ y ∧ (x * y - x - y) % 2 = 0))) %
     (Finset.card ((set.product primes_in_set).filter (λ (t : ℕ × ℕ),
     let (x, y) := t in x ≠ y))) = p) ) :=
sorry

end probability_xy_even_l262_262921


namespace closest_perfect_square_to_350_l262_262271

theorem closest_perfect_square_to_350 :
  let n := 19 in
  let m := 18 in
  18^2 = 324 ∧ 19^2 = 361 ∧ 324 < 350 ∧ 350 < 361 ∧ 
  abs (350 - 324) = 26 ∧ abs (361 - 350) = 11 ∧ 11 < 26 → 
  n^2 = 361 :=
by
  intro n m h
  sorry

end closest_perfect_square_to_350_l262_262271


namespace percentage_increase_in_expenditure_l262_262525

-- Definitions
def original_income (I : ℝ) := I
def expenditure (I : ℝ) := 0.75 * I
def savings (I E : ℝ) := I - E
def new_income (I : ℝ) := 1.2 * I
def new_expenditure (E P : ℝ) := E * (1 + P / 100)
def new_savings (I E P : ℝ) := new_income I - new_expenditure E P

-- Theorem to prove
theorem percentage_increase_in_expenditure (I : ℝ) (P : ℝ) :
  savings I (expenditure I) * 1.5 = new_savings I (expenditure I) P →
  P = 10 :=
by
  intros h
  simp [savings, expenditure, new_income, new_expenditure, new_savings] at h
  sorry

end percentage_increase_in_expenditure_l262_262525


namespace greatest_two_digit_product_12_l262_262790

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l262_262790


namespace distance_between_consecutive_trees_l262_262418

-- Definitions from the problem statement
def yard_length : ℕ := 414
def number_of_trees : ℕ := 24
def number_of_intervals : ℕ := number_of_trees - 1
def distance_between_trees : ℕ := yard_length / number_of_intervals

-- Main theorem we want to prove
theorem distance_between_consecutive_trees :
  distance_between_trees = 18 := by
  -- Proof would go here
  sorry

end distance_between_consecutive_trees_l262_262418


namespace solve_for_x_l262_262213

theorem solve_for_x (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) : 
  (x + 5) / (x - 3) = (x - 4) / (x + 2) → x = 1 / 7 :=
by
  sorry

end solve_for_x_l262_262213


namespace symmetric_points_a_minus_b_l262_262016

theorem symmetric_points_a_minus_b (a b : ℝ) 
  (h1 : a = -5) 
  (h2 : b = -1) :
  a - b = -4 := 
sorry

end symmetric_points_a_minus_b_l262_262016


namespace area_of_triangle_l262_262699

theorem area_of_triangle {a c : ℝ} (B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = 60) :
    (1 / 2) * a * c * Real.sin (B * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end area_of_triangle_l262_262699


namespace more_non_product_eight_digit_numbers_l262_262312

def num_eight_digit_numbers := 10^8 - 10^7
def num_four_digit_numbers := 9999 - 1000 + 1
def num_unique_products := (num_four_digit_numbers.choose 2) + num_four_digit_numbers

theorem more_non_product_eight_digit_numbers :
  (num_eight_digit_numbers - num_unique_products) > num_unique_products := by sorry

end more_non_product_eight_digit_numbers_l262_262312


namespace greatest_two_digit_with_product_12_l262_262765

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l262_262765


namespace trig_expression_l262_262988

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end trig_expression_l262_262988


namespace square_difference_l262_262310

theorem square_difference (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, c^2 = a^2 - b^2 :=
by
  sorry

end square_difference_l262_262310


namespace origin_in_circle_m_gt_5_l262_262166

theorem origin_in_circle_m_gt_5 (m : ℝ) : ((0 - 1)^2 + (0 + 2)^2 < m) → (m > 5) :=
by
  intros h
  sorry

end origin_in_circle_m_gt_5_l262_262166


namespace num_true_propositions_eq_two_l262_262487

open Classical

theorem num_true_propositions_eq_two (p q : Prop) :
  (if (p ∧ q) then 1 else 0) + (if (p ∨ q) then 1 else 0) + (if (¬p) then 1 else 0) + (if (¬q) then 1 else 0) = 2 :=
by sorry

end num_true_propositions_eq_two_l262_262487


namespace certain_number_l262_262556

theorem certain_number (x y z : ℕ) 
  (h1 : x + y = 15) 
  (h2 : y = 7) 
  (h3 : 3 * x = z * y - 11) : 
  z = 5 :=
by sorry

end certain_number_l262_262556


namespace sequence_general_formula_l262_262360

theorem sequence_general_formula (n : ℕ) (h : n ≥ 1) :
  ∃ a : ℕ → ℝ, a 1 = 1 ∧ (∀ n ≥ 1, a (n + 1) = a n / (1 + a n)) ∧ a n = (1 : ℝ) / n :=
by
  sorry

end sequence_general_formula_l262_262360


namespace only_polynomial_is_identity_l262_262623

-- Define the number composed only of digits 1
def Ones (k : ℕ) : ℕ := (10^k - 1) / 9

theorem only_polynomial_is_identity (P : ℕ → ℕ) :
  (∀ k : ℕ, P (Ones k) = Ones k) → (∀ x : ℕ, P x = x) :=
by
  intro h
  sorry

end only_polynomial_is_identity_l262_262623


namespace linear_function_value_l262_262009

theorem linear_function_value
  (a b c : ℝ)
  (h1 : 3 * a + b = 8)
  (h2 : -2 * a + b = 3)
  (h3 : -3 * a + b = c) :
  a^2 + b^2 + c^2 - a * b - b * c - a * c = 13 :=
by
  sorry

end linear_function_value_l262_262009


namespace total_tape_area_l262_262304

theorem total_tape_area 
  (long_side_1 short_side_1 : ℕ) (boxes_1 : ℕ)
  (long_side_2 short_side_2 : ℕ) (boxes_2 : ℕ)
  (long_side_3 short_side_3 : ℕ) (boxes_3 : ℕ)
  (overlap : ℕ) (tape_width : ℕ) :
  long_side_1 = 30 → short_side_1 = 15 → boxes_1 = 5 →
  long_side_2 = 40 → short_side_2 = 40 → boxes_2 = 2 →
  long_side_3 = 50 → short_side_3 = 20 → boxes_3 = 3 →
  overlap = 2 → tape_width = 2 →
  let total_length_1 := boxes_1 * (long_side_1 + overlap + 2 * (short_side_1 + overlap))
  let total_length_2 := boxes_2 * 3 * (long_side_2 + overlap)
  let total_length_3 := boxes_3 * (long_side_3 + overlap + 2 * (short_side_3 + overlap))
  let total_length := total_length_1 + total_length_2 + total_length_3
  let total_area := total_length * tape_width
  total_area = 1740 :=
  by
  -- Add the proof steps here
  -- sorry can be used to skip the proof
  sorry

end total_tape_area_l262_262304


namespace cricket_players_count_l262_262356

-- Define the conditions
def total_players_present : ℕ := 50
def hockey_players : ℕ := 17
def football_players : ℕ := 11
def softball_players : ℕ := 10

-- Define the result to prove
def cricket_players : ℕ := total_players_present - (hockey_players + football_players + softball_players)

-- The theorem stating the equivalence of cricket_players and the correct answer
theorem cricket_players_count : cricket_players = 12 := by
  -- A placeholder for the proof
  sorry

end cricket_players_count_l262_262356


namespace max_sum_42_l262_262355

noncomputable def max_horizontal_vertical_sum (numbers : List ℕ) : ℕ :=
  let a := 14
  let b := 11
  let e := 17
  a + b + e

theorem max_sum_42 : 
  max_horizontal_vertical_sum [2, 5, 8, 11, 14, 17] = 42 := by
  sorry

end max_sum_42_l262_262355


namespace sum_and_product_of_roots_cube_l262_262344

theorem sum_and_product_of_roots_cube (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by {
  sorry
}

end sum_and_product_of_roots_cube_l262_262344


namespace arccos_neg_one_eq_pi_l262_262593

theorem arccos_neg_one_eq_pi : arccos (-1) = π := 
by
  sorry

end arccos_neg_one_eq_pi_l262_262593


namespace sum_of_ages_in_three_years_l262_262363

variable (Josiah Hans : ℕ)

axiom hans_age : Hans = 15
axiom age_relation : Josiah = 3 * Hans

theorem sum_of_ages_in_three_years : Josiah + 3 + (Hans + 3) = 66 :=
by
  simp [hans_age, age_relation]
  sorry

end sum_of_ages_in_three_years_l262_262363


namespace solve_for_q_l262_262381

theorem solve_for_q (x y q : ℚ) 
  (h1 : 7 / 8 = x / 96) 
  (h2 : 7 / 8 = (x + y) / 104) 
  (h3 : 7 / 8 = (q - y) / 144) : 
  q = 133 := 
sorry

end solve_for_q_l262_262381


namespace gcd_12345_6789_eq_3_l262_262757

theorem gcd_12345_6789_eq_3 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_eq_3_l262_262757


namespace trigonometric_expression_value_l262_262353

noncomputable def trigonometric_expression (α : ℝ) : ℝ :=
  (|Real.tan α| / Real.tan α) + (Real.sin α / Real.sqrt ((1 - Real.cos (2 * α)) / 2))

theorem trigonometric_expression_value (α : ℝ) (h : Real.sin α = -Real.cos α) : 
  trigonometric_expression α = 0 ∨ trigonometric_expression α = -2 :=
by 
  sorry

end trigonometric_expression_value_l262_262353


namespace arithmetic_sequence_term_l262_262359

theorem arithmetic_sequence_term (a d n : ℕ) (h₀ : a = 1) (h₁ : d = 3) (h₂ : a + (n - 1) * d = 6019) :
  n = 2007 :=
sorry

end arithmetic_sequence_term_l262_262359


namespace number_of_girls_in_group_l262_262123

open Finset

/-- Given that a tech group consists of 6 students, and 3 people are to be selected to visit an exhibition,
    if there are at least 1 girl among the selected, the number of different selection methods is 16,
    then the number of girls in the group is 2. -/
theorem number_of_girls_in_group :
  ∃ n : ℕ, (n ≥ 1 ∧ n ≤ 6 ∧ 
            (Nat.choose 6 3 - Nat.choose (6 - n) 3 = 16)) → n = 2 :=
by
  sorry

end number_of_girls_in_group_l262_262123


namespace point_A_coordinates_l262_262151

-- Given conditions
def point_A (a : ℝ) : ℝ × ℝ := (a + 1, a^2 - 4)
def negative_half_x_axis (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 = 0

-- Theorem statement
theorem point_A_coordinates (a : ℝ) (h : negative_half_x_axis (point_A a)) :
  point_A a = (-1, 0) :=
sorry

end point_A_coordinates_l262_262151


namespace option_d_not_true_l262_262627

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)

theorem option_d_not_true : (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) := sorry

end option_d_not_true_l262_262627


namespace dodecahedron_interior_diagonals_l262_262670

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l262_262670


namespace cos4_minus_sin4_15_eq_sqrt3_div2_l262_262289

theorem cos4_minus_sin4_15_eq_sqrt3_div2 :
  (Real.cos 15)^4 - (Real.sin 15)^4 = Real.sqrt 3 / 2 :=
sorry

end cos4_minus_sin4_15_eq_sqrt3_div2_l262_262289


namespace height_of_building_l262_262280

-- Define the conditions
def height_flagpole : ℝ := 18
def shadow_flagpole : ℝ := 45
def shadow_building : ℝ := 55

-- State the theorem to prove the height of the building
theorem height_of_building (h : ℝ) : (height_flagpole / shadow_flagpole) = (h / shadow_building) → h = 22 :=
by
  sorry

end height_of_building_l262_262280


namespace dodecahedron_interior_diagonals_l262_262664

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l262_262664


namespace large_square_pattern_l262_262068

theorem large_square_pattern :
  999999^2 = 1000000 * 999998 + 1 :=
by sorry

end large_square_pattern_l262_262068


namespace sequence_term_500_l262_262502

theorem sequence_term_500 :
  ∃ (a : ℕ → ℤ), 
  a 1 = 1001 ∧
  a 2 = 1005 ∧
  (∀ n, 1 ≤ n → (a n + a (n+1) + a (n+2)) = 2 * n) → 
  a 500 = 1334 := 
sorry

end sequence_term_500_l262_262502


namespace trader_sold_meters_l262_262583

-- Defining the context and conditions
def cost_price_per_meter : ℝ := 100
def profit_per_meter : ℝ := 5
def total_selling_price : ℝ := 8925

-- Calculating the selling price per meter
def selling_price_per_meter : ℝ := cost_price_per_meter + profit_per_meter

-- The problem statement: proving the number of meters sold is 85
theorem trader_sold_meters : (total_selling_price / selling_price_per_meter) = 85 :=
by
  sorry

end trader_sold_meters_l262_262583


namespace sum_of_solutions_l262_262414

-- Define the polynomial equation and the condition
def equation (x : ℝ) : Prop := 3 = (x^3 - 3 * x^2 - 12 * x) / (x + 3)

-- Sum of solutions for the given polynomial equation under the constraint
theorem sum_of_solutions :
  (∀ x : ℝ, equation x → x ≠ -3) →
  ∃ (a b : ℝ), equation a ∧ equation b ∧ a + b = 4 := 
by
  intros h
  sorry

end sum_of_solutions_l262_262414


namespace greatest_two_digit_prod_12_l262_262768

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l262_262768


namespace maximum_quadratic_expr_l262_262604

noncomputable def quadratic_expr (x : ℝ) : ℝ :=
  -5 * x^2 + 25 * x - 7

theorem maximum_quadratic_expr : ∃ x : ℝ, quadratic_expr x = 53 / 4 :=
by
  sorry

end maximum_quadratic_expr_l262_262604


namespace avg_waiting_time_l262_262830

theorem avg_waiting_time : 
  let P_G := 1 / 3      -- Probability of green light
  let P_red := 2 / 3    -- Probability of red light
  let E_T_given_G := 0  -- Expected time given green light
  let E_T_given_red := 1 -- Expected time given red light
  (E_T_given_G * P_G) + (E_T_given_red * P_red) = 2 / 3
:= by
  sorry

end avg_waiting_time_l262_262830


namespace infinite_series_limit_l262_262298

noncomputable def geo_series_sum (a r : ℝ) (h : |r| < 1) : ℝ := a / (1 - r)

theorem infinite_series_limit :
  let s := ∑' n : ℕ, (1 / (5 ^ n) + (1 * sqrt 3) / (5 ^ (n+1))) in
  s = 1 / 4 * (5 + sqrt 3) :=
by
  sorry

end infinite_series_limit_l262_262298


namespace min_value_of_y_min_value_achieved_l262_262871

noncomputable def y (x : ℝ) : ℝ := x + 1/x + 16*x / (x^2 + 1)

theorem min_value_of_y : ∀ x > 1, y x ≥ 8 :=
  sorry

theorem min_value_achieved : ∃ x, (x > 1) ∧ (y x = 8) :=
  sorry

end min_value_of_y_min_value_achieved_l262_262871


namespace smallest_number_l262_262949

theorem smallest_number (a b c d : ℤ) (h_a : a = 0) (h_b : b = -1) (h_c : c = -4) (h_d : d = 5) : 
  c < b ∧ c < a ∧ c < d :=
by {
  sorry
}

end smallest_number_l262_262949


namespace dice_sum_probability_15_l262_262560
open Nat

theorem dice_sum_probability_15 (n : ℕ) (h : n = 3432) : 
  ∃ d1 d2 d3 d4 d5 d6 d7 d8 : ℕ,
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ 
  (1 ≤ d3 ∧ d3 ≤ 6) ∧ (1 ≤ d4 ∧ d4 ≤ 6) ∧ 
  (1 ≤ d5 ∧ d5 ≤ 6) ∧ (1 ≤ d6 ∧ d6 ≤ 6) ∧ 
  (1 ≤ d7 ∧ d7 ≤ 6) ∧ (1 ≤ d8 ∧ d8 ≤ 6) ∧ 
  (d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 = 15) :=
by
  sorry

end dice_sum_probability_15_l262_262560


namespace parallel_proof_l262_262717

noncomputable theory
open EuclideanGeometry

variables (A B C H_A H_B H_C D E : Point)

def triangle (A B C : Point) : Prop :=
  ∃ (AB BC CA : Line), line_through AB A B ∧ line_through BC B C ∧ line_through CA C A

def is_altitude_foot (A B C H_A : Point) : Prop :=
  ∃ (AB BC : Line), 
    line_through AB A B ∧ line_through BC B C ∧ perpendicular AB (line_through H_A B C)

def is_projection (H_A D AB : Point) : Prop :=
  perpendicular (line_through H_A D) AB

def is_projection (H_A E BC : Point) : Prop :=
  perpendicular (line_through H_A E) BC

theorem parallel_proof 
  (h_triangle : triangle A B C)
  (h_H_A : is_altitude_foot A B C H_A)
  (h_H_B : is_altitude_foot B A C H_B)
  (h_H_C : is_altitude_foot C A B H_C)
  (h_proj_AB : is_projection H_A D (line_through A B))
  (h_proj_BC : is_projection H_A E (line_through B C)) :
  parallel (line_through D E) (line_through H_B H_C) :=
sorry

end parallel_proof_l262_262717


namespace intersection_A_B_l262_262155

open Set

def isInSetA (x : ℕ) : Prop := ∃ n : ℕ, x = 3 * n + 2
def A : Set ℕ := { x | isInSetA x }
def B : Set ℕ := {6, 8, 10, 12, 14}

theorem intersection_A_B :
  A ∩ B = {8, 14} :=
sorry

end intersection_A_B_l262_262155


namespace convert_kmph_to_mps_l262_262565

theorem convert_kmph_to_mps (speed_kmph : ℝ) (km_to_m : ℝ) (hr_to_s : ℝ) : 
  speed_kmph = 56 → km_to_m = 1000 → hr_to_s = 3600 → 
  (speed_kmph * (km_to_m / hr_to_s) : ℝ) = 15.56 :=
by
  intros
  sorry

end convert_kmph_to_mps_l262_262565


namespace average_death_rate_l262_262051

variable (birth_rate : ℕ) (net_increase_day : ℕ)

noncomputable def death_rate_per_two_seconds (birth_rate net_increase_day : ℕ) : ℕ :=
  let seconds_per_day := 86400
  let net_increase_per_second := net_increase_day / seconds_per_day
  let birth_rate_per_second := birth_rate / 2
  let death_rate_per_second := birth_rate_per_second - net_increase_per_second
  2 * death_rate_per_second

theorem average_death_rate
  (birth_rate : ℕ := 4) 
  (net_increase_day : ℕ := 86400) :
  death_rate_per_two_seconds birth_rate net_increase_day = 2 :=
sorry

end average_death_rate_l262_262051


namespace reduced_rectangle_area_l262_262945

theorem reduced_rectangle_area
  (w h : ℕ) (hw : w = 5) (hh : h = 7)
  (new_w : ℕ) (h_reduced_area : new_w = w - 2 ∧ new_w * h = 21)
  (reduced_h : ℕ) (hr : reduced_h = h - 1) :
  (new_w * reduced_h = 18) :=
by
  sorry

end reduced_rectangle_area_l262_262945


namespace mike_reaches_office_time_l262_262132

-- Define the given conditions
def dave_steps_per_minute : ℕ := 80
def dave_step_length_cm : ℕ := 85
def dave_time_min : ℕ := 20

def mike_steps_per_minute : ℕ := 95
def mike_step_length_cm : ℕ := 70

-- Define Dave's walking speed
def dave_speed_cm_per_min : ℕ := dave_steps_per_minute * dave_step_length_cm

-- Define the total distance to the office
def distance_to_office_cm : ℕ := dave_speed_cm_per_min * dave_time_min

-- Define Mike's walking speed
def mike_speed_cm_per_min : ℕ := mike_steps_per_minute * mike_step_length_cm

-- Define the time it takes Mike to walk to the office
noncomputable def mike_time_to_office_min : ℚ := distance_to_office_cm / mike_speed_cm_per_min

-- State the theorem to prove
theorem mike_reaches_office_time :
  mike_time_to_office_min = 20.45 :=
sorry

end mike_reaches_office_time_l262_262132


namespace closest_perfect_square_to_350_l262_262261

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l262_262261


namespace eliana_total_steps_l262_262614

noncomputable def day1_steps : ℕ := 200 + 300
noncomputable def day2_steps : ℕ := 2 * day1_steps
noncomputable def day3_steps : ℕ := day1_steps + day2_steps + 100

theorem eliana_total_steps : day3_steps = 1600 := by
  sorry

end eliana_total_steps_l262_262614


namespace castle_lego_ratio_l262_262710

def total_legos : ℕ := 500
def legos_put_back : ℕ := 245
def legos_missing : ℕ := 5
def legos_used : ℕ := total_legos - legos_put_back - legos_missing
def ratio (a b : ℕ) : ℚ := a / b

theorem castle_lego_ratio : ratio legos_used total_legos = 1 / 2 :=
by
  unfold ratio legos_used total_legos legos_put_back legos_missing
  norm_num

end castle_lego_ratio_l262_262710


namespace intersection_of_sets_l262_262633

def set_A (x : ℝ) : Prop := |x - 1| < 3
def set_B (x : ℝ) : Prop := (x - 1) / (x - 5) < 0

theorem intersection_of_sets : ∀ x : ℝ, (set_A x ∧ set_B x) ↔ 1 < x ∧ x < 4 := 
by sorry

end intersection_of_sets_l262_262633


namespace flagpole_height_l262_262244

theorem flagpole_height (h : ℕ)
  (shadow_flagpole : ℕ := 72)
  (height_pole : ℕ := 18)
  (shadow_pole : ℕ := 27)
  (ratio_shadow : shadow_flagpole / shadow_pole = 8 / 3) :
  h = 48 :=
by
  sorry

end flagpole_height_l262_262244


namespace manufacturing_cost_before_decrease_l262_262928

def original_manufacturing_cost (P : ℝ) (C_now : ℝ) (profit_rate_now : ℝ) : ℝ :=
  P - profit_rate_now * P

theorem manufacturing_cost_before_decrease
  (P : ℝ)
  (C_now : ℝ)
  (profit_rate_now : ℝ)
  (profit_rate_original : ℝ)
  (H1 : C_now = P - profit_rate_now * P)
  (H2 : profit_rate_now = 0.50)
  (H3 : profit_rate_original = 0.20)
  (H4 : C_now = 50) :
  original_manufacturing_cost P C_now profit_rate_now = 80 :=
sorry

end manufacturing_cost_before_decrease_l262_262928


namespace closest_perfect_square_to_350_l262_262254

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l262_262254


namespace total_steps_eliana_walked_l262_262617

-- Define the conditions of the problem.
def first_day_exercise_steps : Nat := 200
def first_day_additional_steps : Nat := 300
def second_day_multiplier : Nat := 2
def third_day_additional_steps : Nat := 100

-- Define the steps calculation for each day.
def first_day_total_steps : Nat := first_day_exercise_steps + first_day_additional_steps
def second_day_total_steps : Nat := second_day_multiplier * first_day_total_steps
def third_day_total_steps : Nat := second_day_total_steps + third_day_additional_steps

-- Prove that the total number of steps Eliana walked during these three days is 1600.
theorem total_steps_eliana_walked :
  first_day_total_steps + second_day_total_steps + third_day_additional_steps = 1600 :=
by
  -- Conditional values are constants. We can use Lean's deterministic evaluator here.
  -- Hence, there's no need to write out full proof for now. Using sorry to bypass actual proof.
  sorry

end total_steps_eliana_walked_l262_262617


namespace dodecahedron_interior_diagonals_l262_262662

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l262_262662


namespace problem_solution_l262_262980

noncomputable def root1 : ℝ := (3 + Real.sqrt 105) / 4
noncomputable def root2 : ℝ := (3 - Real.sqrt 105) / 4

theorem problem_solution :
  (∀ x : ℝ, x ≠ -2 → x ≠ -3 → (x^3 - x^2 - 4 * x) / (x^2 + 5 * x + 6) + x = -4
    → x = root1 ∨ x = root2) := 
by
  sorry

end problem_solution_l262_262980


namespace jasmine_cookies_l262_262521

theorem jasmine_cookies (J : ℕ) (h1 : 20 + J + (J + 10) = 60) : J = 15 :=
sorry

end jasmine_cookies_l262_262521


namespace doris_needs_weeks_l262_262966

noncomputable def average_weeks_to_cover_expenses (weekly_babysit_hours: ℝ) (saturday_hours: ℝ) : ℝ := 
  let weekday_income := weekly_babysit_hours * 20
  let saturday_income := saturday_hours * (if weekly_babysit_hours > 15 then 15 else 20)
  let teaching_income := 100
  let total_weekly_income := weekday_income + saturday_income + teaching_income
  let monthly_income_before_tax := total_weekly_income * 4
  let monthly_income_after_tax := monthly_income_before_tax * 0.85
  monthly_income_after_tax / 4 / 1200

theorem doris_needs_weeks (weekly_babysit_hours: ℝ) (saturday_hours: ℝ) :
  1200 ≤ (average_weeks_to_cover_expenses weekly_babysit_hours saturday_hours) * 4 * 1200 :=
  by
    sorry

end doris_needs_weeks_l262_262966


namespace sqrt_of_4_l262_262399

theorem sqrt_of_4 : {x : ℤ | x^2 = 4} = {2, -2} :=
by
  sorry

end sqrt_of_4_l262_262399


namespace total_population_l262_262168

-- Definitions based on given conditions
variables (b g t : ℕ)
variables (h1 : b = 4 * g) (h2 : g = 8 * t)

-- Theorem statement
theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) : b + g + t = 41 * t :=
by
  sorry

end total_population_l262_262168


namespace cubic_sum_l262_262347

theorem cubic_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end cubic_sum_l262_262347


namespace probability_each_number_appears_once_l262_262102

-- Define the event that each number from 1 to 6 appears at least once when 10 fair dice are rolled
def each_number_appears_once : ℕ → Prop := λ n, n = 10

-- Define the probability calculation
noncomputable def prob_each_number_appears_once :=
  1 - (6 * (5/6)^10 - 15 * (4/6)^10 + 20 * (3/6)^10 - 15 * (2/6)^10 + 6 * (1/6)^10)

-- Prove that the probability is 0.272
theorem probability_each_number_appears_once :
  each_number_appears_once 10 → prob_each_number_appears_once = 0.272 :=
by
  intro h
  sorry

end probability_each_number_appears_once_l262_262102


namespace fewer_miles_per_gallon_city_l262_262932

-- Define the given conditions.
def miles_per_tankful_highway : ℕ := 420
def miles_per_tankful_city : ℕ := 336
def miles_per_gallon_city : ℕ := 24

-- Define the question as a theorem that proves how many fewer miles per gallon in the city compared to the highway.
theorem fewer_miles_per_gallon_city (G : ℕ) (hG : G = miles_per_tankful_city / miles_per_gallon_city) :
  miles_per_tankful_highway / G - miles_per_gallon_city = 6 :=
by
  -- The proof will be provided here.
  sorry

end fewer_miles_per_gallon_city_l262_262932


namespace karens_class_fund_l262_262364

noncomputable def ratio_of_bills (T W : ℕ) : ℕ × ℕ := (T / Nat.gcd T W, W / Nat.gcd T W)

theorem karens_class_fund (T W : ℕ) (hW : W = 3) (hfund : 10 * T + 20 * W = 120) :
  ratio_of_bills T W = (2, 1) :=
by
  sorry

end karens_class_fund_l262_262364


namespace quadratic_roots_range_quadratic_root_condition_l262_262092

-- Problem 1: Prove that the range of real number \(k\) for which the quadratic 
-- equation \(x^{2} + (2k + 1)x + k^{2} + 1 = 0\) has two distinct real roots is \(k > \frac{3}{4}\). 
theorem quadratic_roots_range (k : ℝ) : 
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x^2 + (2*k+1)*x + k^2 + 1 = 0) ↔ (k > 3/4) := 
sorry

-- Problem 2: Given \(k > \frac{3}{4}\), prove that if the roots \(x₁\) and \(x₂\) of 
-- the equation satisfy \( |x₁| + |x₂| = x₁ \cdot x₂ \), then \( k = 2 \).
theorem quadratic_root_condition (k : ℝ) 
    (hk : k > 3 / 4)
    (x₁ x₂ : ℝ)
    (h₁ : x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0)
    (h₂ : x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0)
    (h3 : |x₁| + |x₂| = x₁ * x₂) : 
    k = 2 := 
sorry

end quadratic_roots_range_quadratic_root_condition_l262_262092


namespace algebraic_expression_value_l262_262490

theorem algebraic_expression_value (a b : ℝ) (h : 4 * b = 3 + 4 * a) :
  a + (a - (a - (a - b) - b) - b) - b = -3 / 2 := by
  sorry

end algebraic_expression_value_l262_262490


namespace original_bananas_total_l262_262279

theorem original_bananas_total (willie_bananas : ℝ) (charles_bananas : ℝ) : willie_bananas = 48.0 → charles_bananas = 35.0 → willie_bananas + charles_bananas = 83.0 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end original_bananas_total_l262_262279


namespace statement_II_and_IV_true_l262_262542

-- Definitions based on the problem's conditions
def AllNewEditions (P : Type) (books : P → Prop) := ∀ x, books x

-- Condition that the statement "All books in the library are new editions." is false
def NotAllNewEditions (P : Type) (books : P → Prop) := ¬ (AllNewEditions P books)

-- Statements to analyze
def SomeBookNotNewEdition (P : Type) (books : P → Prop) := ∃ x, ¬ books x
def NotAllBooksNewEditions (P : Type) (books : P → Prop) := ∃ x, ¬ books x

-- The theorem to prove
theorem statement_II_and_IV_true 
  (P : Type) 
  (books : P → Prop) 
  (h : NotAllNewEditions P books) : 
  SomeBookNotNewEdition P books ∧ NotAllBooksNewEditions P books :=
  by
    sorry

end statement_II_and_IV_true_l262_262542


namespace find_f_2004_l262_262020

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom odd_g : ∀ x : ℝ, g (-x) = -g x
axiom g_eq_f_shift : ∀ x : ℝ, g x = f (x - 1)
axiom g_one : g 1 = 2003

theorem find_f_2004 : f 2004 = 2003 :=
  sorry

end find_f_2004_l262_262020


namespace dodecahedron_interior_diagonals_l262_262656

theorem dodecahedron_interior_diagonals (vertices faces: ℕ) (pentagonal: ∀ face, face = 5) (vertices_per_face : ∀ face ∈ (range faces), 3) :
  vertices = 20 ∧ faces = 12 ∧ interior_diagonals = 160 :=
by
    -- Definitions and problem conditions
    let vertices := 20
    let faces := 12
    let pentagonal := (λ face, face = 5)
    let vertices_per_face := (λ face, 3)

    -- Assertion
    have interior_diagonals : ℕ := (vertices * 16) / 2
    rw Nat.div_eq_of_eq_mul_right _ _ _,
    have h: vertices = 20 ∧ faces = 12 ∧ interior_diagonals = 160 := by
        sorry

    exact h

end dodecahedron_interior_diagonals_l262_262656


namespace total_green_and_yellow_peaches_in_basket_l262_262547

def num_red_peaches := 5
def num_yellow_peaches := 14
def num_green_peaches := 6

theorem total_green_and_yellow_peaches_in_basket :
  num_yellow_peaches + num_green_peaches = 20 :=
by
  sorry

end total_green_and_yellow_peaches_in_basket_l262_262547


namespace total_games_l262_262116

variable (L : ℕ) -- Number of games the team lost

-- Define the number of wins
def Wins := 3 * L + 14

theorem total_games (h_wins : Wins = 101) : (Wins + L = 130) :=
by
  sorry

end total_games_l262_262116


namespace sin_identity_l262_262856

theorem sin_identity (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α + π / 4) ^ 2 = 5 / 6 := 
sorry

end sin_identity_l262_262856


namespace side_length_of_square_l262_262827

theorem side_length_of_square (s : ℝ) (h : s^2 = 2 * (4 * s)) : s = 8 := 
by
  sorry

end side_length_of_square_l262_262827


namespace percentage_decrease_in_savings_l262_262895

theorem percentage_decrease_in_savings (I : ℝ) (F : ℝ) (IncPercent : ℝ) (decPercent : ℝ)
  (h1 : I = 125) (h2 : IncPercent = 0.25) (h3 : F = 125) :
  let P := (I * (1 + IncPercent))
  ∃ decPercent, decPercent = ((P - F) / P) * 100 ∧ decPercent = 20 := 
by
  sorry

end percentage_decrease_in_savings_l262_262895


namespace olivia_pays_in_dollars_l262_262199

theorem olivia_pays_in_dollars (q_chips q_soda : ℕ) 
  (h_chips : q_chips = 4) (h_soda : q_soda = 12) : (q_chips + q_soda) / 4 = 4 := by
  sorry

end olivia_pays_in_dollars_l262_262199


namespace janet_saving_l262_262056

def tile_cost_difference_saving : ℕ :=
  let turquoise_cost_per_tile := 13
  let purple_cost_per_tile := 11
  let area_wall1 := 5 * 8
  let area_wall2 := 7 * 8
  let total_area := area_wall1 + area_wall2
  let tiles_per_square_foot := 4
  let number_of_tiles := total_area * tiles_per_square_foot
  let cost_difference_per_tile := turquoise_cost_per_tile - purple_cost_per_tile
  number_of_tiles * cost_difference_per_tile

theorem janet_saving : tile_cost_difference_saving = 768 := by
  sorry

end janet_saving_l262_262056


namespace trace_bag_weight_l262_262751

-- Definitions for the given problem
def weight_gordon_bag1 := 3
def weight_gordon_bag2 := 7
def total_weight_gordon := weight_gordon_bag1 + weight_gordon_bag2

noncomputable def weight_trace_one_bag : ℕ :=
  sorry

-- Theorem for what we need to prove
theorem trace_bag_weight :
  total_weight_gordon = 10 ∧
  weight_trace_one_bag = total_weight_gordon / 5 :=
sorry

end trace_bag_weight_l262_262751


namespace minimum_value_an_eq_neg28_at_n_eq_3_l262_262993

noncomputable def seq_an (n : ℕ) : ℝ :=
  if n > 0 then (5 / 2) * n^2 - (13 / 2) * n
  else 0

noncomputable def delta_seq_an (n : ℕ) : ℝ := seq_an (n + 1) - seq_an n

noncomputable def delta2_seq_an (n : ℕ) : ℝ := delta_seq_an (n + 1) - delta_seq_an n

theorem minimum_value_an_eq_neg28_at_n_eq_3 : 
  ∃ (n : ℕ), n = 3 ∧ seq_an n = -28 :=
by
  sorry

end minimum_value_an_eq_neg28_at_n_eq_3_l262_262993


namespace line_circle_intersection_l262_262154

theorem line_circle_intersection (k : ℝ) :
  ∃ x y : ℝ, y = k * (x + 1 / 2) ∧ x^2 + y^2 = 1 :=
sorry

end line_circle_intersection_l262_262154


namespace gcd_12345_6789_eq_3_l262_262756

theorem gcd_12345_6789_eq_3 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_eq_3_l262_262756


namespace find_age_of_15th_person_l262_262904

-- Define the conditions given in the problem
def total_age_of_18_persons (avg_18 : ℕ) (num_18 : ℕ) : ℕ := avg_18 * num_18
def total_age_of_5_persons (avg_5 : ℕ) (num_5 : ℕ) : ℕ := avg_5 * num_5
def total_age_of_9_persons (avg_9 : ℕ) (num_9 : ℕ) : ℕ := avg_9 * num_9

-- Define the overall question which is the age of the 15th person
def age_of_15th_person (total_18 : ℕ) (total_5 : ℕ) (total_9 : ℕ) : ℕ :=
  total_18 - total_5 - total_9

-- Statement of the theorem to prove
theorem find_age_of_15th_person :
  let avg_18 := 15
  let num_18 := 18
  let avg_5 := 14
  let num_5 := 5
  let avg_9 := 16
  let num_9 := 9
  let total_18 := total_age_of_18_persons avg_18 num_18 
  let total_5 := total_age_of_5_persons avg_5 num_5
  let total_9 := total_age_of_9_persons avg_9 num_9
  age_of_15th_person total_18 total_5 total_9 = 56 :=
by
  -- Definitions for the total ages
  let avg_18 := 15
  let num_18 := 18
  let avg_5 := 14
  let num_5 := 5
  let avg_9 := 16
  let num_9 := 9
  let total_18 := total_age_of_18_persons avg_18 num_18 
  let total_5 := total_age_of_5_persons avg_5 num_5
  let total_9 := total_age_of_9_persons avg_9 num_9
  
  -- Goal: compute the age of the 15th person
  let answer := age_of_15th_person total_18 total_5 total_9

  -- Prove that the computed age is equal to 56
  show answer = 56
  sorry

end find_age_of_15th_person_l262_262904


namespace find_b_over_a_find_angle_B_l262_262877

-- Definitions and main theorems
noncomputable def sides_in_triangle (A B C a b c : ℝ) : Prop :=
  a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = Real.sqrt 2 * a

noncomputable def cos_law_condition (a b c : ℝ) : Prop :=
  c^2 = b^2 + Real.sqrt 3 * a^2

theorem find_b_over_a {A B C a b c : ℝ} (h : sides_in_triangle A B C a b c) : b / a = Real.sqrt 2 :=
  sorry

theorem find_angle_B {A B C a b c : ℝ} (h1 : sides_in_triangle A B C a b c) (h2 : cos_law_condition a b c)
  (h3 : b / a = Real.sqrt 2) : B = Real.pi / 4 :=
  sorry

end find_b_over_a_find_angle_B_l262_262877


namespace sqrt_four_eq_pm_two_l262_262400

theorem sqrt_four_eq_pm_two : ∀ (x : ℝ), x * x = 4 ↔ x = 2 ∨ x = -2 := 
by
  sorry

end sqrt_four_eq_pm_two_l262_262400


namespace mod_sum_correct_l262_262688

theorem mod_sum_correct (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
    (h1 : a * b * c ≡ 1 [MOD 7])
    (h2 : 5 * c ≡ 2 [MOD 7])
    (h3 : 6 * b ≡ 3 + b [MOD 7]) :
    (a + b + c) % 7 = 4 := sorry

end mod_sum_correct_l262_262688


namespace perpendicular_lines_l262_262149

theorem perpendicular_lines (m : ℝ) :
  (∃ k l : ℝ, k * m + (1 - m) * l = 3 ∧ (m - 1) * k + (2 * m + 3) * l = 2) → m = -3 ∨ m = 1 :=
by sorry

end perpendicular_lines_l262_262149


namespace closest_perfect_square_to_350_l262_262265

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l262_262265


namespace quadrilateral_inequality_l262_262915

theorem quadrilateral_inequality (A C : ℝ) (AB AC AD BC CD : ℝ) (h1 : A + C < 180) (h2 : A > 0) (h3 : C > 0) (h4 : AB > 0) (h5 : AC > 0) (h6 : AD > 0) (h7 : BC > 0) (h8 : CD > 0) : 
  AB * CD + AD * BC < AC * (AB + AD) := 
sorry

end quadrilateral_inequality_l262_262915


namespace first_worker_time_budget_l262_262746

theorem first_worker_time_budget
  (total_time : ℝ := 1)
  (second_worker_time : ℝ := 1 / 3)
  (third_worker_time : ℝ := 1 / 3)
  (x : ℝ) :
  x + second_worker_time + third_worker_time = total_time → x = 1 / 3 :=
by
  sorry

end first_worker_time_budget_l262_262746


namespace area_of_triangle_bounded_by_lines_l262_262315

def line1 (x : ℝ) : ℝ := 2 * x + 3
def line2 (x : ℝ) : ℝ := - x + 5

theorem area_of_triangle_bounded_by_lines :
  let x_intercept_line1 := -3 / 2
  let x_intercept_line2 := 5
  let base := x_intercept_line2 - x_intercept_line1
  let intersection_x := 2 / 3
  let intersection_y := line1 intersection_x
  let height := intersection_y
  let area := (1 / 2) * base * height
  area = 169 / 12 := 
by
  sorry

end area_of_triangle_bounded_by_lines_l262_262315


namespace church_members_l262_262934

theorem church_members (M A C : ℕ) (h1 : A = 4/10 * M)
  (h2 : C = 6/10 * M) (h3 : C = A + 24) : M = 120 := 
  sorry

end church_members_l262_262934


namespace total_bill_is_270_l262_262079

-- Conditions as Lean definitions
def totalBill (T : ℝ) : Prop :=
  let eachShare := T / 10
  9 * (eachShare + 3) = T

-- Theorem stating that the total bill T is 270
theorem total_bill_is_270 (T : ℝ) (h : totalBill T) : T = 270 :=
sorry

end total_bill_is_270_l262_262079


namespace eliana_total_steps_l262_262612

noncomputable def day1_steps : ℕ := 200 + 300
noncomputable def day2_steps : ℕ := 2 * day1_steps
noncomputable def day3_steps : ℕ := day1_steps + day2_steps + 100

theorem eliana_total_steps : day3_steps = 1600 := by
  sorry

end eliana_total_steps_l262_262612


namespace smallest_number_in_systematic_sample_l262_262853

theorem smallest_number_in_systematic_sample (n m x : ℕ) (products : Finset ℕ) :
  n = 80 ∧ m = 5 ∧ products = Finset.range n ∧ x = 42 ∧ x ∈ products ∧ (∃ k : ℕ, x = (n / m) * k + 10) → 10 ∈ products :=
by
  sorry

end smallest_number_in_systematic_sample_l262_262853


namespace profit_is_35_percent_l262_262812

def cost_price (C : ℝ) := C
def initial_selling_price (C : ℝ) := 1.20 * C
def second_selling_price (C : ℝ) := 1.50 * C
def final_selling_price (C : ℝ) := 1.35 * C

theorem profit_is_35_percent (C : ℝ) : 
    final_selling_price C - cost_price C = 0.35 * cost_price C :=
by
    sorry

end profit_is_35_percent_l262_262812


namespace abc_inequality_l262_262378

theorem abc_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_cond : a * b + b * c + c * a = 1) :
  (a + b + c) ≥ Real.sqrt 3 ∧ (a + b + c = Real.sqrt 3 ↔ a = b ∧ b = c ∧ c = Real.sqrt 1 / Real.sqrt 3) :=
by sorry

end abc_inequality_l262_262378


namespace find_a_l262_262183

noncomputable def a : ℝ :=
  let f (x : ℝ) := a^x
  let condition := (a > 0) ∧ (a ≠ 1) ∧ (f 1 - f (-1)) / (f 2 - f (-2)) = 3 / 10
  have condition : Prop := by
    sorry
  have h1 : f (1) = a
  have h2 : f (-1) = 1 / a
  have h3 : f (2) = a^2
  have h4 : f (-2) = 1 / (a^2)
  solve_by_elim

theorem find_a (a : ℝ) (h : (a > 0) ∧ (a ≠ 1) ∧ ((a - 1 / a) / (a^2 - 1 / a^2) = 3 / 10)) : a = 3 :=
  sorry

end find_a_l262_262183


namespace y_intercept_of_line_l262_262541

theorem y_intercept_of_line (m : ℝ) (x₀ : ℝ) (y₀ : ℝ) (h_slope : m = -3) (h_intercept : (x₀, y₀) = (7, 0)) : (0, 21) = (0, (y₀ - m * x₀)) :=
by
  sorry

end y_intercept_of_line_l262_262541


namespace distances_inequality_l262_262468

theorem distances_inequality (x y : ℝ) :
  Real.sqrt ((x + 4)^2 + (y + 2)^2) + 
  Real.sqrt ((x - 5)^2 + (y + 4)^2) ≤ 
  Real.sqrt ((x - 2)^2 + (y - 6)^2) + 
  Real.sqrt ((x - 5)^2 + (y - 6)^2) + 20 :=
  sorry

end distances_inequality_l262_262468


namespace fraction_a_b_l262_262500

variables {a b x y : ℝ}

theorem fraction_a_b (h1 : 4 * x - 2 * y = a) (h2 : 6 * y - 12 * x = b) (hb : b ≠ 0) :
  a / b = -1/3 := 
sorry

end fraction_a_b_l262_262500


namespace last_two_digits_of_power_sequence_l262_262982

noncomputable def power_sequence (n : ℕ) : ℤ :=
  (Int.sqrt 29 + Int.sqrt 21)^(2 * n) + (Int.sqrt 29 - Int.sqrt 21)^(2 * n)

theorem last_two_digits_of_power_sequence :
  (power_sequence 992) % 100 = 71 := by
  sorry

end last_two_digits_of_power_sequence_l262_262982


namespace dodecahedron_interior_diagonals_l262_262679

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l262_262679


namespace sum_of_x_values_satisfying_eq_l262_262415

noncomputable def rational_eq_sum (x : ℝ) : Prop :=
3 = (x^3 - 3 * x^2 - 12 * x) / (x + 3)

theorem sum_of_x_values_satisfying_eq :
  (∃ (x : ℝ), rational_eq_sum x) ∧ (x ≠ -3 → (x_1 + x_2) = 6) :=
sorry

end sum_of_x_values_satisfying_eq_l262_262415


namespace imo_34_l262_262514

-- Define the input conditions
variables (R r ρ : ℝ)

-- The main theorem we need to prove
theorem imo_34 { R r ρ : ℝ } (hR : R = 1) : 
  ρ ≤ 1 - (1/3) * (1 + r)^2 :=
sorry

end imo_34_l262_262514


namespace expression_eval_l262_262453

theorem expression_eval :
    (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
    (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) * 5040 = 
    (5^128 - 4^128) * 5040 := by
  sorry

end expression_eval_l262_262453


namespace max_sum_numbered_cells_max_zero_number_cell_l262_262354

-- Part 1
theorem max_sum_numbered_cells (n : ℕ) (grid : Matrix (Fin (2*n+1)) (Fin (2*n+1)) Cell) (mines : Finset (Fin (2*n+1) × Fin (2*n+1))) 
  (h1 : mines.card = n^2 + 1) :
  ∃ sum : ℕ, sum = 8 * n^2 + 4 := sorry

-- Part 2
theorem max_zero_number_cell (n k : ℕ) (grid : Matrix (Fin n) (Fin n) Cell) (mines : Finset (Fin n × Fin n)) 
  (h1 : mines.card = k) :
  ∃ (k_max : ℕ), k_max = (Nat.floor ((n + 2) / 3) ^ 2) - 1 := sorry

end max_sum_numbered_cells_max_zero_number_cell_l262_262354


namespace problem_statement_l262_262387

theorem problem_statement (x y z : ℝ) :
    2 * x > y^2 + z^2 →
    2 * y > x^2 + z^2 →
    2 * z > y^2 + x^2 →
    x * y * z < 1 := by
  sorry

end problem_statement_l262_262387


namespace votes_cast_l262_262105

theorem votes_cast (candidate_percentage : ℝ) (vote_difference : ℝ) (total_votes : ℝ) 
  (h1 : candidate_percentage = 0.30) 
  (h2 : vote_difference = 1760) 
  (h3 : total_votes = vote_difference / (1 - 2 * candidate_percentage)) 
  : total_votes = 4400 := by
  sorry

end votes_cast_l262_262105


namespace range_of_a_l262_262499

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 4 * x + a^2 ≤ 0 → false) ↔ (a < -2 ∨ a > 2) := 
by
  sorry

end range_of_a_l262_262499


namespace cos_54_deg_l262_262458

-- Define cosine function
noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

-- The main theorem statement
theorem cos_54_deg : cos_deg 54 = (-1 + Real.sqrt 5) / 4 :=
  sorry

end cos_54_deg_l262_262458


namespace color_of_182nd_marble_l262_262947

-- conditions
def pattern_length : ℕ := 15
def blue_length : ℕ := 6
def red_length : ℕ := 5
def green_length : ℕ := 4

def marble_color (n : ℕ) : String :=
  let cycle_pos := n % pattern_length
  if cycle_pos < blue_length then
    "blue"
  else if cycle_pos < blue_length + red_length then
    "red"
  else
    "green"

theorem color_of_182nd_marble : marble_color 182 = "blue" :=
by
  sorry

end color_of_182nd_marble_l262_262947


namespace triangle_angles_ratios_l262_262532

def angles_of_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

theorem triangle_angles_ratios (α β γ : ℝ)
  (h1 : α + β + γ = 180) 
  (h2 : β = 2 * α)
  (h3 : γ = 3 * α) : 
  angles_of_triangle 60 45 75 ∨ angles_of_triangle 45 22.5 112.5 :=
by
  sorry

end triangle_angles_ratios_l262_262532


namespace closest_perfect_square_to_350_l262_262262

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l262_262262


namespace greatest_two_digit_with_product_12_l262_262801

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l262_262801


namespace probability_at_least_two_same_class_l262_262917

/-- Define the number of classes -/
def numClasses : ℕ := 10

/-- Define the number of students -/
def numStudents : ℕ := 3

/-- Define the probability that at least 2 of the students are in the same class -/
theorem probability_at_least_two_same_class 
  (hc : numClasses = 10) 
  (hs : numStudents = 3) :
  (1 - (numClasses * (numClasses - 2) * (numClasses - 1)) / (numClasses * numClasses * numClasses)) = 7 / 25 :=
by
  sorry

end probability_at_least_two_same_class_l262_262917


namespace asha_wins_prob_l262_262391

theorem asha_wins_prob
  (lose_prob : ℚ)
  (h1 : lose_prob = 7/12)
  (h2 : ∀ (win_prob : ℚ), lose_prob + win_prob = 1) :
  ∃ (win_prob : ℚ), win_prob = 5/12 :=
by
  sorry

end asha_wins_prob_l262_262391


namespace sin_alpha_eq_sqrt_5_div_3_l262_262145

variable (α : ℝ)

theorem sin_alpha_eq_sqrt_5_div_3
  (hα : 0 < α ∧ α < Real.pi)
  (h : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) :
  Real.sin α = Real.sqrt 5 / 3 := 
by 
  sorry

end sin_alpha_eq_sqrt_5_div_3_l262_262145


namespace Whitney_money_left_over_l262_262277

theorem Whitney_money_left_over :
  let posters := 2
  let notebooks := 3
  let bookmarks := 2
  let cost_poster := 5
  let cost_notebook := 4
  let cost_bookmark := 2
  let total_cost_posters := posters * cost_poster
  let total_cost_notebooks := notebooks * cost_notebook
  let total_cost_bookmarks := bookmarks * cost_bookmark
  let total_cost := total_cost_posters + total_cost_notebooks + total_cost_bookmarks
  let initial_money := 2 * 20
  let money_left_over := initial_money - total_cost
  in
  money_left_over = 14 := sorry

end Whitney_money_left_over_l262_262277


namespace robot_Y_reaches_B_after_B_reaches_A_l262_262303

-- Definitions for the setup of the problem
def time_J_to_B (t_J_to_B : ℕ) := t_J_to_B = 12
def time_J_catch_up_B (t_J_catch_up_B : ℕ) := t_J_catch_up_B = 9

-- Main theorem to be proved
theorem robot_Y_reaches_B_after_B_reaches_A : 
  ∀ t_J_to_B t_J_catch_up_B, 
    (time_J_to_B t_J_to_B) → 
    (time_J_catch_up_B t_J_catch_up_B) →
    ∃ t : ℕ, t = 56 :=
by 
  sorry

end robot_Y_reaches_B_after_B_reaches_A_l262_262303


namespace minimum_spend_on_boxes_l262_262274

def box_dimensions : ℕ × ℕ × ℕ := (20, 20, 12)
def cost_per_box : ℝ := 0.40
def total_volume : ℕ := 2400000
def volume_of_box (l w h : ℕ) : ℕ := l * w * h
def number_of_boxes (total_vol vol_per_box : ℕ) : ℕ := total_vol / vol_per_box
def total_cost (num_boxes : ℕ) (cost_box : ℝ) : ℝ := num_boxes * cost_box

theorem minimum_spend_on_boxes : total_cost (number_of_boxes total_volume (volume_of_box 20 20 12)) cost_per_box = 200 := by
  sorry

end minimum_spend_on_boxes_l262_262274


namespace relationship_between_a_and_b_l262_262021

variable (a b : ℝ)

-- Conditions: Points lie on the line y = 2x + 1
def point_M (a : ℝ) : Prop := a = 2 * 2 + 1
def point_N (b : ℝ) : Prop := b = 2 * 3 + 1

-- Prove that a < b given the conditions
theorem relationship_between_a_and_b (hM : point_M a) (hN : point_N b) : a < b := 
sorry

end relationship_between_a_and_b_l262_262021


namespace TrishulPercentageLessThanRaghu_l262_262923

-- Define the variables and conditions
variables (R T V : ℝ)

-- Raghu's investment is Rs. 2200
def RaghuInvestment := (R : ℝ) = 2200

-- Vishal invested 10% more than Trishul
def VishalInvestment := (V : ℝ) = 1.10 * T

-- Total sum of investments is Rs. 6358
def TotalInvestment := R + T + V = 6358

-- Define the proof statement
theorem TrishulPercentageLessThanRaghu (R_is_2200 : RaghuInvestment R) 
    (V_is_10_percent_more : VishalInvestment V T) 
    (total_sum_is_6358 : TotalInvestment R T V) : 
  ((2200 - T) / 2200) * 100 = 10 :=
sorry

end TrishulPercentageLessThanRaghu_l262_262923


namespace rajans_position_l262_262880

theorem rajans_position
    (total_boys : ℕ)
    (vinay_position_from_right : ℕ)
    (boys_between_rajan_and_vinay : ℕ)
    (total_boys_eq : total_boys = 24)
    (vinay_position_from_right_eq : vinay_position_from_right = 10)
    (boys_between_eq : boys_between_rajan_and_vinay = 8) :
    ∃ R : ℕ, R = 6 :=
by
  sorry

end rajans_position_l262_262880


namespace max_area_of_inscribed_equilateral_triangle_l262_262396

noncomputable def maxInscribedEquilateralTriangleArea : ℝ :=
  let length : ℝ := 12
  let width : ℝ := 15
  let max_area := 369 * Real.sqrt 3 - 540
  max_area

theorem max_area_of_inscribed_equilateral_triangle :
  maxInscribedEquilateralTriangleArea = 369 * Real.sqrt 3 - 540 := 
by
  sorry

end max_area_of_inscribed_equilateral_triangle_l262_262396


namespace whitney_money_leftover_l262_262278

def poster_cost : ℕ := 5
def notebook_cost : ℕ := 4
def bookmark_cost : ℕ := 2

def posters : ℕ := 2
def notebooks : ℕ := 3
def bookmarks : ℕ := 2

def initial_money : ℕ := 2 * 20

def total_cost : ℕ := posters * poster_cost + notebooks * notebook_cost + bookmarks * bookmark_cost

def money_left_over : ℕ := initial_money - total_cost

theorem whitney_money_leftover : money_left_over = 14 := by
  sorry

end whitney_money_leftover_l262_262278


namespace greatest_two_digit_product_12_l262_262784

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l262_262784


namespace last_number_is_five_l262_262881

theorem last_number_is_five (seq : ℕ → ℕ) (h₀ : seq 0 = 5)
  (h₁ : ∀ n < 32, seq n + seq (n+1) + seq (n+2) + seq (n+3) + seq (n+4) + seq (n+5) = 29) :
  seq 36 = 5 :=
sorry

end last_number_is_five_l262_262881


namespace three_digit_number_property_l262_262138

theorem three_digit_number_property :
  (∃ a b c : ℕ, 100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999 ∧ 100 * a + 10 * b + c = (a + b + c)^3) ↔
  (∃ a b c : ℕ, a = 5 ∧ b = 1 ∧ c = 2 ∧ 100 * a + 10 * b + c = 512) := sorry

end three_digit_number_property_l262_262138


namespace original_concentration_A_l262_262745

-- Definitions of initial conditions and parameters
def mass_A : ℝ := 2000 -- 2 kg in grams
def mass_B : ℝ := 3000 -- 3 kg in grams
def pour_out_A : ℝ := 0.15 -- 15% poured out from bottle A
def pour_out_B : ℝ := 0.30 -- 30% poured out from bottle B
def mixed_concentration1 : ℝ := 27.5 -- 27.5% concentration after first mix
def pour_out_restored : ℝ := 0.40 -- 40% poured out again

-- Using the calculated remaining mass and concentration to solve the proof
theorem original_concentration_A (x y : ℝ) 
  (h1 : 300 * x + 900 * y = 27.5 * (300 + 900)) 
  (h2 : (1700 * x + 300 * 27.5) * 0.4 / (2000 * 0.4) + (2100 * y + 900 * 27.5) * 0.4 / (3000 * 0.4) = 26) : 
  x = 20 :=
by 
  -- Skipping the proof. The proof should involve solving the system of equations.
  sorry

end original_concentration_A_l262_262745


namespace greatest_two_digit_product_12_l262_262783

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l262_262783


namespace number_of_square_tiles_l262_262293

-- A box contains a mix of triangular and square tiles.
-- There are 30 tiles in total with 100 edges altogether.
variable (x y : ℕ) -- where x is the number of triangular tiles and y is the number of square tiles, both must be natural numbers
-- Each triangular tile has 3 edges, and each square tile has 4 edges.

-- Define the conditions
def tile_condition_1 : Prop := x + y = 30
def tile_condition_2 : Prop := 3 * x + 4 * y = 100

-- The goal is to prove the number of square tiles y is 10.
theorem number_of_square_tiles : tile_condition_1 x y → tile_condition_2 x y → y = 10 :=
  by
    intros h1 h2
    sorry

end number_of_square_tiles_l262_262293


namespace gcd_times_xyz_is_square_l262_262716

theorem gcd_times_xyz_is_square (x y z : ℕ) (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) : 
  ∃ k : ℕ, (Nat.gcd x (Nat.gcd y z) * x * y * z) = k ^ 2 :=
sorry

end gcd_times_xyz_is_square_l262_262716


namespace probability_of_error_is_0_05_l262_262552

noncomputable def chi_square : ℝ :=
  50 * ((13 * 20 - 10 * 7) ^ 2 : ℕ) / ((23 : ℕ) * (27 : ℕ) * (20 : ℕ) * (30 : ℕ))

theorem probability_of_error_is_0_05 :
  (3.841 < chi_square) ∧ (chi_square < 6.635) → 0.05 = 0.05 :=
by
  have chi_square := chi_square
  sorry

end probability_of_error_is_0_05_l262_262552


namespace closest_perfect_square_to_350_l262_262258

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l262_262258


namespace part_1_solution_set_part_2_a_range_l262_262574

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part_1_solution_set (a : ℝ) (h : a = 4) :
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
by
  sorry

theorem part_2_a_range :
  {a : ℝ | ∀ x : ℝ, f x a ≥ 4} = {a : ℝ | a ≤ -3 ∨ a ≥ 5} :=
by
  sorry

end part_1_solution_set_part_2_a_range_l262_262574


namespace hall_width_to_length_ratio_l262_262743

def width (w l : ℝ) : Prop := w * l = 578
def length_width_difference (w l : ℝ) : Prop := l - w = 17

theorem hall_width_to_length_ratio (w l : ℝ) (hw : width w l) (hl : length_width_difference w l) : (w / l = 1 / 2) :=
by
  sorry

end hall_width_to_length_ratio_l262_262743


namespace monica_total_students_l262_262189

theorem monica_total_students 
  (c1 : ∀ (i: ℕ), i = 1 → i.students = 20)
  (c2 : ∀ (i: ℕ), (i = 2 ∨ i = 3) → i.students = 25)
  (c3 : ∀ (i: ℕ), i = 4 → i.students = c1 1 / 2)
  (c4 : ∀ (i: ℕ), (i = 5 ∨ i = 6) → i.students = 28)
  : (Σ i, i.students) = 136 := 
by
  sorry

end monica_total_students_l262_262189


namespace sqrt_mixed_number_simplify_l262_262971

open Real

theorem sqrt_mixed_number_simplify :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 :=
by 
  sorry

end sqrt_mixed_number_simplify_l262_262971


namespace dryer_cost_l262_262442

theorem dryer_cost (W D : ℕ) (h1 : W + D = 600) (h2 : W = 3 * D) : D = 150 :=
by
  sorry

end dryer_cost_l262_262442


namespace greatest_two_digit_product_is_12_l262_262775

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l262_262775


namespace price_of_small_bags_l262_262520

theorem price_of_small_bags (price_medium_bag : ℤ) (price_large_bag : ℤ) 
  (money_mark_has : ℤ) (balloons_in_small_bag : ℤ) 
  (balloons_in_medium_bag : ℤ) (balloons_in_large_bag : ℤ) 
  (total_balloons : ℤ) : 
  price_medium_bag = 6 → 
  price_large_bag = 12 → 
  money_mark_has = 24 → 
  balloons_in_small_bag = 50 → 
  balloons_in_medium_bag = 75 → 
  balloons_in_large_bag = 200 → 
  total_balloons = 400 → 
  (money_mark_has / (total_balloons / balloons_in_small_bag)) = 3 :=
by 
  sorry

end price_of_small_bags_l262_262520


namespace max_area_rectangle_l262_262434

theorem max_area_rectangle (l w : ℕ) (h : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
sorry

end max_area_rectangle_l262_262434


namespace unicorn_rope_problem_l262_262441

theorem unicorn_rope_problem
  (d e f : ℕ)
  (h_prime_f : Prime f)
  (h_d : d = 75)
  (h_e : e = 450)
  (h_f : f = 3)
  : d + e + f = 528 := by
  sorry

end unicorn_rope_problem_l262_262441


namespace range_of_k_l262_262326

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x

theorem range_of_k :
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → |f x₁ - f x₂| ≤ k) → k ≥ Real.exp 1 - 1 :=
by
  sorry

end range_of_k_l262_262326


namespace gain_percent_is_33_33_l262_262041
noncomputable def gain_percent_calculation (C S : ℝ) := ((S - C) / C) * 100

theorem gain_percent_is_33_33
  (C S : ℝ)
  (h : 75 * C = 56.25 * S) :
  gain_percent_calculation C S = 33.33 := by
  sorry

end gain_percent_is_33_33_l262_262041


namespace ratio_of_sides_of_rectangles_l262_262171

theorem ratio_of_sides_of_rectangles (s x y : ℝ) 
  (hsx : x + s = 2 * s) 
  (hsy : s + 2 * y = 2 * s)
  (houter_inner_area : (2 * s) ^ 2 = 4 * s ^ 2) : 
  x / y = 2 :=
by
  -- Assuming the conditions hold, we are interested in proving that the ratio x / y = 2
  -- The proof will be provided here
  sorry

end ratio_of_sides_of_rectangles_l262_262171


namespace Olivia_pays_4_dollars_l262_262200

-- Definitions based on the conditions
def quarters_chips : ℕ := 4
def quarters_soda : ℕ := 12
def conversion_rate : ℕ := 4

-- Prove that the total dollars Olivia pays is 4
theorem Olivia_pays_4_dollars (h1 : quarters_chips = 4) (h2 : quarters_soda = 12) (h3 : conversion_rate = 4) : 
  (quarters_chips + quarters_soda) / conversion_rate = 4 :=
by
  -- skipping the proof
  sorry

end Olivia_pays_4_dollars_l262_262200


namespace new_container_volume_l262_262430

-- Define the original volume of the container 
def original_volume : ℝ := 4

-- Define the scale factor of each dimension (quadrupled)
def scale_factor : ℝ := 4

-- Define the new volume, which is original volume * (scale factor ^ 3)
def new_volume (orig_vol : ℝ) (scale : ℝ) : ℝ := orig_vol * (scale ^ 3)

-- The theorem we want to prove
theorem new_container_volume : new_volume original_volume scale_factor = 256 :=
by
  sorry

end new_container_volume_l262_262430


namespace john_bought_three_sodas_l262_262361

-- Define the conditions

def cost_per_soda := 2
def total_money_paid := 20
def change_received := 14

-- Definition indicating the number of sodas bought
def num_sodas_bought := (total_money_paid - change_received) / cost_per_soda

-- Question: Prove that John bought 3 sodas given these conditions
theorem john_bought_three_sodas : num_sodas_bought = 3 := by
  -- Proof: This is an example of how you may structure the proof
  sorry

end john_bought_three_sodas_l262_262361


namespace fraction_checked_by_worker_y_l262_262419

theorem fraction_checked_by_worker_y
  (f_X f_Y : ℝ)
  (h1 : f_X + f_Y = 1)
  (h2 : 0.005 * f_X + 0.008 * f_Y = 0.0074) :
  f_Y = 0.8 :=
by
  sorry

end fraction_checked_by_worker_y_l262_262419


namespace number_of_acceptable_ages_l262_262567

theorem number_of_acceptable_ages (avg_age : ℤ) (std_dev : ℤ) (a b : ℤ) (h_avg : avg_age = 10) (h_std : std_dev = 8)
    (h1 : a = avg_age - std_dev) (h2 : b = avg_age + std_dev) :
    b - a + 1 = 17 :=
by {
    sorry
}

end number_of_acceptable_ages_l262_262567


namespace closest_square_to_350_l262_262268

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l262_262268


namespace T_is_x_plus_3_to_the_4_l262_262509

variable (x : ℝ)

def T : ℝ := (x + 2)^4 + 4 * (x + 2)^3 + 6 * (x + 2)^2 + 4 * (x + 2) + 1

theorem T_is_x_plus_3_to_the_4 : T x = (x + 3)^4 := by
  -- Proof would go here
  sorry

end T_is_x_plus_3_to_the_4_l262_262509


namespace bob_max_candies_l262_262554

theorem bob_max_candies (b : ℕ) (h : b + 2 * b = 30) : b = 10 := 
sorry

end bob_max_candies_l262_262554


namespace common_divisor_l262_262739

theorem common_divisor (d : ℕ) (h1 : 30 % d = 3) (h2 : 40 % d = 4) : d = 9 :=
by 
  sorry

end common_divisor_l262_262739


namespace count_perfect_square_factors_l262_262036

open Nat

def has_larger_square_factor (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ m * m ∣ n

theorem count_perfect_square_factors :
  (Finset.filter has_larger_square_factor (Finset.range 101)).card = 42 := by
sorry

end count_perfect_square_factors_l262_262036


namespace initial_amount_of_water_l262_262562

theorem initial_amount_of_water 
  (W : ℚ) 
  (h1 : W - (7/15) * W - (5/8) * (W - (7/15) * W) - (2/3) * (W - (7/15) * W - (5/8) * (W - (7/15) * W)) = 2.6) 
  : W = 39 := 
sorry

end initial_amount_of_water_l262_262562


namespace find_f_2_l262_262639

variable {f : ℕ → ℤ}

-- Assume the condition given in the problem
axiom h : ∀ x : ℕ, f (x + 1) = x^2 - 1

-- Prove that f(2) = 0
theorem find_f_2 : f 2 = 0 := 
sorry

end find_f_2_l262_262639


namespace combined_tennis_percentage_l262_262590

variable (totalStudentsNorth totalStudentsSouth : ℕ)
variable (percentTennisNorth percentTennisSouth : ℕ)

def studentsPreferringTennisNorth : ℕ := totalStudentsNorth * percentTennisNorth / 100
def studentsPreferringTennisSouth : ℕ := totalStudentsSouth * percentTennisSouth / 100

def totalStudentsBothSchools : ℕ := totalStudentsNorth + totalStudentsSouth
def studentsPreferringTennisBothSchools : ℕ := studentsPreferringTennisNorth totalStudentsNorth percentTennisNorth
                                            + studentsPreferringTennisSouth totalStudentsSouth percentTennisSouth

def combinedPercentTennis : ℕ := studentsPreferringTennisBothSchools totalStudentsNorth totalStudentsSouth percentTennisNorth percentTennisSouth
                                 * 100 / totalStudentsBothSchools totalStudentsNorth totalStudentsSouth

theorem combined_tennis_percentage :
  (totalStudentsNorth = 1800) →
  (totalStudentsSouth = 2700) →
  (percentTennisNorth = 25) →
  (percentTennisSouth = 35) →
  combinedPercentTennis totalStudentsNorth totalStudentsSouth percentTennisNorth percentTennisSouth = 31 :=
by
  intros
  sorry

end combined_tennis_percentage_l262_262590


namespace product_mod_five_l262_262247

theorem product_mod_five (a b c : ℕ) (h₁ : a = 1236) (h₂ : b = 7483) (h₃ : c = 53) :
  (a * b * c) % 5 = 4 :=
by
  sorry

end product_mod_five_l262_262247


namespace number_of_female_students_selected_is_20_l262_262427

noncomputable def number_of_female_students_to_be_selected
(total_students : ℕ) (female_students : ℕ) (students_to_be_selected : ℕ) : ℕ :=
students_to_be_selected * female_students / total_students

theorem number_of_female_students_selected_is_20 :
  number_of_female_students_to_be_selected 2000 800 50 = 20 := 
by
  sorry

end number_of_female_students_selected_is_20_l262_262427


namespace james_total_beverages_l262_262886

-- Define the initial quantities
def initial_sodas := 4 * 10 + 12
def initial_juice_boxes := 3 * 8 + 5
def initial_water_bottles := 2 * 15
def initial_energy_drinks := 7

-- Define the consumption rates
def mon_to_wed_sodas := 3 * 3
def mon_to_wed_juice_boxes := 2 * 3
def mon_to_wed_water_bottles := 1 * 3

def thu_to_sun_sodas := 2 * 4
def thu_to_sun_juice_boxes := 4 * 4
def thu_to_sun_water_bottles := 1 * 4
def thu_to_sun_energy_drinks := 1 * 4

-- Define total beverages consumed
def total_consumed_sodas := mon_to_wed_sodas + thu_to_sun_sodas
def total_consumed_juice_boxes := mon_to_wed_juice_boxes + thu_to_sun_juice_boxes
def total_consumed_water_bottles := mon_to_wed_water_bottles + thu_to_sun_water_bottles
def total_consumed_energy_drinks := thu_to_sun_energy_drinks

-- Define total beverages consumed by the end of the week
def total_beverages_consumed := total_consumed_sodas + total_consumed_juice_boxes + total_consumed_water_bottles + total_consumed_energy_drinks

-- The theorem statement to prove
theorem james_total_beverages : total_beverages_consumed = 50 :=
  by sorry

end james_total_beverages_l262_262886


namespace sqrt_abc_sum_l262_262062

variable (a b c : ℝ)

theorem sqrt_abc_sum (h1 : b + c = 17) (h2 : c + a = 20) (h3 : a + b = 23) :
  Real.sqrt (a * b * c * (a + b + c)) = 10 * Real.sqrt 273 := by
  sorry

end sqrt_abc_sum_l262_262062


namespace median_of_triangle_l262_262074

variable (a b c : ℝ)

noncomputable def AM : ℝ :=
  (Real.sqrt (2 * b * b + 2 * c * c - a * a)) / 2

theorem median_of_triangle :
  abs (((b + c) / 2) - (a / 2)) < AM a b c ∧ 
  AM a b c < (b + c) / 2 := 
by
  sorry

end median_of_triangle_l262_262074


namespace find_number_l262_262040

theorem find_number (x : ℤ) : 45 - (28 - (x - (15 - 16))) = 55 ↔ x = 37 :=
by
  sorry

end find_number_l262_262040


namespace clock_strike_time_l262_262095

theorem clock_strike_time (t : ℕ) (n m : ℕ) (I : ℕ) : 
  t = 12 ∧ n = 3 ∧ m = 6 ∧ 2 * I = t → (m - 1) * I = 30 := by 
  sorry

end clock_strike_time_l262_262095


namespace closest_perfect_square_to_350_l262_262263

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l262_262263


namespace yearly_return_of_1500_investment_is_27_percent_l262_262810

-- Definitions based on conditions
def combined_yearly_return (x : ℝ) : Prop :=
  let investment1 := 500
  let investment2 := 1500
  let total_investment := investment1 + investment2
  let combined_return := 0.22 * total_investment
  let return_from_500 := 0.07 * investment1
  let return_from_1500 := combined_return - return_from_500
  x / 100 * investment2 = return_from_1500

-- Theorem statement to be proven
theorem yearly_return_of_1500_investment_is_27_percent : combined_yearly_return 27 :=
by sorry

end yearly_return_of_1500_investment_is_27_percent_l262_262810


namespace MaryAddedCandy_l262_262894

-- Definitions based on the conditions
def MaryInitialCandyCount (MeganCandyCount : ℕ) : ℕ :=
  3 * MeganCandyCount

-- Given conditions
def MeganCandyCount : ℕ := 5
def MaryTotalCandyCount : ℕ := 25

-- Proof statement
theorem MaryAddedCandy : 
  let MaryInitialCandy := MaryInitialCandyCount MeganCandyCount
  MaryTotalCandyCount - MaryInitialCandy = 10 :=
by 
  sorry

end MaryAddedCandy_l262_262894


namespace sequence_general_formula_l262_262708

theorem sequence_general_formula (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : a 2 = 2)
    (h3 : ∀ n, a (n + 2) = a n + 2) :
    ∀ n, a n = n := by
  sorry

end sequence_general_formula_l262_262708


namespace average_age_when_youngest_born_l262_262096

theorem average_age_when_youngest_born (n : ℕ) (avg_age current_youngest_age total_age_when_youngest_born : ℝ) 
  (h1 : n = 7) (h2 : avg_age = 30) (h3 : current_youngest_age = 8) (h4 : total_age_when_youngest_born = (n * avg_age - n * current_youngest_age)) : 
  total_age_when_youngest_born / n = 22 :=
by
  sorry

end average_age_when_youngest_born_l262_262096


namespace smallest_positive_integer_with_conditions_l262_262831

theorem smallest_positive_integer_with_conditions (n : ℕ) (h₀ : alice_number = 36)
  (h₁ : ∀ p, prime p → p ∣ alice_number → p ∣ n)
  (h₂ : 5 ∣ n) :
  n = 30 :=
by
  let alice_number := 36
  sorry

end smallest_positive_integer_with_conditions_l262_262831


namespace probability_of_negative_m_l262_262498

theorem probability_of_negative_m (m : ℤ) (h₁ : -2 ≤ m) (h₂ : m < (9 : ℤ) / 4) :
  ∃ (neg_count total_count : ℤ), 
    (neg_count = 2) ∧ (total_count = 5) ∧ (m ∈ {i : ℤ | -2 ≤ i ∧ i < 2 ∧ i < 9 / 4}) → 
    (neg_count / total_count = 2 / 5) :=
sorry

end probability_of_negative_m_l262_262498


namespace intersection_M_N_l262_262893

def M (x : ℝ) : Prop := 2 - x > 0
def N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

theorem intersection_M_N:
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l262_262893


namespace selection_structure_count_is_three_l262_262485

def requiresSelectionStructure (problem : ℕ) : Bool :=
  match problem with
  | 1 => true
  | 2 => false
  | 3 => true
  | 4 => true
  | _ => false

def countSelectionStructure : ℕ :=
  (if requiresSelectionStructure 1 then 1 else 0) +
  (if requiresSelectionStructure 2 then 1 else 0) +
  (if requiresSelectionStructure 3 then 1 else 0) +
  (if requiresSelectionStructure 4 then 1 else 0)

theorem selection_structure_count_is_three : countSelectionStructure = 3 :=
  by
    sorry

end selection_structure_count_is_three_l262_262485


namespace dodecahedron_interior_diagonals_l262_262641

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l262_262641


namespace positive_operation_l262_262276

def operation_a := 1 + (-2)
def operation_b := 1 - (-2)
def operation_c := 1 * (-2)
def operation_d := 1 / (-2)

theorem positive_operation : operation_b > 0 ∧ 
  (operation_a <= 0) ∧ (operation_c <= 0) ∧ (operation_d <= 0) := by
  sorry

end positive_operation_l262_262276


namespace not_all_ten_segments_form_triangle_l262_262287

theorem not_all_ten_segments_form_triangle :
  ∃ (segments : Fin 10 → ℕ), ∀ i j k : Fin 10, i < j → j < k → segments i + segments j ≤ segments k := 
sorry

end not_all_ten_segments_form_triangle_l262_262287


namespace train_length_eq_l262_262438

theorem train_length_eq 
  (speed_kmh : ℝ) (time_sec : ℝ) 
  (h_speed_kmh : speed_kmh = 126)
  (h_time_sec : time_sec = 6.856594329596489) : 
  ((speed_kmh * 1000 / 3600) * time_sec) = 239.9808045358781 :=
by
  -- We skip the proof with sorry, as per instructions
  sorry

end train_length_eq_l262_262438


namespace ratio_first_part_l262_262933

theorem ratio_first_part (x : ℝ) (h1 : 180 / 100 * 5 = x) : x = 9 :=
by sorry

end ratio_first_part_l262_262933


namespace find_coefficients_l262_262845

theorem find_coefficients (A B : ℚ) :
  (∀ x : ℚ, 2 * x + 7 = A * (x + 7) + B * (x - 9)) →
  A = 25 / 16 ∧ B = 7 / 16 :=
by
  intro h
  sorry

end find_coefficients_l262_262845


namespace triangle_side_length_range_l262_262631

theorem triangle_side_length_range (x : ℝ) : 
  (1 < x) ∧ (x < 9) → ¬ (x = 10) :=
by
  sorry

end triangle_side_length_range_l262_262631


namespace fabian_initial_hours_l262_262621

-- Define the conditions
def speed : ℕ := 5
def total_distance : ℕ := 30
def additional_time : ℕ := 3

-- The distance Fabian covers in the additional time
def additional_distance := speed * additional_time

-- The initial distance walked by Fabian
def initial_distance := total_distance - additional_distance

-- The initial hours Fabian walked
def initial_hours := initial_distance / speed

theorem fabian_initial_hours : initial_hours = 3 := by
  -- Proof goes here
  sorry

end fabian_initial_hours_l262_262621


namespace age_difference_l262_262424

theorem age_difference (J P : ℕ) 
  (h1 : P = 16 - 10) 
  (h2 : P = (1 / 3) * J) : 
  (J + 10) - 16 = 12 := 
by 
  sorry

end age_difference_l262_262424


namespace shortest_distance_Dasha_Vasya_l262_262884

def distance_Asya_Galia : ℕ := 12
def distance_Galia_Borya : ℕ := 10
def distance_Asya_Borya : ℕ := 8
def distance_Dasha_Galia : ℕ := 15
def distance_Vasya_Galia : ℕ := 17

def distance_Dasha_Vasya : ℕ :=
  distance_Dasha_Galia + distance_Vasya_Galia - distance_Asya_Galia - distance_Galia_Borya + distance_Asya_Borya

theorem shortest_distance_Dasha_Vasya : distance_Dasha_Vasya = 18 :=
by
  -- We assume the distances as given in the conditions. The calculation part is skipped here.
  -- The actual proof steps would go here.
  sorry

end shortest_distance_Dasha_Vasya_l262_262884


namespace find_prime_squares_l262_262976

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

theorem find_prime_squares :
  ∀ (p q : ℕ), is_prime p → is_prime q → is_square (p^(q+1) + q^(p+1)) → (p = 2 ∧ q = 2) :=
by 
  intros p q h_prime_p h_prime_q h_square
  sorry

end find_prime_squares_l262_262976


namespace value_of_fraction_l262_262606

theorem value_of_fraction (y : ℝ) (h : 4 - 9 / y + 9 / (y^2) = 0) : 3 / y = 2 :=
sorry

end value_of_fraction_l262_262606


namespace dodecahedron_interior_diagonals_l262_262684

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l262_262684


namespace min_value_expr_l262_262626

theorem min_value_expr (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a) :
  ∃ x : ℝ, x = a^2 + 1 / (b * (a - b)) ∧ x ≥ 4 :=
by sorry

end min_value_expr_l262_262626


namespace maximum_I_minus_J_l262_262142

open Set Filter

noncomputable def I (f : ℝ → ℝ) : ℝ :=
  ∫ x in 0..1, x^2 * f x

noncomputable def J (f : ℝ → ℝ) : ℝ :=
  ∫ x in 0..1, x * (f x)^2

theorem maximum_I_minus_J (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Icc 0 1)) : 
  I(f) - J(f) ≤ 1/12 :=
sorry

end maximum_I_minus_J_l262_262142


namespace principal_amount_invested_l262_262124

noncomputable def calculate_principal : ℕ := sorry

theorem principal_amount_invested (P : ℝ) (y : ℝ) 
    (h1 : 300 = P * y * 2 / 100) -- Condition for simple interest
    (h2 : 307.50 = P * ((1 + y/100)^2 - 1)) -- Condition for compound interest
    : P = 73.53 := 
sorry

end principal_amount_invested_l262_262124


namespace listK_consecutive_integers_count_l262_262065

-- Given conditions
def listK := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] -- A list K consisting of consecutive integers
def leastInt : Int := -5 -- The least integer in list K
def rangePosInt : Nat := 5 -- The range of the positive integers in list K

-- The theorem to prove
theorem listK_consecutive_integers_count : listK.length = 11 := by
  -- skipping the proof
  sorry

end listK_consecutive_integers_count_l262_262065


namespace evaluate_expression_l262_262103

theorem evaluate_expression :
  2^4 - 4 * 2^3 + 6 * 2^2 - 4 * 2 + 1 = 1 :=
by
  sorry

end evaluate_expression_l262_262103


namespace ratio_of_paper_plates_l262_262828

theorem ratio_of_paper_plates (total_pallets : ℕ) (paper_towels : ℕ) (tissues : ℕ) (paper_cups : ℕ) :
  total_pallets = 20 →
  paper_towels = 20 / 2 →
  tissues = 20 / 4 →
  paper_cups = 1 →
  (total_pallets - (paper_towels + tissues + paper_cups)) / total_pallets = 1 / 5 :=
by
  intros h_total h_towels h_tissues h_cups
  sorry

end ratio_of_paper_plates_l262_262828


namespace libby_quarters_left_l262_262517

theorem libby_quarters_left (initial_quarters : ℕ) (dress_cost_dollars : ℕ) (quarters_per_dollar : ℕ) 
  (h1 : initial_quarters = 160) (h2 : dress_cost_dollars = 35) (h3 : quarters_per_dollar = 4) : 
  initial_quarters - (dress_cost_dollars * quarters_per_dollar) = 20 := by
  sorry

end libby_quarters_left_l262_262517


namespace abs_condition_implies_l262_262109

theorem abs_condition_implies (x : ℝ) 
  (h : |x - 1| < 2) : x < 3 := by
  sorry

end abs_condition_implies_l262_262109


namespace marble_probability_l262_262143

-- Problem Statement
theorem marble_probability :
  let total_marbles := 9
  let chosen_marbles := 4
  let red_marbles := 3
  let blue_marbles := 3
  let green_marbles := 3 in
  let total_ways := (total_marbles.choose chosen_marbles) in
  let red_ways := (red_marbles.choose 1) in
  let blue_ways := (blue_marbles.choose 1) in
  let green_ways := (green_marbles.choose 2) in
  (red_ways * blue_ways * green_ways) / total_ways = 3 / 14 := by
  sorry

end marble_probability_l262_262143


namespace inequality_solution_l262_262137

theorem inequality_solution (x : ℝ) : (x / (x + 1) + (x + 3) / (2 * x) ≥ 2) ↔ (0 < x ∧ x ≤ 1) ∨ x = -3 :=
by
sorry

end inequality_solution_l262_262137


namespace pyramid_values_l262_262175

theorem pyramid_values :
  ∃ (A B C D : ℕ),
    (A = 3000) ∧
    (D = 623) ∧
    (B = 700) ∧
    (C = 253) ∧
    (A = 1100 + 1800) ∧
    (D + 451 ≥ 1065) ∧ (D + 451 ≤ 1075) ∧ -- rounding to nearest ten
    (B + 440 ≥ 1050) ∧ (B + 440 ≤ 1150) ∧
    (B + 1070 ≥ 1700) ∧ (B + 1070 ≤ 1900) ∧
    (C + 188 ≥ 430) ∧ (C + 188 ≤ 450) ∧    -- rounding to nearest ten
    (C + 451 ≥ 695) ∧ (C + 451 ≤ 705) :=  -- using B = 700 for rounding range
sorry

end pyramid_values_l262_262175


namespace nancy_target_amount_l262_262723

theorem nancy_target_amount {rate : ℝ} {hours : ℝ} (h1 : rate = 28 / 4) (h2 : hours = 10) : 28 / 4 * 10 = 70 :=
by
  sorry

end nancy_target_amount_l262_262723


namespace point_inside_circle_l262_262026

theorem point_inside_circle :
  let center := (2, 3)
  let radius := 2
  let P := (1, 2)
  let distance_squared := (P.1 - center.1)^2 + (P.2 - center.2)^2
  distance_squared < radius^2 :=
by
  -- Definitions
  let center := (2, 3)
  let radius := 2
  let P := (1, 2)
  let distance_squared := (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2

  -- Goal
  show distance_squared < radius ^ 2
  
  -- Skip Proof
  sorry

end point_inside_circle_l262_262026


namespace gcd_12345_6789_l262_262752

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l262_262752


namespace passed_percentage_l262_262172

theorem passed_percentage (A B C AB BC AC ABC: ℝ) 
  (hA : A = 0.25) 
  (hB : B = 0.50) 
  (hC : C = 0.30) 
  (hAB : AB = 0.25) 
  (hBC : BC = 0.15) 
  (hAC : AC = 0.10) 
  (hABC : ABC = 0.05) 
  : 100 - (A + B + C - AB - BC - AC + ABC) = 40 := 
by 
  rw [hA, hB, hC, hAB, hBC, hAC, hABC]
  norm_num
  sorry

end passed_percentage_l262_262172


namespace domain_range_a_l262_262860

theorem domain_range_a (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ 1 < a :=
by
  sorry

end domain_range_a_l262_262860


namespace find_number_l262_262230

/-- 
  Given that 23% of a number x is equal to 150, prove that x equals 15000 / 23.
-/
theorem find_number (x : ℝ) (h : (23 / 100) * x = 150) : x = 15000 / 23 :=
by
  sorry

end find_number_l262_262230


namespace nancy_clay_pots_l262_262067

theorem nancy_clay_pots : 
  ∃ M : ℕ, (M + 2 * M + 14 = 50) ∧ M = 12 :=
sorry

end nancy_clay_pots_l262_262067


namespace quadratic_common_root_distinct_real_numbers_l262_262329

theorem quadratic_common_root_distinct_real_numbers:
  ∀ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a →
  (∃ x, x^2 + a * x + b = 0 ∧ x^2 + c * x + a = 0) ∧
  (∃ y, y^2 + a * y + b = 0 ∧ y^2 + b * y + c = 0) ∧
  (∃ z, z^2 + b * z + c = 0 ∧ z^2 + c * z + a = 0) →
  a^2 + b^2 + c^2 = 6 :=
by
  intros a b c h_distinct h_common_root
  sorry

end quadratic_common_root_distinct_real_numbers_l262_262329


namespace closest_perfect_square_to_350_l262_262255

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l262_262255


namespace crayons_total_l262_262876

theorem crayons_total (blue_crayons : ℕ) (red_crayons : ℕ) 
  (H1 : red_crayons = 4 * blue_crayons) (H2 : blue_crayons = 3) : 
  blue_crayons + red_crayons = 15 := 
by
  sorry

end crayons_total_l262_262876


namespace factorize_expression_l262_262622

theorem factorize_expression (x : ℝ) : x^2 - 2023 * x = x * (x - 2023) := 
by 
  sorry

end factorize_expression_l262_262622


namespace probability_divisible_by_5_l262_262543

def is_three_digit_integer (M : ℕ) : Prop :=
  100 ≤ M ∧ M < 1000

def ones_digit_is_4 (M : ℕ) : Prop :=
  (M % 10) = 4

theorem probability_divisible_by_5 (M : ℕ) (h1 : is_three_digit_integer M) (h2 : ones_digit_is_4 M) :
  (∃ p : ℚ, p = 0) :=
by
  sorry

end probability_divisible_by_5_l262_262543


namespace dan_spent_at_music_store_l262_262601

def cost_of_clarinet : ℝ := 130.30
def cost_of_song_book : ℝ := 11.24
def money_left_in_pocket : ℝ := 12.32
def total_spent : ℝ := 129.22

theorem dan_spent_at_music_store : 
  cost_of_clarinet + cost_of_song_book - money_left_in_pocket = total_spent :=
by
  -- Proof omitted.
  sorry

end dan_spent_at_music_store_l262_262601


namespace find_b_l262_262903

variable (f : ℝ → ℝ) (finv : ℝ → ℝ)

-- Defining the function f
def f_def (b : ℝ) (x : ℝ) := 1 / (2 * x + b)

-- Defining the inverse function
def finv_def (x : ℝ) := (2 - 3 * x) / (3 * x)

theorem find_b (b : ℝ) :
  (∀ x : ℝ, f_def b (finv_def x) = x ∧ finv_def (f_def b x) = x) ↔ b = -2 := by
  sorry

end find_b_l262_262903


namespace exists_K_p_l262_262711

noncomputable def constant_K_p (p : ℝ) (hp : p > 1) : ℝ :=
  (p * p) / (p - 1)

theorem exists_K_p (p : ℝ) (hp : p > 1) :
  ∃ K_p > 0, ∀ x y : ℝ, |x|^p + |y|^p = 2 → (x - y)^2 ≤ K_p * (4 - (x + y)^2) :=
by
  use constant_K_p p hp
  sorry

end exists_K_p_l262_262711


namespace probability_relatively_prime_pairs_l262_262555

open Real

-- Define the set of natural numbers in question
def S : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define a function to calculate the greatest common factor (gcd)
noncomputable def gcd (a b : ℕ) : ℕ := nat.gcd a b

-- Define the count of gcd being 1 pairs within the set
noncomputable def relatively_prime_pairs_count : ℕ :=
  (S.product S).filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ gcd p.1 p.2 = 1).card

-- Define the total two-element subsets (pairs)
noncomputable def total_pairs_count : ℕ := (S.product S).filter (λ (p : ℕ × ℕ), p.1 < p.2).card

-- Define the probability as a common fraction
theorem probability_relatively_prime_pairs : 
  (relatively_prime_pairs_count : ℝ) / (total_pairs_count : ℝ) = 3 / 4 :=
by
  sorry

end probability_relatively_prime_pairs_l262_262555


namespace find_y_l262_262481

theorem find_y (x y : ℕ) (h1 : x = 2407) (h2 : x^y + y^x = 2408) : y = 1 :=
sorry

end find_y_l262_262481


namespace incorrect_statement_D_l262_262808

theorem incorrect_statement_D
  (passes_through_center : ∀ (x_vals y_vals : List ℝ), ∃ (regression_line : ℝ → ℝ), 
    regression_line (x_vals.sum / x_vals.length) = (y_vals.sum / y_vals.length))
  (higher_r2_better_fit : ∀ (r2 : ℝ), r2 > 0 → ∃ (residual_sum_squares : ℝ), residual_sum_squares < (1 - r2))
  (slope_interpretation : ∀ (x : ℝ), (0.2 * x + 0.8) - (0.2 * (x - 1) + 0.8) = 0.2)
  (chi_squared_k2 : ∀ (X Y : Type) [Fintype X] [Fintype Y] (k : ℝ), (k > 0) → 
    ∃ (confidence : ℝ), confidence > 0) :
  ¬(∀ (X Y : Type) [Fintype X] [Fintype Y] (k : ℝ), k > 0 → 
    ∃ (confidence : ℝ), confidence < 0) :=
by
  sorry

end incorrect_statement_D_l262_262808


namespace right_handed_total_l262_262107

theorem right_handed_total
  (total_players : ℕ)
  (throwers : ℕ)
  (left_handed_non_throwers : ℕ)
  (right_handed_throwers : ℕ)
  (non_throwers : ℕ)
  (right_handed_non_throwers : ℕ) :
  total_players = 70 →
  throwers = 28 →
  non_throwers = total_players - throwers →
  left_handed_non_throwers = non_throwers / 3 →
  right_handed_non_throwers = non_throwers - left_handed_non_throwers →
  right_handed_throwers = throwers →
  right_handed_throwers + right_handed_non_throwers = 56 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end right_handed_total_l262_262107


namespace calculate_seven_a_sq_minus_four_a_sq_l262_262307

variable (a : ℝ)

theorem calculate_seven_a_sq_minus_four_a_sq : 7 * a^2 - 4 * a^2 = 3 * a^2 := 
by
  sorry

end calculate_seven_a_sq_minus_four_a_sq_l262_262307


namespace proof_problem_l262_262634

variable (a b c : ℝ)

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : ∀ x, abs (x + a) - abs (x - b) + c ≤ 10) :
  a + b + c = 10 ∧ 
  (∀ (h5 : a + b + c = 10), 
    (∃ a' b' c', a' = 11/3 ∧ b' = 8/3 ∧ c' = 11/3 ∧ 
                (∀ a'' b'' c'', a'' = a ∧ b'' = b ∧ c'' = c → 
                (1/4 * (a - 1)^2 + (b - 2)^2 + (c - 3)^2) ≥ 8/3 ∧ 
                (1/4 * (a' - 1)^2 + (b' - 2)^2 + (c' - 3)^2) = 8 / 3 ))) := by
  sorry

end proof_problem_l262_262634


namespace shell_count_l262_262464

theorem shell_count (initial_shells : ℕ) (ed_limpet : ℕ) (ed_oyster : ℕ) (ed_conch : ℕ) (jacob_extra : ℕ)
  (h1 : initial_shells = 2)
  (h2 : ed_limpet = 7) 
  (h3 : ed_oyster = 2) 
  (h4 : ed_conch = 4) 
  (h5 : jacob_extra = 2) : 
  (initial_shells + ed_limpet + ed_oyster + ed_conch + (ed_limpet + ed_oyster + ed_conch + jacob_extra)) = 30 := 
by 
  sorry

end shell_count_l262_262464


namespace simplify_expression_l262_262732

theorem simplify_expression : 
  (1 / (1 / (1 / 3) ^ 1 + 1 / (1 / 3) ^ 2 + 1 / (1 / 3) ^ 3 + 1 / (1 / 3) ^ 4)) = 1 / 120 :=
by
  sorry

end simplify_expression_l262_262732


namespace trace_bag_weight_l262_262749

-- Define the weights of Gordon's bags
def gordon_bag1_weight : ℕ := 3
def gordon_bag2_weight : ℕ := 7

-- Define the number of Trace's bags
def trace_num_bags : ℕ := 5

-- Define what we are trying to prove: the weight of one of Trace's shopping bags
theorem trace_bag_weight :
  (gordon_bag1_weight + gordon_bag2_weight) = (trace_num_bags * 2) :=
by
  sorry

end trace_bag_weight_l262_262749


namespace odd_prime_2wy_factors_l262_262497

theorem odd_prime_2wy_factors (w y : ℕ) (h1 : Nat.Prime w) (h2 : Nat.Prime y) (h3 : ¬ Even w) (h4 : ¬ Even y) (h5 : w < y) (h6 : Nat.totient (2 * w * y) = 8) :
  w = 3 :=
sorry

end odd_prime_2wy_factors_l262_262497


namespace ticket_price_l262_262336

variable (x : ℝ)

def tickets_condition1 := 3 * x
def tickets_condition2 := 5 * x
def total_spent := 3 * x + 5 * x

theorem ticket_price : total_spent x = 32 → x = 4 :=
by
  -- Proof steps will be provided here.
  sorry

end ticket_price_l262_262336


namespace simplify_complex_expression_l262_262240

theorem simplify_complex_expression :
  (1 / (-8 ^ 2) ^ 4 * (-8) ^ 9) = -8 := by
  sorry

end simplify_complex_expression_l262_262240


namespace minimum_value_function_l262_262133

theorem minimum_value_function (x : ℝ) (h : x > 1) : 
  ∃ y, y = (16 - 2 * Real.sqrt 7) / 3 ∧ ∀ x > 1, (4*x^2 + 2*x + 5) / (x^2 + x + 1) ≥ y :=
sorry

end minimum_value_function_l262_262133


namespace greatest_two_digit_with_product_12_l262_262799

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l262_262799


namespace greatest_multiple_of_30_less_than_1000_l262_262759

theorem greatest_multiple_of_30_less_than_1000 : ∃ (n : ℕ), n < 1000 ∧ n % 30 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 30 = 0 → m ≤ n := 
by 
  use 990
  sorry

end greatest_multiple_of_30_less_than_1000_l262_262759


namespace dodecahedron_interior_diagonals_l262_262663

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l262_262663


namespace aladdin_no_profit_l262_262301

theorem aladdin_no_profit (x : ℕ) :
  (x + 1023000) / 1024 <= x :=
by
  sorry

end aladdin_no_profit_l262_262301


namespace part_a_part_b_l262_262285

-- Part (a)
theorem part_a
  (initial_deposit : ℝ)
  (initial_exchange_rate : ℝ)
  (annual_return_rate : ℝ)
  (final_exchange_rate : ℝ)
  (conversion_fee_rate : ℝ)
  (broker_commission_rate : ℝ) :
  initial_deposit = 12000 →
  initial_exchange_rate = 60 →
  annual_return_rate = 0.12 →
  final_exchange_rate = 80 →
  conversion_fee_rate = 0.04 →
  broker_commission_rate = 0.25 →
  let deposit_in_dollars := 12000 / 60
  let profit_in_dollars := deposit_in_dollars * 0.12
  let total_in_dollars := deposit_in_dollars + profit_in_dollars
  let broker_commission := profit_in_dollars * 0.25
  let amount_before_conversion := total_in_dollars - broker_commission
  let amount_in_rubles := amount_before_conversion * 80
  let conversion_fee := amount_in_rubles * 0.04
  let final_amount := amount_in_rubles - conversion_fee
  final_amount = 16742.4 := sorry

-- Part (b)
theorem part_b
  (initial_deposit : ℝ)
  (final_amount : ℝ) :
  initial_deposit = 12000 →
  final_amount = 16742.4 →
  let effective_return := (16742.4 / 12000) - 1
  effective_return * 100 = 39.52 := sorry

end part_a_part_b_l262_262285


namespace area_of_roof_l262_262094

-- Definitions and conditions
def length (w : ℝ) := 4 * w
def difference_eq (l w : ℝ) := l - w = 39
def area (l w : ℝ) := l * w

-- Theorem statement
theorem area_of_roof (w l : ℝ) (h_length : l = length w) (h_diff : difference_eq l w) : area l w = 676 :=
by
  sorry

end area_of_roof_l262_262094


namespace ratio_of_siblings_l262_262887

/-- Let's define the sibling relationships and prove the ratio of Janet's to Masud's siblings is 3 to 1. -/
theorem ratio_of_siblings (masud_siblings : ℕ) (carlos_siblings janet_siblings : ℕ)
  (h1 : masud_siblings = 60)
  (h2 : carlos_siblings = 3 * masud_siblings / 4)
  (h3 : janet_siblings = carlos_siblings + 135) 
  (h4 : janet_siblings < some_mul * masud_siblings) : 
  janet_siblings / masud_siblings = 3 :=
by
  sorry

end ratio_of_siblings_l262_262887


namespace dodecahedron_interior_diagonals_l262_262676

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l262_262676


namespace tangent_line_at_1_l262_262908

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x
def tangent_line_eq : ℝ × ℝ → ℝ := fun ⟨x, y⟩ => x - y - 1

theorem tangent_line_at_1 : tangent_line_eq (1, f 1) = 0 := by
  -- Proof would go here
  sorry

end tangent_line_at_1_l262_262908


namespace largest_common_divisor_414_345_l262_262245

theorem largest_common_divisor_414_345 : ∃ d, d ∣ 414 ∧ d ∣ 345 ∧ 
                                      (∀ e, e ∣ 414 ∧ e ∣ 345 → e ≤ d) ∧ d = 69 :=
by 
  sorry

end largest_common_divisor_414_345_l262_262245


namespace max_levels_passed_prob_pass_three_levels_l262_262912

-- Define the conditions of the game
def max_roll (n : ℕ) : ℕ := 6 * n
def pass_condition (n : ℕ) : ℕ := 2^n

-- Problem 1: Prove the maximum number of levels a person can pass
theorem max_levels_passed : ∃ n : ℕ, (∀ m : ℕ, m > n → max_roll m ≤ pass_condition m) ∧ (∀ m : ℕ, m ≤ n → max_roll m > pass_condition m) :=
by sorry

-- Define the probabilities for passing each level
def prob_pass_level_1 : ℚ := 4 / 6
def prob_pass_level_2 : ℚ := 30 / 36
def prob_pass_level_3 : ℚ := 160 / 216

-- Problem 2: Prove the probability of passing the first three levels consecutively
theorem prob_pass_three_levels : prob_pass_level_1 * prob_pass_level_2 * prob_pass_level_3 = 100 / 243 :=
by sorry

end max_levels_passed_prob_pass_three_levels_l262_262912


namespace gcd_a_b_eq_1023_l262_262459

def a : ℕ := 2^1010 - 1
def b : ℕ := 2^1000 - 1

theorem gcd_a_b_eq_1023 : Nat.gcd a b = 1023 := 
by
  sorry

end gcd_a_b_eq_1023_l262_262459


namespace eggs_used_to_bake_cake_l262_262371

theorem eggs_used_to_bake_cake
    (initial_eggs : ℕ)
    (omelet_eggs : ℕ)
    (aunt_eggs : ℕ)
    (meal_eggs : ℕ)
    (num_meals : ℕ)
    (remaining_eggs_after_omelet : initial_eggs - omelet_eggs = 22)
    (eggs_given_to_aunt : 2 * aunt_eggs = initial_eggs - omelet_eggs)
    (remaining_eggs_after_aunt : initial_eggs - omelet_eggs - aunt_eggs = 11)
    (total_eggs_for_meals : meal_eggs * num_meals = 9)
    (remaining_eggs_after_meals : initial_eggs - omelet_eggs - aunt_eggs - meal_eggs * num_meals = 2) :
  initial_eggs - omelet_eggs - aunt_eggs - meal_eggs * num_meals = 2 :=
sorry

end eggs_used_to_bake_cake_l262_262371


namespace path_area_and_cost_l262_262435

-- Define the initial conditions
def field_length : ℝ := 65
def field_width : ℝ := 55
def path_width : ℝ := 2.5
def cost_per_sq_m : ℝ := 2

-- Define the extended dimensions including the path
def extended_length := field_length + 2 * path_width
def extended_width := field_width + 2 * path_width

-- Define the areas
def area_with_path := extended_length * extended_width
def area_of_field := field_length * field_width
def area_of_path := area_with_path - area_of_field

-- Define the cost
def cost_of_constructing_path := area_of_path * cost_per_sq_m

theorem path_area_and_cost :
  area_of_path = 625 ∧ cost_of_constructing_path = 1250 :=
by
  sorry

end path_area_and_cost_l262_262435


namespace tank_empty_time_when_inlet_open_l262_262939

-- Define the conditions
def leak_empty_time : ℕ := 6
def tank_capacity : ℕ := 4320
def inlet_rate_per_minute : ℕ := 6

-- Calculate rates from conditions
def leak_rate_per_hour : ℕ := tank_capacity / leak_empty_time
def inlet_rate_per_hour : ℕ := inlet_rate_per_minute * 60

-- Proof Problem: Prove the time for the tank to empty when both leak and inlet are open
theorem tank_empty_time_when_inlet_open :
  tank_capacity / (leak_rate_per_hour - inlet_rate_per_hour) = 12 :=
by
  sorry

end tank_empty_time_when_inlet_open_l262_262939


namespace quadrilateral_offset_l262_262000

-- Define the problem statement
theorem quadrilateral_offset
  (d : ℝ) (x : ℝ) (y : ℝ) (A : ℝ)
  (h₀ : d = 10) 
  (h₁ : y = 3) 
  (h₂ : A = 50) :
  x = 7 :=
by
  -- Assuming the given conditions
  have h₃ : A = 1/2 * d * x + 1/2 * d * y :=
  by
    -- specific formula for area of the quadrilateral
    sorry
  
  -- Given A = 50, d = 10, y = 3, solve for x to show x = 7
  sorry

end quadrilateral_offset_l262_262000


namespace radius_area_tripled_l262_262081

theorem radius_area_tripled (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = (n * (Real.sqrt 3 - 1)) / 2 :=
by {
  sorry
}

end radius_area_tripled_l262_262081


namespace planting_trees_equation_l262_262822

theorem planting_trees_equation (x : ℝ) (h1 : x > 0) : 
  20 / x - 20 / ((1 + 0.1) * x) = 4 :=
sorry

end planting_trees_equation_l262_262822


namespace simplify_expression_l262_262237

theorem simplify_expression : (1 / (-8^2)^4) * (-8)^9 = -8 := 
by
  sorry

end simplify_expression_l262_262237


namespace coin_collection_problem_l262_262553

variable (n d q : ℚ)

theorem coin_collection_problem 
  (h1 : n + d + q = 30)
  (h2 : 5 * n + 10 * d + 20 * q = 340)
  (h3 : d = 2 * n) :
  q - n = 2 / 7 := by
  sorry

end coin_collection_problem_l262_262553


namespace part_a_part_b_part_c_l262_262569

-- The conditions for quadrilateral ABCD
variables (a b c d e f m n S : ℝ)
variables (S_nonneg : 0 ≤ S)

-- Prove Part (a)
theorem part_a (a b c d e f : ℝ) (S : ℝ) (h : S ≤ 1/4 * (e^2 + f^2)) : S <= 1/4 * (e^2 + f^2) :=
by 
  exact h

-- Prove Part (b)
theorem part_b (a b c d e f m n S: ℝ) (h : S ≤ 1/2 * (m^2 + n^2)) : S <= 1/2 * (m^2 + n^2) :=
by 
  exact h

-- Prove Part (c)
theorem part_c (a b c d e f m n S: ℝ) (h : S ≤ 1/4 * (a + c) * (b + d)) : S <= 1/4 * (a + c) * (b + d) :=
by 
  exact h

#eval "This Lean code defines the correctness statement of each part of the problem."

end part_a_part_b_part_c_l262_262569


namespace weight_of_11_25m_rod_l262_262691

noncomputable def weight_per_meter (total_weight : ℝ) (length : ℝ) : ℝ :=
  total_weight / length

def weight_of_rod (weight_per_length : ℝ) (length : ℝ) : ℝ :=
  weight_per_length * length

theorem weight_of_11_25m_rod :
  let total_weight_8m := 30.4
  let length_8m := 8.0
  let length_11_25m := 11.25
  let weight_per_length := weight_per_meter total_weight_8m length_8m
  weight_of_rod weight_per_length length_11_25m = 42.75 :=
by sorry

end weight_of_11_25m_rod_l262_262691


namespace percent_fair_hair_l262_262576

theorem percent_fair_hair 
  (total_employees : ℕ) 
  (percent_women_fair_hair : ℝ) 
  (percent_fair_hair_women : ℝ)
  (total_women_fair_hair : ℕ)
  (total_fair_hair : ℕ)
  (h1 : percent_women_fair_hair = 30 / 100)
  (h2 : percent_fair_hair_women = 40 / 100)
  (h3 : total_women_fair_hair = percent_women_fair_hair * total_employees)
  (h4 : percent_fair_hair_women * total_fair_hair = total_women_fair_hair)
  : total_fair_hair = 75 / 100 * total_employees := 
by
  sorry

end percent_fair_hair_l262_262576


namespace find_abs_product_abc_l262_262182

theorem find_abs_product_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h : a + 1 / b = b + 1 / c ∧ b + 1 / c = c + 1 / a) : |a * b * c| = 1 :=
sorry

end find_abs_product_abc_l262_262182


namespace class_size_is_44_l262_262608

theorem class_size_is_44 (n : ℕ) : 
  (n - 1) % 2 = 1 ∧ (n - 1) % 7 = 1 → n = 44 := 
by 
  sorry

end class_size_is_44_l262_262608


namespace second_number_is_255_l262_262221

theorem second_number_is_255 (x : ℝ) (n : ℝ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 90) 
  (h2 : (128 + n + 511 + 1023 + x) / 5 = 423) : 
  n = 255 :=
sorry

end second_number_is_255_l262_262221


namespace find_tan_theta_l262_262158

theorem find_tan_theta
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioc 0 (Real.pi / 4))
  (h2 : Real.sin θ + Real.cos θ = 17 / 13) :
  Real.tan θ = 5 / 12 :=
sorry

end find_tan_theta_l262_262158


namespace more_than_half_good_l262_262232

open Finset

def is_good_permutation {n : ℕ} (p : Perm (Fin (2 * n))) : Prop :=
  ∃ i : Fin (2 * n - 1), abs (p i).val - (p (i + 1)).val = n

noncomputable def count_good_permutations {n : ℕ} : Nat :=
  (univ.perm.card) / 2

theorem more_than_half_good {n : ℕ} (h : 0 < n) :
  ↑(count_good_permutations) < (2 * n)! :=
sorry

end more_than_half_good_l262_262232


namespace reflect_across_y_axis_l262_262215

theorem reflect_across_y_axis (x y : ℝ) :
  (x, y) = (1, 2) → (-x, y) = (-1, 2) :=
by
  intro h
  cases h
  sorry

end reflect_across_y_axis_l262_262215
