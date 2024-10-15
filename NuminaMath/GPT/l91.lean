import Mathlib

namespace NUMINAMATH_GPT_cover_condition_l91_9147

theorem cover_condition (n : ℕ) :
  (∃ (f : ℕ) (h1 : f = n^2), f % 2 = 0) ↔ (n % 2 = 0) := 
sorry

end NUMINAMATH_GPT_cover_condition_l91_9147


namespace NUMINAMATH_GPT_red_marbles_difference_l91_9188

theorem red_marbles_difference 
  (x y : ℕ) 
  (h1 : 7 * x + 3 * x = 140) 
  (h2 : 3 * y + 2 * y = 140)
  (h3 : 10 * x = 5 * y) : 
  7 * x - 3 * y = 20 := 
by 
  sorry

end NUMINAMATH_GPT_red_marbles_difference_l91_9188


namespace NUMINAMATH_GPT_min_f_value_f_achieves_min_l91_9137

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x ^ 2 + 1) + (x * (x + 3)) / (x ^ 2 + 2) + (3 * (x + 1)) / (x * (x ^ 2 + 2))

theorem min_f_value (x : ℝ) (hx : x > 0) : f x ≥ 3 :=
sorry

theorem f_achieves_min (x : ℝ) (hx : x > 0) : ∃ x, f x = 3 :=
sorry

end NUMINAMATH_GPT_min_f_value_f_achieves_min_l91_9137


namespace NUMINAMATH_GPT_bob_speed_l91_9112

theorem bob_speed (v : ℝ) : (∀ v_a : ℝ, v_a > 120 → 30 / v_a < 30 / v - 0.5) → v = 40 :=
by
  sorry

end NUMINAMATH_GPT_bob_speed_l91_9112


namespace NUMINAMATH_GPT_asha_savings_l91_9164

theorem asha_savings (brother father mother granny spending remaining total borrowed_gifted savings : ℤ) 
  (h1 : brother = 20)
  (h2 : father = 40)
  (h3 : mother = 30)
  (h4 : granny = 70)
  (h5 : spending = 3 * total / 4)
  (h6 : remaining = 65)
  (h7 : remaining = total - spending)
  (h8 : total = brother + father + mother + granny + savings)
  (h9 : borrowed_gifted = brother + father + mother + granny) :
  savings = 100 := by
    sorry

end NUMINAMATH_GPT_asha_savings_l91_9164


namespace NUMINAMATH_GPT_integer_roots_of_quadratic_eq_are_neg3_and_neg7_l91_9103

theorem integer_roots_of_quadratic_eq_are_neg3_and_neg7 :
  {k : ℤ | ∃ x : ℤ, k * x^2 - 2 * (3 * k - 1) * x + 9 * k - 1 = 0} = {-3, -7} :=
by
  sorry

end NUMINAMATH_GPT_integer_roots_of_quadratic_eq_are_neg3_and_neg7_l91_9103


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l91_9156

variable {a b : ℝ}

theorem sufficient_but_not_necessary (ha : a > 0) (hb : b > 0) : 
  (ab > 1) → (a + b > 2) ∧ ¬ (a + b > 2 → ab > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l91_9156


namespace NUMINAMATH_GPT_speed_difference_between_lucy_and_sam_l91_9105

noncomputable def average_speed (distance : ℚ) (time_minutes : ℚ) : ℚ :=
  distance / (time_minutes / 60)

theorem speed_difference_between_lucy_and_sam :
  let distance := 6
  let lucy_time := 15
  let sam_time := 45
  let lucy_speed := average_speed distance lucy_time
  let sam_speed := average_speed distance sam_time
  (lucy_speed - sam_speed) = 16 :=
by
  sorry

end NUMINAMATH_GPT_speed_difference_between_lucy_and_sam_l91_9105


namespace NUMINAMATH_GPT_quadratic_equation_root_form_l91_9152

theorem quadratic_equation_root_form
  (a b c : ℤ) (m n p : ℤ)
  (ha : a = 3)
  (hb : b = -4)
  (hc : c = -7)
  (h_discriminant : b^2 - 4 * a * c = n)
  (hgcd_mn : Int.gcd m n = 1)
  (hgcd_mp : Int.gcd m p = 1)
  (hgcd_np : Int.gcd n p = 1) :
  n = 100 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_root_form_l91_9152


namespace NUMINAMATH_GPT_bread_remaining_is_26_85_l91_9132

noncomputable def bread_leftover (jimin_cm : ℕ) (taehyung_m original_length : ℝ) : ℝ :=
  original_length - (jimin_cm / 100 + taehyung_m)

theorem bread_remaining_is_26_85 :
  bread_leftover 150 1.65 30 = 26.85 :=
by
  sorry

end NUMINAMATH_GPT_bread_remaining_is_26_85_l91_9132


namespace NUMINAMATH_GPT_find_a10_l91_9102

def arith_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

variables (a : ℕ → ℚ) (d : ℚ)

-- Conditions
def condition1 := a 4 + a 11 = 16  -- translates to a_5 + a_12 = 16
def condition2 := a 6 = 1  -- translates to a_7 = 1
def condition3 := arith_seq a d  -- a is an arithmetic sequence with common difference d

-- The main theorem
theorem find_a10 : condition1 a ∧ condition2 a ∧ condition3 a d → a 9 = 15 := sorry

end NUMINAMATH_GPT_find_a10_l91_9102


namespace NUMINAMATH_GPT_cube_painting_problem_l91_9104

theorem cube_painting_problem (n : ℕ) (hn : n > 0) :
  (6 * n^2 = (6 * n^3) / 3) ↔ n = 3 :=
by sorry

end NUMINAMATH_GPT_cube_painting_problem_l91_9104


namespace NUMINAMATH_GPT_rectangle_diagonal_l91_9174

theorem rectangle_diagonal (P A: ℝ) (hP : P = 46) (hA : A = 120) : ∃ d : ℝ, d = 17 :=
by
  -- Sorry provides the placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_l91_9174


namespace NUMINAMATH_GPT_triangle_lattice_points_l91_9195

-- Given lengths of the legs of the right triangle
def DE : Nat := 15
def EF : Nat := 20

-- Calculate the hypotenuse using the Pythagorean theorem
def DF : Nat := Nat.sqrt (DE ^ 2 + EF ^ 2)

-- Calculate the area of the triangle
def Area : Nat := (DE * EF) / 2

-- Calculate the number of boundary points
def B : Nat :=
  let points_DE := DE + 1
  let points_EF := EF + 1
  let points_DF := DF + 1
  points_DE + points_EF + points_DF - 3

-- Calculate the number of interior points using Pick's Theorem
def I : Int := Area - (B / 2 - 1)

-- Calculate the total number of lattice points
def total_lattice_points : Int := I + Int.ofNat B

-- The theorem statement
theorem triangle_lattice_points : total_lattice_points = 181 := by
  -- The actual proof goes here
  sorry

end NUMINAMATH_GPT_triangle_lattice_points_l91_9195


namespace NUMINAMATH_GPT_expression_divisible_by_1961_l91_9173

theorem expression_divisible_by_1961 (n : ℕ) : 
  (5^(2*n) * 3^(4*n) - 2^(6*n)) % 1961 = 0 := by
  sorry

end NUMINAMATH_GPT_expression_divisible_by_1961_l91_9173


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l91_9140

-- Definitions based on the conditions
def point1_lies_on_line (a : ℝ) : Prop := a = (2/3 : ℝ) * (-1 : ℝ) - 3
def point2_lies_on_line (b : ℝ) : Prop := b = (2/3 : ℝ) * (1/2 : ℝ) - 3

-- The main theorem to prove the relationship between a and b
theorem relationship_between_a_and_b (a b : ℝ) 
  (h1 : point1_lies_on_line a)
  (h2 : point2_lies_on_line b) : a < b :=
by
  -- Skipping the actual proof. Including sorry to indicate it's not provided.
  sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l91_9140


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l91_9161

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  ((1 / a < 1 ↔ a < 0 ∨ a > 1) ∧ ¬(1 / a < 1 → a ≤ 0 ∨ a ≤ 1)) := 
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l91_9161


namespace NUMINAMATH_GPT_cube_sum_eq_one_l91_9175

theorem cube_sum_eq_one (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 2) (h3 : abc = 1) : a^3 + b^3 + c^3 = 1 :=
sorry

end NUMINAMATH_GPT_cube_sum_eq_one_l91_9175


namespace NUMINAMATH_GPT_symmetrical_character_is_C_l91_9162

-- Definitions of the characters and the concept of symmetry
def is_symmetrical (char: Char): Prop := 
  match char with
  | '中' => True
  | _ => False

-- The options given in the problem
def optionA := '爱'
def optionB := '我'
def optionC := '中'
def optionD := '国'

-- The problem statement: Prove that among the given options, the symmetrical character is 中.
theorem symmetrical_character_is_C : (is_symmetrical optionA = False) ∧ (is_symmetrical optionB = False) ∧ (is_symmetrical optionC = True) ∧ (is_symmetrical optionD = False) :=
by
  sorry

end NUMINAMATH_GPT_symmetrical_character_is_C_l91_9162


namespace NUMINAMATH_GPT_fish_caught_by_twentieth_fisherman_l91_9199

theorem fish_caught_by_twentieth_fisherman :
  ∀ (total_fishermen total_fish fish_per_fisherman nineten_fishermen : ℕ),
  total_fishermen = 20 →
  total_fish = 10000 →
  fish_per_fisherman = 400 →
  nineten_fishermen = 19 →
  (total_fishermen * fish_per_fisherman) - (nineten_fishermen * fish_per_fisherman) = 2400 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fish_caught_by_twentieth_fisherman_l91_9199


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t2_l91_9138

def displacement (t : ℝ) : ℝ := 14 * t - t ^ 2

theorem instantaneous_velocity_at_t2 : (deriv displacement 2) = 10 := by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t2_l91_9138


namespace NUMINAMATH_GPT_length_of_train_l91_9126

theorem length_of_train
  (L : ℝ) 
  (h1 : ∀ S, S = L / 8)
  (h2 : L + 267 = (L / 8) * 20) :
  L = 178 :=
sorry

end NUMINAMATH_GPT_length_of_train_l91_9126


namespace NUMINAMATH_GPT_arithmetic_sequence_a8_value_l91_9196

theorem arithmetic_sequence_a8_value
  (a : ℕ → ℤ) 
  (h1 : a 1 + 3 * a 8 + a 15 = 120)
  (h2 : a 1 + a 15 = 2 * a 8) :
  a 8 = 24 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a8_value_l91_9196


namespace NUMINAMATH_GPT_decagon_adjacent_vertices_probability_l91_9148

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end NUMINAMATH_GPT_decagon_adjacent_vertices_probability_l91_9148


namespace NUMINAMATH_GPT_n_cubed_minus_9n_plus_27_not_div_by_81_l91_9107

theorem n_cubed_minus_9n_plus_27_not_div_by_81 (n : ℤ) : ¬ 81 ∣ (n^3 - 9 * n + 27) :=
sorry

end NUMINAMATH_GPT_n_cubed_minus_9n_plus_27_not_div_by_81_l91_9107


namespace NUMINAMATH_GPT_xy_sum_proof_l91_9133

-- Define the given list of numbers
def original_list := [201, 202, 204, 205, 206, 209, 209, 210, 212]

-- Define the target new average and sum of numbers
def target_average : ℕ := 207
def sum_xy : ℕ := 417

-- Calculate the original sum
def original_sum : ℕ := 201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + 212

-- The new total sum calculation with x and y included
def new_total_sum := original_sum + sum_xy

-- Number of elements in the new list
def new_num_elements : ℕ := 11

-- Target new sum based on the new average and number of elements
def target_new_sum := target_average * new_num_elements

theorem xy_sum_proof : new_total_sum = target_new_sum := by
  sorry

end NUMINAMATH_GPT_xy_sum_proof_l91_9133


namespace NUMINAMATH_GPT_average_grade_of_female_students_is_92_l91_9142

noncomputable def female_average_grade 
  (overall_avg : ℝ) (male_avg : ℝ) (num_males : ℕ) (num_females : ℕ) : ℝ :=
  let total_students := num_males + num_females
  let total_score := total_students * overall_avg
  let male_total_score := num_males * male_avg
  let female_total_score := total_score - male_total_score
  female_total_score / num_females

theorem average_grade_of_female_students_is_92 :
  female_average_grade 90 83 8 28 = 92 := 
by
  -- Proof steps to be completed
  sorry

end NUMINAMATH_GPT_average_grade_of_female_students_is_92_l91_9142


namespace NUMINAMATH_GPT_prime_intersect_even_l91_9125

-- Definitions for prime numbers and even numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Sets P and Q
def P : Set ℕ := { n | is_prime n }
def Q : Set ℕ := { n | is_even n }

-- Proof statement
theorem prime_intersect_even : P ∩ Q = {2} :=
by
  sorry

end NUMINAMATH_GPT_prime_intersect_even_l91_9125


namespace NUMINAMATH_GPT_card_2015_in_box_3_l91_9153

-- Define the pattern function for placing cards
def card_placement (n : ℕ) : ℕ :=
  let cycle_length := 12
  let cycle_pos := (n - 1) % cycle_length + 1
  if cycle_pos ≤ 7 then cycle_pos
  else 14 - cycle_pos

-- Define the theorem to prove the position of the 2015th card
theorem card_2015_in_box_3 : card_placement 2015 = 3 := by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_card_2015_in_box_3_l91_9153


namespace NUMINAMATH_GPT_complex_projective_form_and_fixed_points_l91_9194

noncomputable def complex_projective_transformation (a b c d : ℂ) (z : ℂ) : ℂ :=
  (a * z + b) / (c * z + d)

theorem complex_projective_form_and_fixed_points (a b c d : ℂ) (h : d ≠ 0) :
  (∃ (f : ℂ → ℂ), ∀ z, f z = complex_projective_transformation a b c d z)
  ∧ ∃ (z₁ z₂ : ℂ), complex_projective_transformation a b c d z₁ = z₁ ∧ complex_projective_transformation a b c d z₂ = z₂ :=
by
  -- omitted proof, this is just the statement
  sorry

end NUMINAMATH_GPT_complex_projective_form_and_fixed_points_l91_9194


namespace NUMINAMATH_GPT_expand_expression_l91_9179

variable {R : Type} [CommRing R]
variable (a b x : R)

theorem expand_expression (a b x : R) :
  (a * x^2 + b) * (5 * x^3) = 35 * x^5 + (-15) * x^3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_expand_expression_l91_9179


namespace NUMINAMATH_GPT_fish_bird_apple_fraction_l91_9186

theorem fish_bird_apple_fraction (M : ℝ) (hM : 0 < M) :
  let R_fish := 120
  let R_bird := 60
  let R_total := 180
  let T := M / R_total
  let fish_fraction := (R_fish * T) / M
  let bird_fraction := (R_bird * T) / M
  fish_fraction = 2/3 ∧ bird_fraction = 1/3 := by
  sorry

end NUMINAMATH_GPT_fish_bird_apple_fraction_l91_9186


namespace NUMINAMATH_GPT_number_is_76_l91_9124

theorem number_is_76 (x : ℝ) (h : (3 / 4) * x = x - 19) : x = 76 :=
sorry

end NUMINAMATH_GPT_number_is_76_l91_9124


namespace NUMINAMATH_GPT_reciprocal_equality_l91_9100

theorem reciprocal_equality (a b : ℝ) (h1 : 1 / a = -8) (h2 : 1 / -b = 8) : a = b :=
sorry

end NUMINAMATH_GPT_reciprocal_equality_l91_9100


namespace NUMINAMATH_GPT_option_d_is_deductive_reasoning_l91_9101

-- Define the conditions of the problem
def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ c q : ℤ, c * q ≠ 0 ∧ ∀ n : ℕ, a n = c * q ^ n

-- Define the specific sequence {-2^n}
def a (n : ℕ) : ℤ := -2^n

-- State the proof problem
theorem option_d_is_deductive_reasoning :
  is_geometric_sequence a :=
sorry

end NUMINAMATH_GPT_option_d_is_deductive_reasoning_l91_9101


namespace NUMINAMATH_GPT_painting_time_l91_9131

noncomputable def work_rate (t : ℕ) : ℚ := 1 / t

theorem painting_time (shawn_time karen_time alex_time total_work_rate : ℚ)
  (h_shawn : shawn_time = 18)
  (h_karen : karen_time = 12)
  (h_alex : alex_time = 15) :
  total_work_rate = 1 / (shawn_time + karen_time + alex_time) :=
by
  sorry

end NUMINAMATH_GPT_painting_time_l91_9131


namespace NUMINAMATH_GPT_find_y_l91_9187

theorem find_y (y : ℝ) (h₁ : (y^2 - 7*y + 12) / (y - 3) + (3*y^2 + 5*y - 8) / (3*y - 1) = -8) : y = -6 :=
sorry

end NUMINAMATH_GPT_find_y_l91_9187


namespace NUMINAMATH_GPT_smallest_n_square_area_l91_9145

theorem smallest_n_square_area (n : ℕ) (n_positive : 0 < n) : ∃ k : ℕ, 14 * n = k^2 ↔ n = 14 := 
sorry

end NUMINAMATH_GPT_smallest_n_square_area_l91_9145


namespace NUMINAMATH_GPT_certain_multiple_l91_9155

theorem certain_multiple (n m : ℤ) (h : n = 5) (eq : 7 * n - 15 = m * n + 10) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_certain_multiple_l91_9155


namespace NUMINAMATH_GPT_shaded_area_ratio_l91_9129

noncomputable def ratio_of_shaded_area_to_circle_area (AB r : ℝ) : ℝ :=
  let AC := r
  let CB := 2 * r
  let radius_semicircle_AB := 3 * r / 2
  let area_semicircle_AB := (1 / 2) * (Real.pi * (radius_semicircle_AB ^ 2))
  let radius_semicircle_AC := r / 2
  let area_semicircle_AC := (1 / 2) * (Real.pi * (radius_semicircle_AC ^ 2))
  let radius_semicircle_CB := r
  let area_semicircle_CB := (1 / 2) * (Real.pi * (radius_semicircle_CB ^ 2))
  let total_area_semicircles := area_semicircle_AB + area_semicircle_AC + area_semicircle_CB
  let non_overlapping_area_semicircle_AB := area_semicircle_AB - (area_semicircle_AC + area_semicircle_CB)
  let shaded_area := non_overlapping_area_semicircle_AB
  let area_circle_CD := Real.pi * (r ^ 2)
  shaded_area / area_circle_CD

theorem shaded_area_ratio (AB r : ℝ) : ratio_of_shaded_area_to_circle_area AB r = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_ratio_l91_9129


namespace NUMINAMATH_GPT_prime_large_factor_l91_9128

theorem prime_large_factor (p : ℕ) (hp : Nat.Prime p) (hp_ge_3 : p ≥ 3) (x : ℕ) (hx_large : ∃ N, x ≥ N) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ (p + 3) / 2 ∧ (∃ q : ℕ, Nat.Prime q ∧ q > p ∧ q ∣ (x + i)) := by
  sorry

end NUMINAMATH_GPT_prime_large_factor_l91_9128


namespace NUMINAMATH_GPT_seating_arrangement_l91_9197

-- Define the problem in Lean
theorem seating_arrangement :
  let n := 9   -- Total number of people
  let r := 7   -- Number of seats at the circular table
  let combinations := Nat.choose n 2  -- Ways to select 2 people not seated
  let factorial (k : ℕ) := Nat.recOn k 1 (λ k' acc => (k' + 1) * acc)
  let arrangements := factorial (r - 1)  -- Ways to seat 7 people around a circular table
  combinations * arrangements = 25920 :=
by
  -- In Lean, sorry is used to indicate that we skip the proof for now.
  sorry

end NUMINAMATH_GPT_seating_arrangement_l91_9197


namespace NUMINAMATH_GPT_files_missing_is_15_l91_9141

def total_files : ℕ := 60
def morning_files : ℕ := total_files / 2
def afternoon_files : ℕ := 15
def organized_files : ℕ := morning_files + afternoon_files
def missing_files : ℕ := total_files - organized_files

theorem files_missing_is_15 : missing_files = 15 :=
  sorry

end NUMINAMATH_GPT_files_missing_is_15_l91_9141


namespace NUMINAMATH_GPT_product_of_abcd_l91_9113

noncomputable def a (c : ℚ) : ℚ := 33 * c + 16
noncomputable def b (c : ℚ) : ℚ := 8 * c + 4
noncomputable def d (c : ℚ) : ℚ := c + 1

theorem product_of_abcd :
  (2 * a c + 3 * b c + 5 * c + 8 * d c = 45) →
  (4 * (d c + c) = b c) →
  (4 * (b c) + c = a c) →
  (c + 1 = d c) →
  a c * b c * c * d c = ((1511 : ℚ) / 103) * ((332 : ℚ) / 103) * (-(7 : ℚ) / 103) * ((96 : ℚ) / 103) :=
by
  intros
  sorry

end NUMINAMATH_GPT_product_of_abcd_l91_9113


namespace NUMINAMATH_GPT_num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10_l91_9184

def num_valid_pairs : Nat := 34

def num_valid_first_digits : Nat := 6

def num_valid_last_digits : Nat := 10

theorem num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10 :
  (num_valid_first_digits * num_valid_pairs * num_valid_last_digits) = 2040 :=
by
  sorry

end NUMINAMATH_GPT_num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10_l91_9184


namespace NUMINAMATH_GPT_find_value_of_p_l91_9180

theorem find_value_of_p (p : ℝ) :
  (∀ x y, (x = 0 ∧ y = -2) → y = p*x^2 + 5*x + p) ∧
  (∀ x y, (x = 1/2 ∧ y = 0) → y = p*x^2 + 5*x + p) ∧
  (∀ x y, (x = 2 ∧ y = 0) → y = p*x^2 + 5*x + p) →
  p = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_p_l91_9180


namespace NUMINAMATH_GPT_range_of_g_l91_9190

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : 
  Set.range g = Set.Icc ((π / 2) - (π / 3)) ((π / 2) + (π / 3)) := by
  sorry

end NUMINAMATH_GPT_range_of_g_l91_9190


namespace NUMINAMATH_GPT_like_terms_constants_l91_9193

theorem like_terms_constants :
  ∀ (a b : ℚ), a = 1/2 → b = -1/3 → (a = 1/2 ∧ b = -1/3) → a + b = 1/2 + -1/3 :=
by
  intros a b ha hb h
  sorry

end NUMINAMATH_GPT_like_terms_constants_l91_9193


namespace NUMINAMATH_GPT_value_of_y_l91_9159

theorem value_of_y (y : ℕ) (hy : (1 / 8) * 2^36 = 8^y) : y = 11 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l91_9159


namespace NUMINAMATH_GPT_sqrt_three_is_irrational_and_infinite_non_repeating_decimal_l91_9170

theorem sqrt_three_is_irrational_and_infinite_non_repeating_decimal :
    ∀ r : ℝ, r = Real.sqrt 3 → ¬ ∃ (m n : ℤ), n ≠ 0 ∧ r = m / n := by
    sorry

end NUMINAMATH_GPT_sqrt_three_is_irrational_and_infinite_non_repeating_decimal_l91_9170


namespace NUMINAMATH_GPT_simplify_expression_l91_9122

theorem simplify_expression :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 10) / 4) = 13 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l91_9122


namespace NUMINAMATH_GPT_lines_intersect_l91_9123

-- Condition definitions
def line1 (t : ℝ) : ℝ × ℝ :=
  ⟨2 + t * -1, 3 + t * 5⟩

def line2 (u : ℝ) : ℝ × ℝ :=
  ⟨u * -1, 7 + u * 4⟩

-- Theorem statement
theorem lines_intersect :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (6, -17) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_l91_9123


namespace NUMINAMATH_GPT_batsman_average_after_25th_innings_l91_9110

theorem batsman_average_after_25th_innings (A : ℝ) (h_pre_avg : (25 * (A + 3)) = (24 * A + 80))
  : A + 3 = 8 := 
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_25th_innings_l91_9110


namespace NUMINAMATH_GPT_line_through_point_parallel_to_given_l91_9165

open Real

theorem line_through_point_parallel_to_given (x y : ℝ) :
  (∃ (m : ℝ), (y - 0 = m * (x - 1)) ∧ x - 2*y - 1 = 0) ↔
  (x = 1 ∧ y = 0 ∧ ∃ l, x - 2*y - l = 0) :=
by sorry

end NUMINAMATH_GPT_line_through_point_parallel_to_given_l91_9165


namespace NUMINAMATH_GPT_proteges_57_l91_9169

def divisors (n : ℕ) : List ℕ := (List.range (n + 1)).filter (λ d => n % d = 0)

def units_digit (n : ℕ) : ℕ := n % 10

def proteges (n : ℕ) : List ℕ := (divisors n).map units_digit

theorem proteges_57 : proteges 57 = [1, 3, 9, 7] :=
sorry

end NUMINAMATH_GPT_proteges_57_l91_9169


namespace NUMINAMATH_GPT_opposite_of_abs_frac_l91_9181

theorem opposite_of_abs_frac (h : 0 < (1 : ℝ) / 2023) : -|((1 : ℝ) / 2023)| = -(1 / 2023) := by
  sorry

end NUMINAMATH_GPT_opposite_of_abs_frac_l91_9181


namespace NUMINAMATH_GPT_solution_inequality_l91_9134

theorem solution_inequality (θ x : ℝ)
  (h : |x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) : 
  -1 ≤ x ∧ x ≤ -Real.cos (2 * θ) :=
sorry

end NUMINAMATH_GPT_solution_inequality_l91_9134


namespace NUMINAMATH_GPT_y_range_l91_9109

variable (a b : ℝ)
variable (h₀ : 0 < a) (h₁ : 0 < b)

theorem y_range (x : ℝ) (y : ℝ) (h₂ : y = (a * Real.sin x + b) / (a * Real.sin x - b)) : 
  y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b) :=
sorry

end NUMINAMATH_GPT_y_range_l91_9109


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l91_9168

variable (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)

theorem hyperbola_asymptotes (e : ℝ) (h_ecc : e = (Real.sqrt 5) / 2)
  (h_hyperbola : e = Real.sqrt (1 + (b^2 / a^2))) :
  (∀ x : ℝ, y = x * (b / a) ∨ y = -x * (b / a)) :=
by
  -- Here, the proof would follow logically from the given conditions.
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l91_9168


namespace NUMINAMATH_GPT_xiaohong_home_to_school_distance_l91_9146

noncomputable def driving_distance : ℝ := 1000
noncomputable def total_travel_time : ℝ := 22.5
noncomputable def walking_speed : ℝ := 80
noncomputable def biking_time : ℝ := 40
noncomputable def biking_speed_offset : ℝ := 800

theorem xiaohong_home_to_school_distance (d : ℝ) (v_d : ℝ) :
    let t_w := (d - driving_distance) / walking_speed
    let t_d := driving_distance / v_d
    let v_b := v_d - biking_speed_offset
    (t_d + t_w = total_travel_time)
    → (d / v_b = biking_time)
    → d = 2720 :=
by
  sorry

end NUMINAMATH_GPT_xiaohong_home_to_school_distance_l91_9146


namespace NUMINAMATH_GPT_sum_mod_9_l91_9144

theorem sum_mod_9 (h1 : 34125 % 9 = 1) (h2 : 34126 % 9 = 2) (h3 : 34127 % 9 = 3)
                  (h4 : 34128 % 9 = 4) (h5 : 34129 % 9 = 5) (h6 : 34130 % 9 = 6)
                  (h7 : 34131 % 9 = 7) :
  (34125 + 34126 + 34127 + 34128 + 34129 + 34130 + 34131) % 9 = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_9_l91_9144


namespace NUMINAMATH_GPT_perpendicular_line_sum_l91_9136

theorem perpendicular_line_sum (a b c : ℝ) 
  (h1 : -a / 4 * 2 / 5 = -1)
  (h2 : 10 * 1 + 4 * c - 2 = 0)
  (h3 : 2 * 1 - 5 * c + b = 0) : 
  a + b + c = -4 :=
sorry

end NUMINAMATH_GPT_perpendicular_line_sum_l91_9136


namespace NUMINAMATH_GPT_inequality_proof_l91_9176

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  (1 + 1/x) * (1 + 1/y) ≥ 4 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l91_9176


namespace NUMINAMATH_GPT_div_neg_21_by_3_l91_9111

theorem div_neg_21_by_3 : (-21 : ℤ) / 3 = -7 :=
by sorry

end NUMINAMATH_GPT_div_neg_21_by_3_l91_9111


namespace NUMINAMATH_GPT_remainder_problem_l91_9121

theorem remainder_problem (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) : N % 15 = 10 :=
by
  sorry

end NUMINAMATH_GPT_remainder_problem_l91_9121


namespace NUMINAMATH_GPT_guests_did_not_respond_l91_9157

theorem guests_did_not_respond (n : ℕ) (p_yes p_no : ℝ) (hn : n = 200)
    (hp_yes : p_yes = 0.83) (hp_no : p_no = 0.09) : 
    n - (n * p_yes + n * p_no) = 16 :=
by sorry

end NUMINAMATH_GPT_guests_did_not_respond_l91_9157


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_count_l91_9108

theorem arithmetic_sequence_terms_count :
  ∃ n : ℕ, ∀ a d l, 
    a = 13 → 
    d = 3 → 
    l = 73 → 
    l = a + (n - 1) * d ∧ n = 21 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_count_l91_9108


namespace NUMINAMATH_GPT_monkeys_and_apples_l91_9166

theorem monkeys_and_apples
  {x a : ℕ}
  (h1 : a = 3 * x + 6)
  (h2 : 0 < a - 4 * (x - 1) ∧ a - 4 * (x - 1) < 4)
  : (x = 7 ∧ a = 27) ∨ (x = 8 ∧ a = 30) ∨ (x = 9 ∧ a = 33) :=
sorry

end NUMINAMATH_GPT_monkeys_and_apples_l91_9166


namespace NUMINAMATH_GPT_least_cost_grass_seed_l91_9191

variable (cost_5_pound_bag : ℕ) [Fact (cost_5_pound_bag = 1380)]
variable (cost_10_pound_bag : ℕ) [Fact (cost_10_pound_bag = 2043)]
variable (cost_25_pound_bag : ℕ) [Fact (cost_25_pound_bag = 3225)]
variable (min_weight : ℕ) [Fact (min_weight = 65)]
variable (max_weight : ℕ) [Fact (max_weight = 80)]

theorem least_cost_grass_seed :
  ∃ (n5 n10 n25 : ℕ),
    n5 * 5 + n10 * 10 + n25 * 25 ≥ min_weight ∧
    n5 * 5 + n10 * 10 + n25 * 25 ≤ max_weight ∧
    n5 * cost_5_pound_bag + n10 * cost_10_pound_bag + n25 * cost_25_pound_bag = 9675 :=
  sorry

end NUMINAMATH_GPT_least_cost_grass_seed_l91_9191


namespace NUMINAMATH_GPT_lines_parallel_lines_perpendicular_l91_9171

-- Definition of lines
def l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) := x + (a - 1) * y + a ^ 2 - 1 = 0

-- Parallel condition proof problem
theorem lines_parallel (a : ℝ) : (a = -1) → ∀ x y : ℝ, l1 a x y ∧ l2 a x y →  
        (-(a / 2) = (1 / (1 - a))) ∧ (-3 ≠ -a - 1) :=
by
  intros
  sorry

-- Perpendicular condition proof problem
theorem lines_perpendicular (a : ℝ) : (a = 2 / 3) → ∀ x y : ℝ, l1 a x y ∧ l2 a x y → 
        (- (a / 2) * (1 / (1 - a)) = -1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_lines_parallel_lines_perpendicular_l91_9171


namespace NUMINAMATH_GPT_point_M_coordinates_l91_9139

/- Define the conditions -/

def isInFourthQuadrant (M : ℝ × ℝ) : Prop :=
  M.1 > 0 ∧ M.2 < 0

def distanceToXAxis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  abs M.2 = d

def distanceToYAxis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  abs M.1 = d

/- Write the Lean theorem statement -/

theorem point_M_coordinates :
  ∀ (M : ℝ × ℝ), isInFourthQuadrant M ∧ distanceToXAxis M 3 ∧ distanceToYAxis M 4 → M = (4, -3) :=
by
  intro M
  sorry

end NUMINAMATH_GPT_point_M_coordinates_l91_9139


namespace NUMINAMATH_GPT_poly_coeff_sum_l91_9189

variable {a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ}

theorem poly_coeff_sum :
  (∀ x : ℝ, (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 + 6 * a_6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_poly_coeff_sum_l91_9189


namespace NUMINAMATH_GPT_FindDotsOnFaces_l91_9160

-- Define the structure of a die with specific dot distribution
structure Die where
  three_dots_face : ℕ
  two_dots_faces : ℕ
  one_dot_faces : ℕ

-- Define the problem scenario of 7 identical dice forming 'П' shape
noncomputable def SevenIdenticalDiceFormP (A B C : ℕ) : Prop :=
  ∃ (d : Die), 
    d.three_dots_face = 3 ∧
    d.two_dots_faces = 2 ∧
    d.one_dot_faces = 1 ∧
    (d.three_dots_face + d.two_dots_faces + d.one_dot_faces = 6) ∧
    (A = 2) ∧
    (B = 2) ∧
    (C = 3) 

-- State the theorem to prove A = 2, B = 2, C = 3 given the conditions
theorem FindDotsOnFaces (A B C : ℕ) (h : SevenIdenticalDiceFormP A B C) : A = 2 ∧ B = 2 ∧ C = 3 :=
  by sorry

end NUMINAMATH_GPT_FindDotsOnFaces_l91_9160


namespace NUMINAMATH_GPT_sum_of_cubes_l91_9116

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l91_9116


namespace NUMINAMATH_GPT_geometric_Sn_over_n_sum_first_n_terms_l91_9163

-- The first problem statement translation to Lean 4
theorem geometric_Sn_over_n (a S : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n+1) = (n + 2) * S n) :
  ∃ r : ℕ, (r = 2 ∧ ∃ b : ℕ, b = 1 ∧ 
    ∀ n : ℕ, 0 < n → (S (n + 1)) / (n + 1) = r * (S n) / n) := 
sorry

-- The second problem statement translation to Lean 4
theorem sum_first_n_terms (a S : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n + 1) = (n + 2) * S n)
  (h3 : ∀ n : ℕ, S n = n * 2^(n - 1)) :
  ∀ n : ℕ, T n = (n - 1) * 2^n + 1 :=
sorry

end NUMINAMATH_GPT_geometric_Sn_over_n_sum_first_n_terms_l91_9163


namespace NUMINAMATH_GPT_irrationals_among_examples_l91_9151

theorem irrationals_among_examples :
  ¬ ∃ (r : ℚ), r = π ∧
  (∃ (a b : ℚ), a * a = 4) ∧
  (∃ (r : ℚ), r = 0) ∧
  (∃ (r : ℚ), r = -22 / 7) := 
sorry

end NUMINAMATH_GPT_irrationals_among_examples_l91_9151


namespace NUMINAMATH_GPT_passed_boys_avg_marks_l91_9120

theorem passed_boys_avg_marks (total_boys : ℕ) (avg_marks_all_boys : ℕ) (avg_marks_failed_boys : ℕ) (passed_boys : ℕ) 
  (h1 : total_boys = 120)
  (h2 : avg_marks_all_boys = 35)
  (h3 : avg_marks_failed_boys = 15)
  (h4 : passed_boys = 100) : 
  (39 = (35 * 120 - 15 * (total_boys - passed_boys)) / passed_boys) :=
  sorry

end NUMINAMATH_GPT_passed_boys_avg_marks_l91_9120


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l91_9149

theorem sum_of_squares_of_consecutive_integers :
  ∃ x : ℕ, x * (x + 1) * (x + 2) = 12 * (x + (x + 1) + (x + 2)) ∧ (x^2 + (x + 1)^2 + (x + 2)^2 = 77) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l91_9149


namespace NUMINAMATH_GPT_angle_bisector_proportion_l91_9130

theorem angle_bisector_proportion
  (p q r : ℝ)
  (u v : ℝ)
  (h1 : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < u ∧ 0 < v)
  (h2 : u + v = p)
  (h3 : u * q = v * r) :
  u / p = r / (r + q) :=
sorry

end NUMINAMATH_GPT_angle_bisector_proportion_l91_9130


namespace NUMINAMATH_GPT_total_revenue_is_correct_l91_9150

-- Joan decided to sell all of her old books.
-- She had 33 books in total.
-- She sold 15 books at $4 each.
-- She sold 6 books at $7 each.
-- The rest of the books were sold at $10 each.
-- We need to prove that the total revenue is $222.

def totalBooks := 33
def booksAt4 := 15
def priceAt4 := 4
def booksAt7 := 6
def priceAt7 := 7
def priceAt10 := 10
def remainingBooks := totalBooks - (booksAt4 + booksAt7)
def revenueAt4 := booksAt4 * priceAt4
def revenueAt7 := booksAt7 * priceAt7
def revenueAt10 := remainingBooks * priceAt10
def totalRevenue := revenueAt4 + revenueAt7 + revenueAt10

theorem total_revenue_is_correct : totalRevenue = 222 := by
  sorry

end NUMINAMATH_GPT_total_revenue_is_correct_l91_9150


namespace NUMINAMATH_GPT_sum_of_first_15_terms_l91_9135

-- Given an arithmetic sequence {a_n} such that a_4 + a_6 + a_8 + a_10 + a_12 = 40
-- we need to prove that the sum of the first 15 terms is 120

theorem sum_of_first_15_terms 
  (a_4 a_6 a_8 a_10 a_12 : ℤ)
  (h1 : a_4 + a_6 + a_8 + a_10 + a_12 = 40)
  (a1 d : ℤ)
  (h2 : a_4 = a1 + 3*d)
  (h3 : a_6 = a1 + 5*d)
  (h4 : a_8 = a1 + 7*d)
  (h5 : a_10 = a1 + 9*d)
  (h6 : a_12 = a1 + 11*d) :
  (15 * (a1 + 7*d) = 120) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_15_terms_l91_9135


namespace NUMINAMATH_GPT_trash_can_ratio_l91_9183

theorem trash_can_ratio (streets_trash_cans total_trash_cans : ℕ) 
(h_streets : streets_trash_cans = 14) 
(h_total : total_trash_cans = 42) : 
(total_trash_cans - streets_trash_cans) / streets_trash_cans = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_trash_can_ratio_l91_9183


namespace NUMINAMATH_GPT_problem1_proof_l91_9185

-- Define the mathematical conditions and problems
def problem1_expression (x y : ℝ) : ℝ := y * (4 * x - 3 * y) + (x - 2 * y) ^ 2

-- State the theorem with the simplified form as the conclusion
theorem problem1_proof (x y : ℝ) : problem1_expression x y = x^2 + y^2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_proof_l91_9185


namespace NUMINAMATH_GPT_four_digit_integers_correct_five_digit_integers_correct_l91_9118

-- Definition for the four-digit integers problem
def num_four_digit_integers := ∃ digits : Finset (Fin 5), 4 * 24 = 96

theorem four_digit_integers_correct : num_four_digit_integers := 
by
  sorry

-- Definition for the five-digit integers problem without repetition and greater than 21000
def num_five_digit_integers := ∃ digits : Finset (Fin 5), 48 + 18 = 66

theorem five_digit_integers_correct : num_five_digit_integers := 
by
  sorry

end NUMINAMATH_GPT_four_digit_integers_correct_five_digit_integers_correct_l91_9118


namespace NUMINAMATH_GPT_simplify_fraction_l91_9154

-- Define factorial (or use the existing factorial definition if available in Mathlib)
def fact : ℕ → ℕ 
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Problem statement
theorem simplify_fraction :
  (5 * fact 7 + 35 * fact 6) / fact 8 = 5 / 4 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l91_9154


namespace NUMINAMATH_GPT_parabolas_intersect_l91_9182

theorem parabolas_intersect :
  let eq1 (x : ℝ) := 3 * x^2 - 4 * x + 2
  let eq2 (x : ℝ) := -x^2 + 6 * x + 8
  (∃ x y : ℝ, y = eq1 x ∧ y = eq2 x ∧ x = -0.5 ∧ y = 4.75) ∧
  (∃ x y : ℝ, y = eq1 x ∧ y = eq2 x ∧ x = 3 ∧ y = 17) :=
by sorry

end NUMINAMATH_GPT_parabolas_intersect_l91_9182


namespace NUMINAMATH_GPT_quadratic_sum_roots_twice_difference_l91_9114

theorem quadratic_sum_roots_twice_difference
  (a b c x₁ x₂ : ℝ)
  (h_eq : a * x₁^2 + b * x₁ + c = 0)
  (h_eq2 : a * x₂^2 + b * x₂ + c = 0)
  (h_sum_twice_diff: x₁ + x₂ = 2 * (x₁ - x₂)) :
  3 * b^2 = 16 * a * c :=
sorry

end NUMINAMATH_GPT_quadratic_sum_roots_twice_difference_l91_9114


namespace NUMINAMATH_GPT_transformed_line_theorem_l91_9177

theorem transformed_line_theorem (k b : ℝ) (h₁ : k = 1) (h₂ : b = 1) (x : ℝ) :
  (k * x + b > 0) ↔ (x > -1) :=
by sorry

end NUMINAMATH_GPT_transformed_line_theorem_l91_9177


namespace NUMINAMATH_GPT_problem_proof_l91_9143

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 2

theorem problem_proof : f (1 + g 3) = 32 := by
  sorry

end NUMINAMATH_GPT_problem_proof_l91_9143


namespace NUMINAMATH_GPT_cube_sum_l91_9119

theorem cube_sum (x y : ℝ) (h1 : x * y = 15) (h2 : x + y = 11) : x^3 + y^3 = 836 := 
by
  sorry

end NUMINAMATH_GPT_cube_sum_l91_9119


namespace NUMINAMATH_GPT_foci_distance_of_hyperbola_l91_9115

theorem foci_distance_of_hyperbola : 
  let a_squared := 32
  let b_squared := 8
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  2 * c = 4 * Real.sqrt 10 :=
by
  -- Definitions based on conditions
  let a_squared := 32
  let b_squared := 8
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  
  -- Proof outline here (using sorry to skip proof details)
  sorry

end NUMINAMATH_GPT_foci_distance_of_hyperbola_l91_9115


namespace NUMINAMATH_GPT_jessica_older_than_claire_l91_9167

-- Define the current age of Claire
def claire_current_age := 20 - 2

-- Define the current age of Jessica
def jessica_current_age := 24

-- Prove that Jessica is 6 years older than Claire
theorem jessica_older_than_claire : jessica_current_age - claire_current_age = 6 :=
by
  -- Definitions of the ages
  let claire_current_age := 18
  let jessica_current_age := 24

  -- Prove the age difference
  sorry

end NUMINAMATH_GPT_jessica_older_than_claire_l91_9167


namespace NUMINAMATH_GPT_average_of_remaining_numbers_l91_9158

theorem average_of_remaining_numbers (S : ℕ) 
  (h₁ : S = 85 * 10) 
  (S' : ℕ) 
  (h₂ : S' = S - 70 - 76) : 
  S' / 8 = 88 := 
sorry

end NUMINAMATH_GPT_average_of_remaining_numbers_l91_9158


namespace NUMINAMATH_GPT_sin_225_plus_alpha_l91_9198

theorem sin_225_plus_alpha (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = 5 / 13) :
    Real.sin (5 * Real.pi / 4 + α) = -5 / 13 :=
by
  sorry

end NUMINAMATH_GPT_sin_225_plus_alpha_l91_9198


namespace NUMINAMATH_GPT_employee_total_weekly_pay_l91_9106

-- Define the conditions
def hours_per_day_first_3_days : ℕ := 6
def hours_per_day_last_2_days : ℕ := 2 * hours_per_day_first_3_days
def first_40_hours_pay_rate : ℕ := 30
def overtime_multiplier : ℕ := 3 / 2 -- 50% more pay, i.e., 1.5 times

-- Functions to compute total hours worked and total pay
def hours_first_3_days (d : ℕ) : ℕ := d * hours_per_day_first_3_days
def hours_last_2_days (d : ℕ) : ℕ := d * hours_per_day_last_2_days
def total_hours_worked : ℕ := (hours_first_3_days 3) + (hours_last_2_days 2)
def regular_hours : ℕ := min 40 total_hours_worked
def overtime_hours : ℕ := total_hours_worked - regular_hours
def regular_pay : ℕ := regular_hours * first_40_hours_pay_rate
def overtime_pay_rate : ℕ := first_40_hours_pay_rate + (first_40_hours_pay_rate / 2) -- 50% more
def overtime_pay : ℕ := overtime_hours * overtime_pay_rate
def total_pay : ℕ := regular_pay + overtime_pay

-- The statement to be proved
theorem employee_total_weekly_pay : total_pay = 1290 := by
  sorry

end NUMINAMATH_GPT_employee_total_weekly_pay_l91_9106


namespace NUMINAMATH_GPT_find_ab_l91_9127

theorem find_ab (a b : ℤ) :
  (∀ x : ℤ, a * (x - 3) + b * (3 * x + 1) = 5 * (x + 1)) →
  a = -1 ∧ b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l91_9127


namespace NUMINAMATH_GPT_new_bucket_capacity_l91_9192

theorem new_bucket_capacity (init_buckets : ℕ) (init_capacity : ℕ) (new_buckets : ℕ) (total_volume : ℕ) :
  init_buckets * init_capacity = total_volume →
  new_buckets * 9 = total_volume →
  9 = total_volume / new_buckets :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_new_bucket_capacity_l91_9192


namespace NUMINAMATH_GPT_f_properties_l91_9178

theorem f_properties (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^2) - f (y^2) ≤ (f x + y) * (x - f y)) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end NUMINAMATH_GPT_f_properties_l91_9178


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l91_9172

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 1 ≤ x then x^2 - 2 * a * x + a
  else if 0 < x then 2 * x + a / x
  else 0 -- Undefined for x ≤ 0

theorem problem1 (a : ℝ) :
  (∀ x y : ℝ, (0 < x ∧ x < y) → f a x < f a y) ↔ (a ≤ -1 / 2) :=
sorry
  
theorem problem2 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ f a x1 = 1 ∧ f a x2 = 1 ∧ f a x3 = 1) ↔ (0 < a ∧ a < 1 / 8) :=
sorry

theorem problem3 (a : ℝ) :
  (∀ x : ℝ, f a x ≥ x - 2 * a) ↔ (0 ≤ a ∧ a ≤ 1 + Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l91_9172


namespace NUMINAMATH_GPT_ones_digit_of_first_in_sequence_l91_9117

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
  
def in_arithmetic_sequence (a d : ℕ) (n : ℕ) : Prop :=
  ∃ k, a = k * d + n

theorem ones_digit_of_first_in_sequence {p q r s t : ℕ}
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (ht : is_prime t)
  (hseq : in_arithmetic_sequence p 10 q ∧ 
          in_arithmetic_sequence q 10 r ∧
          in_arithmetic_sequence r 10 s ∧
          in_arithmetic_sequence s 10 t)
  (hincr : p < q ∧ q < r ∧ r < s ∧ s < t)
  (hstart : p > 5) :
  p % 10 = 1 := sorry

end NUMINAMATH_GPT_ones_digit_of_first_in_sequence_l91_9117
