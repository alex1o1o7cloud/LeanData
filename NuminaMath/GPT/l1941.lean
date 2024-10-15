import Mathlib

namespace NUMINAMATH_GPT_possible_quadrilateral_areas_l1941_194145

-- Define the problem set up
structure Point where
  x : ℝ
  y : ℝ

structure Square where
  side_length : ℝ
  A : Point
  B : Point
  C : Point
  D : Point

-- Defines the division points on each side of the square
def division_points (A B C D : Point) : List Point :=
  [
    -- Points on AB
    { x := 1, y := 4 }, { x := 2, y := 4 }, { x := 3, y := 4 },
    -- Points on BC
    { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 4, y := 1 },
    -- Points on CD
    { x := 3, y := 0 }, { x := 2, y := 0 }, { x := 1, y := 0 },
    -- Points on DA
    { x := 0, y := 3 }, { x := 0, y := 2 }, { x := 0, y := 1 }
  ]

-- Possible areas calculation using the Shoelace Theorem
def quadrilateral_areas : List ℝ :=
  [6, 7, 7.5, 8, 8.5, 9, 10]

-- Math proof problem in Lean, we need to prove that the quadrilateral areas match the given values
theorem possible_quadrilateral_areas (ABCD : Square) (pts : List Point) :
    (division_points ABCD.A ABCD.B ABCD.C ABCD.D) = [
      { x := 1, y := 4 }, { x := 2, y := 4 }, { x := 3, y := 4 },
      { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 4, y := 1 },
      { x := 3, y := 0 }, { x := 2, y := 0 }, { x := 1, y := 0 },
      { x := 0, y := 3 }, { x := 0, y := 2 }, { x := 0, y := 1 }
    ] → 
    (∃ areas, areas ⊆ quadrilateral_areas) := by
  sorry

end NUMINAMATH_GPT_possible_quadrilateral_areas_l1941_194145


namespace NUMINAMATH_GPT_simplify_polynomial_l1941_194133

/-- Simplification of the polynomial expression -/
theorem simplify_polynomial (x : ℝ) :
  x * (4 * x^2 - 2) - 5 * (x^2 - 3 * x + 5) = 4 * x^3 - 5 * x^2 + 13 * x - 25 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1941_194133


namespace NUMINAMATH_GPT_arithmetic_sequence_inequality_l1941_194151

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → 2 * a n = a (n - 1) + a (n + 1)) :
  a 2 * a 4 ≤ a 3 ^ 2 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_inequality_l1941_194151


namespace NUMINAMATH_GPT_pour_tea_into_containers_l1941_194177

-- Define the total number of containers
def total_containers : ℕ := 80

-- Define the amount of tea that Geraldo drank in terms of containers
def geraldo_drank_containers : ℚ := 3.5

-- Define the amount of tea that Geraldo consumed in terms of pints
def geraldo_drank_pints : ℕ := 7

-- Define the conversion factor from pints to gallons
def pints_per_gallon : ℕ := 8

-- Question: How many gallons of tea were poured into the containers?
theorem pour_tea_into_containers 
  (total_containers : ℕ)
  (geraldo_drank_containers : ℚ)
  (geraldo_drank_pints : ℕ)
  (pints_per_gallon : ℕ) :
  (total_containers * (geraldo_drank_pints / geraldo_drank_containers) / pints_per_gallon) = 20 :=
by
  sorry

end NUMINAMATH_GPT_pour_tea_into_containers_l1941_194177


namespace NUMINAMATH_GPT_simplify_expression_l1941_194165

theorem simplify_expression :
  (4 + 2 + 6) / 3 - (2 + 1) / 3 = 3 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1941_194165


namespace NUMINAMATH_GPT_range_of_sum_l1941_194124

theorem range_of_sum (a b c : ℝ) (h1: a > b) (h2 : b > c) (h3 : a + b + c = 1) (h4 : a^2 + b^2 + c^2 = 3) :
-2/3 < b + c ∧ b + c < 0 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_sum_l1941_194124


namespace NUMINAMATH_GPT_product_of_four_consecutive_integers_is_perfect_square_l1941_194172

-- Define the main statement we want to prove
theorem product_of_four_consecutive_integers_is_perfect_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_product_of_four_consecutive_integers_is_perfect_square_l1941_194172


namespace NUMINAMATH_GPT_carnations_in_first_bouquet_l1941_194176

theorem carnations_in_first_bouquet 
  (c2 : ℕ) (c3 : ℕ) (avg : ℕ) (n : ℕ) (total_carnations : ℕ) : 
  c2 = 14 → c3 = 13 → avg = 12 → n = 3 → total_carnations = avg * n →
  (total_carnations - (c2 + c3) = 9) :=
by
  sorry

end NUMINAMATH_GPT_carnations_in_first_bouquet_l1941_194176


namespace NUMINAMATH_GPT_cube_surface_area_ratio_l1941_194192

variable (x : ℝ) (hx : x > 0)

theorem cube_surface_area_ratio (hx : x > 0):
  let side1 := 7 * x
  let side2 := x
  let SA1 := 6 * side1^2
  let SA2 := 6 * side2^2
  (SA1 / SA2) = 49 := 
by 
  sorry

end NUMINAMATH_GPT_cube_surface_area_ratio_l1941_194192


namespace NUMINAMATH_GPT_ratio_second_third_l1941_194137

theorem ratio_second_third (S T : ℕ) (h_sum : 200 + S + T = 500) (h_third : T = 100) : S / T = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_second_third_l1941_194137


namespace NUMINAMATH_GPT_books_from_first_shop_l1941_194140

theorem books_from_first_shop (x : ℕ) (h : (2080 : ℚ) / (x + 50) = 18.08695652173913) : x = 65 :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_books_from_first_shop_l1941_194140


namespace NUMINAMATH_GPT_physical_education_class_min_size_l1941_194190

theorem physical_education_class_min_size :
  ∃ (x : Nat), 3 * x + 2 * (x + 1) > 50 ∧ 5 * x + 2 = 52 := by
  sorry

end NUMINAMATH_GPT_physical_education_class_min_size_l1941_194190


namespace NUMINAMATH_GPT_a_eq_b_pow_n_l1941_194115

variables (a b n : ℕ)
variable (h : ∀ (k : ℕ), k ≠ b → b - k ∣ a - k^n)

theorem a_eq_b_pow_n : a = b^n := 
by
  sorry

end NUMINAMATH_GPT_a_eq_b_pow_n_l1941_194115


namespace NUMINAMATH_GPT_polygon_perpendiculars_length_l1941_194155

noncomputable def RegularPolygon := { n : ℕ // n ≥ 3 }

structure Perpendiculars (P : RegularPolygon) (i : ℕ) :=
  (d_i     : ℝ)
  (d_i_minus_1 : ℝ)
  (d_i_plus_1 : ℝ)
  (line_crosses_interior : Bool)

theorem polygon_perpendiculars_length {P : RegularPolygon} {i : ℕ}
  (hyp : Perpendiculars P i) :
  hyp.d_i = if hyp.line_crosses_interior 
            then hyp.d_i_minus_1 + hyp.d_i_plus_1 
            else abs (hyp.d_i_minus_1 - hyp.d_i_plus_1) :=
sorry

end NUMINAMATH_GPT_polygon_perpendiculars_length_l1941_194155


namespace NUMINAMATH_GPT_number_of_solutions_l1941_194126

theorem number_of_solutions :
  (∃(x y : ℤ), x^4 + y^2 = 6 * y - 8) ∧ ∃!(x y : ℤ), x^4 + y^2 = 6 * y - 8 := 
sorry

end NUMINAMATH_GPT_number_of_solutions_l1941_194126


namespace NUMINAMATH_GPT_maze_paths_unique_l1941_194108

-- Define the conditions and branching points
def maze_structure (x : ℕ) (b : ℕ) : Prop :=
  x > 0 ∧ b > 0 ∧
  -- This represents the structure and unfolding paths at each point
  ∀ (i : ℕ), i < x → ∃ j < b, True

-- Define a function to count the number of unique paths given the number of branching points
noncomputable def count_paths (x : ℕ) (b : ℕ) : ℕ :=
  x * (2 ^ b)

-- State the main theorem
theorem maze_paths_unique : ∃ x b, maze_structure x b ∧ count_paths x b = 16 :=
by
  -- The proof contents are skipped for now
  sorry

end NUMINAMATH_GPT_maze_paths_unique_l1941_194108


namespace NUMINAMATH_GPT_perpendicular_lines_have_a_zero_l1941_194174

theorem perpendicular_lines_have_a_zero {a : ℝ} :
  ∀ x y : ℝ, (ax + y - 1 = 0) ∧ (x + a*y - 1 = 0) → a = 0 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_have_a_zero_l1941_194174


namespace NUMINAMATH_GPT_smallest_n_proof_l1941_194112

-- Given conditions and the problem statement in Lean 4
noncomputable def smallest_n : ℕ := 11

theorem smallest_n_proof :
  ∃ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) ∧ (smallest_n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 11) :=
sorry

end NUMINAMATH_GPT_smallest_n_proof_l1941_194112


namespace NUMINAMATH_GPT_largest_circle_center_is_A_l1941_194129

-- Define the given lengths of the pentagon's sides
def AB : ℝ := 16
def BC : ℝ := 14
def CD : ℝ := 17
def DE : ℝ := 13
def AE : ℝ := 14

-- Define the radii of the circles centered at points A, B, C, D, E
variables (R_A R_B R_C R_D R_E : ℝ)

-- Conditions based on the problem statement
def radius_conditions : Prop :=
  R_A + R_B = AB ∧
  R_B + R_C = BC ∧
  R_C + R_D = CD ∧
  R_D + R_E = DE ∧
  R_E + R_A = AE

-- The main theorem to prove
theorem largest_circle_center_is_A (h : radius_conditions R_A R_B R_C R_D R_E) :
  10 ≥ R_A ∧ R_A ≥ R_B ∧ R_A ≥ R_C ∧ R_A ≥ R_D ∧ R_A ≥ R_E :=
by sorry

end NUMINAMATH_GPT_largest_circle_center_is_A_l1941_194129


namespace NUMINAMATH_GPT_roots_square_sum_l1941_194153

theorem roots_square_sum (a b : ℝ) 
  (h1 : a^2 - 4 * a + 4 = 0) 
  (h2 : b^2 - 4 * b + 4 = 0) 
  (h3 : a = b) :
  a^2 + b^2 = 8 := 
sorry

end NUMINAMATH_GPT_roots_square_sum_l1941_194153


namespace NUMINAMATH_GPT_increasing_digits_count_l1941_194120

theorem increasing_digits_count : 
  ∃ n, n = 120 ∧ ∀ x : ℕ, x ≤ 1000 → (∀ i j : ℕ, i < j → ((x / 10^i % 10) < (x / 10^j % 10)) → 
  x ≤ 1000 ∧ (x / 10^i % 10) ≠ (x / 10^j % 10)) :=
sorry

end NUMINAMATH_GPT_increasing_digits_count_l1941_194120


namespace NUMINAMATH_GPT_calculate_expression_l1941_194104

theorem calculate_expression : ((18^18 / 18^17)^3 * 8^3) / 2^9 = 5832 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1941_194104


namespace NUMINAMATH_GPT_gift_card_value_l1941_194191

def latte_cost : ℝ := 3.75
def croissant_cost : ℝ := 3.50
def daily_treat_cost : ℝ := latte_cost + croissant_cost
def weekly_treat_cost : ℝ := daily_treat_cost * 7

def cookie_cost : ℝ := 1.25
def total_cookie_cost : ℝ := cookie_cost * 5

def total_spent : ℝ := weekly_treat_cost + total_cookie_cost
def remaining_balance : ℝ := 43.00

theorem gift_card_value : (total_spent + remaining_balance) = 100 := 
by sorry

end NUMINAMATH_GPT_gift_card_value_l1941_194191


namespace NUMINAMATH_GPT_Vann_total_teeth_cleaned_l1941_194150

theorem Vann_total_teeth_cleaned :
  let dogs := 7
  let cats := 12
  let pigs := 9
  let horses := 4
  let rabbits := 15
  let dogs_teeth := 42
  let cats_teeth := 30
  let pigs_teeth := 44
  let horses_teeth := 40
  let rabbits_teeth := 28
  (dogs * dogs_teeth) + (cats * cats_teeth) + (pigs * pigs_teeth) + (horses * horses_teeth) + (rabbits * rabbits_teeth) = 1630 :=
by
  sorry

end NUMINAMATH_GPT_Vann_total_teeth_cleaned_l1941_194150


namespace NUMINAMATH_GPT_min_value_of_u_l1941_194121

theorem min_value_of_u (x y : ℝ) (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) (hxy : x * y = -1) :
  (∀ u, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) → u ≥ (12 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_u_l1941_194121


namespace NUMINAMATH_GPT_total_cups_for_8_batches_l1941_194170

def cups_of_flour (batches : ℕ) : ℝ := 4 * batches
def cups_of_sugar (batches : ℕ) : ℝ := 1.5 * batches
def total_cups (batches : ℕ) : ℝ := cups_of_flour batches + cups_of_sugar batches

theorem total_cups_for_8_batches : total_cups 8 = 44 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_total_cups_for_8_batches_l1941_194170


namespace NUMINAMATH_GPT_abs_add_conditions_l1941_194181

theorem abs_add_conditions (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a < b) :
  a + b = 1 ∨ a + b = 7 :=
by
  sorry

end NUMINAMATH_GPT_abs_add_conditions_l1941_194181


namespace NUMINAMATH_GPT_division_result_l1941_194103

theorem division_result:
    35 / 0.07 = 500 := by
  sorry

end NUMINAMATH_GPT_division_result_l1941_194103


namespace NUMINAMATH_GPT_units_digit_n_is_7_l1941_194175

def units_digit (x : ℕ) : ℕ := x % 10

theorem units_digit_n_is_7 (m n : ℕ) (h1 : m * n = 31 ^ 4) (h2 : units_digit m = 6) :
  units_digit n = 7 :=
sorry

end NUMINAMATH_GPT_units_digit_n_is_7_l1941_194175


namespace NUMINAMATH_GPT_selected_numbers_count_l1941_194168

noncomputable def check_num_of_selected_numbers : ℕ := 
  let n := 2015
  let max_num := n * n
  let common_difference := 15
  let starting_number := 14
  let count := (max_num - starting_number) / common_difference + 1
  count

theorem selected_numbers_count : check_num_of_selected_numbers = 270681 := by
  -- Skipping the actual proof
  sorry

end NUMINAMATH_GPT_selected_numbers_count_l1941_194168


namespace NUMINAMATH_GPT_quadratic_discriminant_l1941_194117

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end NUMINAMATH_GPT_quadratic_discriminant_l1941_194117


namespace NUMINAMATH_GPT_journey_total_distance_l1941_194179

theorem journey_total_distance :
  let speed1 := 40 -- in kmph
  let time1 := 3 -- in hours
  let speed2 := 60 -- in kmph
  let totalTime := 5 -- in hours
  let distance1 := speed1 * time1
  let time2 := totalTime - time1
  let distance2 := speed2 * time2
  let totalDistance := distance1 + distance2
  totalDistance = 240 := 
by
  sorry

end NUMINAMATH_GPT_journey_total_distance_l1941_194179


namespace NUMINAMATH_GPT_range_of_a_l1941_194102

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 1| + |x - 2| ≤ a^2 + a + 1)) → -1 < a ∧ a < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1941_194102


namespace NUMINAMATH_GPT_inequality_problem_l1941_194173

theorem inequality_problem
  (a b c : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( ( (2 * a + b + c) ^ 2 ) / ( 2 * a ^ 2 + (b + c) ^ 2 ) ) +
  ( ( (a + 2 * b + c) ^ 2 ) / ( 2 * b ^ 2 + (c + a) ^ 2 ) ) +
  ( ( (a + b + 2 * c) ^ 2 ) / ( 2 * c ^ 2 + (a + b) ^ 2 ) ) ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l1941_194173


namespace NUMINAMATH_GPT_ratio_of_sides_l1941_194163

variable {A B C a b c : ℝ}

theorem ratio_of_sides
  (h1 : 2 * b * Real.sin (2 * A) = 3 * a * Real.sin B)
  (h2 : c = 2 * b) :
  a / b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_sides_l1941_194163


namespace NUMINAMATH_GPT_decision_making_system_reliability_l1941_194146

theorem decision_making_system_reliability (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (10 * p^3 - 15 * p^4 + 6 * p^5 > 3 * p^2 - 2 * p^3) -> (1 / 2 < p) ∧ (p < 1) :=
by
  sorry

end NUMINAMATH_GPT_decision_making_system_reliability_l1941_194146


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1941_194135

theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℕ), (∀ n, a n.succ = a n + 2) → a 1 = 2 → a 5 = 10 :=
by
  intros a h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1941_194135


namespace NUMINAMATH_GPT_trapezoid_shorter_base_length_l1941_194101

theorem trapezoid_shorter_base_length (longer_base : ℕ) (segment_length : ℕ) (shorter_base : ℕ) 
  (h1 : longer_base = 120) (h2 : segment_length = 7)
  (h3 : segment_length = (longer_base - shorter_base) / 2) : 
  shorter_base = 106 := by
  sorry

end NUMINAMATH_GPT_trapezoid_shorter_base_length_l1941_194101


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1941_194188

noncomputable def a : ℝ := (1 / 3) ^ 3
noncomputable def b (x : ℝ) : ℝ := x ^ 3
noncomputable def c (x : ℝ) : ℝ := Real.log x

theorem relationship_among_a_b_c (x : ℝ) (h : x > 2) : a < c x ∧ c x < b x :=
by {
  -- proof steps are skipped
  sorry
}

end NUMINAMATH_GPT_relationship_among_a_b_c_l1941_194188


namespace NUMINAMATH_GPT_sum_square_ends_same_digit_l1941_194193

theorem sum_square_ends_same_digit {a b : ℤ} (h : (a + b) % 10 = 0) :
  (a^2 % 10) = (b^2 % 10) :=
by
  sorry

end NUMINAMATH_GPT_sum_square_ends_same_digit_l1941_194193


namespace NUMINAMATH_GPT_remainder_of_3x_plus_5y_l1941_194123

-- Conditions and parameter definitions
def x (k : ℤ) := 13 * k + 7
def y (m : ℤ) := 17 * m + 11

-- Proof statement
theorem remainder_of_3x_plus_5y (k m : ℤ) : (3 * x k + 5 * y m) % 221 = 76 := by
  sorry

end NUMINAMATH_GPT_remainder_of_3x_plus_5y_l1941_194123


namespace NUMINAMATH_GPT_highest_y_coordinate_l1941_194119

-- Define the conditions
def ellipse_condition (x y : ℝ) : Prop :=
  (x^2 / 25) + ((y - 3)^2 / 9) = 1

-- The theorem to prove
theorem highest_y_coordinate : ∃ x : ℝ, ∀ y : ℝ, ellipse_condition x y → y ≤ 6 :=
sorry

end NUMINAMATH_GPT_highest_y_coordinate_l1941_194119


namespace NUMINAMATH_GPT_arithmetic_identity_l1941_194162

theorem arithmetic_identity : 3 * 5 * 7 + 15 / 3 = 110 := by
  sorry

end NUMINAMATH_GPT_arithmetic_identity_l1941_194162


namespace NUMINAMATH_GPT_relationship_f_l1941_194156

-- Define the function f which is defined on the reals and even
variable (f : ℝ → ℝ)
-- Condition: f is an even function
axiom even_f : ∀ x, f (-x) = f x
-- Condition: (x₁ - x₂)[f(x₁) - f(x₂)] > 0 for all x₁, x₂ ∈ [0, +∞)
axiom increasing_cond : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem relationship_f : f (1/2) < f 1 ∧ f 1 < f (-2) := by
  sorry

end NUMINAMATH_GPT_relationship_f_l1941_194156


namespace NUMINAMATH_GPT_solve_system_of_equations_l1941_194144

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : 6.751 * x + 3.249 * y = 26.751) 
  (h2 : 3.249 * x + 6.751 * y = 23.249) : 
  x = 3 ∧ y = 2 := 
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1941_194144


namespace NUMINAMATH_GPT_number_of_toddlers_l1941_194157

-- Definitions based on the conditions provided in the problem
def total_children := 40
def newborns := 4
def toddlers (T : ℕ) := T
def teenagers (T : ℕ) := 5 * T

-- The theorem to prove
theorem number_of_toddlers : ∃ T : ℕ, newborns + toddlers T + teenagers T = total_children ∧ T = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_toddlers_l1941_194157


namespace NUMINAMATH_GPT_interview_room_count_l1941_194159

-- Define the number of people in the waiting room
def people_in_waiting_room : ℕ := 22

-- Define the increase in number of people
def extra_people_arrive : ℕ := 3

-- Define the total number of people after more people arrive
def total_people_after_arrival : ℕ := people_in_waiting_room + extra_people_arrive

-- Define the relationship between people in waiting room and interview room
def relation (x : ℕ) : Prop := total_people_after_arrival = 5 * x

theorem interview_room_count : ∃ x : ℕ, relation x ∧ x = 5 :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_interview_room_count_l1941_194159


namespace NUMINAMATH_GPT_find_sum_2017_l1941_194105

-- Define the arithmetic sequence and its properties
def is_arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a n) / 2

-- Given conditions
variables (a : ℕ → ℤ)
axiom h1 : is_arithmetic_sequence a
axiom h2 : sum_first_n_terms a 2011 = -2011
axiom h3 : a 1012 = 3

-- Theorem to be proven
theorem find_sum_2017 : sum_first_n_terms a 2017 = 2017 :=
by sorry

end NUMINAMATH_GPT_find_sum_2017_l1941_194105


namespace NUMINAMATH_GPT_circle_radius_value_l1941_194110

theorem circle_radius_value (k : ℝ) :
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + k = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25) → k = 16 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_value_l1941_194110


namespace NUMINAMATH_GPT_ratio_3_2_l1941_194136

theorem ratio_3_2 (m n : ℕ) (h1 : m + n = 300) (h2 : m > 100) (h3 : n > 100) : m / n = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_3_2_l1941_194136


namespace NUMINAMATH_GPT_cat_food_insufficient_l1941_194111

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end NUMINAMATH_GPT_cat_food_insufficient_l1941_194111


namespace NUMINAMATH_GPT_average_speed_of_car_l1941_194141

noncomputable def average_speed (total_distance total_time : ℕ) : ℕ :=
  total_distance / total_time

theorem average_speed_of_car :
  let d1 := 80
  let d2 := 40
  let d3 := 60
  let d4 := 50
  let d5 := 90
  let d6 := 100
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  average_speed total_distance total_time = 70 :=
by
  let d1 := 80
  let d2 := 40
  let d3 := 60
  let d4 := 50
  let d5 := 90
  let d6 := 100
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  exact sorry

end NUMINAMATH_GPT_average_speed_of_car_l1941_194141


namespace NUMINAMATH_GPT_exists_integers_A_B_C_l1941_194186

theorem exists_integers_A_B_C (a b : ℚ) (N_star : Set ℕ) (Q : Set ℚ)
  (h : ∀ x ∈ N_star, (a * (x : ℚ) + b) / (x : ℚ) ∈ Q) : 
  ∃ A B C : ℤ, ∀ x ∈ N_star, 
    (a * (x : ℚ) + b) / (x : ℚ) = (A * (x : ℚ) + B) / (C * (x : ℚ)) := 
sorry

end NUMINAMATH_GPT_exists_integers_A_B_C_l1941_194186


namespace NUMINAMATH_GPT_composite_sum_l1941_194164

theorem composite_sum (a b : ℤ) (h : 56 * a = 65 * b) : ∃ m n : ℤ,  m > 1 ∧ n > 1 ∧ a + b = m * n :=
sorry

end NUMINAMATH_GPT_composite_sum_l1941_194164


namespace NUMINAMATH_GPT_milan_rate_per_minute_l1941_194139

-- Definitions based on the conditions
def monthly_fee : ℝ := 2.0
def total_bill : ℝ := 23.36
def total_minutes : ℕ := 178
def expected_rate_per_minute : ℝ := 0.12

-- Theorem statement based on the question
theorem milan_rate_per_minute :
  (total_bill - monthly_fee) / total_minutes = expected_rate_per_minute := 
by 
  sorry

end NUMINAMATH_GPT_milan_rate_per_minute_l1941_194139


namespace NUMINAMATH_GPT_area_of_triangle_ABF_l1941_194189

theorem area_of_triangle_ABF :
  let C : Set (ℝ × ℝ) := {p | (p.1 ^ 2 / 4) + (p.2 ^ 2 / 3) = 1}
  let line : Set (ℝ × ℝ) := {p | p.1 - p.2 - 1 = 0}
  let F : ℝ × ℝ := (-1, 0)
  let AB := C ∩ line
  ∃ A B : ℝ × ℝ, A ∈ AB ∧ B ∈ AB ∧ A ≠ B ∧ 
  (1/2) * (2 : ℝ) * (12 * Real.sqrt (2 : ℝ) / 7) = (12 * Real.sqrt (2 : ℝ) / 7) :=
sorry

end NUMINAMATH_GPT_area_of_triangle_ABF_l1941_194189


namespace NUMINAMATH_GPT_multiplication_correct_l1941_194167

theorem multiplication_correct (x : ℤ) (h : x - 6 = 51) : x * 6 = 342 := by
  sorry

end NUMINAMATH_GPT_multiplication_correct_l1941_194167


namespace NUMINAMATH_GPT_distinct_parenthesizations_of_3_3_3_3_l1941_194127

theorem distinct_parenthesizations_of_3_3_3_3 : 
  ∃ (v1 v2 v3 v4 v5 : ℕ), 
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ 
    v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ 
    v3 ≠ v4 ∧ v3 ≠ v5 ∧ 
    v4 ≠ v5 ∧ 
    v1 = 3 ^ (3 ^ (3 ^ 3)) ∧ 
    v2 = 3 ^ ((3 ^ 3) ^ 3) ∧ 
    v3 = (3 ^ 3) ^ (3 ^ 3) ∧ 
    v4 = ((3 ^ 3) ^ 3) ^ 3 ∧ 
    v5 = 3 ^ (27 ^ 27) :=
  sorry

end NUMINAMATH_GPT_distinct_parenthesizations_of_3_3_3_3_l1941_194127


namespace NUMINAMATH_GPT_problem_l1941_194116

open Real 

noncomputable def sqrt_log_a (a : ℝ) : ℝ := sqrt (log a / log 10)
noncomputable def sqrt_log_b (b : ℝ) : ℝ := sqrt (log b / log 10)

theorem problem (a b : ℝ) 
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (condition1 : sqrt_log_a a + 2 * sqrt_log_b b + 2 * log (sqrt a) / log 10 + log (sqrt b) / log 10 = 150)
  (int_sqrt_log_a : ∃ (m : ℕ), sqrt_log_a a = m)
  (int_sqrt_log_b : ∃ (n : ℕ), sqrt_log_b b = n)
  (condition2 : a^2 * b = 10^81) :
  a * b = 10^85 :=
sorry

end NUMINAMATH_GPT_problem_l1941_194116


namespace NUMINAMATH_GPT_cos_seven_pi_over_six_l1941_194183

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_seven_pi_over_six_l1941_194183


namespace NUMINAMATH_GPT_ratio_of_numbers_l1941_194154

theorem ratio_of_numbers (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) (h₄ : a + b = 7 * (a - b) + 14) : a / b = 4 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_numbers_l1941_194154


namespace NUMINAMATH_GPT_find_factor_l1941_194100

theorem find_factor (x f : ℝ) (h1 : x = 6)
    (h2 : (2 * x + 9) * f = 63) : f = 3 :=
sorry

end NUMINAMATH_GPT_find_factor_l1941_194100


namespace NUMINAMATH_GPT_arithmetic_sequence_difference_l1941_194128

theorem arithmetic_sequence_difference 
  (a b c : ℝ) 
  (h1: 2 + (7 / 4) = a)
  (h2: 2 + 2 * (7 / 4) = b)
  (h3: 2 + 3 * (7 / 4) = c)
  (h4: 2 + 4 * (7 / 4) = 9):
  c - a = 3.5 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_difference_l1941_194128


namespace NUMINAMATH_GPT_distance_ratio_l1941_194182

variables (KD DM : ℝ)

theorem distance_ratio : 
  KD = 4 ∧ (KD + DM + DM + KD = 12) → (KD / DM = 2) := 
by
  sorry

end NUMINAMATH_GPT_distance_ratio_l1941_194182


namespace NUMINAMATH_GPT_algebraic_expression_value_l1941_194184

theorem algebraic_expression_value (a : ℝ) (h : (a^2 - 3) * (a^2 + 1) = 0) : a^2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1941_194184


namespace NUMINAMATH_GPT_abs_eq_1_solution_set_l1941_194161

theorem abs_eq_1_solution_set (x : ℝ) : (|x| + |x + 1| = 1) ↔ (x ∈ Set.Icc (-1 : ℝ) 0) := by
  sorry

end NUMINAMATH_GPT_abs_eq_1_solution_set_l1941_194161


namespace NUMINAMATH_GPT_number_of_bouncy_balls_per_package_l1941_194194

theorem number_of_bouncy_balls_per_package (x : ℕ) (h : 4 * x + 8 * x + 4 * x = 160) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_bouncy_balls_per_package_l1941_194194


namespace NUMINAMATH_GPT_total_spent_in_may_l1941_194130

-- Conditions as definitions
def cost_per_weekday : ℕ := (2 * 15) + (2 * 18)
def cost_per_weekend_day : ℕ := (3 * 12) + (2 * 20)
def weekdays_in_may : ℕ := 22
def weekend_days_in_may : ℕ := 9

-- The statement to prove
theorem total_spent_in_may :
  cost_per_weekday * weekdays_in_may + cost_per_weekend_day * weekend_days_in_may = 2136 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_in_may_l1941_194130


namespace NUMINAMATH_GPT_correct_option_l1941_194113

-- Definitions based on the problem's conditions
def option_A (x : ℝ) : Prop := x^2 * x^4 = x^8
def option_B (x : ℝ) : Prop := (x^2)^3 = x^5
def option_C (x : ℝ) : Prop := x^2 + x^2 = 2 * x^2
def option_D (x : ℝ) : Prop := (3 * x)^2 = 3 * x^2

-- Theorem stating that out of the given options, option C is correct
theorem correct_option (x : ℝ) : option_C x :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_option_l1941_194113


namespace NUMINAMATH_GPT_area_to_paint_l1941_194199

def wall_height : ℕ := 10
def wall_length : ℕ := 15
def bookshelf_height : ℕ := 3
def bookshelf_length : ℕ := 5

theorem area_to_paint : (wall_height * wall_length) - (bookshelf_height * bookshelf_length) = 135 :=
by 
  sorry

end NUMINAMATH_GPT_area_to_paint_l1941_194199


namespace NUMINAMATH_GPT_find_m_value_l1941_194134

variable (m : ℝ)

theorem find_m_value (h1 : m^2 - 3 * m = 4)
                     (h2 : m^2 = 5 * m + 6) : m = -1 :=
sorry

end NUMINAMATH_GPT_find_m_value_l1941_194134


namespace NUMINAMATH_GPT_solve_for_x_l1941_194125

theorem solve_for_x (x : ℝ) (h : Real.exp (Real.log 7) = 9 * x + 2) : x = 5 / 9 :=
by {
    -- Proof needs to be filled here
    sorry
}

end NUMINAMATH_GPT_solve_for_x_l1941_194125


namespace NUMINAMATH_GPT_new_average_rent_l1941_194149

theorem new_average_rent 
  (n : ℕ) (h_n : n = 4) 
  (avg_old : ℝ) (h_avg_old : avg_old = 800) 
  (inc_rate : ℝ) (h_inc_rate : inc_rate = 0.16) 
  (old_rent : ℝ) (h_old_rent : old_rent = 1250) 
  (new_rent : ℝ) (h_new_rent : new_rent = old_rent * (1 + inc_rate)) 
  (total_rent_old : ℝ) (h_total_rent_old : total_rent_old = n * avg_old)
  (total_rent_new : ℝ) (h_total_rent_new : total_rent_new = total_rent_old - old_rent + new_rent)
  (avg_new : ℝ) (h_avg_new : avg_new = total_rent_new / n) : 
  avg_new = 850 := 
sorry

end NUMINAMATH_GPT_new_average_rent_l1941_194149


namespace NUMINAMATH_GPT_tangent_addition_l1941_194169

theorem tangent_addition (y : ℝ) (h : Real.tan y = -1) : Real.tan (y + Real.pi / 3) = -1 :=
sorry

end NUMINAMATH_GPT_tangent_addition_l1941_194169


namespace NUMINAMATH_GPT_surface_area_of_cube_l1941_194171

noncomputable def cube_edge_length : ℝ := 20

theorem surface_area_of_cube (edge_length : ℝ) (h : edge_length = cube_edge_length) : 
    6 * edge_length ^ 2 = 2400 :=
by
  rw [h]
  sorry  -- proof placeholder

end NUMINAMATH_GPT_surface_area_of_cube_l1941_194171


namespace NUMINAMATH_GPT_negation_proposition_l1941_194114

theorem negation_proposition :
  (¬ (∀ x : ℝ, x > 0 → x^2 - 3 * x + 2 < 0)) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3 * x + 2 ≥ 0) := 
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1941_194114


namespace NUMINAMATH_GPT_find_special_number_l1941_194166

theorem find_special_number : 
  ∃ n, 
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
  (n % 2 = 1) ∧ 
  (n = 27719) :=
  sorry

end NUMINAMATH_GPT_find_special_number_l1941_194166


namespace NUMINAMATH_GPT_quadratic_has_real_roots_iff_l1941_194132

theorem quadratic_has_real_roots_iff (k : ℝ) : (∃ x : ℝ, x^2 + 2*x - k = 0) ↔ k ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_iff_l1941_194132


namespace NUMINAMATH_GPT_selling_price_equivalence_l1941_194180

noncomputable def cost_price_25_profit : ℝ := 1750 / 1.25
def selling_price_profit := 1520
def selling_price_loss := 1280

theorem selling_price_equivalence
  (cp : ℝ)
  (h1 : cp = cost_price_25_profit)
  (h2 : cp = 1400) :
  (selling_price_profit - cp = cp - selling_price_loss) → (selling_price_loss = 1280) := 
  by
  unfold cost_price_25_profit at h1
  simp [h1] at h2
  sorry

end NUMINAMATH_GPT_selling_price_equivalence_l1941_194180


namespace NUMINAMATH_GPT_at_least_100_valid_pairs_l1941_194197

-- Define the conditions
def boots_distribution (L41 L42 L43 R41 R42 R43 : ℕ) : Prop :=
  L41 + L42 + L43 = 300 ∧ R41 + R42 + R43 = 300 ∧
  (L41 = 200 ∨ L42 = 200 ∨ L43 = 200) ∧
  (R41 = 200 ∨ R42 = 200 ∨ R43 = 200)

-- Define the theorem to be proven
theorem at_least_100_valid_pairs (L41 L42 L43 R41 R42 R43 : ℕ) :
  boots_distribution L41 L42 L43 R41 R42 R43 → 
  (L41 ≥ 100 ∧ R41 ≥ 100 ∨ L42 ≥ 100 ∧ R42 ≥ 100 ∨ L43 ≥ 100 ∧ R43 ≥ 100) → 100 ≤ min L41 R41 ∨ 100 ≤ min L42 R42 ∨ 100 ≤ min L43 R43 :=
  sorry

end NUMINAMATH_GPT_at_least_100_valid_pairs_l1941_194197


namespace NUMINAMATH_GPT_physics_class_size_l1941_194148

theorem physics_class_size (total_students physics_only math_only both : ℕ) 
  (h1 : total_students = 53)
  (h2 : both = 7)
  (h3 : physics_only = 2 * (math_only + both))
  (h4 : total_students = physics_only + math_only + both) :
  physics_only + both = 40 :=
by
  sorry

end NUMINAMATH_GPT_physics_class_size_l1941_194148


namespace NUMINAMATH_GPT_weight_of_new_person_l1941_194195

theorem weight_of_new_person (W : ℝ) : 
  let avg_original := W / 5
  let avg_new := avg_original + 4
  let total_new := 5 * avg_new
  let weight_new_person := total_new - (W - 50)
  weight_new_person = 70 :=
by
  let avg_original := W / 5
  let avg_new := avg_original + 4
  let total_new := 5 * avg_new
  let weight_new_person := total_new - (W - 50)
  have : weight_new_person = 70 := sorry
  exact this

end NUMINAMATH_GPT_weight_of_new_person_l1941_194195


namespace NUMINAMATH_GPT_volume_truncated_cone_l1941_194147

/-- 
Given a truncated right circular cone with a large base radius of 10 cm,
a smaller base radius of 3 cm, and a height of 9 cm, 
prove that the volume of the truncated cone is 417 π cubic centimeters.
-/
theorem volume_truncated_cone :
  let R := 10
  let r := 3
  let h := 9
  let V := (1/3) * Real.pi * h * (R^2 + R*r + r^2)
  V = 417 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_volume_truncated_cone_l1941_194147


namespace NUMINAMATH_GPT_factorize_x4_plus_16_l1941_194106

theorem factorize_x4_plus_16: ∀ (x : ℝ), x^4 + 16 = (x^2 + 2 * x + 2) * (x^2 - 2 * x + 2) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factorize_x4_plus_16_l1941_194106


namespace NUMINAMATH_GPT_sum_of_bases_l1941_194107

theorem sum_of_bases (S₁ S₂ G₁ G₂ : ℚ)
  (h₁ : G₁ = 4 * S₁ / (S₁^2 - 1) + 8 / (S₁^2 - 1))
  (h₂ : G₂ = 8 * S₁ / (S₁^2 - 1) + 4 / (S₁^2 - 1))
  (h₃ : G₁ = 3 * S₂ / (S₂^2 - 1) + 6 / (S₂^2 - 1))
  (h₄ : G₂ = 6 * S₂ / (S₂^2 - 1) + 3 / (S₂^2 - 1)) :
  S₁ + S₂ = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_bases_l1941_194107


namespace NUMINAMATH_GPT_victoria_donuts_cost_l1941_194160

theorem victoria_donuts_cost (n : ℕ) (cost_per_dozen : ℝ) (total_donuts_needed : ℕ) 
  (dozens_needed : ℕ) (actual_total_donuts : ℕ) (total_cost : ℝ) :
  total_donuts_needed ≥ 550 ∧ cost_per_dozen = 7.49 ∧ (total_donuts_needed = 12 * dozens_needed) ∧
  (dozens_needed = Nat.ceil (total_donuts_needed / 12)) ∧ 
  (actual_total_donuts = 12 * dozens_needed) ∧ actual_total_donuts ≥ 550 ∧ 
  (total_cost = dozens_needed * cost_per_dozen) →
  total_cost = 344.54 :=
by
  sorry

end NUMINAMATH_GPT_victoria_donuts_cost_l1941_194160


namespace NUMINAMATH_GPT_numberOfTrucks_l1941_194142

-- Conditions
def numberOfTanksPerTruck : ℕ := 3
def capacityPerTank : ℕ := 150
def totalWaterCapacity : ℕ := 1350

-- Question and proof goal
theorem numberOfTrucks : 
  (totalWaterCapacity / (numberOfTanksPerTruck * capacityPerTank) = 3) := 
by 
  sorry

end NUMINAMATH_GPT_numberOfTrucks_l1941_194142


namespace NUMINAMATH_GPT_triangle_perimeter_l1941_194109

theorem triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 6)
  (c1 c2 : ℝ) (h3 : (c1 - 2) * (c1 - 4) = 0) (h4 : (c2 - 2) * (c2 - 4) = 0) :
  c1 = 2 ∨ c1 = 4 → c2 = 2 ∨ c2 = 4 → 
  (c1 ≠ 2 ∧ c1 = 4 ∨ c2 ≠ 2 ∧ c2 = 4) → 
  (a + b + c1 = 13 ∨ a + b + c2 = 13) :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1941_194109


namespace NUMINAMATH_GPT_smallest_prime_with_prime_digit_sum_l1941_194152

def is_prime (n : ℕ) : Prop := ¬ ∃ m, m ∣ n ∧ 1 < m ∧ m < n

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_prime_digit_sum :
  ∃ p, is_prime p ∧ is_prime (digit_sum p) ∧ 10 < digit_sum p ∧ p = 29 :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_with_prime_digit_sum_l1941_194152


namespace NUMINAMATH_GPT_C_share_per_rs_equals_l1941_194196

-- Definitions based on given conditions
def A_share_per_rs (x : ℝ) : ℝ := x
def B_share_per_rs : ℝ := 0.65
def C_share : ℝ := 48
def total_sum : ℝ := 246

-- The target statement to prove
theorem C_share_per_rs_equals : C_share / total_sum = 0.195122 :=
by
  sorry

end NUMINAMATH_GPT_C_share_per_rs_equals_l1941_194196


namespace NUMINAMATH_GPT_exists_n_prime_factors_m_exp_n_plus_n_exp_m_l1941_194143

theorem exists_n_prime_factors_m_exp_n_plus_n_exp_m (m k : ℕ) (hm : m > 0) (hm_odd : m % 2 = 1) (hk : k > 0) :
  ∃ n : ℕ, n > 0 ∧ (∃ primes : Finset ℕ, primes.card ≥ k ∧ ∀ p ∈ primes, p.Prime ∧ p ∣ m ^ n + n ^ m) := 
sorry

end NUMINAMATH_GPT_exists_n_prime_factors_m_exp_n_plus_n_exp_m_l1941_194143


namespace NUMINAMATH_GPT_concentric_circles_false_statement_l1941_194138

theorem concentric_circles_false_statement
  (a b c : ℝ)
  (h1 : a < b)
  (h2 : b < c) :
  ¬ (b + a = c + b) :=
sorry

end NUMINAMATH_GPT_concentric_circles_false_statement_l1941_194138


namespace NUMINAMATH_GPT_river_flow_speed_eq_l1941_194122

-- Definitions of the given conditions
def ship_speed : ℝ := 30
def distance_downstream : ℝ := 144
def distance_upstream : ℝ := 96

-- Lean 4 statement to prove the condition
theorem river_flow_speed_eq (v : ℝ) :
  (distance_downstream / (ship_speed + v) = distance_upstream / (ship_speed - v)) :=
by { sorry }

end NUMINAMATH_GPT_river_flow_speed_eq_l1941_194122


namespace NUMINAMATH_GPT_smallest_x_for_perfect_cube_l1941_194158

theorem smallest_x_for_perfect_cube (M : ℤ) :
  ∃ x : ℕ, 1680 * x = M^3 ∧ ∀ y : ℕ, 1680 * y = M^3 → 44100 ≤ y := 
sorry

end NUMINAMATH_GPT_smallest_x_for_perfect_cube_l1941_194158


namespace NUMINAMATH_GPT_problem_l1941_194118

theorem problem (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : f 1 = f 3) 
  (h2 : f 1 > f 4) 
  (hf : ∀ x, f x = a * x ^ 2 + b * x + c) :
  a < 0 ∧ 4 * a + b = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1941_194118


namespace NUMINAMATH_GPT_r_minus_s_l1941_194185

-- Define the equation whose roots are r and s
def equation (x : ℝ) := (6 * x - 18) / (x ^ 2 + 4 * x - 21) = x + 3

-- Define the condition that r and s are distinct roots of the equation and r > s
def is_solution_pair (r s : ℝ) :=
  equation r ∧ equation s ∧ r ≠ s ∧ r > s

-- The main theorem we need to prove
theorem r_minus_s (r s : ℝ) (h : is_solution_pair r s) : r - s = 12 :=
by
  sorry

end NUMINAMATH_GPT_r_minus_s_l1941_194185


namespace NUMINAMATH_GPT_greatest_difference_is_124_l1941_194178

-- Define the variables a, b, c, and x
variables (a b c x : ℕ)

-- Define the conditions of the problem
def conditions (a b c : ℕ) := 
  (4 * a = 2 * b) ∧ 
  (4 * a = c) ∧ 
  (a > 0) ∧ 
  (a < 10) ∧ 
  (b < 10) ∧ 
  (c < 10)

-- Define the value of a number given its digits
def number (a b c : ℕ) := 100 * a + 10 * b + c

-- Define the maximum and minimum values of x
def max_val (a : ℕ) := number a (2 * a) (4 * a)
def min_val (a : ℕ) := number a (2 * a) (4 * a)

-- Define the greatest difference
def greatest_difference := max_val 2 - min_val 1

-- Prove that the greatest difference is 124
theorem greatest_difference_is_124 : greatest_difference = 124 :=
by 
  unfold greatest_difference 
  unfold max_val 
  unfold min_val 
  unfold number 
  sorry

end NUMINAMATH_GPT_greatest_difference_is_124_l1941_194178


namespace NUMINAMATH_GPT_pumpkin_pie_degrees_l1941_194198

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

end NUMINAMATH_GPT_pumpkin_pie_degrees_l1941_194198


namespace NUMINAMATH_GPT_inscribed_square_side_length_l1941_194131

theorem inscribed_square_side_length (AC BC : ℝ) (h₀ : AC = 6) (h₁ : BC = 8) :
  ∃ x : ℝ, x = 24 / 7 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_square_side_length_l1941_194131


namespace NUMINAMATH_GPT_lizzie_garbage_l1941_194187

/-- Let G be the amount of garbage Lizzie's group collected. 
We are given that the second group collected G - 39 pounds of garbage,
and the total amount collected by both groups is 735 pounds.
We need to prove that G is 387 pounds. -/
theorem lizzie_garbage (G : ℕ) (h1 : G + (G - 39) = 735) : G = 387 :=
sorry

end NUMINAMATH_GPT_lizzie_garbage_l1941_194187
