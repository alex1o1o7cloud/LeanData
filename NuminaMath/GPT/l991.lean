import Mathlib

namespace NUMINAMATH_GPT_compute_expr_l991_99195

open Real

-- Define the polynomial and its roots.
def polynomial (x : ℝ) := 3 * x^2 - 5 * x - 2

-- Given conditions: p and q are roots of the polynomial.
def is_root (p q : ℝ) : Prop := 
  polynomial p = 0 ∧ polynomial q = 0

-- The main theorem.
theorem compute_expr (p q : ℝ) (h : is_root p q) : 
  ∃ k : ℝ, k = p - q ∧ (p ≠ q) → (9 * p^3 + 9 * q^3) / (p - q) = 215 / (3 * (p - q)) :=
sorry

end NUMINAMATH_GPT_compute_expr_l991_99195


namespace NUMINAMATH_GPT_initial_toys_count_l991_99170

-- Definitions for the conditions
def initial_toys (X : ℕ) : ℕ := X
def lost_toys (X : ℕ) : ℕ := X - 6
def found_toys (X : ℕ) : ℕ := (lost_toys X) + 9
def borrowed_toys (X : ℕ) : ℕ := (found_toys X) + 5
def traded_toys (X : ℕ) : ℕ := (borrowed_toys X) - 3

-- Statement to prove
theorem initial_toys_count (X : ℕ) : traded_toys X = 43 → X = 38 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_initial_toys_count_l991_99170


namespace NUMINAMATH_GPT_no_solution_outside_intervals_l991_99149

theorem no_solution_outside_intervals (x a : ℝ) :
  (a < 0 ∨ a > 10) → 3 * |x + 3 * a| + |x + a^2| + 2 * x ≠ a :=
by {
  sorry
}

end NUMINAMATH_GPT_no_solution_outside_intervals_l991_99149


namespace NUMINAMATH_GPT_volume_frustum_l991_99171

noncomputable def volume_of_frustum (base_edge_original : ℝ) (altitude_original : ℝ) 
(base_edge_smaller : ℝ) (altitude_smaller : ℝ) : ℝ :=
let volume_original := (1 / 3) * (base_edge_original ^ 2) * altitude_original
let volume_smaller := (1 / 3) * (base_edge_smaller ^ 2) * altitude_smaller
(volume_original - volume_smaller)

theorem volume_frustum
  (base_edge_original : ℝ) (altitude_original : ℝ) 
  (base_edge_smaller : ℝ) (altitude_smaller : ℝ)
  (h_base_edge_original : base_edge_original = 10)
  (h_altitude_original : altitude_original = 10)
  (h_base_edge_smaller : base_edge_smaller = 5)
  (h_altitude_smaller : altitude_smaller = 5) :
  volume_of_frustum base_edge_original altitude_original base_edge_smaller altitude_smaller = (875 / 3) :=
by
  rw [h_base_edge_original, h_altitude_original, h_base_edge_smaller, h_altitude_smaller]
  simp [volume_of_frustum]
  sorry

end NUMINAMATH_GPT_volume_frustum_l991_99171


namespace NUMINAMATH_GPT_smallest_value_y_l991_99111

theorem smallest_value_y (y : ℝ) : (|y - 8| = 15) → y = -7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_y_l991_99111


namespace NUMINAMATH_GPT_real_part_of_solution_l991_99192

theorem real_part_of_solution (a b : ℝ) (z : ℂ) (h : z = a + b * Complex.I): 
  z * (z + Complex.I) * (z + 2 * Complex.I) = 1800 * Complex.I → a = 20.75 := by
  sorry

end NUMINAMATH_GPT_real_part_of_solution_l991_99192


namespace NUMINAMATH_GPT_find_s_l991_99139

noncomputable def is_monic (p : Polynomial ℝ) : Prop :=
  p.leadingCoeff = 1

variables (f g : Polynomial ℝ) (s : ℝ)
variables (r1 r2 r3 r4 r5 r6 : ℝ)

-- Conditions
def conditions : Prop :=
  is_monic f ∧ is_monic g ∧
  (f.roots = [s + 2, s + 8, r1] ∨ f.roots = [s + 8, s + 2, r1] ∨ f.roots = [s + 2, r1, s + 8] ∨
   f.roots = [r1, s + 2, s + 8] ∨ f.roots = [r1, s + 8, s + 2]) ∧
  (g.roots = [s + 4, s + 10, r2] ∨ g.roots = [s + 10, s + 4, r2] ∨ g.roots = [s + 4, r2, s + 10] ∨
   g.roots = [r2, s + 4, s + 10] ∨ g.roots = [r2, s + 10, s + 4]) ∧
  ∀ (x : ℝ), f.eval x - g.eval x = 2 * s

-- Theorem statement

theorem find_s (h : conditions f g r1 r2 s) : s = 288 / 14 :=
sorry

end NUMINAMATH_GPT_find_s_l991_99139


namespace NUMINAMATH_GPT_find_x_l991_99151

theorem find_x (x : ℕ) : (x % 6 = 0) ∧ (x^2 > 200) ∧ (x < 30) → (x = 18 ∨ x = 24) :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_x_l991_99151


namespace NUMINAMATH_GPT_units_digit_of_fraction_l991_99125

-- Define the problem
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fraction :
  units_digit ((30 * 31 * 32 * 33 * 34 * 35) / 2500) = 2 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_fraction_l991_99125


namespace NUMINAMATH_GPT_book_price_range_l991_99189

variable (x : ℝ) -- Assuming x is a real number

theorem book_price_range 
    (hA : ¬(x ≥ 20)) 
    (hB : ¬(x ≤ 15)) : 
    15 < x ∧ x < 20 := 
by
  sorry

end NUMINAMATH_GPT_book_price_range_l991_99189


namespace NUMINAMATH_GPT_molecular_weight_8_moles_N2O_l991_99186

-- Definitions for atomic weights and the number of moles
def atomic_weight_N : Float := 14.01
def atomic_weight_O : Float := 16.00
def moles_N2O : Float := 8.0

-- Definition for molecular weight of N2O
def molecular_weight_N2O : Float := 
  (2 * atomic_weight_N) + (1 * atomic_weight_O)

-- Target statement to prove
theorem molecular_weight_8_moles_N2O :
  moles_N2O * molecular_weight_N2O = 352.16 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_8_moles_N2O_l991_99186


namespace NUMINAMATH_GPT_determine_value_of_a_l991_99143

theorem determine_value_of_a :
  ∃ b, (∀ x : ℝ, (4 * x^2 + 12 * x + (b^2)) = (2 * x + b)^2) :=
sorry

end NUMINAMATH_GPT_determine_value_of_a_l991_99143


namespace NUMINAMATH_GPT_evaluate_at_3_l991_99146

def f (x : ℕ) : ℕ := x ^ 2

theorem evaluate_at_3 : f 3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_at_3_l991_99146


namespace NUMINAMATH_GPT_actual_average_height_correct_l991_99167

theorem actual_average_height_correct : 
  (∃ (avg_height : ℚ), avg_height = 181 ) →
  (∃ (num_boys : ℕ), num_boys = 35) →
  (∃ (incorrect_height : ℚ), incorrect_height = 166) →
  (∃ (actual_height : ℚ), actual_height = 106) →
  (179.29 : ℚ) = 
    (round ((6315 + 106 : ℚ) / 35 * 100) / 100 ) :=
by
sorry

end NUMINAMATH_GPT_actual_average_height_correct_l991_99167


namespace NUMINAMATH_GPT_number_of_common_divisors_l991_99173

theorem number_of_common_divisors :
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  let divisors_count := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  gcd_ab = 420 ∧ divisors_count = 24 :=
by
  let a := 9240
  let b := 8820
  let gcd_ab := Nat.gcd a b
  have h1 : gcd_ab = 420 := sorry
  have h2 : (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 24 := by norm_num
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_number_of_common_divisors_l991_99173


namespace NUMINAMATH_GPT_range_of_b_l991_99107

theorem range_of_b (b : ℝ) : 
  (¬ (4 ≤ 3 * 3 + b) ∧ (4 ≤ 3 * 4 + b)) ↔ (-8 ≤ b ∧ b < -5) := 
by
  sorry

end NUMINAMATH_GPT_range_of_b_l991_99107


namespace NUMINAMATH_GPT_vishal_investment_more_than_trishul_l991_99179

theorem vishal_investment_more_than_trishul:
  ∀ (V T R : ℝ),
  R = 2100 →
  T = 0.90 * R →
  V + T + R = 6069 →
  ((V - T) / T) * 100 = 10 :=
by
  intros V T R hR hT hSum
  sorry

end NUMINAMATH_GPT_vishal_investment_more_than_trishul_l991_99179


namespace NUMINAMATH_GPT_original_selling_price_is_440_l991_99153

variable (P : ℝ)

-- Condition: Bill made a profit of 10% by selling a product.
def original_selling_price := 1.10 * P

-- Condition: He had purchased the product for 10% less.
def new_purchase_price := 0.90 * P

-- Condition: With a 30% profit on the new purchase price, the new selling price.
def new_selling_price := 1.17 * P

-- Condition: The new selling price is $28 more than the original selling price.
def price_difference_condition : Prop := new_selling_price P = original_selling_price P + 28

-- Conclusion: The original selling price was \$440
theorem original_selling_price_is_440
    (h : price_difference_condition P) : original_selling_price P = 440 :=
sorry

end NUMINAMATH_GPT_original_selling_price_is_440_l991_99153


namespace NUMINAMATH_GPT_cannot_form_3x3_square_l991_99137

def square_pieces (squares : ℕ) (rectangles : ℕ) (triangles : ℕ) := 
  squares = 4 ∧ rectangles = 1 ∧ triangles = 1

def area (squares : ℕ) (rectangles : ℕ) (triangles : ℕ) : ℕ := 
  squares * 1 * 1 + rectangles * 2 * 1 + triangles * (1 * 1 / 2)

theorem cannot_form_3x3_square : 
  ∀ squares rectangles triangles, 
  square_pieces squares rectangles triangles → 
  area squares rectangles triangles < 9 := by
  intros squares rectangles triangles h
  unfold square_pieces at h
  unfold area
  sorry

end NUMINAMATH_GPT_cannot_form_3x3_square_l991_99137


namespace NUMINAMATH_GPT_obtain_1_after_3_operations_obtain_1_after_4_operations_obtain_1_after_5_operations_l991_99155

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 3

theorem obtain_1_after_3_operations:
  (operation (operation (operation 1)) = 1) ∨ 
  (operation (operation (operation 8)) = 1) := by
  sorry

theorem obtain_1_after_4_operations:
  (operation (operation (operation (operation 1))) = 1) ∨ 
  (operation (operation (operation (operation 5))) = 1) ∨ 
  (operation (operation (operation (operation 16))) = 1) := by
  sorry

theorem obtain_1_after_5_operations:
  (operation (operation (operation (operation (operation 4)))) = 1) ∨ 
  (operation (operation (operation (operation (operation 10)))) = 1) ∨ 
  (operation (operation (operation (operation (operation 13)))) = 1) := by
  sorry

end NUMINAMATH_GPT_obtain_1_after_3_operations_obtain_1_after_4_operations_obtain_1_after_5_operations_l991_99155


namespace NUMINAMATH_GPT_geom_seq_general_formula_sum_first_n_terms_formula_l991_99118

namespace GeometricArithmeticSequences

def geom_seq_general (a_n : ℕ → ℝ) (n : ℕ) : Prop :=
  a_n 1 = 1 ∧ (2 * a_n 3 = a_n 2) → a_n n = 1 / (2 ^ (n - 1))

def sum_first_n_terms (a_n b_n : ℕ → ℝ) (S_n T_n : ℕ → ℝ) (n : ℕ) : Prop :=
  b_n 1 = 2 ∧ S_n 3 = b_n 2 + 6 → 
  T_n n = 6 - (n + 3) / (2 ^ (n - 1))

theorem geom_seq_general_formula :
  ∀ a_n : ℕ → ℝ, ∀ n : ℕ, geom_seq_general a_n n :=
by sorry

theorem sum_first_n_terms_formula :
  ∀ a_n b_n : ℕ → ℝ, ∀ S_n T_n : ℕ → ℝ, ∀ n : ℕ, sum_first_n_terms a_n b_n S_n T_n n :=
by sorry

end GeometricArithmeticSequences

end NUMINAMATH_GPT_geom_seq_general_formula_sum_first_n_terms_formula_l991_99118


namespace NUMINAMATH_GPT_prove_ellipse_and_sum_constant_l991_99138

-- Define the ellipse properties
def ellipse_center_origin (a b : ℝ) : Prop :=
  a = 4 ∧ b^2 = 12

-- Standard equation of the ellipse
def ellipse_standard_eqn (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 12) = 1

-- Define the conditions for m and n given point M(1, 3)
def condition_m_n (m n : ℝ) (x0 : ℝ) : Prop :=
  (9 * m^2 + 96 * m + 48 - (13/4) * x0^2 = 0) ∧ (9 * n^2 + 96 * n + 48 - (13/4) * x0^2 = 0)

-- Prove the standard equation of the ellipse and m+n constant properties
theorem prove_ellipse_and_sum_constant (a b x y m n x0 : ℝ) 
  (h1 : ellipse_center_origin a b)
  (h2 : ellipse_standard_eqn x y)
  (h3 : condition_m_n m n x0) :
  m + n = -32/3 := 
sorry

end NUMINAMATH_GPT_prove_ellipse_and_sum_constant_l991_99138


namespace NUMINAMATH_GPT_fly_least_distance_l991_99109

noncomputable def leastDistance (r : ℝ) (h : ℝ) (start_dist : ℝ) (end_dist : ℝ) : ℝ := 
  let C := 2 * Real.pi * r
  let R := Real.sqrt (r^2 + h^2)
  let θ := C / R
  let A := (start_dist, 0)
  let B := (Real.cos (θ / 2) * end_dist, Real.sin (θ / 2) * end_dist)
  Real.sqrt ((B.fst - A.fst)^2 + (B.snd - A.snd)^2)

theorem fly_least_distance : 
  leastDistance 600 (200 * Real.sqrt 7) 125 (375 * Real.sqrt 2) = 625 := 
sorry

end NUMINAMATH_GPT_fly_least_distance_l991_99109


namespace NUMINAMATH_GPT_Creekview_science_fair_l991_99180

/-- Given the total number of students at Creekview High School is 1500,
    900 of these students participate in a science fair, where three-quarters
    of the girls participate and two-thirds of the boys participate,
    prove that 900 girls participate in the science fair. -/
theorem Creekview_science_fair
  (g b : ℕ)
  (h1 : g + b = 1500)
  (h2 : (3 / 4) * g + (2 / 3) * b = 900) :
  (3 / 4) * g = 900 := by
sorry

end NUMINAMATH_GPT_Creekview_science_fair_l991_99180


namespace NUMINAMATH_GPT_combined_area_correct_l991_99169

-- Define the given dimensions and border width
def length : ℝ := 0.6
def width : ℝ := 0.35
def border_width : ℝ := 0.05

-- Define the area of the rectangle, the new dimensions with the border, 
-- and the combined area of the rectangle and the border
def rectangle_area : ℝ := length * width
def new_length : ℝ := length + 2 * border_width
def new_width : ℝ := width + 2 * border_width
def combined_area : ℝ := new_length * new_width

-- The statement we want to prove
theorem combined_area_correct : combined_area = 0.315 := by
  sorry

end NUMINAMATH_GPT_combined_area_correct_l991_99169


namespace NUMINAMATH_GPT_find_ordered_pairs_l991_99110

theorem find_ordered_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m ∣ 3 * n - 2 ∧ 2 * n ∣ 3 * m - 2) ↔ (m, n) = (2, 2) ∨ (m, n) = (10, 14) ∨ (m, n) = (14, 10) :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pairs_l991_99110


namespace NUMINAMATH_GPT_mike_total_games_l991_99131

-- Define the number of games Mike went to this year
def games_this_year : ℕ := 15

-- Define the number of games Mike went to last year
def games_last_year : ℕ := 39

-- Prove the total number of games Mike went to
theorem mike_total_games : games_this_year + games_last_year = 54 :=
by
  sorry

end NUMINAMATH_GPT_mike_total_games_l991_99131


namespace NUMINAMATH_GPT_budget_allocations_and_percentage_changes_l991_99140

theorem budget_allocations_and_percentage_changes (X : ℝ) :
  (14 * X / 100, 24 * X / 100, 15 * X / 100, 19 * X / 100, 8 * X / 100, 20 * X / 100) = 
  (0.14 * X, 0.24 * X, 0.15 * X, 0.19 * X, 0.08 * X, 0.20 * X) ∧
  ((14 - 12) / 12 * 100 = 16.67 ∧
   (24 - 22) / 22 * 100 = 9.09 ∧
   (15 - 13) / 13 * 100 = 15.38 ∧
   (19 - 18) / 18 * 100 = 5.56 ∧
   (8 - 7) / 7 * 100 = 14.29 ∧
   ((20 - (100 - (12 + 22 + 13 + 18 + 7))) / (100 - (12 + 22 + 13 + 18 + 7)) * 100) = -28.57) := by
  sorry

end NUMINAMATH_GPT_budget_allocations_and_percentage_changes_l991_99140


namespace NUMINAMATH_GPT_gcd_increase_by_9_l991_99108

theorem gcd_increase_by_9 (m n d : ℕ) (h1 : d = Nat.gcd m n) (h2 : 9 * d = Nat.gcd (m + 6) n) : d = 3 ∨ d = 6 :=
by
  sorry

end NUMINAMATH_GPT_gcd_increase_by_9_l991_99108


namespace NUMINAMATH_GPT_negation_q_sufficient_not_necessary_negation_p_l991_99106

theorem negation_q_sufficient_not_necessary_negation_p :
  (∃ x : ℝ, (∃ p : 16 - x^2 < 0, (x ∈ [-4, 4]))) →
  (∃ x : ℝ, (∃ q : x^2 + x - 6 > 0, (x ∈ [-3, 2]))) :=
sorry

end NUMINAMATH_GPT_negation_q_sufficient_not_necessary_negation_p_l991_99106


namespace NUMINAMATH_GPT_count_integer_values_l991_99114

-- Statement of the problem in Lean 4
theorem count_integer_values (x : ℤ) : 
  (7 * x^2 + 23 * x + 20 ≤ 30) → 
  ∃ (n : ℕ), n = 6 :=
sorry

end NUMINAMATH_GPT_count_integer_values_l991_99114


namespace NUMINAMATH_GPT_johns_umbrellas_in_house_l991_99162

-- Definitions based on the conditions
def umbrella_cost : Nat := 8
def total_amount_paid : Nat := 24
def umbrella_in_car : Nat := 1

-- The goal is to prove that the number of umbrellas in John's house is 2
theorem johns_umbrellas_in_house : 
  (total_amount_paid / umbrella_cost) - umbrella_in_car = 2 :=
by sorry

end NUMINAMATH_GPT_johns_umbrellas_in_house_l991_99162


namespace NUMINAMATH_GPT_saleswoman_commission_l991_99117

theorem saleswoman_commission (x : ℝ) (h1 : ∀ sale : ℝ, sale = 800) (h2 : (x / 100) * 500 + 0.25 * (800 - 500) = 0.21875 * 800) : x = 20 := by
  sorry

end NUMINAMATH_GPT_saleswoman_commission_l991_99117


namespace NUMINAMATH_GPT_plane_division_99_lines_l991_99141

theorem plane_division_99_lines (m : ℕ) (n : ℕ) : 
  m = 99 ∧ n < 199 → (n = 100 ∨ n = 198) :=
by 
  sorry

end NUMINAMATH_GPT_plane_division_99_lines_l991_99141


namespace NUMINAMATH_GPT_inspection_arrangements_l991_99101

-- Definitions based on conditions
def liberal_arts_classes : ℕ := 2
def science_classes : ℕ := 3
def num_students (classes : ℕ) : ℕ := classes

-- Main theorem statement
theorem inspection_arrangements (liberal_arts_classes science_classes : ℕ)
  (h1: liberal_arts_classes = 2) (h2: science_classes = 3) : 
  num_students liberal_arts_classes * num_students science_classes = 24 :=
by {
  -- Given there are 2 liberal arts classes and 3 science classes,
  -- there are exactly 24 ways to arrange the inspections as per the conditions provided.
  sorry
}

end NUMINAMATH_GPT_inspection_arrangements_l991_99101


namespace NUMINAMATH_GPT_time_for_one_kid_to_wash_six_whiteboards_l991_99163

-- Define the conditions as a function
def time_taken (k : ℕ) (w : ℕ) : ℕ := 20 * 4 * w / k

theorem time_for_one_kid_to_wash_six_whiteboards :
  time_taken 1 6 = 160 := by
-- Proof omitted
sorry

end NUMINAMATH_GPT_time_for_one_kid_to_wash_six_whiteboards_l991_99163


namespace NUMINAMATH_GPT_gcd_exponentiation_l991_99129

def m : ℕ := 2^2050 - 1
def n : ℕ := 2^2040 - 1

theorem gcd_exponentiation : Nat.gcd m n = 1023 := by
  sorry

end NUMINAMATH_GPT_gcd_exponentiation_l991_99129


namespace NUMINAMATH_GPT_count_triangles_in_figure_l991_99121

-- Define the structure of the grid with the given properties.
def grid_structure : Prop :=
  ∃ (n1 n2 n3 n4 : ℕ), 
  n1 = 3 ∧  -- First row: 3 small triangles
  n2 = 2 ∧  -- Second row: 2 small triangles
  n3 = 1 ∧  -- Third row: 1 small triangle
  n4 = 1    -- 1 large inverted triangle

-- The problem statement
theorem count_triangles_in_figure (h : grid_structure) : 
  ∃ (total_triangles : ℕ), total_triangles = 9 :=
sorry

end NUMINAMATH_GPT_count_triangles_in_figure_l991_99121


namespace NUMINAMATH_GPT_number_of_spinsters_l991_99126

-- Given conditions
variables (S C : ℕ)
axiom ratio_condition : S / C = 2 / 9
axiom difference_condition : C = S + 63

-- Theorem to prove
theorem number_of_spinsters : S = 18 :=
sorry

end NUMINAMATH_GPT_number_of_spinsters_l991_99126


namespace NUMINAMATH_GPT_recipe_calls_for_nine_cups_of_flour_l991_99194

def cups_of_flour (x : ℕ) := 
  ∃ cups_added_sugar : ℕ, 
    cups_added_sugar = (6 - 4) ∧ 
    x = cups_added_sugar + 7

theorem recipe_calls_for_nine_cups_of_flour : cups_of_flour 9 :=
by
  sorry

end NUMINAMATH_GPT_recipe_calls_for_nine_cups_of_flour_l991_99194


namespace NUMINAMATH_GPT_tan_of_acute_angle_l991_99105

theorem tan_of_acute_angle (α : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : Real.cos (π / 2 + α) = -3/5) : Real.tan α = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_of_acute_angle_l991_99105


namespace NUMINAMATH_GPT_find_m_value_l991_99148

variable (m a0 a1 a2 a3 a4 a5 : ℚ)

-- Defining the conditions given in the problem
def poly_expansion_condition : Prop := (m * 1 - 1)^5 = a5 * 1^5 + a4 * 1^4 + a3 * 1^3 + a2 * 1^2 + a1 * 1 + a0
def a1_a2_a3_a4_a5_condition : Prop := a1 + a2 + a3 + a4 + a5 = 33

-- We are required to prove that given these conditions, m = 3.
theorem find_m_value (h1 : a0 = -1) (h2 : poly_expansion_condition m a0 a1 a2 a3 a4 a5) 
(h3 : a1_a2_a3_a4_a5_condition a1 a2 a3 a4 a5) : m = 3 := by
  sorry

end NUMINAMATH_GPT_find_m_value_l991_99148


namespace NUMINAMATH_GPT_central_angle_of_regular_hexagon_l991_99102

-- Define the total degrees in a circle
def total_degrees_in_circle : ℝ := 360

-- Define the number of sides in a regular hexagon
def sides_in_hexagon : ℕ := 6

-- Theorems to prove that the central angle of a regular hexagon is 60°
theorem central_angle_of_regular_hexagon :
  total_degrees_in_circle / sides_in_hexagon = 60 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_regular_hexagon_l991_99102


namespace NUMINAMATH_GPT_water_speed_l991_99175

theorem water_speed (swim_speed : ℝ) (time : ℝ) (distance : ℝ) (v : ℝ) 
  (h1: swim_speed = 10) (h2: time = 2) (h3: distance = 12) 
  (h4: distance = (swim_speed - v) * time) : 
  v = 4 :=
by
  sorry

end NUMINAMATH_GPT_water_speed_l991_99175


namespace NUMINAMATH_GPT_convex_polygon_obtuse_sum_l991_99190
open Int

def convex_polygon_sides (n : ℕ) (S : ℕ) : Prop :=
  180 * (n - 2) = 3000 + S ∧ (S = 60 ∨ S = 240)

theorem convex_polygon_obtuse_sum (n : ℕ) (hn : 3 ≤ n) :
  (∃ S, convex_polygon_sides n S) ↔ (n = 19 ∨ n = 20) :=
by
  sorry

end NUMINAMATH_GPT_convex_polygon_obtuse_sum_l991_99190


namespace NUMINAMATH_GPT_truck_speed_in_mph_l991_99156

-- Definitions based on the conditions
def truck_length : ℝ := 66  -- Truck length in feet
def tunnel_length : ℝ := 330  -- Tunnel length in feet
def exit_time : ℝ := 6  -- Exit time in seconds
def feet_to_miles : ℝ := 5280  -- Feet per mile

-- Problem statement
theorem truck_speed_in_mph :
  ((tunnel_length + truck_length) / exit_time) * (3600 / feet_to_miles) = 45 := 
sorry

end NUMINAMATH_GPT_truck_speed_in_mph_l991_99156


namespace NUMINAMATH_GPT_problem1_problem2_l991_99184

theorem problem1 : -20 + 3 + 5 - 7 = -19 := by
  sorry

theorem problem2 : (-3)^2 * 5 + (-2)^3 / 4 - |-3| = 40 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l991_99184


namespace NUMINAMATH_GPT_employee_payment_l991_99113

theorem employee_payment (X Y : ℝ) (h1 : X + Y = 528) (h2 : X = 1.2 * Y) : Y = 240 :=
by
  sorry

end NUMINAMATH_GPT_employee_payment_l991_99113


namespace NUMINAMATH_GPT_marvin_number_is_correct_l991_99185

theorem marvin_number_is_correct (y : ℤ) (h : y - 5 = 95) : y + 5 = 105 := by
  sorry

end NUMINAMATH_GPT_marvin_number_is_correct_l991_99185


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l991_99164

open Set

-- Define the universal set U
def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 10}

-- Define sets A, B, and C
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

-- Problem Statements
theorem problem_1 : A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10} := by
  sorry

theorem problem_2 : (A ∩ B) ∩ C = ∅ := by
  sorry

theorem problem_3 : (U \ A) ∩ (U \ B) = {0, 3} := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l991_99164


namespace NUMINAMATH_GPT_ratio_of_second_to_first_l991_99123

noncomputable def building_heights (H1 H2 H3 : ℝ) : Prop :=
  H1 = 600 ∧ H3 = 3 * (H1 + H2) ∧ H1 + H2 + H3 = 7200

theorem ratio_of_second_to_first (H1 H2 H3 : ℝ) (h : building_heights H1 H2 H3) :
  H1 ≠ 0 → (H2 / H1 = 2) :=
by
  unfold building_heights at h
  rcases h with ⟨h1, h3, h_total⟩
  sorry -- Steps of solving are skipped

end NUMINAMATH_GPT_ratio_of_second_to_first_l991_99123


namespace NUMINAMATH_GPT_range_of_m_l991_99132

-- Defining the conditions p and q
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

-- Defining the main theorem
theorem range_of_m (m : ℝ) : (∀ x : ℝ, q x → p x m) ∧ ¬ (∀ x : ℝ, p x m → q x) ↔ (-1/3 ≤ m ∧ m ≤ 3/2) :=
sorry

end NUMINAMATH_GPT_range_of_m_l991_99132


namespace NUMINAMATH_GPT_area_of_circle_2pi_distance_AB_sqrt6_l991_99178

/- Definition of the circle in polar coordinates -/
def circle_polar := ∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

/- Definition of the line in polar coordinates -/
def line_polar := ∀ θ, ∃ ρ : ℝ, ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0

/- The area of the circle -/
theorem area_of_circle_2pi : 
  (∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) → 
  ∃ A : ℝ, A = 2 * Real.pi :=
by
  intro h
  sorry

/- The distance between two intersection points A and B -/
theorem distance_AB_sqrt6 : 
  (∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) → 
  (∀ θ, ∃ ρ : ℝ, ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0) → 
  ∃ d : ℝ, d = Real.sqrt 6 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_area_of_circle_2pi_distance_AB_sqrt6_l991_99178


namespace NUMINAMATH_GPT_restaurant_cost_l991_99182

section Restaurant
variable (total_people kids adults : ℕ) 
variable (meal_cost : ℕ)
variable (total_cost : ℕ)

def calculate_adults (total_people kids : ℕ) : ℕ := 
  total_people - kids

def calculate_total_cost (adults meal_cost : ℕ) : ℕ :=
  adults * meal_cost

theorem restaurant_cost (total_people kids meal_cost : ℕ) :
  total_people = 13 →
  kids = 9 →
  meal_cost = 7 →
  calculate_adults total_people kids = 4 →
  calculate_total_cost 4 meal_cost = 28 :=
by
  intros
  simp [calculate_adults, calculate_total_cost]
  sorry -- Proof would be added here
end Restaurant

end NUMINAMATH_GPT_restaurant_cost_l991_99182


namespace NUMINAMATH_GPT_trapezoid_base_length_l991_99127

-- Definitions from the conditions
def trapezoid_area (a b h : ℕ) : ℕ := (1 / 2) * (a + b) * h

theorem trapezoid_base_length (b : ℕ) (h : ℕ) (a : ℕ) (A : ℕ) (H_area : A = 222) (H_upper_side : a = 23) (H_height : h = 12) :
  A = trapezoid_area a b h ↔ b = 14 :=
by sorry

end NUMINAMATH_GPT_trapezoid_base_length_l991_99127


namespace NUMINAMATH_GPT_tangent_circles_l991_99183

theorem tangent_circles (a b c : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 = a^2 → (x-b)^2 + (y-c)^2 = a^2) →
    ( (b^2 + c^2) / (a^2) = 4 ) :=
by
  intro h
  have h_dist : (b^2 + c^2) = (2 * a) ^ 2 := sorry
  have h_div : (b^2 + c^2) / (a^2) = 4 := sorry
  exact h_div

end NUMINAMATH_GPT_tangent_circles_l991_99183


namespace NUMINAMATH_GPT_circle_shaded_region_perimeter_l991_99166

theorem circle_shaded_region_perimeter
  (O P Q : Type) [MetricSpace O]
  (r : ℝ) (OP OQ : ℝ) (arc_PQ : ℝ)
  (hOP : OP = 8)
  (hOQ : OQ = 8)
  (h_arc_PQ : arc_PQ = 8 * Real.pi) :
  (OP + OQ + arc_PQ = 16 + 8 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_circle_shaded_region_perimeter_l991_99166


namespace NUMINAMATH_GPT_eval_polynomial_at_4_using_horners_method_l991_99136

noncomputable def polynomial : (x : ℝ) → ℝ :=
  λ x => 3 * x^5 - 2 * x^4 + 5 * x^3 - 2.5 * x^2 + 1.5 * x - 0.7

theorem eval_polynomial_at_4_using_horners_method :
  polynomial 4 = 2845.3 :=
by
  sorry

end NUMINAMATH_GPT_eval_polynomial_at_4_using_horners_method_l991_99136


namespace NUMINAMATH_GPT_blake_change_l991_99116

def cost_oranges : ℕ := 40
def cost_apples : ℕ := 50
def cost_mangoes : ℕ := 60
def initial_money : ℕ := 300

def total_cost : ℕ := cost_oranges + cost_apples + cost_mangoes
def change : ℕ := initial_money - total_cost

theorem blake_change : change = 150 := by
  sorry

end NUMINAMATH_GPT_blake_change_l991_99116


namespace NUMINAMATH_GPT_problem_statement_l991_99157

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^2021 + a^2022 = 2 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l991_99157


namespace NUMINAMATH_GPT_initial_amount_l991_99128

theorem initial_amount (cost_bread cost_butter cost_juice total_remain total_amount : ℕ) :
  cost_bread = 2 →
  cost_butter = 3 →
  cost_juice = 2 * cost_bread →
  total_remain = 6 →
  total_amount = cost_bread + cost_butter + cost_juice + total_remain →
  total_amount = 15 := by
  intros h_bread h_butter h_juice h_remain h_total
  sorry

end NUMINAMATH_GPT_initial_amount_l991_99128


namespace NUMINAMATH_GPT_largest_subset_no_multiples_l991_99168

theorem largest_subset_no_multiples : ∀ (S : Finset ℕ), (S = Finset.range 101) → 
  ∃ (A : Finset ℕ), A ⊆ S ∧ (∀ x ∈ A, ∀ y ∈ A, x ≠ y → ¬(x ∣ y) ∧ ¬(y ∣ x)) ∧ A.card = 50 :=
by
  sorry

end NUMINAMATH_GPT_largest_subset_no_multiples_l991_99168


namespace NUMINAMATH_GPT_fifteen_percent_of_x_is_ninety_l991_99100

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end NUMINAMATH_GPT_fifteen_percent_of_x_is_ninety_l991_99100


namespace NUMINAMATH_GPT_find_a_value_l991_99161

noncomputable def f (x a : ℝ) : ℝ := (x^2 + a) / (x + 1)

def slope_of_tangent_line (a : ℝ) : Prop :=
  (deriv (fun x => f x a) 1) = -1

theorem find_a_value : ∃ a : ℝ, slope_of_tangent_line a ∧ a = 7 := by
  sorry

end NUMINAMATH_GPT_find_a_value_l991_99161


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l991_99199

theorem min_value_of_reciprocal_sum {a b : ℝ} (h : a > 0 ∧ b > 0)
  (h_circle1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (h_circle2 : ∀ x y : ℝ, (x - 2)^2 + (y - 2)^2 = 4)
  (h_common_chord : a + b = 2) :
  (1 / a + 9 / b = 8) := 
sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l991_99199


namespace NUMINAMATH_GPT_structure_burns_in_65_seconds_l991_99124

noncomputable def toothpick_grid_burn_time (m n : ℕ) (toothpicks : ℕ) (burn_time : ℕ) : ℕ :=
  if (m = 3 ∧ n = 5 ∧ toothpicks = 38 ∧ burn_time = 10) then 65 else 0

theorem structure_burns_in_65_seconds : toothpick_grid_burn_time 3 5 38 10 = 65 := by
  sorry

end NUMINAMATH_GPT_structure_burns_in_65_seconds_l991_99124


namespace NUMINAMATH_GPT_determine_all_cards_l991_99144

noncomputable def min_cards_to_determine_positions : ℕ :=
  2

theorem determine_all_cards {k : ℕ} (h : k = min_cards_to_determine_positions) :
  ∀ (placed_cards : ℕ → ℕ × ℕ),
  (∀ n, 1 ≤ n ∧ n ≤ 300 → placed_cards n = placed_cards (n + 1) ∨ placed_cards n + (1, 0) = placed_cards (n + 1) ∨ placed_cards n + (0, 1) = placed_cards (n + 1))
  → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_all_cards_l991_99144


namespace NUMINAMATH_GPT_correct_operation_l991_99172

variable (a b : ℝ)

theorem correct_operation : (-2 * a ^ 2) ^ 2 = 4 * a ^ 4 := by
  sorry

end NUMINAMATH_GPT_correct_operation_l991_99172


namespace NUMINAMATH_GPT_opposite_reciprocal_of_neg_five_l991_99150

theorem opposite_reciprocal_of_neg_five : 
  ∀ x : ℝ, x = -5 → - (1 / x) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_opposite_reciprocal_of_neg_five_l991_99150


namespace NUMINAMATH_GPT_second_car_distance_l991_99181

theorem second_car_distance (x : ℝ) : 
  let d_initial : ℝ := 150
  let d_first_car_initial : ℝ := 25
  let d_right_turn : ℝ := 15
  let d_left_turn : ℝ := 25
  let d_final_gap : ℝ := 65
  (d_initial - x = d_final_gap) → x = 85 := by
  sorry

end NUMINAMATH_GPT_second_car_distance_l991_99181


namespace NUMINAMATH_GPT_beth_total_crayons_l991_99104

theorem beth_total_crayons :
  let packs := 4
  let crayons_per_pack := 10
  let extra_crayons := 6
  packs * crayons_per_pack + extra_crayons = 46 :=
by
  let packs := 4
  let crayons_per_pack := 10
  let extra_crayons := 6
  show packs * crayons_per_pack + extra_crayons = 46
  sorry

end NUMINAMATH_GPT_beth_total_crayons_l991_99104


namespace NUMINAMATH_GPT_prime_addition_equality_l991_99134

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_addition_equality (x y : ℕ)
  (hx : is_prime x)
  (hy : is_prime y)
  (hxy : x < y)
  (hsum : x + y = 36) : 4 * x + y = 51 :=
sorry

end NUMINAMATH_GPT_prime_addition_equality_l991_99134


namespace NUMINAMATH_GPT_symmetric_circle_eq_l991_99135

theorem symmetric_circle_eq (C_1_eq : ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 1)
    (line_eq : ∀ x y : ℝ, x - y - 2 = 0) :
    ∀ x y : ℝ, (x - 1)^2 + y^2 = 1 :=
sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l991_99135


namespace NUMINAMATH_GPT_parabola_normal_intersect_l991_99193

theorem parabola_normal_intersect {x y : ℝ} (h₁ : y = x^2) (A : ℝ × ℝ) (hA : A = (-1, 1)) :
  ∃ B : ℝ × ℝ, B = (1.5, 2.25) ∧ ∀ x : ℝ, (y - 1) = 1/2 * (x + 1) →
  ∀ x : ℝ, y = x^2 ∧ B = (1.5, 2.25) :=
sorry

end NUMINAMATH_GPT_parabola_normal_intersect_l991_99193


namespace NUMINAMATH_GPT_number_of_diagonals_of_nonagon_l991_99142

theorem number_of_diagonals_of_nonagon:
  (9 * (9 - 3)) / 2 = 27 := by
  sorry

end NUMINAMATH_GPT_number_of_diagonals_of_nonagon_l991_99142


namespace NUMINAMATH_GPT_reese_spending_l991_99154

-- Definitions used in Lean 4 statement
variable (S : ℝ := 11000)
variable (M : ℝ := 0.4 * S)
variable (A : ℝ := 1500)
variable (L : ℝ := 2900)

-- Lean 4 verification statement
theorem reese_spending :
  ∃ (P : ℝ), S - (P * S + M + A) = L ∧ P * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_reese_spending_l991_99154


namespace NUMINAMATH_GPT_find_rates_l991_99196

theorem find_rates
  (d b p t_p t_b t_w: ℕ)
  (rp rb rw: ℚ)
  (h1: d = b + 10)
  (h2: b = 3 * p)
  (h3: p = 50)
  (h4: t_p = 4)
  (h5: t_b = 2)
  (h6: t_w = 5)
  (h7: rp = p / t_p)
  (h8: rb = b / t_b)
  (h9: rw = d / t_w):
  rp = 12.5 ∧ rb = 75 ∧ rw = 32 := by
  sorry

end NUMINAMATH_GPT_find_rates_l991_99196


namespace NUMINAMATH_GPT_assignment_statement_meaning_l991_99158

-- Define the meaning of the assignment statement
def is_assignment_statement (s: String) : Prop := s = "Variable = Expression"

-- Define the specific assignment statement we are considering
def assignment_statement : String := "i = i + 1"

-- Define the meaning of the specific assignment statement
def assignment_meaning (s: String) : Prop := s = "Add 1 to the original value of i and then assign it back to i, the value of i increases by 1"

-- The proof statement
theorem assignment_statement_meaning :
  is_assignment_statement "Variable = Expression" → assignment_meaning "i = i + 1" :=
by
  intros
  sorry

end NUMINAMATH_GPT_assignment_statement_meaning_l991_99158


namespace NUMINAMATH_GPT_length_segment_pq_l991_99187

theorem length_segment_pq 
  (P Q R S T : ℝ)
  (h1 : (dist P Q + dist P R + dist P S + dist P T = 67))
  (h2 : (dist Q P + dist Q R + dist Q S + dist Q T = 34)) :
  dist P Q = 11 :=
sorry

end NUMINAMATH_GPT_length_segment_pq_l991_99187


namespace NUMINAMATH_GPT_problem2_l991_99159

theorem problem2 (x y : ℝ) (h1 : x^2 + x * y = 3) (h2 : x * y + y^2 = -2) : 
  2 * x^2 - x * y - 3 * y^2 = 12 := 
by 
  sorry

end NUMINAMATH_GPT_problem2_l991_99159


namespace NUMINAMATH_GPT_triangle_angle_sum_l991_99188

theorem triangle_angle_sum (P Q R : ℝ) (h1 : P + Q = 60) (h2 : P + Q + R = 180) : R = 120 := by
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l991_99188


namespace NUMINAMATH_GPT_coordinates_of_point_P_l991_99177

theorem coordinates_of_point_P 
  (x y : ℝ)
  (h1 : y = x^3 - x)
  (h2 : (3 * x^2 - 1) = 2)
  (h3 : ∀ x y, x + 2 * y = 0 → ∃ m, -1/(m) = 2) :
  (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_point_P_l991_99177


namespace NUMINAMATH_GPT_difference_of_roots_l991_99176

noncomputable def r_and_s (r s : ℝ) : Prop :=
(∃ (r s : ℝ), (r, s) ≠ (s, r) ∧ r > s ∧ (5 * r - 15) / (r ^ 2 + 3 * r - 18) = r + 3
  ∧ (5 * s - 15) / (s ^ 2 + 3 * s - 18) = s + 3)

theorem difference_of_roots (r s : ℝ) (h : r_and_s r s) : r - s = Real.sqrt 29 := by
  sorry

end NUMINAMATH_GPT_difference_of_roots_l991_99176


namespace NUMINAMATH_GPT_customer_C_weight_l991_99133

def weights : List ℕ := [22, 25, 28, 31, 34, 36, 38, 40, 45]

-- Definitions for customer A and B such that customer A's total weight equals twice of customer B's total weight
variable {A B : List ℕ}

-- Condition on weights distribution
def valid_distribution (A B : List ℕ) : Prop :=
  (A.sum = 2 * B.sum) ∧ (A ++ B).sum + 38 = 299

-- Prove the weight of the bag received by customer C
theorem customer_C_weight :
  ∃ (C : ℕ), C ∈ weights ∧ C = 38 := by
  sorry

end NUMINAMATH_GPT_customer_C_weight_l991_99133


namespace NUMINAMATH_GPT_gcd_of_sum_and_fraction_l991_99147

theorem gcd_of_sum_and_fraction (p : ℕ) (a b : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1)
  (hcoprime : Nat.gcd a b = 1) : Nat.gcd (a + b) ((a^p + b^p) / (a + b)) = p := 
sorry

end NUMINAMATH_GPT_gcd_of_sum_and_fraction_l991_99147


namespace NUMINAMATH_GPT_tagged_fish_in_second_catch_l991_99152

theorem tagged_fish_in_second_catch
  (N : ℕ)
  (initial_catch tagged_returned : ℕ)
  (second_catch : ℕ)
  (approximate_pond_fish : ℕ)
  (condition_1 : initial_catch = 60)
  (condition_2 : tagged_returned = 60)
  (condition_3 : second_catch = 60)
  (condition_4 : approximate_pond_fish = 1800) :
  (tagged_returned * second_catch) / approximate_pond_fish = 2 :=
by
  sorry

end NUMINAMATH_GPT_tagged_fish_in_second_catch_l991_99152


namespace NUMINAMATH_GPT_simplify_expr1_simplify_and_evaluate_l991_99174

-- First problem: simplify and prove equality.
theorem simplify_expr1 (a : ℝ) :
  -2 * a^2 + 3 - (3 * a^2 - 6 * a + 1) + 3 = -5 * a^2 + 6 * a + 2 :=
by sorry

-- Second problem: simplify and evaluate under given conditions.
theorem simplify_and_evaluate (x y : ℝ) (h_x : x = -2) (h_y : y = -3) :
  (1 / 2) * x - 2 * (x - (1 / 3) * y^2) + (-(3 / 2) * x + (1 / 3) * y^2) = 15 :=
by sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_and_evaluate_l991_99174


namespace NUMINAMATH_GPT_determine_d_l991_99115

def Q (x d : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

theorem determine_d (d : ℝ) : (∃ d, Q (-2) d = 0) → d = -14 := by
  sorry

end NUMINAMATH_GPT_determine_d_l991_99115


namespace NUMINAMATH_GPT_max_value_seq_l991_99145

noncomputable def a_n (n : ℕ) : ℝ := n / (n^2 + 90)

theorem max_value_seq : ∃ n : ℕ, a_n n = 1 / 19 :=
by
  sorry

end NUMINAMATH_GPT_max_value_seq_l991_99145


namespace NUMINAMATH_GPT_polynomial_square_binomial_l991_99120

-- Define the given polynomial and binomial
def polynomial (x : ℚ) (a : ℚ) : ℚ :=
  25 * x^2 + 40 * x + a

def binomial (x b : ℚ) : ℚ :=
  (5 * x + b)^2

-- Theorem to state the problem
theorem polynomial_square_binomial (a : ℚ) : 
  (∃ b, polynomial x a = binomial x b) ↔ a = 16 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_square_binomial_l991_99120


namespace NUMINAMATH_GPT_expand_x_plus_3y_squared_expand_2x_plus_3y_squared_expand_m3_plus_n5_squared_expand_5x_minus_3y_squared_expand_3m5_minus_4n2_squared_l991_99130

-- Proof for (x + 3y)^2 = x^2 + 6xy + 9y^2
theorem expand_x_plus_3y_squared (x y : ℝ) : 
  (x + 3 * y) ^ 2 = x ^ 2 + 6 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (2x + 3y)^2 = 4x^2 + 12xy + 9y^2
theorem expand_2x_plus_3y_squared (x y : ℝ) : 
  (2 * x + 3 * y) ^ 2 = 4 * x ^ 2 + 12 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (m^3 + n^5)^2 = m^6 + 2m^3n^5 + n^10
theorem expand_m3_plus_n5_squared (m n : ℝ) : 
  (m ^ 3 + n ^ 5) ^ 2 = m ^ 6 + 2 * m ^ 3 * n ^ 5 + n ^ 10 := 
  sorry

-- Proof for (5x - 3y)^2 = 25x^2 - 30xy + 9y^2
theorem expand_5x_minus_3y_squared (x y : ℝ) : 
  (5 * x - 3 * y) ^ 2 = 25 * x ^ 2 - 30 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (3m^5 - 4n^2)^2 = 9m^10 - 24m^5n^2 + 16n^4
theorem expand_3m5_minus_4n2_squared (m n : ℝ) : 
  (3 * m ^ 5 - 4 * n ^ 2) ^ 2 = 9 * m ^ 10 - 24 * m ^ 5 * n ^ 2 + 16 * n ^ 4 := 
  sorry

end NUMINAMATH_GPT_expand_x_plus_3y_squared_expand_2x_plus_3y_squared_expand_m3_plus_n5_squared_expand_5x_minus_3y_squared_expand_3m5_minus_4n2_squared_l991_99130


namespace NUMINAMATH_GPT_john_dimes_l991_99119

theorem john_dimes :
  ∀ (d : ℕ), 
  (4 * 25 + d * 10 + 5) = 135 → (5: ℕ) + (d: ℕ) * 10 + 4 = 4 + 131 + 3*d → d = 3 :=
by
  sorry

end NUMINAMATH_GPT_john_dimes_l991_99119


namespace NUMINAMATH_GPT_product_of_numbers_l991_99165

theorem product_of_numbers (a b : ℕ) (hcf_val lcm_val : ℕ) 
  (h_hcf : Nat.gcd a b = hcf_val) 
  (h_lcm : Nat.lcm a b = lcm_val) 
  (hcf_eq : hcf_val = 33) 
  (lcm_eq : lcm_val = 2574) : 
  a * b = 84942 := 
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l991_99165


namespace NUMINAMATH_GPT_constant_term_exists_l991_99112

theorem constant_term_exists (n : ℕ) (h : n = 6) : 
  (∃ r : ℕ, 2 * n - 3 * r = 0) ∧ 
  (∃ n' r' : ℕ, n' ≠ 6 ∧ 2 * n' - 3 * r' = 0) := by
  sorry

end NUMINAMATH_GPT_constant_term_exists_l991_99112


namespace NUMINAMATH_GPT_solve_for_x_l991_99198

theorem solve_for_x (x : ℝ) 
  (h : 6 * x + 12 * x = 558 - 9 * (x - 4)) : 
  x = 22 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l991_99198


namespace NUMINAMATH_GPT_min_value_four_over_a_plus_nine_over_b_l991_99191

theorem min_value_four_over_a_plus_nine_over_b :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → (∀ x y, x > 0 → y > 0 → x + y ≥ 2 * Real.sqrt (x * y)) →
  (∃ (min_val : ℝ), min_val = (4 / a + 9 / b) ∧ min_val = 25) :=
by
  intros a b ha hb hab am_gm
  sorry

end NUMINAMATH_GPT_min_value_four_over_a_plus_nine_over_b_l991_99191


namespace NUMINAMATH_GPT_complex_modulus_square_l991_99103

open Complex

theorem complex_modulus_square (a b : ℝ) (h : 5 * (a + b * I) + 3 * Complex.abs (a + b * I) = 15 - 16 * I) :
  (Complex.abs (a + b * I))^2 = 256 / 25 :=
by sorry

end NUMINAMATH_GPT_complex_modulus_square_l991_99103


namespace NUMINAMATH_GPT_solid_is_cone_l991_99160

-- Definitions for the conditions
structure Solid where
  front_view : Type
  side_view : Type
  top_view : Type

def is_isosceles_triangle (shape : Type) : Prop := sorry
def is_circle (shape : Type) : Prop := sorry

-- Define the solid based on the given conditions
noncomputable def my_solid : Solid := {
  front_view := sorry,
  side_view := sorry,
  top_view := sorry
}

-- Prove that the solid is a cone given the provided conditions
theorem solid_is_cone (s : Solid) : 
  is_isosceles_triangle s.front_view → 
  is_isosceles_triangle s.side_view → 
  is_circle s.top_view → 
  s = my_solid :=
by
  sorry

end NUMINAMATH_GPT_solid_is_cone_l991_99160


namespace NUMINAMATH_GPT_range_of_2_cos_sq_l991_99122

theorem range_of_2_cos_sq :
  ∀ x : ℝ, 0 ≤ 2 * (Real.cos x) ^ 2 ∧ 2 * (Real.cos x) ^ 2 ≤ 2 :=
by sorry

end NUMINAMATH_GPT_range_of_2_cos_sq_l991_99122


namespace NUMINAMATH_GPT_competition_arrangements_l991_99197

noncomputable def count_arrangements (students : Fin 4) (events : Fin 3) : Nat :=
  -- The actual counting function is not implemented
  sorry

theorem competition_arrangements (students : Fin 4) (events : Fin 3) :
  let count := count_arrangements students events
  (∃ (A B C D : Fin 4), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
    B ≠ C ∧ B ≠ D ∧ 
    C ≠ D ∧ 
    (A ≠ 0) ∧ 
    count = 24) := sorry

end NUMINAMATH_GPT_competition_arrangements_l991_99197
