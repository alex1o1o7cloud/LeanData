import Mathlib

namespace NUMINAMATH_CALUDE_first_pumpkin_weight_l2676_267674

/-- The weight of the first pumpkin given the total weight of two pumpkins and the weight of the second pumpkin -/
theorem first_pumpkin_weight (total_weight second_weight : ℝ) 
  (h1 : total_weight = 12.7)
  (h2 : second_weight = 8.7) : 
  total_weight - second_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_pumpkin_weight_l2676_267674


namespace NUMINAMATH_CALUDE_four_pockets_sixteen_coins_l2676_267669

/-- The total number of coins in multiple pockets -/
def total_coins (num_pockets : ℕ) (coins_per_pocket : ℕ) : ℕ :=
  num_pockets * coins_per_pocket

/-- Theorem: Given 4 pockets with 16 coins each, the total number of coins is 64 -/
theorem four_pockets_sixteen_coins : total_coins 4 16 = 64 := by
  sorry

end NUMINAMATH_CALUDE_four_pockets_sixteen_coins_l2676_267669


namespace NUMINAMATH_CALUDE_purple_pairs_coincide_l2676_267629

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  yellow : ℕ
  green : ℕ
  purple : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  yellow_yellow : ℕ
  green_green : ℕ
  yellow_purple : ℕ
  purple_purple : ℕ

/-- The main theorem to prove -/
theorem purple_pairs_coincide 
  (counts : TriangleCounts)
  (pairs : CoincidingPairs)
  (h1 : counts.yellow = 4)
  (h2 : counts.green = 6)
  (h3 : counts.purple = 10)
  (h4 : pairs.yellow_yellow = 3)
  (h5 : pairs.green_green = 4)
  (h6 : pairs.yellow_purple = 3) :
  pairs.purple_purple = 5 := by
  sorry

end NUMINAMATH_CALUDE_purple_pairs_coincide_l2676_267629


namespace NUMINAMATH_CALUDE_amp_composition_l2676_267634

-- Define the operations
def amp (x : ℤ) : ℤ := 10 - x
def amp_prefix (x : ℤ) : ℤ := x - 10

-- State the theorem
theorem amp_composition : amp_prefix (amp 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_amp_composition_l2676_267634


namespace NUMINAMATH_CALUDE_smallest_rectangle_containing_circle_l2676_267639

/-- The diameter of the circle -/
def circle_diameter : ℝ := 10

/-- The area of the smallest rectangle containing the circle -/
def smallest_rectangle_area : ℝ := 120

/-- Theorem stating that the area of the smallest rectangle containing a circle
    with diameter 10 units is 120 square units -/
theorem smallest_rectangle_containing_circle :
  smallest_rectangle_area = circle_diameter * (circle_diameter + 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_rectangle_containing_circle_l2676_267639


namespace NUMINAMATH_CALUDE_expected_girls_left_10_7_l2676_267604

/-- The expected number of girls standing to the left of all boys in a random arrangement -/
def expected_girls_left (num_boys num_girls : ℕ) : ℚ :=
  num_girls / (num_boys + 1)

/-- Theorem: In a random arrangement of 10 boys and 7 girls, 
    the expected number of girls standing to the left of all boys is 7/11 -/
theorem expected_girls_left_10_7 : 
  expected_girls_left 10 7 = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expected_girls_left_10_7_l2676_267604


namespace NUMINAMATH_CALUDE_residue_products_l2676_267671

theorem residue_products (m k : ℕ) (hm : m > 0) (hk : k > 0) :
  (Nat.gcd m k = 1 →
    ∃ (a : Fin m → ℤ) (b : Fin k → ℤ),
      ∀ (i j i' j' : ℕ) (hi : i < m) (hj : j < k) (hi' : i' < m) (hj' : j' < k),
        (i ≠ i' ∨ j ≠ j') →
        (a ⟨i, hi⟩ * b ⟨j, hj⟩) % (m * k) ≠ (a ⟨i', hi'⟩ * b ⟨j', hj'⟩) % (m * k)) ∧
  (Nat.gcd m k > 1 →
    ∀ (a : Fin m → ℤ) (b : Fin k → ℤ),
      ∃ (i j i' j' : ℕ) (hi : i < m) (hj : j < k) (hi' : i' < m) (hj' : j' < k),
        (i ≠ i' ∨ j ≠ j') ∧
        (a ⟨i, hi⟩ * b ⟨j, hj⟩) % (m * k) = (a ⟨i', hi'⟩ * b ⟨j', hj'⟩) % (m * k)) :=
by sorry

end NUMINAMATH_CALUDE_residue_products_l2676_267671


namespace NUMINAMATH_CALUDE_intersection_point_l2676_267618

theorem intersection_point (x y : ℚ) : 
  (8 * x - 5 * y = 10) ∧ (6 * x + 2 * y = 22) ↔ (x = 65/23 ∧ y = -137/23) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l2676_267618


namespace NUMINAMATH_CALUDE_expression_evaluation_l2676_267601

theorem expression_evaluation :
  let x : ℝ := (Real.pi - 3) ^ 0
  let y : ℝ := (-1/3)⁻¹
  ((2*x - y)^2 - (y + 2*x) * (y - 2*x)) / (-1/2 * x) = -40 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2676_267601


namespace NUMINAMATH_CALUDE_gary_gold_amount_l2676_267694

/-- Proves that Gary has 30 grams of gold given the conditions of the problem -/
theorem gary_gold_amount (gary_cost_per_gram : ℝ) (anna_amount : ℝ) (anna_cost_per_gram : ℝ) (total_cost : ℝ)
  (h1 : gary_cost_per_gram = 15)
  (h2 : anna_amount = 50)
  (h3 : anna_cost_per_gram = 20)
  (h4 : total_cost = 1450)
  (h5 : gary_cost_per_gram * gary_amount + anna_amount * anna_cost_per_gram = total_cost) :
  gary_amount = 30 := by
  sorry

end NUMINAMATH_CALUDE_gary_gold_amount_l2676_267694


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l2676_267698

/-- Vovochka's addition method for three-digit numbers -/
def vovochkaAdd (a b c d e f : ℕ) : ℕ :=
  1000 * (a + d) + 100 * (b + e) + (c + f)

/-- Regular addition for three-digit numbers -/
def regularAdd (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The difference between Vovochka's addition and regular addition -/
def addDifference (a b c d e f : ℕ) : ℤ :=
  (vovochkaAdd a b c d e f : ℤ) - (regularAdd a b c d e f : ℤ)

/-- Theorem: The smallest positive difference between Vovochka's addition and regular addition is 1800 -/
theorem smallest_positive_difference :
  ∃ (a b c d e f : ℕ),
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10) ∧
    (a + d > 0) ∧
    (addDifference a b c d e f > 0) ∧
    (∀ (x y z u v w : ℕ),
      (x < 10 ∧ y < 10 ∧ z < 10 ∧ u < 10 ∧ v < 10 ∧ w < 10) →
      (x + u > 0) →
      (addDifference x y z u v w > 0) →
      (addDifference a b c d e f ≤ addDifference x y z u v w)) ∧
    (addDifference a b c d e f = 1800) :=
  sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l2676_267698


namespace NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l2676_267642

/-- Represents the process of drawing segments as described in the problem -/
structure SegmentDrawing where
  initial_segment : Unit  -- Represents the initial segment OA
  branch_factor : Nat     -- Number of segments drawn from each point (5 in this case)
  free_ends : Nat         -- Number of free ends

/-- Calculates the number of free ends after k iterations of drawing segments -/
def free_ends_after_iterations (k : Nat) : Nat :=
  1 + 4 * k

/-- Theorem stating that it's possible to have exactly 1001 free ends -/
theorem exists_k_for_1001_free_ends :
  ∃ k : Nat, free_ends_after_iterations k = 1001 := by
  sorry

#check exists_k_for_1001_free_ends

end NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l2676_267642


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2676_267686

theorem polynomial_coefficient_sum : 
  ∀ (A B C D E : ℝ), 
  (∀ x : ℝ, (2*x + 3)*(4*x^3 - 2*x^2 + x - 7) = A*x^4 + B*x^3 + C*x^2 + D*x + E) →
  A + B + C + D + E = -20 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2676_267686


namespace NUMINAMATH_CALUDE_prescription_rebate_calculation_l2676_267693

/-- Calculates the mail-in rebate amount for a prescription purchase -/
def calculate_rebate (original_cost cashback_percent final_cost : ℚ) : ℚ :=
  let cashback := original_cost * (cashback_percent / 100)
  let cost_after_cashback := original_cost - cashback
  cost_after_cashback - final_cost

theorem prescription_rebate_calculation :
  let original_cost : ℚ := 150
  let cashback_percent : ℚ := 10
  let final_cost : ℚ := 110
  calculate_rebate original_cost cashback_percent final_cost = 25 := by
  sorry

#eval calculate_rebate 150 10 110

end NUMINAMATH_CALUDE_prescription_rebate_calculation_l2676_267693


namespace NUMINAMATH_CALUDE_raine_initial_payment_l2676_267621

/-- The price of a bracelet in dollars -/
def bracelet_price : ℕ := 15

/-- The price of a gold heart necklace in dollars -/
def necklace_price : ℕ := 10

/-- The price of a personalized coffee mug in dollars -/
def mug_price : ℕ := 20

/-- The number of bracelets Raine bought -/
def bracelets_bought : ℕ := 3

/-- The number of gold heart necklaces Raine bought -/
def necklaces_bought : ℕ := 2

/-- The number of personalized coffee mugs Raine bought -/
def mugs_bought : ℕ := 1

/-- The amount of change Raine received in dollars -/
def change_received : ℕ := 15

/-- The theorem stating the amount Raine initially gave -/
theorem raine_initial_payment : 
  bracelet_price * bracelets_bought + 
  necklace_price * necklaces_bought + 
  mug_price * mugs_bought + 
  change_received = 100 := by
  sorry

end NUMINAMATH_CALUDE_raine_initial_payment_l2676_267621


namespace NUMINAMATH_CALUDE_book_price_adjustment_l2676_267684

theorem book_price_adjustment (x : ℝ) :
  (1 + x / 100) * (1 - x / 100) = 0.75 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_price_adjustment_l2676_267684


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l2676_267685

/-- Time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120) 
  (h2 : train_speed_kmh = 60) 
  (h3 : bridge_length = 170) : 
  ∃ (time : ℝ), abs (time - 17.40) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l2676_267685


namespace NUMINAMATH_CALUDE_inequality_proof_l2676_267667

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : y * z + z * x + x * y = 1) : 
  x * (1 - y)^2 * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ 4 / (9 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2676_267667


namespace NUMINAMATH_CALUDE_three_primes_in_list_l2676_267638

def number_list : List Nat := [11, 12, 13, 14, 15, 16, 17]

theorem three_primes_in_list :
  (number_list.filter Nat.Prime).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_primes_in_list_l2676_267638


namespace NUMINAMATH_CALUDE_total_books_read_formula_l2676_267663

/-- The total number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month := 5
  let months_per_year := 12
  let books_per_student_per_year := books_per_month * months_per_year
  books_per_student_per_year * c * s

/-- Theorem stating the total number of books read by the entire student body in one year -/
theorem total_books_read_formula (c s : ℕ) :
  total_books_read c s = 60 * c * s :=
by sorry

end NUMINAMATH_CALUDE_total_books_read_formula_l2676_267663


namespace NUMINAMATH_CALUDE_slope_MN_constant_l2676_267632

/-- Definition of curve C -/
def curve_C (x y : ℝ) : Prop := y^2 = 4*x + 4 ∧ x ≥ 0

/-- Definition of point D on curve C -/
def point_D : ℝ × ℝ := (0, 2)

/-- Definition of complementary slopes -/
def complementary_slopes (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1

/-- Theorem: The slope of line MN is constant and equal to -1 -/
theorem slope_MN_constant (k : ℝ) (M N : ℝ × ℝ) :
  curve_C M.1 M.2 →
  curve_C N.1 N.2 →
  curve_C point_D.1 point_D.2 →
  complementary_slopes k (-k) →
  (M.2 - point_D.2) = k * (M.1 - point_D.1) →
  (N.2 - point_D.2) = (-k) * (N.1 - point_D.1) →
  M ≠ point_D →
  N ≠ point_D →
  (N.2 - M.2) / (N.1 - M.1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_slope_MN_constant_l2676_267632


namespace NUMINAMATH_CALUDE_divide_100_by_0_25_l2676_267611

theorem divide_100_by_0_25 : (100 : ℝ) / 0.25 = 400 := by
  sorry

end NUMINAMATH_CALUDE_divide_100_by_0_25_l2676_267611


namespace NUMINAMATH_CALUDE_bakers_ovens_l2676_267665

/-- Baker's bread production problem -/
theorem bakers_ovens :
  let loaves_per_hour_per_oven : ℕ := 5
  let weekday_hours : ℕ := 5
  let weekday_count : ℕ := 5
  let weekend_hours : ℕ := 2
  let weekend_count : ℕ := 2
  let weeks : ℕ := 3
  let total_loaves : ℕ := 1740
  
  let weekly_hours := weekday_hours * weekday_count + weekend_hours * weekend_count
  let weekly_loaves_per_oven := weekly_hours * loaves_per_hour_per_oven
  let total_loaves_per_oven := weekly_loaves_per_oven * weeks
  
  total_loaves / total_loaves_per_oven = 4 := by
  sorry


end NUMINAMATH_CALUDE_bakers_ovens_l2676_267665


namespace NUMINAMATH_CALUDE_convergence_of_iterative_process_l2676_267670

theorem convergence_of_iterative_process (a b : ℝ) (h : a > b) :
  ∃ k : ℕ, 2^(-k : ℤ) * (a - b) < (1 : ℝ) / 2002 := by
  sorry

end NUMINAMATH_CALUDE_convergence_of_iterative_process_l2676_267670


namespace NUMINAMATH_CALUDE_tangent_line_of_conic_section_l2676_267641

/-- Conic section equation -/
def ConicSection (A B C D E F : ℝ) (x y : ℝ) : Prop :=
  A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0

/-- Tangent line equation -/
def TangentLine (A B C D E F x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  2 * A * x₀ * x + B * (x₀ * y + x * y₀) + 2 * C * y₀ * y + 
  D * (x₀ + x) + E * (y₀ + y) + 2 * F = 0

theorem tangent_line_of_conic_section 
  (A B C D E F x₀ y₀ : ℝ) :
  ConicSection A B C D E F x₀ y₀ →
  ∃ ε > 0, ∀ x y : ℝ, 
    0 < (x - x₀)^2 + (y - y₀)^2 ∧ (x - x₀)^2 + (y - y₀)^2 < ε^2 →
    ConicSection A B C D E F x y →
    TangentLine A B C D E F x₀ y₀ x y := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_of_conic_section_l2676_267641


namespace NUMINAMATH_CALUDE_vector_magnitude_range_l2676_267613

theorem vector_magnitude_range (a b : EuclideanSpace ℝ (Fin 3)) :
  (norm b = 2) → (norm a = 2 * norm (b - a)) → (4/3 : ℝ) ≤ norm a ∧ norm a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_range_l2676_267613


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_sqrt_three_squared_l2676_267654

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

theorem sqrt_three_squared : Real.sqrt (3 ^ 2) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_sqrt_three_squared_l2676_267654


namespace NUMINAMATH_CALUDE_experience_difference_l2676_267689

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions of the problem -/
def problemConditions (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.roger = 50 - 8 ∧
  e.peter = 19 - 7 ∧
  e.tom = 2 * e.robert ∧
  e.robert = e.peter - 4 ∧
  e.robert > e.mike

/-- The theorem to prove -/
theorem experience_difference (e : Experience) :
  problemConditions e → e.robert - e.mike = 2 := by
  sorry

end NUMINAMATH_CALUDE_experience_difference_l2676_267689


namespace NUMINAMATH_CALUDE_simplify_star_expression_l2676_267666

/-- Custom binary operation ※ for rational numbers -/
def star (a b : ℚ) : ℚ := 2 * a - b

/-- Theorem stating the equivalence of the expression and its simplified form -/
theorem simplify_star_expression (x y : ℚ) : 
  star (star (x - y) (x + y)) (-3 * y) = 2 * x - 3 * y :=
sorry

end NUMINAMATH_CALUDE_simplify_star_expression_l2676_267666


namespace NUMINAMATH_CALUDE_f_value_at_2_l2676_267673

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8) (h2 : f (-2) = 10) :
  f 2 = -26 := by sorry

end NUMINAMATH_CALUDE_f_value_at_2_l2676_267673


namespace NUMINAMATH_CALUDE_five_number_difference_l2676_267661

theorem five_number_difference (a b c d e : ℝ) 
  (h1 : (a + b + c + d) / 4 + e = 74)
  (h2 : (a + b + c + e) / 4 + d = 80)
  (h3 : (a + b + d + e) / 4 + c = 98)
  (h4 : (a + c + d + e) / 4 + b = 116)
  (h5 : (b + c + d + e) / 4 + a = 128) :
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 126 := by
  sorry

end NUMINAMATH_CALUDE_five_number_difference_l2676_267661


namespace NUMINAMATH_CALUDE_equation_solution_l2676_267651

theorem equation_solution : ∃ x : ℚ, (3*x + 5*x = 600 - (4*x + 6*x)) ∧ x = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2676_267651


namespace NUMINAMATH_CALUDE_right_triangle_angles_l2676_267658

theorem right_triangle_angles (A B C : Real) (h1 : A + B + C = 180) (h2 : C = 90) (h3 : A = 50) : B = 40 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l2676_267658


namespace NUMINAMATH_CALUDE_line_plane_relationship_l2676_267609

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpendicular : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the intersection relation between a line and a plane
variable (intersects : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_relationship 
  (a b : Line) (α : Plane)
  (h1 : perpendicular a b)
  (h2 : parallel_line_plane a α) :
  intersects b α ∨ subset_line_plane b α ∨ parallel_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l2676_267609


namespace NUMINAMATH_CALUDE_square_of_difference_l2676_267612

theorem square_of_difference (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l2676_267612


namespace NUMINAMATH_CALUDE_spaghetti_to_manicotti_ratio_l2676_267659

/-- The ratio of students who preferred spaghetti to those who preferred manicotti -/
def pasta_preference_ratio (spaghetti_count manicotti_count : ℕ) : ℚ :=
  spaghetti_count / manicotti_count

/-- The total number of students surveyed -/
def total_students : ℕ := 650

/-- The theorem stating the ratio of spaghetti preference to manicotti preference -/
theorem spaghetti_to_manicotti_ratio : 
  pasta_preference_ratio 250 100 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_spaghetti_to_manicotti_ratio_l2676_267659


namespace NUMINAMATH_CALUDE_sum_of_powers_l2676_267615

theorem sum_of_powers (a b : ℝ) : 
  a + b = 1 →
  a^2 + b^2 = 3 →
  a^3 + b^3 = 4 →
  a^4 + b^4 = 7 →
  a^5 + b^5 = 11 →
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l2676_267615


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l2676_267622

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l2676_267622


namespace NUMINAMATH_CALUDE_equation_rewrite_l2676_267688

theorem equation_rewrite (x y : ℝ) : (2 * x + y = 5) ↔ (y = 5 - 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l2676_267688


namespace NUMINAMATH_CALUDE_total_salaries_l2676_267620

/-- The total amount of A and B's salaries given the specified conditions -/
theorem total_salaries (A_salary B_salary : ℝ) : 
  A_salary = 1500 →
  A_salary * 0.05 = B_salary * 0.15 →
  A_salary + B_salary = 2000 := by
  sorry

end NUMINAMATH_CALUDE_total_salaries_l2676_267620


namespace NUMINAMATH_CALUDE_second_journey_half_time_l2676_267699

/-- Represents a journey with distance and speed -/
structure Journey where
  distance : ℝ
  speed : ℝ

/-- Theorem stating that under given conditions, the time of the second journey is half of the first -/
theorem second_journey_half_time (j1 j2 : Journey) 
  (h1 : j1.distance = 80)
  (h2 : j2.distance = 160)
  (h3 : j2.speed = 4 * j1.speed) :
  (j2.distance / j2.speed) = (1/2) * (j1.distance / j1.speed) := by
  sorry

#check second_journey_half_time

end NUMINAMATH_CALUDE_second_journey_half_time_l2676_267699


namespace NUMINAMATH_CALUDE_marion_score_l2676_267656

theorem marion_score (total_items : Nat) (ella_incorrect : Nat) (marion_additional : Nat) :
  total_items = 40 →
  ella_incorrect = 4 →
  marion_additional = 6 →
  (total_items - ella_incorrect) / 2 + marion_additional = 24 := by
  sorry

end NUMINAMATH_CALUDE_marion_score_l2676_267656


namespace NUMINAMATH_CALUDE_square_neq_iff_neq_and_neq_neg_l2676_267627

theorem square_neq_iff_neq_and_neq_neg (x y : ℝ) :
  x^2 ≠ y^2 ↔ x ≠ y ∧ x ≠ -y := by
  sorry

end NUMINAMATH_CALUDE_square_neq_iff_neq_and_neq_neg_l2676_267627


namespace NUMINAMATH_CALUDE_non_congruent_triangles_count_l2676_267614

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A 2x4 array of points -/
def PointArray : Array (Array Point) :=
  #[#[{x := 0, y := 0}, {x := 1, y := 0}, {x := 2, y := 0}, {x := 3, y := 0}],
    #[{x := 0, y := 1}, {x := 1, y := 1}, {x := 2, y := 1}, {x := 3, y := 1}]]

/-- Check if two triangles are congruent -/
def are_congruent (t1 t2 : Array Point) : Prop := sorry

/-- Count non-congruent triangles in the point array -/
def count_non_congruent_triangles (arr : Array (Array Point)) : ℕ := sorry

/-- Theorem: The number of non-congruent triangles in the given 2x4 array is 3 -/
theorem non_congruent_triangles_count :
  count_non_congruent_triangles PointArray = 3 := by sorry

end NUMINAMATH_CALUDE_non_congruent_triangles_count_l2676_267614


namespace NUMINAMATH_CALUDE_circle_ratio_theorem_l2676_267680

theorem circle_ratio_theorem (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) 
  (h : π * r₂^2 - π * r₁^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_theorem_l2676_267680


namespace NUMINAMATH_CALUDE_integral_2sqrt_minus_sin_l2676_267626

open MeasureTheory Interval Real

theorem integral_2sqrt_minus_sin : ∫ x in (-1)..1, (2 * Real.sqrt (1 - x^2) - Real.sin x) = π := by
  sorry

end NUMINAMATH_CALUDE_integral_2sqrt_minus_sin_l2676_267626


namespace NUMINAMATH_CALUDE_largest_three_digit_in_pascal_l2676_267610

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Pascal's triangle entry at row n and position k -/
def pascal (n k : ℕ) : ℕ := binomial n k

/-- The largest three-digit number -/
def largest_three_digit : ℕ := 999

/-- The row where the largest three-digit number first appears -/
def first_appearance_row : ℕ := 1000

/-- The position in the row where the largest three-digit number first appears -/
def first_appearance_pos : ℕ := 500

theorem largest_three_digit_in_pascal :
  (∀ n k, n < first_appearance_row → pascal n k ≤ largest_three_digit) ∧
  pascal first_appearance_row first_appearance_pos = largest_three_digit ∧
  (∀ n k, n > first_appearance_row → pascal n k > largest_three_digit) :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_in_pascal_l2676_267610


namespace NUMINAMATH_CALUDE_tangent_circles_radii_relation_l2676_267624

/-- Three circles with centers O₁, O₂, and O₃, which are tangent to each other and a line -/
structure TangentCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  O₃ : ℝ × ℝ
  R₁ : ℝ
  R₂ : ℝ
  R₃ : ℝ
  tangent_to_line : Bool
  tangent_to_each_other : Bool

/-- The theorem stating the relationship between the radii of three tangent circles -/
theorem tangent_circles_radii_relation (tc : TangentCircles) :
  1 / Real.sqrt tc.R₂ = 1 / Real.sqrt tc.R₁ + 1 / Real.sqrt tc.R₃ :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_radii_relation_l2676_267624


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2676_267603

def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2676_267603


namespace NUMINAMATH_CALUDE_set_range_with_given_mean_median_l2676_267687

/-- Given a set of three real numbers with mean and median both equal to 5,
    and the smallest number being 2, the range of the set is 6. -/
theorem set_range_with_given_mean_median (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordered set of three numbers
  a = 2 →  -- Smallest number is 2
  (a + b + c) / 3 = 5 →  -- Mean is 5
  b = 5 →  -- Median is 5 (for three numbers, the median is the middle number)
  c - a = 6 :=  -- Range is 6
by sorry

end NUMINAMATH_CALUDE_set_range_with_given_mean_median_l2676_267687


namespace NUMINAMATH_CALUDE_roses_money_proof_l2676_267607

/-- The amount of money Rose already has -/
def roses_money : ℝ := 7.10

/-- The cost of the paintbrush -/
def paintbrush_cost : ℝ := 2.40

/-- The cost of the set of paints -/
def paints_cost : ℝ := 9.20

/-- The cost of the easel -/
def easel_cost : ℝ := 6.50

/-- The additional amount Rose needs -/
def additional_needed : ℝ := 11

theorem roses_money_proof :
  roses_money + additional_needed = paintbrush_cost + paints_cost + easel_cost :=
by sorry

end NUMINAMATH_CALUDE_roses_money_proof_l2676_267607


namespace NUMINAMATH_CALUDE_vector_arithmetic_l2676_267617

/-- Given two 2D vectors a and b, prove that 3a + 4b equals the expected result. -/
theorem vector_arithmetic (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (-3, 4)) :
  (3 : ℝ) • a + (4 : ℝ) • b = (-6, 19) := by
  sorry

end NUMINAMATH_CALUDE_vector_arithmetic_l2676_267617


namespace NUMINAMATH_CALUDE_cubic_polynomial_value_at_5_l2676_267643

/-- A cubic polynomial satisfying specific conditions -/
def cubicPolynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a * x^3 + b * x^2 + c * x + d) ∧
  (p 1 = 1) ∧ (p 2 = 1/8) ∧ (p 3 = 1/27) ∧ (p 4 = 1/64)

/-- Theorem stating that a cubic polynomial satisfying the given conditions has p(5) = 0 -/
theorem cubic_polynomial_value_at_5 (p : ℝ → ℝ) (h : cubicPolynomial p) : p 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_value_at_5_l2676_267643


namespace NUMINAMATH_CALUDE_triangle_height_to_bc_l2676_267623

/-- In a triangle ABC, given side lengths and an angle, prove the height to a specific side. -/
theorem triangle_height_to_bc (a b c h : ℝ) (B : ℝ) : 
  a = 2 → 
  b = Real.sqrt 7 → 
  B = π / 3 →
  c^2 = a^2 + b^2 - 2*a*c*(Real.cos B) →
  h = (a * c * Real.sin B) / a →
  h = 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_to_bc_l2676_267623


namespace NUMINAMATH_CALUDE_equation_solution_l2676_267648

theorem equation_solution (x : ℚ) : 
  (x + 10) / (x - 4) = (x - 4) / (x + 8) → x = -32 / 13 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2676_267648


namespace NUMINAMATH_CALUDE_distribute_problems_l2676_267653

theorem distribute_problems (n m : ℕ) (hn : n = 7) (hm : m = 15) :
  (Nat.choose m n) * (Nat.factorial n) = 32432400 := by
  sorry

end NUMINAMATH_CALUDE_distribute_problems_l2676_267653


namespace NUMINAMATH_CALUDE_inner_circle_radius_l2676_267640

theorem inner_circle_radius (r : ℝ) : 
  r > 0 →
  (π * ((10 : ℝ)^2 - (0.5 * r)^2) = 3.25 * π * (8^2 - r^2)) →
  r = 6 := by
sorry

end NUMINAMATH_CALUDE_inner_circle_radius_l2676_267640


namespace NUMINAMATH_CALUDE_recurrence_solution_l2676_267646

-- Define the recurrence relation
def a : ℕ → ℤ
  | 0 => 3
  | n + 1 => 2 * a n + 2^(n + 1)

-- State the theorem
theorem recurrence_solution (n : ℕ) : a n = (n + 3) * 2^n := by
  sorry

end NUMINAMATH_CALUDE_recurrence_solution_l2676_267646


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l2676_267679

theorem no_real_roots_for_nonzero_k (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, x^2 + k*x + 2*k^2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l2676_267679


namespace NUMINAMATH_CALUDE_panthers_second_half_score_l2676_267682

theorem panthers_second_half_score 
  (total_first_half : ℕ)
  (cougars_lead_first_half : ℕ)
  (total_game : ℕ)
  (cougars_lead_total : ℕ)
  (h1 : total_first_half = 38)
  (h2 : cougars_lead_first_half = 16)
  (h3 : total_game = 58)
  (h4 : cougars_lead_total = 22) :
  ∃ (cougars_first cougars_second panthers_first panthers_second : ℕ),
    cougars_first + panthers_first = total_first_half ∧
    cougars_first = panthers_first + cougars_lead_first_half ∧
    cougars_first + cougars_second + panthers_first + panthers_second = total_game ∧
    (cougars_first + cougars_second) - (panthers_first + panthers_second) = cougars_lead_total ∧
    panthers_second = 7 :=
by sorry

end NUMINAMATH_CALUDE_panthers_second_half_score_l2676_267682


namespace NUMINAMATH_CALUDE_molar_mass_calculation_l2676_267692

/-- Given that 3 moles of a substance weigh 264 grams, prove that its molar mass is 88 grams/mole -/
theorem molar_mass_calculation (total_weight : ℝ) (num_moles : ℝ) (h1 : total_weight = 264) (h2 : num_moles = 3) :
  total_weight / num_moles = 88 := by
  sorry

end NUMINAMATH_CALUDE_molar_mass_calculation_l2676_267692


namespace NUMINAMATH_CALUDE_jacob_painting_fraction_l2676_267681

/-- Jacob's painting rate in houses per minute -/
def painting_rate : ℚ := 1 / 60

/-- Time given to paint in minutes -/
def paint_time : ℚ := 15

/-- Theorem: If Jacob can paint a house in 60 minutes, then he can paint 1/4 of the house in 15 minutes -/
theorem jacob_painting_fraction :
  painting_rate * paint_time = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_jacob_painting_fraction_l2676_267681


namespace NUMINAMATH_CALUDE_price_increase_theorem_l2676_267662

theorem price_increase_theorem (original_price : ℝ) (original_price_pos : original_price > 0) :
  let price_a := original_price * 1.2 * 1.15
  let price_b := original_price * 1.3 * 0.9
  let price_c := original_price * 1.25 * 1.2
  let total_increase := (price_a + price_b + price_c) - 3 * original_price
  let percent_increase := total_increase / (3 * original_price) * 100
  percent_increase = 35 := by
sorry

end NUMINAMATH_CALUDE_price_increase_theorem_l2676_267662


namespace NUMINAMATH_CALUDE_new_person_weight_l2676_267650

/-- Given a group of 8 people, when one person weighing 20 kg is replaced by a new person,
    and the average weight increases by 2.5 kg, the weight of the new person is 40 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_removed : Real) (avg_increase : Real) :
  initial_count = 8 →
  weight_removed = 20 →
  avg_increase = 2.5 →
  (initial_count : Real) * avg_increase + weight_removed = 40 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2676_267650


namespace NUMINAMATH_CALUDE_minimum_blocks_for_wall_l2676_267697

/-- Represents the dimensions of a wall -/
structure WallDimensions where
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  height : ℝ
  length1 : ℝ
  length2 : ℝ

/-- Calculates the minimum number of blocks needed for a wall -/
def minimumBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the minimum number of blocks needed for the specific wall -/
theorem minimum_blocks_for_wall :
  let wall := WallDimensions.mk 150 8
  let block := BlockDimensions.mk 1 2 1.5
  minimumBlocksNeeded wall block = 604 :=
by sorry

end NUMINAMATH_CALUDE_minimum_blocks_for_wall_l2676_267697


namespace NUMINAMATH_CALUDE_farm_hens_count_l2676_267625

/-- Given a farm with roosters and hens, proves that the number of hens is 67 -/
theorem farm_hens_count (roosters hens : ℕ) : 
  hens = 9 * roosters - 5 →
  hens + roosters = 75 →
  hens = 67 := by sorry

end NUMINAMATH_CALUDE_farm_hens_count_l2676_267625


namespace NUMINAMATH_CALUDE_dot_product_range_l2676_267608

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse x^2 + y^2/9 = 1 -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 + p.y^2/9 = 1

/-- Checks if two points are symmetric about the origin -/
def areSymmetric (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = -q.y

/-- Calculates the dot product of vectors CA and CB -/
def dotProduct (a b c : Point) : ℝ :=
  (a.x - c.x) * (b.x - c.x) + (a.y - c.y) * (b.y - c.y)

theorem dot_product_range :
  ∀ (a b : Point),
    isOnEllipse a →
    isOnEllipse b →
    areSymmetric a b →
    let c := Point.mk 5 5
    41 ≤ dotProduct a b c ∧ dotProduct a b c ≤ 49 := by
  sorry


end NUMINAMATH_CALUDE_dot_product_range_l2676_267608


namespace NUMINAMATH_CALUDE_f_greater_than_f_prime_plus_three_halves_l2676_267619

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - log x) + (2 * x - 1) / x^2

theorem f_greater_than_f_prime_plus_three_halves (x : ℝ) (hx : x ∈ Set.Icc 1 2) :
  f 1 x > (deriv (f 1)) x + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_f_greater_than_f_prime_plus_three_halves_l2676_267619


namespace NUMINAMATH_CALUDE_fraction_sum_equals_four_l2676_267676

theorem fraction_sum_equals_four : 
  (2 : ℚ) / 15 + 4 / 15 + 6 / 15 + 8 / 15 + 10 / 15 + 30 / 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_four_l2676_267676


namespace NUMINAMATH_CALUDE_trig_identity_for_point_l2676_267606

/-- Given a point P on the terminal side of angle α with coordinates (4a, -3a) where a < 0,
    prove that 2sin(α) + cos(α) = 2/5 -/
theorem trig_identity_for_point (a : ℝ) (α : ℝ) (h : a < 0) :
  let x : ℝ := 4 * a
  let y : ℝ := -3 * a
  let r : ℝ := Real.sqrt (x^2 + y^2)
  2 * (y / r) + (x / r) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_trig_identity_for_point_l2676_267606


namespace NUMINAMATH_CALUDE_not_right_triangle_3_4_5_squared_l2676_267652

-- Define a function to check if three numbers can form a right triangle
def isRightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that 3^2, 4^2, 5^2 cannot form a right triangle
theorem not_right_triangle_3_4_5_squared :
  ¬ isRightTriangle (3^2) (4^2) (5^2) := by
  sorry


end NUMINAMATH_CALUDE_not_right_triangle_3_4_5_squared_l2676_267652


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2676_267675

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_roots : a 3 * a 7 = 256)
  (h_a4 : a 4 = 8) :
  a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2676_267675


namespace NUMINAMATH_CALUDE_total_time_is_14_25_years_l2676_267605

def time_to_get_in_shape : ℕ := 2 * 12  -- 2 years in months
def time_to_learn_climbing : ℕ := 2 * time_to_get_in_shape
def time_for_survival_skills : ℕ := 9
def time_for_photography : ℕ := 3
def downtime : ℕ := 1
def time_for_summits : List ℕ := [4, 5, 6, 8, 7, 9, 10]
def time_to_learn_diving : ℕ := 13
def time_for_cave_diving : ℕ := 2 * 12  -- 2 years in months

theorem total_time_is_14_25_years :
  let total_months : ℕ := time_to_get_in_shape + time_to_learn_climbing +
                          time_for_survival_skills + time_for_photography +
                          downtime + (time_for_summits.sum) +
                          time_to_learn_diving + time_for_cave_diving
  (total_months : ℚ) / 12 = 14.25 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_14_25_years_l2676_267605


namespace NUMINAMATH_CALUDE_unique_solution_system_l2676_267690

theorem unique_solution_system : 
  ∃! (x y : ℕ+), (x : ℝ)^(y : ℝ) + 3 = (y : ℝ)^(x : ℝ) + 1 ∧ 
                 2 * (x : ℝ)^(y : ℝ) + 4 = (y : ℝ)^(x : ℝ) + 9 ∧
                 x = 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2676_267690


namespace NUMINAMATH_CALUDE_second_hand_store_shirt_price_l2676_267637

/-- The price of a shirt sold to the second-hand store -/
def shirt_price : ℚ := 4

theorem second_hand_store_shirt_price :
  let pants_sold : ℕ := 3
  let shorts_sold : ℕ := 5
  let shirts_sold : ℕ := 5
  let pants_price : ℚ := 5
  let shorts_price : ℚ := 3
  let new_shirts_bought : ℕ := 2
  let new_shirt_price : ℚ := 10
  let remaining_money : ℚ := 30

  shirt_price * shirts_sold + 
  pants_price * pants_sold + 
  shorts_price * shorts_sold = 
  remaining_money + new_shirt_price * new_shirts_bought := by sorry

end NUMINAMATH_CALUDE_second_hand_store_shirt_price_l2676_267637


namespace NUMINAMATH_CALUDE_negation_equivalence_l2676_267644

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) ↔ (∃ x : ℝ, x^2 + 2*x + 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2676_267644


namespace NUMINAMATH_CALUDE_odd_function_properties_l2676_267695

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property f(x+1) = f(x-1)
def property_f (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (x - 1)

-- Define periodicity with period 2
def periodic_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

-- Define symmetry about (k, 0) for all integer k
def symmetric_about_int (f : ℝ → ℝ) : Prop :=
  ∀ (k : ℤ) (x : ℝ), f (2 * k - x) = -f x

theorem odd_function_properties (f : ℝ → ℝ) 
  (h_odd : odd_function f) (h_prop : property_f f) :
  periodic_2 f ∧ symmetric_about_int f := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_l2676_267695


namespace NUMINAMATH_CALUDE_divisibility_by_hundred_l2676_267602

theorem divisibility_by_hundred (N : ℕ) : 
  N = 2^5 * 3^2 * 7 * 75 → 100 ∣ N := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_hundred_l2676_267602


namespace NUMINAMATH_CALUDE_sphere_surface_area_equals_volume_l2676_267636

/-- For a sphere with radius 3, its surface area is numerically equal to its volume. -/
theorem sphere_surface_area_equals_volume :
  let r : ℝ := 3
  let surface_area : ℝ := 4 * Real.pi * r^2
  let volume : ℝ := (4/3) * Real.pi * r^3
  surface_area = volume := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_equals_volume_l2676_267636


namespace NUMINAMATH_CALUDE_sum_of_smaller_angles_is_180_l2676_267600

/-- A convex pentagon with all diagonals drawn. -/
structure ConvexPentagonWithDiagonals where
  -- We don't need to define the specific properties here, just the structure

/-- The sum of the smaller angles formed by intersecting diagonals in a convex pentagon. -/
def sumOfSmallerAngles (p : ConvexPentagonWithDiagonals) : ℝ := sorry

/-- Theorem: The sum of the smaller angles formed by intersecting diagonals in a convex pentagon is always 180°. -/
theorem sum_of_smaller_angles_is_180 (p : ConvexPentagonWithDiagonals) :
  sumOfSmallerAngles p = 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_smaller_angles_is_180_l2676_267600


namespace NUMINAMATH_CALUDE_ellipse_focal_chord_properties_l2676_267657

/-- An ellipse with eccentricity e and a line segment PQ passing through its left focus -/
structure EllipseWithFocalChord where
  e : ℝ
  b : ℝ
  hb : b > 0
  pq_not_vertical : True  -- Represents that PQ is not perpendicular to x-axis
  equilateral_exists : True  -- Represents that there exists R making PQR equilateral

/-- The range of eccentricity and slope of PQ for an ellipse with a special focal chord -/
theorem ellipse_focal_chord_properties (E : EllipseWithFocalChord) :
  E.e > Real.sqrt 3 / 3 ∧ E.e < 1 ∧
  ∃ (k : ℝ), (k = 1 / Real.sqrt (3 * E.e^2 - 1) ∨ k = -1 / Real.sqrt (3 * E.e^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_chord_properties_l2676_267657


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2676_267683

def A (θ : Real) : Set Real := {1, Real.sin θ}
def B : Set Real := {1/2, 2}

theorem sufficient_but_not_necessary :
  (∀ θ : Real, θ = 5 * Real.pi / 6 → A θ ∩ B = {1/2}) ∧
  (∃ θ : Real, θ ≠ 5 * Real.pi / 6 ∧ A θ ∩ B = {1/2}) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2676_267683


namespace NUMINAMATH_CALUDE_greater_number_proof_l2676_267635

theorem greater_number_proof (a b : ℝ) (h_sum : a + b = 36) (h_diff : a - b = 8) (h_greater : a > b) : a = 22 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l2676_267635


namespace NUMINAMATH_CALUDE_complex_fraction_power_l2676_267649

theorem complex_fraction_power (i : ℂ) : i^2 = -1 → ((1 + i) / (1 - i))^2017 = i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l2676_267649


namespace NUMINAMATH_CALUDE_bisection_method_condition_l2676_267616

/-- A continuous function on a closed interval -/
def ContinuousOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  Continuous f ∧ a ≤ b

/-- The bisection method is applicable on an interval -/
def BisectionApplicable (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOnInterval f a b ∧ f a * f b < 0

/-- Theorem: For the bisection method to be applicable on an interval [a, b],
    the function f must satisfy f(a) · f(b) < 0 -/
theorem bisection_method_condition (f : ℝ → ℝ) (a b : ℝ) :
  BisectionApplicable f a b → f a * f b < 0 := by
  sorry


end NUMINAMATH_CALUDE_bisection_method_condition_l2676_267616


namespace NUMINAMATH_CALUDE_player_positions_satisfy_distances_l2676_267647

/-- Represents the positions of four soccer players on a number line -/
def PlayerPositions : Fin 4 → ℝ
  | 0 => 0
  | 1 => 1
  | 2 => 4
  | 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end NUMINAMATH_CALUDE_player_positions_satisfy_distances_l2676_267647


namespace NUMINAMATH_CALUDE_hot_air_balloon_problem_l2676_267664

theorem hot_air_balloon_problem (initial_balloons : ℕ) 
  (h1 : initial_balloons = 200)
  (h2 : initial_balloons > 0) : 
  let first_blown_up := initial_balloons / 5
  let second_blown_up := 2 * first_blown_up
  let total_blown_up := first_blown_up + second_blown_up
  initial_balloons - total_blown_up = 80 := by
sorry

end NUMINAMATH_CALUDE_hot_air_balloon_problem_l2676_267664


namespace NUMINAMATH_CALUDE_shekars_english_score_l2676_267691

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def biology_score : ℕ := 95
def average_score : ℕ := 77
def total_subjects : ℕ := 5

theorem shekars_english_score :
  let known_scores_sum := math_score + science_score + social_studies_score + biology_score
  let total_score := average_score * total_subjects
  total_score - known_scores_sum = 67 := by
  sorry

end NUMINAMATH_CALUDE_shekars_english_score_l2676_267691


namespace NUMINAMATH_CALUDE_loop_requirement_correct_l2676_267696

/-- Represents a mathematical operation that may or may not require a loop statement --/
inductive MathOperation
  | GeometricSum
  | CompareNumbers
  | PiecewiseFunction
  | LargestNaturalNumber

/-- Determines if a given mathematical operation requires a loop statement --/
def requires_loop (op : MathOperation) : Prop :=
  match op with
  | MathOperation.GeometricSum => true
  | MathOperation.CompareNumbers => false
  | MathOperation.PiecewiseFunction => false
  | MathOperation.LargestNaturalNumber => true

theorem loop_requirement_correct :
  (requires_loop MathOperation.GeometricSum) ∧
  (¬requires_loop MathOperation.CompareNumbers) ∧
  (¬requires_loop MathOperation.PiecewiseFunction) ∧
  (requires_loop MathOperation.LargestNaturalNumber) :=
by sorry

#check loop_requirement_correct

end NUMINAMATH_CALUDE_loop_requirement_correct_l2676_267696


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l2676_267628

/-- The number of different books in the 'crazy silly school' series -/
def num_books : ℕ := 16

/-- The number of movies watched -/
def movies_watched : ℕ := 19

/-- The difference between movies watched and books read -/
def movie_book_difference : ℕ := 3

theorem crazy_silly_school_books :
  num_books = movies_watched - movie_book_difference :=
by sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l2676_267628


namespace NUMINAMATH_CALUDE_rainy_days_last_week_l2676_267645

theorem rainy_days_last_week (n : ℤ) : 
  (∃ (R NR : ℕ), 
    R + NR = 7 ∧ 
    n * R + 3 * NR = 20 ∧ 
    3 * NR = n * R + 10) →
  (∃ (R : ℕ), R = 2) :=
sorry

end NUMINAMATH_CALUDE_rainy_days_last_week_l2676_267645


namespace NUMINAMATH_CALUDE_kim_cousins_count_l2676_267655

theorem kim_cousins_count (gum_per_cousin : ℕ) (total_gum : ℕ) (h1 : gum_per_cousin = 5) (h2 : total_gum = 20) :
  total_gum / gum_per_cousin = 4 := by
sorry

end NUMINAMATH_CALUDE_kim_cousins_count_l2676_267655


namespace NUMINAMATH_CALUDE_expression_value_l2676_267678

theorem expression_value : 
  let a : ℚ := 1/3
  let b : ℚ := 3
  (2 * a⁻¹ + a⁻¹ / b) / a = 21 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2676_267678


namespace NUMINAMATH_CALUDE_correct_seat_ratio_l2676_267630

/-- The ratio of coach class seats to first-class seats in an airplane -/
def seat_ratio (total_seats first_class_seats : ℕ) : ℚ × ℚ :=
  let coach_seats := total_seats - first_class_seats
  (coach_seats, first_class_seats)

/-- Theorem stating the correct ratio of coach to first-class seats -/
theorem correct_seat_ratio :
  seat_ratio 387 77 = (310, 77) := by
  sorry

#eval seat_ratio 387 77

end NUMINAMATH_CALUDE_correct_seat_ratio_l2676_267630


namespace NUMINAMATH_CALUDE_canoe_production_sum_l2676_267631

/-- Represents the number of canoes built in the first month -/
def first_month_canoes : ℕ := 7

/-- Represents the ratio of canoes built between consecutive months -/
def monthly_ratio : ℕ := 3

/-- Represents the number of months considered -/
def num_months : ℕ := 6

/-- Calculates the sum of a geometric sequence -/
def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem canoe_production_sum :
  geometric_sum first_month_canoes monthly_ratio num_months = 2548 := by
  sorry

end NUMINAMATH_CALUDE_canoe_production_sum_l2676_267631


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2676_267660

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x = 2 ↔ (x - 1)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2676_267660


namespace NUMINAMATH_CALUDE_library_books_before_grant_l2676_267677

/-- The number of books purchased with the grant -/
def books_purchased : ℕ := 2647

/-- The total number of books after the grant -/
def total_books : ℕ := 8582

/-- The number of books before the grant -/
def books_before : ℕ := total_books - books_purchased

theorem library_books_before_grant : books_before = 5935 := by
  sorry

end NUMINAMATH_CALUDE_library_books_before_grant_l2676_267677


namespace NUMINAMATH_CALUDE_max_abc_value_l2676_267668

theorem max_abc_value (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a * b + b * c + a * c = 1) : 
  a * b * c ≤ Real.sqrt 3 / 9 := by
sorry

end NUMINAMATH_CALUDE_max_abc_value_l2676_267668


namespace NUMINAMATH_CALUDE_percy_swimming_hours_l2676_267633

/-- Percy's swimming schedule and total hours over 4 weeks -/
theorem percy_swimming_hours :
  let weekday_hours : ℕ := 2 -- 1 hour before school + 1 hour after school
  let weekdays_per_week : ℕ := 5
  let weekend_hours : ℕ := 3
  let weekend_days : ℕ := 2
  let weeks : ℕ := 4
  
  let total_hours_per_week : ℕ := weekday_hours * weekdays_per_week + weekend_hours * weekend_days
  let total_hours_four_weeks : ℕ := total_hours_per_week * weeks
  
  total_hours_four_weeks = 64
  := by sorry

end NUMINAMATH_CALUDE_percy_swimming_hours_l2676_267633


namespace NUMINAMATH_CALUDE_negative_three_cubed_equality_l2676_267672

theorem negative_three_cubed_equality : (-3)^3 = -3^3 := by sorry

end NUMINAMATH_CALUDE_negative_three_cubed_equality_l2676_267672
