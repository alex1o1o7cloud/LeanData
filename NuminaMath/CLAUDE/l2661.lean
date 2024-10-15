import Mathlib

namespace NUMINAMATH_CALUDE_total_factories_to_check_l2661_266115

theorem total_factories_to_check (first_group : ℕ) (second_group : ℕ) (remaining : ℕ) :
  first_group = 69 → second_group = 52 → remaining = 48 →
  first_group + second_group + remaining = 169 := by
  sorry

end NUMINAMATH_CALUDE_total_factories_to_check_l2661_266115


namespace NUMINAMATH_CALUDE_fifth_power_sum_l2661_266139

theorem fifth_power_sum (x : ℝ) (h : x + 1/x = -5) : x^5 + 1/x^5 = -2525 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l2661_266139


namespace NUMINAMATH_CALUDE_problem_solution_l2661_266193

def f (x : ℝ) := |x| - |2*x - 1|

def M := {x : ℝ | f x > -1}

theorem problem_solution :
  (M = {x : ℝ | 0 < x ∧ x < 2}) ∧
  (∀ a ∈ M,
    (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
    (a = 1 → a^2 - a + 1 = 1/a) ∧
    (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2661_266193


namespace NUMINAMATH_CALUDE_two_card_selection_l2661_266197

theorem two_card_selection (deck_size : ℕ) (h : deck_size = 60) : 
  deck_size * (deck_size - 1) = 3540 :=
by sorry

end NUMINAMATH_CALUDE_two_card_selection_l2661_266197


namespace NUMINAMATH_CALUDE_remaining_perimeter_l2661_266144

/-- The perimeter of the remaining shape after cutting out two squares from a rectangle. -/
theorem remaining_perimeter (rectangle_length rectangle_width square1_side square2_side : ℕ) :
  rectangle_length = 50 ∧ 
  rectangle_width = 20 ∧ 
  square1_side = 12 ∧ 
  square2_side = 4 →
  2 * (rectangle_length + rectangle_width) + 4 * square1_side + 4 * square2_side = 204 := by
  sorry

end NUMINAMATH_CALUDE_remaining_perimeter_l2661_266144


namespace NUMINAMATH_CALUDE_max_value_cos_sin_sum_l2661_266145

theorem max_value_cos_sin_sum :
  ∃ (M : ℝ), M = 5 ∧ ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_sum_l2661_266145


namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l2661_266107

theorem rectangle_cylinder_volume_ratio :
  let rectangle_length : ℝ := 10
  let rectangle_width : ℝ := 6
  let cylinder_A_height : ℝ := rectangle_width
  let cylinder_A_circumference : ℝ := rectangle_length
  let cylinder_B_height : ℝ := rectangle_length
  let cylinder_B_circumference : ℝ := rectangle_width
  let cylinder_A_volume : ℝ := (cylinder_A_circumference^2 * cylinder_A_height) / (4 * π)
  let cylinder_B_volume : ℝ := (cylinder_B_circumference^2 * cylinder_B_height) / (4 * π)
  max cylinder_A_volume cylinder_B_volume / min cylinder_A_volume cylinder_B_volume = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l2661_266107


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l2661_266157

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (h1 : (2*x + 3) % 7 = 0) 
  (h2 : (3*y - 4) % 7 = 0) : 
  (∃ n : ℕ+, (3*x^2 + 2*x*y + y^2 + n) % 7 = 0 ∧ 
   ∀ m : ℕ+, m < n → (3*x^2 + 2*x*y + y^2 + m) % 7 ≠ 0) → 
  (∃ n : ℕ+, n = 4 ∧ (3*x^2 + 2*x*y + y^2 + n) % 7 = 0 ∧ 
   ∀ m : ℕ+, m < n → (3*x^2 + 2*x*y + y^2 + m) % 7 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l2661_266157


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l2661_266165

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) : 
  total_students = 42 →
  boy_ratio = 3 →
  girl_ratio = 4 →
  ∃ (k : ℕ), 
    (boy_ratio * k + girl_ratio * k = total_students) ∧
    (girl_ratio * k - boy_ratio * k = 6) :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l2661_266165


namespace NUMINAMATH_CALUDE_school_population_after_additions_l2661_266161

theorem school_population_after_additions 
  (initial_girls : ℕ) 
  (initial_boys : ℕ) 
  (initial_teachers : ℕ) 
  (additional_girls : ℕ) 
  (additional_boys : ℕ) 
  (additional_teachers : ℕ) 
  (h1 : initial_girls = 732) 
  (h2 : initial_boys = 761) 
  (h3 : initial_teachers = 54) 
  (h4 : additional_girls = 682) 
  (h5 : additional_boys = 8) 
  (h6 : additional_teachers = 3) : 
  initial_girls + initial_boys + initial_teachers + 
  additional_girls + additional_boys + additional_teachers = 2240 :=
by
  sorry


end NUMINAMATH_CALUDE_school_population_after_additions_l2661_266161


namespace NUMINAMATH_CALUDE_complex_square_eq_abs_square_iff_real_l2661_266106

open Complex

theorem complex_square_eq_abs_square_iff_real (z : ℂ) :
  (z - 1)^2 = abs (z - 1)^2 ↔ z.im = 0 :=
sorry

end NUMINAMATH_CALUDE_complex_square_eq_abs_square_iff_real_l2661_266106


namespace NUMINAMATH_CALUDE_power_sum_equality_l2661_266142

theorem power_sum_equality : 2^300 + (-2^301) = -2^300 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2661_266142


namespace NUMINAMATH_CALUDE_range_of_f_l2661_266122

open Real

noncomputable def f (x : ℝ) : ℝ := (1 + cos x)^2023 + (1 - cos x)^2023

theorem range_of_f : 
  ∀ y ∈ Set.range (f ∘ (fun x => x * π / 3) ∘ fun t => t * 2 - 1), 2 ≤ y ∧ y ≤ 2^2023 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2661_266122


namespace NUMINAMATH_CALUDE_expression_change_l2661_266143

theorem expression_change (x a : ℝ) (b : ℝ) (h : a > 0) : 
  let f := fun x => x^3 - b
  let δ := fun (ε : ℝ) => f (x + ε) - (b + a^2) - (f x - b)
  (δ a = 3*x^2*a + 3*x*a^2 + a^3 - a^2) ∧ 
  (δ (-a) = -3*x^2*a + 3*x*a^2 - a^3 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_expression_change_l2661_266143


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2661_266175

/-- Given the ages of three people A, B, and C, prove that the ratio of B's age to C's age is 2:1 --/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 27 →
  b = 10 →
  b / c = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2661_266175


namespace NUMINAMATH_CALUDE_select_products_l2661_266183

theorem select_products (total : ℕ) (qualified : ℕ) (unqualified : ℕ) (select : ℕ) 
    (h1 : total = qualified + unqualified) 
    (h2 : total = 50) 
    (h3 : qualified = 47) 
    (h4 : unqualified = 3) 
    (h5 : select = 4) : 
    (Nat.choose unqualified 1 * Nat.choose qualified 3 + 
     Nat.choose unqualified 2 * Nat.choose qualified 2 + 
     Nat.choose unqualified 3 * Nat.choose qualified 1) = 
    (Nat.choose total 4 - Nat.choose qualified 4) := by
  sorry

end NUMINAMATH_CALUDE_select_products_l2661_266183


namespace NUMINAMATH_CALUDE_division_reduction_l2661_266184

theorem division_reduction (x : ℝ) : (45 / x = 45 - 30) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_reduction_l2661_266184


namespace NUMINAMATH_CALUDE_or_false_sufficient_not_necessary_for_and_false_l2661_266149

theorem or_false_sufficient_not_necessary_for_and_false (p q : Prop) :
  (¬(p ∨ q) → ¬p ∧ ¬q) ∧ ∃ (p q : Prop), (¬p ∧ ¬q) ∧ ¬¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_or_false_sufficient_not_necessary_for_and_false_l2661_266149


namespace NUMINAMATH_CALUDE_not_always_increasing_sum_of_increasing_and_decreasing_l2661_266162

-- Define the concept of an increasing function
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the concept of a decreasing function
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem not_always_increasing_sum_of_increasing_and_decreasing :
  ¬(∀ f g : ℝ → ℝ, Increasing f → Decreasing g → Increasing (λ x ↦ f x + g x)) :=
sorry

end NUMINAMATH_CALUDE_not_always_increasing_sum_of_increasing_and_decreasing_l2661_266162


namespace NUMINAMATH_CALUDE_extreme_value_implies_m_plus_n_11_l2661_266121

/-- A function f with an extreme value of 0 at x = -1 -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + 3*m*x^2 + n*x + m^2

/-- The derivative of f -/
def f' (m n : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*m*x + n

theorem extreme_value_implies_m_plus_n_11 (m n : ℝ) :
  (f m n (-1) = 0) →
  (f' m n (-1) = 0) →
  (m + n = 11) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_m_plus_n_11_l2661_266121


namespace NUMINAMATH_CALUDE_cubic_inequality_reciprocal_l2661_266167

theorem cubic_inequality_reciprocal (a b : ℝ) (h1 : a^3 > b^3) (h2 : a * b > 0) :
  1 / a < 1 / b := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_reciprocal_l2661_266167


namespace NUMINAMATH_CALUDE_john_is_25_l2661_266156

-- Define John's age and his mother's age
def john_age : ℕ := sorry
def mother_age : ℕ := sorry

-- State the conditions
axiom age_difference : mother_age = john_age + 30
axiom sum_of_ages : john_age + mother_age = 80

-- Theorem to prove
theorem john_is_25 : john_age = 25 := by sorry

end NUMINAMATH_CALUDE_john_is_25_l2661_266156


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2661_266116

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 1 → 1 / x < 1) ∧
  (∃ x, 1 / x < 1 ∧ ¬(x > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2661_266116


namespace NUMINAMATH_CALUDE_petya_strategy_exists_l2661_266109

theorem petya_strategy_exists (opponent_choice : ℚ → ℚ) : 
  ∃ (a b c : ℚ), 
    ∃ (x y : ℂ), 
      (x^3 + a*x^2 + b*x + c = 0) ∧ 
      (y^3 + a*y^2 + b*y + c = 0) ∧ 
      (y - x = 2014) ∧
      ((a = opponent_choice b ∧ c = opponent_choice 0) ∨ 
       (b = opponent_choice a ∧ c = opponent_choice 0) ∨ 
       (a = opponent_choice c ∧ b = opponent_choice 0)) :=
by sorry

end NUMINAMATH_CALUDE_petya_strategy_exists_l2661_266109


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l2661_266113

theorem quadratic_always_nonnegative : ∀ x : ℝ, x^2 - x + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l2661_266113


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l2661_266146

theorem lcm_from_product_and_hcf (a b : ℕ+) :
  a * b = 82500 → Nat.gcd a b = 55 → Nat.lcm a b = 1500 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l2661_266146


namespace NUMINAMATH_CALUDE_no_matching_pyramids_l2661_266170

/-- Represents a convex n-sided pyramid -/
structure NSidedPyramid (n : ℕ) :=
  (convex : Bool)
  (dihedralAngles : Fin n → ℝ)

/-- Represents a triangular pyramid -/
structure TriangularPyramid :=
  (dihedralAngles : Fin 4 → ℝ)

/-- The theorem stating that no such pair of pyramids exists -/
theorem no_matching_pyramids :
  ∀ (n : ℕ) (nPyramid : NSidedPyramid n) (tPyramid : TriangularPyramid),
    n ≥ 4 →
    nPyramid.convex = true →
    (∃ (i j k l : Fin n),
      i ≠ j ∧ i ≠ k ∧ i ≠ l ∧
      j ≠ k ∧ j ≠ l ∧
      k ≠ l ∧
      nPyramid.dihedralAngles i = tPyramid.dihedralAngles 0 ∧
      nPyramid.dihedralAngles j = tPyramid.dihedralAngles 1 ∧
      nPyramid.dihedralAngles k = tPyramid.dihedralAngles 2 ∧
      nPyramid.dihedralAngles l = tPyramid.dihedralAngles 3) →
    False :=
by sorry

end NUMINAMATH_CALUDE_no_matching_pyramids_l2661_266170


namespace NUMINAMATH_CALUDE_car_travel_time_l2661_266141

/-- Proves that given the conditions of two cars A and B, the time taken by Car B to reach its destination is 1 hour. -/
theorem car_travel_time (speed_A speed_B : ℝ) (time_A : ℝ) (ratio : ℝ) : 
  speed_A = 50 →
  speed_B = 100 →
  time_A = 6 →
  ratio = 3 →
  (speed_A * time_A) / (speed_B * (speed_A * time_A / (ratio * speed_B))) = 1 := by
  sorry


end NUMINAMATH_CALUDE_car_travel_time_l2661_266141


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2661_266105

/-- Given a quadratic equation x^2 + mx - 2 = 0 where -1 is a root,
    prove that m = -1 and the other root is 2 -/
theorem quadratic_equation_roots (m : ℝ) : 
  ((-1 : ℝ)^2 + m*(-1) - 2 = 0) → 
  (m = -1 ∧ ∃ r : ℝ, r ≠ -1 ∧ r^2 + m*r - 2 = 0 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2661_266105


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l2661_266101

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (h_pig : pig_value = 350) (h_goat : goat_value = 250) :
  (∃ (d : ℕ), d > 0 ∧ 
   (∃ (p g : ℤ), d = pig_value * p + goat_value * g) ∧
   (∀ (d' : ℕ), d' > 0 → d' < d → 
    ¬(∃ (p' g' : ℤ), d' = pig_value * p' + goat_value * g'))) →
  (∃ (d : ℕ), d = 50 ∧ d > 0 ∧ 
   (∃ (p g : ℤ), d = pig_value * p + goat_value * g) ∧
   (∀ (d' : ℕ), d' > 0 → d' < d → 
    ¬(∃ (p' g' : ℤ), d' = pig_value * p' + goat_value * g'))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l2661_266101


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2661_266126

/-- Proves that the weight of a replaced person is 65 kg given the conditions of the problem -/
theorem replaced_person_weight (n : ℕ) (old_avg new_avg new_weight : ℝ) : 
  n = 8 ∧ 
  new_avg - old_avg = 3.5 ∧
  new_weight = 93 →
  (n * new_avg - new_weight) / (n - 1) = 65 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2661_266126


namespace NUMINAMATH_CALUDE_book_distribution_l2661_266132

theorem book_distribution (x : ℕ) : x > 0 → (x / 3 : ℚ) + 2 = (x - 9 : ℚ) / 2 :=
  sorry

end NUMINAMATH_CALUDE_book_distribution_l2661_266132


namespace NUMINAMATH_CALUDE_bounded_sequence_with_distance_condition_l2661_266118

theorem bounded_sequence_with_distance_condition :
  ∃ (a : ℕ → ℝ), 
    (∃ (C D : ℝ), ∀ n, C ≤ a n ∧ a n ≤ D) ∧ 
    (∀ (n m : ℕ), n > m → |a m - a n| ≥ 1 / (n - m : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_bounded_sequence_with_distance_condition_l2661_266118


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_tetrahedron_l2661_266158

/-- Given a tetrahedron with volume V, face areas S₁, S₂, S₃, S₄, and an inscribed sphere of radius R,
    prove that R = 3V / (S₁ + S₂ + S₃ + S₄) -/
theorem inscribed_sphere_radius_tetrahedron 
  (V : ℝ) 
  (S₁ S₂ S₃ S₄ : ℝ) 
  (R : ℝ) 
  (h_volume : V > 0)
  (h_areas : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0)
  (h_inscribed : R > 0) :
  R = 3 * V / (S₁ + S₂ + S₃ + S₄) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_tetrahedron_l2661_266158


namespace NUMINAMATH_CALUDE_archie_antibiotics_duration_l2661_266172

/-- Calculates the number of days Archie can take antibiotics given the cost, 
    daily frequency, and available money. -/
def daysOfAntibiotics (costPerDose : ℚ) (dosesPerDay : ℕ) (availableMoney : ℚ) : ℚ :=
  availableMoney / (costPerDose * dosesPerDay)

/-- Proves that Archie can take antibiotics for 7 days given the specified conditions. -/
theorem archie_antibiotics_duration :
  daysOfAntibiotics 3 3 63 = 7 := by
sorry

end NUMINAMATH_CALUDE_archie_antibiotics_duration_l2661_266172


namespace NUMINAMATH_CALUDE_triangle_properties_l2661_266138

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def triangle_conditions (t : Triangle) : Prop :=
  acute_triangle t ∧
  t.a = 2 * t.b * Real.sin t.A ∧
  t.a = 3 * Real.sqrt 3 ∧
  t.c = 5

-- State the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = Real.pi/6 ∧ 
  t.b = Real.sqrt 7 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B) = (15 * Real.sqrt 3)/4 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2661_266138


namespace NUMINAMATH_CALUDE_no_real_solutions_l2661_266152

theorem no_real_solutions :
  ¬∃ x : ℝ, (3 * x^2) / (x - 2) - (5 * x + 4) / 4 + (10 - 9 * x) / (x - 2) + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2661_266152


namespace NUMINAMATH_CALUDE_students_dislike_both_l2661_266155

/-- Given a class of students and their food preferences, calculate the number of students who don't like either food. -/
theorem students_dislike_both (total : ℕ) (like_fries : ℕ) (like_burgers : ℕ) (like_both : ℕ) 
  (h1 : total = 25)
  (h2 : like_fries = 15)
  (h3 : like_burgers = 10)
  (h4 : like_both = 6)
  (h5 : like_both ≤ like_fries ∧ like_both ≤ like_burgers) :
  total - (like_fries + like_burgers - like_both) = 6 := by
  sorry

#check students_dislike_both

end NUMINAMATH_CALUDE_students_dislike_both_l2661_266155


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2661_266190

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2661_266190


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l2661_266103

theorem gcd_lcm_product_90_135 : Nat.gcd 90 135 * Nat.lcm 90 135 = 12150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l2661_266103


namespace NUMINAMATH_CALUDE_log_division_simplification_l2661_266189

theorem log_division_simplification : 
  Real.log 8 / Real.log (1/8) = -1 := by sorry

end NUMINAMATH_CALUDE_log_division_simplification_l2661_266189


namespace NUMINAMATH_CALUDE_not_p_neither_sufficient_nor_necessary_for_q_l2661_266173

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

-- Define what it means for ¬p to be neither sufficient nor necessary for q
def neither_sufficient_nor_necessary : Prop :=
  (∃ x : ℝ, ¬(p x) ∧ ¬(q x)) ∧ 
  (∃ y : ℝ, ¬(p y) ∧ q y) ∧ 
  (∃ z : ℝ, p z ∧ q z)

-- Theorem statement
theorem not_p_neither_sufficient_nor_necessary_for_q : 
  neither_sufficient_nor_necessary :=
sorry

end NUMINAMATH_CALUDE_not_p_neither_sufficient_nor_necessary_for_q_l2661_266173


namespace NUMINAMATH_CALUDE_count_pairs_satisfying_condition_l2661_266179

theorem count_pairs_satisfying_condition : 
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2 < 50 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 50) (Finset.range 50))).card = 204 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_satisfying_condition_l2661_266179


namespace NUMINAMATH_CALUDE_triangle_inequality_check_l2661_266100

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_inequality_check :
  can_form_triangle 9 6 13 ∧
  ¬(can_form_triangle 6 8 16) ∧
  ¬(can_form_triangle 18 9 8) ∧
  ¬(can_form_triangle 3 5 9) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_check_l2661_266100


namespace NUMINAMATH_CALUDE_least_number_divisible_by_four_primes_ge_5_l2661_266185

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem least_number_divisible_by_four_primes_ge_5 :
  ∃ (n : Nat) (p₁ p₂ p₃ p₄ : Nat),
    n > 0 ∧
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧
    p₁ ≥ 5 ∧ p₂ ≥ 5 ∧ p₃ ≥ 5 ∧ p₄ ≥ 5 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧
    (∀ m : Nat, m > 0 ∧ m < n →
      ¬(∃ (q₁ q₂ q₃ q₄ : Nat),
        is_prime q₁ ∧ is_prime q₂ ∧ is_prime q₃ ∧ is_prime q₄ ∧
        q₁ ≥ 5 ∧ q₂ ≥ 5 ∧ q₃ ≥ 5 ∧ q₄ ≥ 5 ∧
        q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
        m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0)) ∧
    n = 5005 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_four_primes_ge_5_l2661_266185


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2661_266137

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 > 0}
def B : Set ℝ := {x : ℝ | x / (x - 1) < 0}

-- State the theorem
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2661_266137


namespace NUMINAMATH_CALUDE_trig_identity_l2661_266163

theorem trig_identity : 
  Real.sin (155 * π / 180) * Real.sin (55 * π / 180) + 
  Real.cos (25 * π / 180) * Real.cos (55 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l2661_266163


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2661_266177

theorem complex_modulus_problem (z : ℂ) : z = (2 * I) / (1 - I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2661_266177


namespace NUMINAMATH_CALUDE_cookie_shop_purchases_l2661_266102

/-- The number of different types of cookies available. -/
def num_cookies : ℕ := 7

/-- The number of different types of milk available. -/
def num_milk : ℕ := 4

/-- The total number of items Gamma and Delta purchase collectively. -/
def total_items : ℕ := 4

/-- The number of ways Gamma can choose items without repeats. -/
def gamma_choices (k : ℕ) : ℕ := Nat.choose (num_cookies + num_milk) k

/-- The number of ways Delta can choose k cookies with possible repeats. -/
def delta_choices (k : ℕ) : ℕ := 
  (Nat.choose num_cookies k) +  -- All different cookies
  (if k > 1 then num_cookies * (Nat.choose (k - 1 + num_cookies - 1) (num_cookies - 1)) else 0)  -- With repeats

/-- The total number of ways Gamma and Delta can purchase 4 items collectively. -/
def total_ways : ℕ := 
  (gamma_choices 4) +  -- Gamma 4, Delta 0
  (gamma_choices 3) * num_cookies +  -- Gamma 3, Delta 1
  (gamma_choices 2) * (delta_choices 2) +  -- Gamma 2, Delta 2
  (gamma_choices 1) * (delta_choices 3) +  -- Gamma 1, Delta 3
  (delta_choices 4)  -- Gamma 0, Delta 4

theorem cookie_shop_purchases : total_ways = 4096 := by
  sorry

end NUMINAMATH_CALUDE_cookie_shop_purchases_l2661_266102


namespace NUMINAMATH_CALUDE_max_side_length_of_triangle_l2661_266117

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b → b < c → a + b + c = 24 → c ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_max_side_length_of_triangle_l2661_266117


namespace NUMINAMATH_CALUDE_good_carrots_count_l2661_266181

theorem good_carrots_count (vanessa_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : vanessa_carrots = 17)
  (h2 : mom_carrots = 14)
  (h3 : bad_carrots = 7) :
  vanessa_carrots + mom_carrots - bad_carrots = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_good_carrots_count_l2661_266181


namespace NUMINAMATH_CALUDE_work_completion_time_l2661_266128

theorem work_completion_time 
  (n : ℕ) -- number of persons
  (t : ℝ) -- time to complete the work
  (h : t = 12) -- given condition that work is completed in 12 days
  : (2 * n) * (3 : ℝ) = n * t / 2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2661_266128


namespace NUMINAMATH_CALUDE_number_always_divisible_by_396_l2661_266180

/-- Represents a permutation of digits 0 to 9 -/
def DigitPermutation := Fin 10 → Fin 10

/-- Constructs the number based on the given permutation -/
def constructNumber (p : DigitPermutation) : ℕ :=
  -- Implementation details omitted
  sorry

/-- The theorem to be proved -/
theorem number_always_divisible_by_396 (p : DigitPermutation) :
  396 ∣ constructNumber p := by
  sorry

end NUMINAMATH_CALUDE_number_always_divisible_by_396_l2661_266180


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l2661_266187

theorem factorization_of_quadratic (m : ℝ) : m^2 - 4*m = m*(m - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l2661_266187


namespace NUMINAMATH_CALUDE_unique_lottery_number_l2661_266191

/-- A five-digit number -/
def FiveDigitNumber := ℕ

/-- Check if a number is a five-digit number -/
def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sumOfDigits (n / 10)

/-- Neighbor's age -/
def neighborAge : ℕ := 45

/-- Theorem: The only five-digit number where the sum of its digits equals 45
    and can be easily solved is 99999 -/
theorem unique_lottery_number :
  ∃! (n : FiveDigitNumber), 
    isFiveDigitNumber n ∧ 
    sumOfDigits n = neighborAge ∧
    (∀ (m : FiveDigitNumber), isFiveDigitNumber m → sumOfDigits m = neighborAge → m = n) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_lottery_number_l2661_266191


namespace NUMINAMATH_CALUDE_square_plot_area_l2661_266199

/-- The area of a square plot with side length 50.5 m is 2550.25 square meters. -/
theorem square_plot_area : 
  let side_length : ℝ := 50.5
  let area : ℝ := side_length * side_length
  area = 2550.25 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_area_l2661_266199


namespace NUMINAMATH_CALUDE_thousand_power_division_l2661_266127

theorem thousand_power_division :
  1000 * (1000^1000) / (500^1000) = 2^1001 * 500 := by
  sorry

end NUMINAMATH_CALUDE_thousand_power_division_l2661_266127


namespace NUMINAMATH_CALUDE_students_per_group_l2661_266111

theorem students_per_group 
  (total_students : ℕ) 
  (students_not_picked : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 36) 
  (h2 : students_not_picked = 9) 
  (h3 : num_groups = 3) : 
  (total_students - students_not_picked) / num_groups = 9 := by
sorry

end NUMINAMATH_CALUDE_students_per_group_l2661_266111


namespace NUMINAMATH_CALUDE_intersection_point_inside_circle_l2661_266169

/-- The intersection point of two lines is inside a circle iff a is within a specific range -/
theorem intersection_point_inside_circle (a : ℝ) :
  let P : ℝ × ℝ := (a, 3 * a)  -- Intersection point of y = x + 2a and y = 2x + a
  (P.1 - 1)^2 + (P.2 - 1)^2 < 4 ↔ -1/5 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_inside_circle_l2661_266169


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2661_266194

/-- The diagonal of a rectangle with side lengths 40√3 cm and 30√3 cm is 50√3 cm. -/
theorem rectangle_diagonal (a b d : ℝ) (ha : a = 40 * Real.sqrt 3) (hb : b = 30 * Real.sqrt 3) 
  (hd : d ^ 2 = a ^ 2 + b ^ 2) : d = 50 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_diagonal_l2661_266194


namespace NUMINAMATH_CALUDE_matrix_N_property_l2661_266108

theorem matrix_N_property :
  ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
    (∀ (u : Fin 3 → ℝ), N.mulVec u = (3 : ℝ) • u) ∧
    N = ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_property_l2661_266108


namespace NUMINAMATH_CALUDE_distance_between_tangent_circles_l2661_266136

/-- The distance between centers of two internally tangent circles -/
def distance_between_centers (r₁ r₂ : ℝ) : ℝ := |r₂ - r₁|

/-- Theorem: The distance between centers of two internally tangent circles
    with radii 3 and 4 is 1 -/
theorem distance_between_tangent_circles :
  let r₁ : ℝ := 3
  let r₂ : ℝ := 4
  distance_between_centers r₁ r₂ = 1 := by sorry

end NUMINAMATH_CALUDE_distance_between_tangent_circles_l2661_266136


namespace NUMINAMATH_CALUDE_work_efficiency_l2661_266134

/-- Given two workers A and B, where A can finish a work in 18 days and B can do the same work in half the time taken by A, this theorem proves that working together, they can finish 1/6 of the work in one day. -/
theorem work_efficiency (days_A : ℕ) (days_B : ℕ) : 
  days_A = 18 → 
  days_B = days_A / 2 → 
  (1 : ℚ) / days_A + (1 : ℚ) / days_B = (1 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_work_efficiency_l2661_266134


namespace NUMINAMATH_CALUDE_number_multiplied_by_9999_l2661_266133

theorem number_multiplied_by_9999 :
  ∃ x : ℕ, x * 9999 = 724817410 ∧ x = 72492 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_9999_l2661_266133


namespace NUMINAMATH_CALUDE_fourth_term_is_27_l2661_266124

-- Define the sequence sum function
def S (n : ℕ) : ℤ := 4 * n^2 - n - 8

-- Define the sequence term function
def a (n : ℕ) : ℤ := S n - S (n - 1)

-- Theorem statement
theorem fourth_term_is_27 : a 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_27_l2661_266124


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l2661_266154

theorem fraction_meaningful_condition (x : ℝ) :
  (∃ y : ℝ, y = 3 / (x - 1)) ↔ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l2661_266154


namespace NUMINAMATH_CALUDE_rubiks_cube_probabilities_l2661_266171

/-- The probability of person A solving the cube within 30 seconds -/
def prob_A : ℝ := 0.8

/-- The probability of person B solving the cube within 30 seconds -/
def prob_B : ℝ := 0.6

/-- The probability of person A succeeding on their third attempt -/
def prob_A_third_attempt : ℝ := (1 - prob_A) * (1 - prob_A) * prob_A

/-- The probability that at least one of them succeeds on their first attempt -/
def prob_at_least_one_first_attempt : ℝ := 1 - (1 - prob_A) * (1 - prob_B)

theorem rubiks_cube_probabilities :
  prob_A_third_attempt = 0.032 ∧ prob_at_least_one_first_attempt = 0.92 := by
  sorry

end NUMINAMATH_CALUDE_rubiks_cube_probabilities_l2661_266171


namespace NUMINAMATH_CALUDE_unique_root_of_sqrt_equation_l2661_266150

theorem unique_root_of_sqrt_equation :
  ∃! x : ℝ, x + 9 ≥ 0 ∧ x - 2 ≥ 0 ∧ Real.sqrt (x + 9) - Real.sqrt (x - 2) = 3 :=
by
  -- The unique solution is x = 19/9
  use 19/9
  sorry

end NUMINAMATH_CALUDE_unique_root_of_sqrt_equation_l2661_266150


namespace NUMINAMATH_CALUDE_second_number_divisible_by_seven_l2661_266125

theorem second_number_divisible_by_seven (a b c : ℕ+) 
  (ha : a = 105)
  (hc : c = 2436)
  (h_gcd : Nat.gcd a (Nat.gcd b c) = 7) :
  7 ∣ b := by
sorry

end NUMINAMATH_CALUDE_second_number_divisible_by_seven_l2661_266125


namespace NUMINAMATH_CALUDE_increasing_cubic_function_l2661_266123

/-- A function f(x) = x³ + ax - 2 is increasing on [1, +∞) if and only if a ≥ -3 -/
theorem increasing_cubic_function (a : ℝ) :
  (∀ x ≥ 1, Monotone (fun x => x^3 + a*x - 2)) ↔ a ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_l2661_266123


namespace NUMINAMATH_CALUDE_break_even_point_correct_l2661_266130

/-- The cost to mold each handle in dollars -/
def moldCost : ℝ := 0.60

/-- The fixed cost to run the molding machine per week in dollars -/
def fixedCost : ℝ := 7640

/-- The selling price per handle in dollars -/
def sellingPrice : ℝ := 4.60

/-- The number of handles needed to break even -/
def breakEvenPoint : ℕ := 1910

/-- Theorem stating that the calculated break-even point is correct -/
theorem break_even_point_correct :
  ↑breakEvenPoint * (sellingPrice - moldCost) = fixedCost :=
sorry

end NUMINAMATH_CALUDE_break_even_point_correct_l2661_266130


namespace NUMINAMATH_CALUDE_two_pairs_four_shoes_l2661_266159

/-- Given that a person buys a certain number of pairs of shoes, and each pair consists of a certain number of shoes, calculate the total number of new shoes. -/
def total_new_shoes (pairs_bought : ℕ) (shoes_per_pair : ℕ) : ℕ :=
  pairs_bought * shoes_per_pair

/-- Theorem stating that buying 2 pairs of shoes, with 2 shoes per pair, results in 4 new shoes. -/
theorem two_pairs_four_shoes :
  total_new_shoes 2 2 = 4 := by
  sorry

#eval total_new_shoes 2 2

end NUMINAMATH_CALUDE_two_pairs_four_shoes_l2661_266159


namespace NUMINAMATH_CALUDE_blood_donation_theorem_l2661_266168

/-- Represents the number of people for each blood type -/
structure BloodDonors where
  typeO : Nat
  typeA : Nat
  typeB : Nat
  typeAB : Nat

/-- Calculates the number of ways to select one person to donate blood -/
def selectOneDonor (donors : BloodDonors) : Nat :=
  donors.typeO + donors.typeA + donors.typeB + donors.typeAB

/-- Calculates the number of ways to select one person from each blood type -/
def selectFourDonors (donors : BloodDonors) : Nat :=
  donors.typeO * donors.typeA * donors.typeB * donors.typeAB

/-- Theorem statement for the blood donation problem -/
theorem blood_donation_theorem (donors : BloodDonors) :
  selectOneDonor donors = donors.typeO + donors.typeA + donors.typeB + donors.typeAB ∧
  selectFourDonors donors = donors.typeO * donors.typeA * donors.typeB * donors.typeAB := by
  sorry

/-- Example with the given numbers -/
def example_donors : BloodDonors :=
  { typeO := 28, typeA := 7, typeB := 9, typeAB := 3 }

#eval selectOneDonor example_donors  -- Expected: 47
#eval selectFourDonors example_donors  -- Expected: 5292

end NUMINAMATH_CALUDE_blood_donation_theorem_l2661_266168


namespace NUMINAMATH_CALUDE_workshop_2_production_l2661_266112

/-- Represents the production and sampling data for a factory with three workshops -/
structure FactoryData where
  total_production : ℕ
  sample_1 : ℕ
  sample_2 : ℕ
  sample_3 : ℕ

/-- Checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- The main theorem about the factory's production -/
theorem workshop_2_production (data : FactoryData) 
    (h_total : data.total_production = 3600)
    (h_arithmetic : isArithmeticSequence data.sample_1 data.sample_2 data.sample_3) :
    data.sample_2 = 1200 := by
  sorry


end NUMINAMATH_CALUDE_workshop_2_production_l2661_266112


namespace NUMINAMATH_CALUDE_sine_negative_half_solutions_l2661_266160

theorem sine_negative_half_solutions : 
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, 0 ≤ x ∧ x < 2*π ∧ Real.sin x = -0.5) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_negative_half_solutions_l2661_266160


namespace NUMINAMATH_CALUDE_min_value_theorem_l2661_266166

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 7) :
  (1 / (1 + a)) + (4 / (2 + b)) ≥ (13 + 4 * Real.sqrt 3) / 14 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 7 ∧
    (1 / (1 + a₀)) + (4 / (2 + b₀)) = (13 + 4 * Real.sqrt 3) / 14 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2661_266166


namespace NUMINAMATH_CALUDE_money_distribution_l2661_266119

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (bc_sum : B + C = 350)
  (c_amount : C = 50) :
  A + C = 200 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l2661_266119


namespace NUMINAMATH_CALUDE_faster_train_length_l2661_266198

/-- Calculates the length of a faster train given the speeds of two trains and the time taken for the faster train to cross a man in the slower train. -/
theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (crossing_time : ℝ)
  (h1 : faster_speed = 72)
  (h2 : slower_speed = 36)
  (h3 : crossing_time = 37)
  (h4 : faster_speed > slower_speed) :
  let relative_speed := faster_speed - slower_speed
  let relative_speed_ms := relative_speed * (1000 / 3600)
  relative_speed_ms * crossing_time = 370 := by
  sorry

#check faster_train_length

end NUMINAMATH_CALUDE_faster_train_length_l2661_266198


namespace NUMINAMATH_CALUDE_sum_of_three_hexagons_l2661_266186

theorem sum_of_three_hexagons :
  ∀ (square hexagon : ℚ),
  (3 * square + 2 * hexagon = 18) →
  (2 * square + 3 * hexagon = 20) →
  (3 * hexagon = 72 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_hexagons_l2661_266186


namespace NUMINAMATH_CALUDE_diamonds_in_figure_l2661_266129

-- Define the sequence of figures
def F (n : ℕ) : ℕ :=
  2 * n^2 - 2 * n + 1

-- State the theorem
theorem diamonds_in_figure (n : ℕ) (h : n ≥ 1) : 
  F n = 2 * n^2 - 2 * n + 1 :=
by sorry

-- Verify the result for F_20
example : F 20 = 761 :=
by sorry

end NUMINAMATH_CALUDE_diamonds_in_figure_l2661_266129


namespace NUMINAMATH_CALUDE_find_number_l2661_266148

theorem find_number (x n : ℚ) : 
  x = 4 → 
  5 * x + 3 = n * (x - 17) → 
  n = -23 / 13 := by
sorry

end NUMINAMATH_CALUDE_find_number_l2661_266148


namespace NUMINAMATH_CALUDE_negative_roots_existence_l2661_266192

theorem negative_roots_existence (p : ℝ) :
  p > 3/5 →
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^4 + 2*p*x₁^3 + p*x₁^2 + x₁^2 + 2*p*x₁ + 1 = 0 ∧
  x₂^4 + 2*p*x₂^3 + p*x₂^2 + x₂^2 + 2*p*x₂ + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_roots_existence_l2661_266192


namespace NUMINAMATH_CALUDE_harry_apples_l2661_266135

theorem harry_apples (x : ℕ) : x + 5 = 84 → x = 79 := by
  sorry

end NUMINAMATH_CALUDE_harry_apples_l2661_266135


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l2661_266188

theorem monomial_sum_condition (a b : ℝ) (m n : ℕ) :
  (∃ k : ℝ, ∃ p q : ℕ, 2 * a^(m+2) * b^(2*n+2) + a^3 * b^8 = k * a^p * b^q) →
  m = 1 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l2661_266188


namespace NUMINAMATH_CALUDE_chloe_shoes_altered_l2661_266140

/-- Given the cost per shoe and total cost, calculate the number of pairs of shoes to be altered. -/
def shoesAltered (costPerShoe : ℕ) (totalCost : ℕ) : ℕ :=
  (totalCost / costPerShoe) / 2

/-- Theorem: Given the specific costs, prove that Chloe wants to get 14 pairs of shoes altered. -/
theorem chloe_shoes_altered :
  shoesAltered 37 1036 = 14 := by
  sorry

end NUMINAMATH_CALUDE_chloe_shoes_altered_l2661_266140


namespace NUMINAMATH_CALUDE_min_bullseyes_for_victory_l2661_266164

/-- Represents the possible scores in the archery tournament -/
inductive Score
  | bullseye : Score
  | ten : Score
  | five : Score
  | three : Score
  | zero : Score

/-- Convert a Score to its numerical value -/
def score_value : Score → Nat
  | Score.bullseye => 12
  | Score.ten => 10
  | Score.five => 5
  | Score.three => 3
  | Score.zero => 0

/-- The total number of shots in the tournament -/
def total_shots : Nat := 120

/-- The number of shots already taken -/
def shots_taken : Nat := 60

/-- Alex's lead after half the tournament -/
def alex_lead : Nat := 70

/-- Alex's minimum score per shot -/
def alex_min_score : Nat := 5

/-- The maximum possible score per shot -/
def max_score_per_shot : Nat := 12

/-- Theorem: The minimum number of consecutive bullseyes Alex needs to guarantee victory is 51 -/
theorem min_bullseyes_for_victory :
  ∀ n : Nat,
  (∀ m : Nat, m < n → 
    ∃ opponent_score : Nat,
    opponent_score ≤ (total_shots - shots_taken) * max_score_per_shot ∧
    alex_lead + n * score_value Score.bullseye + (total_shots - shots_taken - n) * alex_min_score ≤ opponent_score) ∧
  (∀ opponent_score : Nat,
   opponent_score ≤ (total_shots - shots_taken) * max_score_per_shot →
   alex_lead + n * score_value Score.bullseye + (total_shots - shots_taken - n) * alex_min_score > opponent_score) →
  n = 51 := by
  sorry

end NUMINAMATH_CALUDE_min_bullseyes_for_victory_l2661_266164


namespace NUMINAMATH_CALUDE_problem_part1_problem_part2_l2661_266176

-- Part 1
theorem problem_part1 (m n : ℕ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : 3 * m + 2 * n = 225) (h4 : Nat.gcd m n = 15) : 
  m + n = 105 := by
  sorry

-- Part 2
theorem problem_part2 (m n : ℕ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : 3 * m + 2 * n = 225) (h4 : Nat.lcm m n = 45) : 
  m + n = 90 := by
  sorry

end NUMINAMATH_CALUDE_problem_part1_problem_part2_l2661_266176


namespace NUMINAMATH_CALUDE_f_sum_symmetric_max_a_bound_l2661_266131

noncomputable def f (x : ℝ) : ℝ := (3 * Real.exp x) / (1 + Real.exp x)

theorem f_sum_symmetric (x : ℝ) : f x + f (-x) = 3 := by sorry

theorem max_a_bound (a : ℝ) :
  (∀ x > 0, f (4 - a * x) + f (x^2) ≥ 3) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_max_a_bound_l2661_266131


namespace NUMINAMATH_CALUDE_curve_equation_and_cosine_value_l2661_266196

-- Define the curves C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ^2 * (3 + Real.sin θ^2) = 12
def C₂ (x y t α : ℝ) : Prop := x = 1 + t * Real.cos α ∧ y = t * Real.sin α

-- Define the condition for α
def α_condition (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (α : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), C₂ A.1 A.2 t₁ α ∧ C₂ B.1 B.2 t₂ α ∧
  (A.1^2 / 4 + A.2^2 / 3 = 1) ∧ (B.1^2 / 4 + B.2^2 / 3 = 1)

-- Define the distance condition
def distance_condition (A B P : ℝ × ℝ) : Prop :=
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) +
  Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 7/2

theorem curve_equation_and_cosine_value
  (α : ℝ) (A B P : ℝ × ℝ)
  (h_α : α_condition α)
  (h_int : intersection_points A B α)
  (h_dist : distance_condition A B P) :
  (∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) ↔ (∃ (ρ θ : ℝ), C₁ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)) ∧
  Real.cos α = 2 * Real.sqrt 7 / 7 :=
sorry

end NUMINAMATH_CALUDE_curve_equation_and_cosine_value_l2661_266196


namespace NUMINAMATH_CALUDE_right_triangle_area_l2661_266178

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  b = (2/3) * a →
  b = (2/3) * c →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 32/9 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2661_266178


namespace NUMINAMATH_CALUDE_shop_length_calculation_l2661_266114

/-- Given a shop with specified rent and dimensions, calculate its length -/
theorem shop_length_calculation (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) :
  monthly_rent = 2400 →
  width = 8 →
  annual_rent_per_sqft = 360 →
  (monthly_rent * 12) / (width * annual_rent_per_sqft) = 10 := by
  sorry

end NUMINAMATH_CALUDE_shop_length_calculation_l2661_266114


namespace NUMINAMATH_CALUDE_not_both_follow_control_principle_option_d_is_incorrect_l2661_266182

/-- Represents an experimental approach -/
inductive ExperimentalApproach
| BlankControl
| RepeatWithSameSoil

/-- Represents a scientific principle -/
inductive ScientificPrinciple
| Control
| Repeatability

/-- Function to determine which principle an approach follows -/
def principleFollowed (approach : ExperimentalApproach) : ScientificPrinciple :=
  match approach with
  | ExperimentalApproach.BlankControl => ScientificPrinciple.Control
  | ExperimentalApproach.RepeatWithSameSoil => ScientificPrinciple.Repeatability

/-- Theorem stating that not both approaches follow the control principle -/
theorem not_both_follow_control_principle :
  ¬(principleFollowed ExperimentalApproach.BlankControl = ScientificPrinciple.Control ∧
     principleFollowed ExperimentalApproach.RepeatWithSameSoil = ScientificPrinciple.Control) :=
by sorry

/-- Main theorem proving that the statement in option D is incorrect -/
theorem option_d_is_incorrect :
  ¬(∀ (approach : ExperimentalApproach), principleFollowed approach = ScientificPrinciple.Control) :=
by sorry

end NUMINAMATH_CALUDE_not_both_follow_control_principle_option_d_is_incorrect_l2661_266182


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2661_266104

theorem sufficient_not_necessary (a b : ℝ) :
  (((a - b) * a^2 < 0) → (a < b)) ∧
  (∃ a b : ℝ, (a < b) ∧ ((a - b) * a^2 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2661_266104


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2661_266147

theorem polar_to_rectangular_conversion :
  let r : ℝ := 8
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 4 * Real.sqrt 2) ∧ (y = 4 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2661_266147


namespace NUMINAMATH_CALUDE_square_equal_implies_abs_equal_l2661_266151

theorem square_equal_implies_abs_equal (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  sorry

end NUMINAMATH_CALUDE_square_equal_implies_abs_equal_l2661_266151


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l2661_266120

def total_people : ℕ := 10
def num_men : ℕ := 7
def num_women : ℕ := 3
def selection_size : ℕ := 3

theorem probability_at_least_one_woman :
  let prob_no_women := (num_men.choose selection_size : ℚ) / (total_people.choose selection_size : ℚ)
  (1 : ℚ) - prob_no_women = 17 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l2661_266120


namespace NUMINAMATH_CALUDE_cycle_iff_minimal_cut_l2661_266153

-- Define a planar multigraph
structure PlanarMultigraph where
  V : Type*  -- Vertex set
  E : Type*  -- Edge set
  is_planar : Bool
  is_connected : Bool

-- Define a dual graph
def DualGraph (G : PlanarMultigraph) : PlanarMultigraph := sorry

-- Define a cycle in a graph
def is_cycle (G : PlanarMultigraph) (E : Set G.E) : Prop := sorry

-- Define a cut in a graph
def is_cut (G : PlanarMultigraph) (E : Set G.E) : Prop := sorry

-- Define a minimal cut
def is_minimal_cut (G : PlanarMultigraph) (E : Set G.E) : Prop := sorry

-- Define the dual edge set
def dual_edge_set (G : PlanarMultigraph) (E : Set G.E) : Set (DualGraph G).E := sorry

-- Main theorem
theorem cycle_iff_minimal_cut (G : PlanarMultigraph) (E : Set G.E) :
  is_cycle G E ↔ is_minimal_cut (DualGraph G) (dual_edge_set G E) := by sorry

end NUMINAMATH_CALUDE_cycle_iff_minimal_cut_l2661_266153


namespace NUMINAMATH_CALUDE_apple_pies_count_l2661_266195

def total_apple_weight : ℕ := 120
def applesauce_fraction : ℚ := 1/2
def pounds_per_pie : ℕ := 4

theorem apple_pies_count :
  (total_apple_weight * (1 - applesauce_fraction) / pounds_per_pie : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_pies_count_l2661_266195


namespace NUMINAMATH_CALUDE_line_plane_intersection_l2661_266110

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary operations and relations
variable (intersect : Plane → Plane → Line)
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersects : Line → Line → Prop)

-- State the theorem
theorem line_plane_intersection 
  (m n : Line) (α β : Plane) :
  intersect α β = m → subset n α → 
  (parallel m n) ∨ (intersects m n) := by
  sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l2661_266110


namespace NUMINAMATH_CALUDE_exactly_one_real_solution_iff_l2661_266174

/-- The cubic equation in x with parameter a -/
def cubic_equation (a x : ℝ) : ℝ :=
  x^3 - a*x^2 - (a+1)*x + a^2 - 2

/-- The condition for exactly one real solution -/
def has_exactly_one_real_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation a x = 0

/-- Theorem stating the condition for exactly one real solution -/
theorem exactly_one_real_solution_iff (a : ℝ) :
  has_exactly_one_real_solution a ↔ a < 7/4 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_real_solution_iff_l2661_266174
