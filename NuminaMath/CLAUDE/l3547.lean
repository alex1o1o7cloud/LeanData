import Mathlib

namespace sufficient_but_not_necessary_l3547_354789

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  ¬((a - 1) * (a - 2) = 0 → a = 2) :=
by sorry

end sufficient_but_not_necessary_l3547_354789


namespace range_of_a_l3547_354795

theorem range_of_a (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 5 ∧ ∀ x ∈ s, (1 + a ≤ x ∧ x < 2)) → 
  (-5 < a ∧ a ≤ -4) := by
sorry

end range_of_a_l3547_354795


namespace chimney_bricks_l3547_354773

/-- The number of bricks in the chimney -/
def h : ℕ := 360

/-- Brenda's time to build the chimney alone (in hours) -/
def brenda_time : ℕ := 8

/-- Brandon's time to build the chimney alone (in hours) -/
def brandon_time : ℕ := 12

/-- Efficiency decrease when working together (in bricks per hour) -/
def efficiency_decrease : ℕ := 15

/-- Time taken to build the chimney together (in hours) -/
def time_together : ℕ := 6

theorem chimney_bricks : 
  time_together * ((h / brenda_time + h / brandon_time) - efficiency_decrease) = h := by
  sorry

#check chimney_bricks

end chimney_bricks_l3547_354773


namespace rectangle_division_l3547_354739

/-- Given a rectangle with length 3y and width y, divided into a smaller rectangle
    of length x and width y-x surrounded by four congruent right-angled triangles,
    this theorem proves the perimeter of one triangle and the area of the smaller rectangle. -/
theorem rectangle_division (x y : ℝ) : 
  let triangle_perimeter := 3 * y + Real.sqrt (2 * x^2 - 6 * y * x + 9 * y^2)
  let smaller_rectangle_area := x * y - x^2
  ∀ (triangle_side_a triangle_side_b : ℝ),
    triangle_side_a = x ∧ 
    triangle_side_b = 3 * y - x →
    triangle_perimeter = triangle_side_a + triangle_side_b + 
      Real.sqrt (triangle_side_a^2 + triangle_side_b^2) ∧
    smaller_rectangle_area = x * (y - x) := by
  sorry

end rectangle_division_l3547_354739


namespace unique_injective_function_l3547_354767

/-- Iterate a function n times -/
def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (iterate f n x)

/-- The property that f must satisfy -/
def satisfies_equation (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, iterate f (f a) b * iterate f (f b) a = (f (a + b))^2

/-- The main theorem statement -/
theorem unique_injective_function :
  ∀ f : ℕ → ℕ, Function.Injective f → satisfies_equation f → ∀ x : ℕ, f x = x + 1 := by
  sorry


end unique_injective_function_l3547_354767


namespace prob_A_value_l3547_354748

/-- The probability that person A speaks the truth -/
def prob_A : ℝ := sorry

/-- The probability that person B speaks the truth -/
def prob_B : ℝ := 0.6

/-- The probability that both A and B speak the truth simultaneously -/
def prob_A_and_B : ℝ := 0.48

/-- The events of A and B speaking the truth are independent -/
axiom independence : prob_A_and_B = prob_A * prob_B

theorem prob_A_value : prob_A = 0.8 := by
  sorry

end prob_A_value_l3547_354748


namespace sqrt_n_squared_plus_n_bounds_l3547_354723

theorem sqrt_n_squared_plus_n_bounds (n : ℕ) :
  (n : ℝ) + 0.4 < Real.sqrt ((n : ℝ)^2 + n) ∧ Real.sqrt ((n : ℝ)^2 + n) < (n : ℝ) + 0.5 :=
by sorry

end sqrt_n_squared_plus_n_bounds_l3547_354723


namespace point_on_x_axis_l3547_354793

theorem point_on_x_axis (m : ℝ) :
  (m + 5, 2 * m + 8) = (1, 0) ↔ (m + 5, 2 * m + 8).2 = 0 := by
sorry

end point_on_x_axis_l3547_354793


namespace simplify_trig_expression_l3547_354777

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin 4 * Real.cos 4) = Real.cos 4 - Real.sin 4 := by
  sorry

end simplify_trig_expression_l3547_354777


namespace fraction_to_decimal_l3547_354700

theorem fraction_to_decimal : (47 : ℚ) / 160 = 0.29375 := by sorry

end fraction_to_decimal_l3547_354700


namespace tan_product_identity_l3547_354711

theorem tan_product_identity : (1 + Real.tan (18 * π / 180)) * (1 + Real.tan (27 * π / 180)) = 2 := by
  sorry

end tan_product_identity_l3547_354711


namespace extreme_value_of_f_l3547_354770

-- Define the function
def f (x : ℝ) : ℝ := (x^2 - 1)^3 + 1

-- State the theorem
theorem extreme_value_of_f :
  ∃ (e : ℝ), e = 0 ∧ ∀ (x : ℝ), f x ≥ e :=
sorry

end extreme_value_of_f_l3547_354770


namespace binomial_30_3_minus_10_l3547_354763

theorem binomial_30_3_minus_10 : Nat.choose 30 3 - 10 = 4050 := by
  sorry

end binomial_30_3_minus_10_l3547_354763


namespace line_circle_intersection_k_range_l3547_354727

/-- Given a line y = kx + 3 intersecting a circle (x-4)^2 + (y-3)^2 = 4 at two points M and N,
    where |MN| ≥ 2√3, prove that -√15/15 ≤ k ≤ √15/15 -/
theorem line_circle_intersection_k_range (k : ℝ) :
  (∃ M N : ℝ × ℝ,
    (M.1 - 4)^2 + (M.2 - 3)^2 = 4 ∧
    (N.1 - 4)^2 + (N.2 - 3)^2 = 4 ∧
    M.2 = k * M.1 + 3 ∧
    N.2 = k * N.1 + 3 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) →
  -Real.sqrt 15 / 15 ≤ k ∧ k ≤ Real.sqrt 15 / 15 := by
  sorry


end line_circle_intersection_k_range_l3547_354727


namespace height_classification_groups_l3547_354752

/-- Given the heights of students in a class, calculate the number of groups needed for classification --/
theorem height_classification_groups 
  (tallest_height : ℕ) 
  (shortest_height : ℕ) 
  (class_width : ℕ) 
  (h1 : tallest_height = 175) 
  (h2 : shortest_height = 150) 
  (h3 : class_width = 3) : 
  ℕ := by
  sorry

#check height_classification_groups

end height_classification_groups_l3547_354752


namespace B_subset_M_M_closed_under_mult_l3547_354745

-- Define the set M
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

-- Define the set B
def B : Set ℤ := {b | ∃ n : ℕ, b = 2*n + 1}

-- Theorem 1: B is a subset of M
theorem B_subset_M : B ⊆ M := by sorry

-- Theorem 2: M is closed under multiplication
theorem M_closed_under_mult : ∀ a₁ a₂ : ℤ, a₁ ∈ M → a₂ ∈ M → (a₁ * a₂) ∈ M := by sorry

end B_subset_M_M_closed_under_mult_l3547_354745


namespace complex_product_QED_l3547_354780

theorem complex_product_QED : 
  let Q : ℂ := 5 + 3 * Complex.I
  let E : ℂ := 2 * Complex.I
  let D : ℂ := 5 - 3 * Complex.I
  Q * E * D = 68 * Complex.I := by
sorry

end complex_product_QED_l3547_354780


namespace verna_haley_weight_difference_l3547_354730

/-- Given the weights of Verna, Haley, and Sherry, prove that Verna weighs 17 pounds more than Haley -/
theorem verna_haley_weight_difference :
  ∀ (verna_weight haley_weight sherry_weight : ℕ),
    verna_weight > haley_weight →
    verna_weight = sherry_weight / 2 →
    haley_weight = 103 →
    verna_weight + sherry_weight = 360 →
    verna_weight - haley_weight = 17 := by
  sorry

end verna_haley_weight_difference_l3547_354730


namespace min_max_values_of_f_l3547_354728

def f (x : ℝ) : ℝ := 4 * x^2 + 8 * x + 16

theorem min_max_values_of_f :
  let a := -3
  let b := 2
  ∃ (x_min x_max : ℝ), a ≤ x_min ∧ x_min ≤ b ∧ a ≤ x_max ∧ x_max ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    f x_min = 12 ∧ f x_max = 48 :=
by sorry

end min_max_values_of_f_l3547_354728


namespace remainder_theorem_l3547_354742

theorem remainder_theorem (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := by
  sorry

end remainder_theorem_l3547_354742


namespace tenth_term_of_specific_arithmetic_sequence_l3547_354759

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

/-- The 10th term of the arithmetic sequence with first term 10 and common difference -2 is -8 -/
theorem tenth_term_of_specific_arithmetic_sequence :
  arithmeticSequence 10 (-2) 10 = -8 := by
  sorry

end tenth_term_of_specific_arithmetic_sequence_l3547_354759


namespace yogurt_combinations_l3547_354722

theorem yogurt_combinations (flavors : ℕ) (toppings : ℕ) (sizes : ℕ) :
  flavors = 5 → toppings = 8 → sizes = 3 →
  flavors * (toppings.choose 2) * sizes = 420 :=
by sorry

end yogurt_combinations_l3547_354722


namespace alien_martian_limb_difference_l3547_354782

/-- The number of arms an Alien has -/
def alien_arms : ℕ := 3

/-- The number of legs an Alien has -/
def alien_legs : ℕ := 8

/-- The number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- The number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

/-- The total number of limbs for one Alien -/
def alien_limbs : ℕ := alien_arms + alien_legs

/-- The total number of limbs for one Martian -/
def martian_limbs : ℕ := martian_arms + martian_legs

/-- The number of Aliens and Martians we're comparing -/
def number_of_creatures : ℕ := 5

theorem alien_martian_limb_difference :
  number_of_creatures * alien_limbs - number_of_creatures * martian_limbs = 5 := by
  sorry

end alien_martian_limb_difference_l3547_354782


namespace inequality_proof_l3547_354791

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c - a) + b^2 * (a + c - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end inequality_proof_l3547_354791


namespace roots_equation_value_l3547_354747

theorem roots_equation_value (α β : ℝ) :
  α^2 - 3*α - 2 = 0 →
  β^2 - 3*β - 2 = 0 →
  5 * α^4 + 12 * β^3 = 672.5 + 31.5 * Real.sqrt 17 := by
  sorry

end roots_equation_value_l3547_354747


namespace quadratic_function_properties_l3547_354769

/-- A quadratic function satisfying specific conditions -/
def f (x : ℝ) : ℝ := 2 * (x - 2)^2 - 8

/-- The theorem stating the properties of the quadratic function f -/
theorem quadratic_function_properties :
  (∀ x, f (x + 2) = f (2 - x)) ∧
  (∀ x, f x ≥ -8) ∧
  (f 1 = -6) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) 4, -8 ≤ f x ∧ f x < f (-1)) := by
  sorry

#check quadratic_function_properties

end quadratic_function_properties_l3547_354769


namespace geometric_sequence_sum_minimum_l3547_354714

theorem geometric_sequence_sum_minimum (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  q > 1 →
  (∀ n, a n = a 1 * q^(n-1)) →
  (∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) →
  S 4 = 2 * S 2 + 1 →
  ∃ S_6_min : ℝ, S_6_min = 2 * Real.sqrt 3 + 3 ∧ S 6 ≥ S_6_min :=
by sorry

end geometric_sequence_sum_minimum_l3547_354714


namespace shekars_average_marks_l3547_354743

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 47
def biology_score : ℕ := 85

def total_subjects : ℕ := 5

theorem shekars_average_marks :
  (mathematics_score + science_score + social_studies_score + english_score + biology_score) / total_subjects = 71 := by
  sorry

end shekars_average_marks_l3547_354743


namespace min_value_of_a2_plus_b2_l3547_354754

/-- Given a quadratic function f(x) = x^2 + ax + b - 3 that passes through (2, 0),
    the minimum value of a^2 + b^2 is 1/5 -/
theorem min_value_of_a2_plus_b2 (a b : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + b - 3 = 0) → x = 2) → 
  (∃ m : ℝ, m = (1 : ℝ) / 5 ∧ ∀ a' b' : ℝ, (∀ x : ℝ, (x^2 + a'*x + b' - 3 = 0) → x = 2) → a'^2 + b'^2 ≥ m) :=
sorry

end min_value_of_a2_plus_b2_l3547_354754


namespace triangle_angle_measure_l3547_354788

/-- Given a triangle ABC with side lengths a, b, and c, if a^2 + b^2 - c^2 = ab, 
    then the measure of angle C is 60°. -/
theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
    (h_eq : a^2 + b^2 - c^2 = a * b) : 
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 3 := by
  sorry

end triangle_angle_measure_l3547_354788


namespace equation_solutions_l3547_354776

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x - 2)^6 + (x - 6)^6 = 432

-- Define the approximate solutions
def solution1 : ℝ := 4.795
def solution2 : ℝ := 3.205

-- State the theorem
theorem equation_solutions :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (∀ (x : ℝ), equation x → (|x - solution1| < ε ∨ |x - solution2| < ε)) :=
sorry

end equation_solutions_l3547_354776


namespace three_intersections_implies_a_value_l3547_354751

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then Real.sin x else x^3 - 9*x^2 + 25*x + a

theorem three_intersections_implies_a_value (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = x₁ ∧ f a x₂ = x₂ ∧ f a x₃ = x₃) →
  a = -20 ∨ a = -16 := by
  sorry

end three_intersections_implies_a_value_l3547_354751


namespace triangle_value_l3547_354794

theorem triangle_value (triangle q r : ℚ) 
  (eq1 : triangle + q = 75)
  (eq2 : (triangle + q) + r = 138)
  (eq3 : r = q / 3) :
  triangle = -114 := by
sorry

end triangle_value_l3547_354794


namespace cubic_equation_solution_l3547_354735

theorem cubic_equation_solution :
  ∃ x : ℝ, x^3 + 3*x^2 + 3*x + 7 = 0 ∧ x = -1 - Real.rpow 6 (1/3 : ℝ) :=
by sorry

end cubic_equation_solution_l3547_354735


namespace same_terminal_side_M_subset_N_l3547_354787

/-- Represents an angle in degrees -/
structure Angle :=
  (value : ℝ)

/-- Defines the terminal side of an angle -/
def terminalSide (a : Angle) : ℝ × ℝ := sorry

/-- Defines set M -/
def M : Set ℝ := {x | ∃ k : ℤ, x = 45 + k * 90}

/-- Defines set N -/
def N : Set ℝ := {y | ∃ k : ℤ, y = 90 + k * 45}

/-- Theorem stating that angles α and β have the same terminal side -/
theorem same_terminal_side (k : ℤ) :
  terminalSide (Angle.mk ((2 * k + 1) * 180)) = terminalSide (Angle.mk ((4 * k + 1) * 180)) ∧
  terminalSide (Angle.mk ((2 * k + 1) * 180)) = terminalSide (Angle.mk ((4 * k - 1) * 180)) :=
sorry

/-- Theorem stating that M is a subset of N -/
theorem M_subset_N : M ⊆ N :=
sorry

end same_terminal_side_M_subset_N_l3547_354787


namespace complement_of_P_l3547_354779

def U : Set ℝ := Set.univ

def P : Set ℝ := {x : ℝ | x^2 - 5*x - 6 ≥ 0}

theorem complement_of_P (x : ℝ) : x ∈ Set.compl P ↔ x ∈ Set.Ioo (-1) 6 := by
  sorry

end complement_of_P_l3547_354779


namespace seventh_term_of_geometric_sequence_l3547_354760

def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℤ) (q : ℤ) 
  (h_seq : geometric_sequence a q)
  (h_a4 : a 4 = 27)
  (h_q : q = -3) :
  a 7 = -729 := by
  sorry

end seventh_term_of_geometric_sequence_l3547_354760


namespace problem_1_problem_2_problem_3_l3547_354792

-- Problem 1
theorem problem_1 (x : ℚ) : 
  16 * (6*x - 1) * (2*x - 1) * (3*x + 1) * (x - 1) + 25 = (24*x^2 - 16*x - 3)^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℚ) : 
  (6*x - 1) * (2*x - 1) * (3*x - 1) * (x - 1) + x^2 = (6*x^2 - 6*x + 1)^2 := by sorry

-- Problem 3
theorem problem_3 (x : ℚ) : 
  (6*x - 1) * (4*x - 1) * (3*x - 1) * (x - 1) + 9*x^4 = (9*x^2 - 7*x + 1)^2 := by sorry

end problem_1_problem_2_problem_3_l3547_354792


namespace nursing_home_flowers_l3547_354731

/-- The number of flower sets bought by Mayor Harvey -/
def num_sets : ℕ := 3

/-- The number of flowers in each set -/
def flowers_per_set : ℕ := 90

/-- The total number of flowers bought for the nursing home -/
def total_flowers : ℕ := num_sets * flowers_per_set

theorem nursing_home_flowers : total_flowers = 270 := by
  sorry

end nursing_home_flowers_l3547_354731


namespace complex_magnitude_problem_l3547_354775

theorem complex_magnitude_problem (z : ℂ) (h : 3 * z * Complex.I = -6 + 2 * Complex.I) :
  Complex.abs z = 2 * Real.sqrt 10 / 3 := by
  sorry

end complex_magnitude_problem_l3547_354775


namespace complex_modulus_problem_l3547_354799

theorem complex_modulus_problem (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (2 - i) / (a + i) = b * i) (b : ℝ) : 
  Complex.abs (4 * a + Complex.I * Real.sqrt 2) = Real.sqrt 6 := by
  sorry

end complex_modulus_problem_l3547_354799


namespace right_triangle_side_length_l3547_354708

noncomputable def triangle_side_length (angleA : Real) (sideBC : Real) : Real :=
  sideBC * Real.tan angleA

theorem right_triangle_side_length :
  let angleA : Real := 30 * π / 180  -- Convert 30° to radians
  let sideBC : Real := 12
  let sideAB : Real := triangle_side_length angleA sideBC
  ∀ ε > 0, |sideAB - 6.9| < ε :=
sorry

end right_triangle_side_length_l3547_354708


namespace nested_root_evaluation_l3547_354712

theorem nested_root_evaluation (N : ℝ) (h : N > 1) :
  (N * (N * (N ^ (1/3)) ^ (1/4))) ^ (1/3) = N ^ (4/9) := by
  sorry

end nested_root_evaluation_l3547_354712


namespace permutation_formula_l3547_354781

def A (n k : ℕ) : ℕ :=
  (List.range k).foldl (fun acc i => acc * (n - i)) n

theorem permutation_formula (n k : ℕ) (h : k ≤ n) :
  A n k = (List.range k).foldl (fun acc i => acc * (n - i)) n :=
by sorry

end permutation_formula_l3547_354781


namespace power_equality_l3547_354768

theorem power_equality : 32^4 * 4^5 = 2^30 := by
  sorry

end power_equality_l3547_354768


namespace max_sum_of_products_l3547_354746

/-- The maximum sum of products for four distinct values from {3, 4, 5, 6} -/
theorem max_sum_of_products : 
  ∀ (f g h j : ℕ), 
    f ∈ ({3, 4, 5, 6} : Set ℕ) → 
    g ∈ ({3, 4, 5, 6} : Set ℕ) → 
    h ∈ ({3, 4, 5, 6} : Set ℕ) → 
    j ∈ ({3, 4, 5, 6} : Set ℕ) → 
    f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j → 
    f * g + g * h + h * j + j * f ≤ 80 :=
by sorry

end max_sum_of_products_l3547_354746


namespace minimum_employees_needed_l3547_354729

theorem minimum_employees_needed 
  (water_pollution : ℕ) 
  (air_pollution : ℕ) 
  (both : ℕ) 
  (h1 : water_pollution = 85) 
  (h2 : air_pollution = 73) 
  (h3 : both = 27) 
  (h4 : both ≤ water_pollution ∧ both ≤ air_pollution) : 
  water_pollution + air_pollution - both = 131 :=
sorry

end minimum_employees_needed_l3547_354729


namespace iodine_atom_radius_scientific_notation_l3547_354704

theorem iodine_atom_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000000133 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -8 :=
by sorry

end iodine_atom_radius_scientific_notation_l3547_354704


namespace rectangle_minimum_width_l3547_354750

/-- A rectangle with length 1.5 times its width and area at least 450 square feet has a minimum width of 10√3 feet. -/
theorem rectangle_minimum_width (w : ℝ) (h_positive : w > 0) : 
  1.5 * w * w ≥ 450 → w ≥ 10 * Real.sqrt 3 :=
by
  sorry

#check rectangle_minimum_width

end rectangle_minimum_width_l3547_354750


namespace hyperbola_eccentricity_l3547_354718

/-- The eccentricity of a hyperbola with equation y² - x²/4 = 1 is √5 -/
theorem hyperbola_eccentricity : 
  let hyperbola := fun (x y : ℝ) => y^2 - x^2/4 = 1
  ∃ e : ℝ, e = Real.sqrt 5 ∧ 
    ∀ x y : ℝ, hyperbola x y → 
      e = Real.sqrt ((1 + 4) / 1) := by
        sorry

end hyperbola_eccentricity_l3547_354718


namespace balls_color_probability_l3547_354721

def num_balls : ℕ := 6
def probability_black : ℚ := 1/2
def probability_white : ℚ := 1/2

theorem balls_color_probability :
  let favorable_outcomes := (num_balls.choose (num_balls / 2))
  let total_outcomes := 2^num_balls
  (favorable_outcomes : ℚ) / total_outcomes = 5/16 := by
sorry

end balls_color_probability_l3547_354721


namespace least_common_addition_of_primes_l3547_354778

theorem least_common_addition_of_primes (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x < y → 4 * x + y = 87 → x + y = 81 := by
  sorry

end least_common_addition_of_primes_l3547_354778


namespace mark_and_carolyn_money_sum_l3547_354706

theorem mark_and_carolyn_money_sum :
  let mark_money : ℚ := 7/8
  let carolyn_money : ℚ := 2/5
  (mark_money + carolyn_money : ℚ) = 1.275 := by sorry

end mark_and_carolyn_money_sum_l3547_354706


namespace sticker_theorem_l3547_354757

def sticker_problem (initial_stickers : ℕ) (stickers_per_friend : ℕ) (num_friends : ℕ) 
  (remaining_stickers : ℕ) (justin_diff : ℕ) : Prop :=
  let total_to_friends := stickers_per_friend * num_friends
  let total_given_away := initial_stickers - remaining_stickers
  let mandy_and_justin := total_given_away - total_to_friends
  let mandy_stickers := (mandy_and_justin + justin_diff) / 2
  mandy_stickers - total_to_friends = 2

theorem sticker_theorem : 
  sticker_problem 72 4 3 42 10 := by sorry

end sticker_theorem_l3547_354757


namespace interesting_2018_gon_after_marked_removal_l3547_354740

/-- A convex polygon with n vertices --/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry

/-- A coloring of vertices in two colors --/
def Coloring (n : ℕ) := Fin n → Bool

/-- The sum of angles at vertices of a given color in a polygon --/
def sumAngles (p : ConvexPolygon n) (c : Coloring n) (color : Bool) : ℝ := sorry

/-- A polygon is interesting if the sum of angles of one color equals the sum of angles of the other color --/
def isInteresting (p : ConvexPolygon n) (c : Coloring n) : Prop :=
  sumAngles p c true = sumAngles p c false

/-- Remove a vertex from a polygon --/
def removeVertex (p : ConvexPolygon (n + 1)) (i : Fin (n + 1)) : ConvexPolygon n := sorry

/-- The theorem to be proved --/
theorem interesting_2018_gon_after_marked_removal
  (p : ConvexPolygon 2019)
  (marked : Fin 2019)
  (h : ∀ (i : Fin 2019), i ≠ marked → ∃ (c : Coloring 2018), isInteresting (removeVertex p i) c) :
  ∃ (c : Coloring 2018), isInteresting (removeVertex p marked) c :=
sorry

end interesting_2018_gon_after_marked_removal_l3547_354740


namespace seventh_observation_l3547_354798

theorem seventh_observation (n : ℕ) (x : ℝ) (y : ℝ) :
  n = 6 →
  x = 16 →
  y = x - 1 →
  (n * x + 9) / (n + 1) = y →
  9 = 9 :=
by sorry

end seventh_observation_l3547_354798


namespace thirty_is_seventy_five_percent_of_forty_l3547_354720

theorem thirty_is_seventy_five_percent_of_forty :
  ∀ x : ℝ, (75 / 100) * x = 30 → x = 40 := by
  sorry

end thirty_is_seventy_five_percent_of_forty_l3547_354720


namespace range_of_a_l3547_354737

def p (x : ℝ) : Prop := 0 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a :
  (∀ x a : ℝ, q x a → p x) ∧
  (∃ x : ℝ, p x ∧ ∀ a : ℝ, ¬(q x a)) →
  ∀ a : ℝ, (0 ≤ a ∧ a ≤ 1/2) ↔ (∃ x : ℝ, q x a) :=
sorry

end range_of_a_l3547_354737


namespace quadratic_one_root_l3547_354701

/-- A quadratic function f(x) = x^2 - 3x + m + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 3*x + m + 2

/-- The discriminant of the quadratic function f -/
def discriminant (m : ℝ) : ℝ := (-3)^2 - 4*(1)*(m+2)

theorem quadratic_one_root (m : ℝ) : 
  (∃! x, f m x = 0) → m = 1/4 := by
  sorry

end quadratic_one_root_l3547_354701


namespace prob_different_suits_l3547_354772

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suits in a standard deck -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- Function to get the suit of a card -/
def getSuit (card : Fin 52) : Suit := sorry

/-- Probability of drawing three cards of different suits -/
def probDifferentSuits (d : Deck) : ℚ :=
  (39 : ℚ) / 51 * (26 : ℚ) / 50

/-- Theorem stating the probability of drawing three cards of different suits -/
theorem prob_different_suits (d : Deck) :
  probDifferentSuits d = 169 / 425 := by
  sorry

end prob_different_suits_l3547_354772


namespace symmetric_point_and_line_in_quadrant_l3547_354790

-- Define the symmetric point function
def symmetric_point (x y : ℝ) (a b c : ℝ) : ℝ × ℝ := sorry

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x + y + m - 1 = 0

theorem symmetric_point_and_line_in_quadrant :
  -- Statement C
  symmetric_point 1 0 1 (-1) 1 = (-1, 2) ∧
  -- Statement D
  ∀ m : ℝ, line_equation m (-1) 1 := by sorry

end symmetric_point_and_line_in_quadrant_l3547_354790


namespace wall_height_proof_l3547_354733

/-- Given a wall and a painting, proves that the wall height is 5 feet -/
theorem wall_height_proof (wall_width painting_width painting_height painting_area_percentage : ℝ) :
  wall_width = 10 ∧ 
  painting_width = 2 ∧ 
  painting_height = 4 ∧ 
  painting_area_percentage = 0.16 ∧
  painting_width * painting_height = painting_area_percentage * (wall_width * (wall_width * painting_height / (painting_width * painting_height))) →
  wall_width * painting_height / (painting_width * painting_height) = 5 := by
sorry

end wall_height_proof_l3547_354733


namespace impossible_last_digit_match_l3547_354713

theorem impossible_last_digit_match (n : ℕ) (h_n : n = 111) :
  ¬ ∃ (S : Finset ℕ),
    Finset.card S = n ∧
    (∀ x ∈ S, x ≤ 500) ∧
    (∀ x ∈ S, ∀ y ∈ S, x ≠ y → x ≠ y) ∧
    (∀ x ∈ S, x % 10 = (Finset.sum S id - x) % 10) :=
by sorry

end impossible_last_digit_match_l3547_354713


namespace triangular_prism_skew_lines_l3547_354758

/-- A triangular prism -/
structure TriangularPrism where
  vertices : Finset (ℝ × ℝ × ℝ)
  edges : Finset (Finset (ℝ × ℝ × ℝ))
  is_valid : vertices.card = 6 ∧ edges.card = 9

/-- A line in 3D space -/
def Line3D := Finset (ℝ × ℝ × ℝ)

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- The set of all lines passing through any two vertices of the prism -/
def all_lines (p : TriangularPrism) : Finset Line3D := sorry

/-- The set of all pairs of skew lines in the prism -/
def skew_line_pairs (p : TriangularPrism) : Finset (Line3D × Line3D) := sorry

theorem triangular_prism_skew_lines (p : TriangularPrism) :
  (all_lines p).card = 15 → (skew_line_pairs p).card = 36 := by
  sorry

end triangular_prism_skew_lines_l3547_354758


namespace consecutive_integers_sum_of_squares_l3547_354761

theorem consecutive_integers_sum_of_squares : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a * (a + 1) * (a + 2) = 12 * (3 * a + 3)) → 
  (a^2 + (a + 1)^2 + (a + 2)^2 = 149) := by
  sorry

end consecutive_integers_sum_of_squares_l3547_354761


namespace quadratic_rewrite_sum_l3547_354715

theorem quadratic_rewrite_sum (a b c : ℝ) :
  (∀ x, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) →
  a + b + c = 171 := by
sorry

end quadratic_rewrite_sum_l3547_354715


namespace profit_sharing_ratio_l3547_354755

def johnsons_share : ℕ := 2500
def mikes_shirt_cost : ℕ := 200
def mikes_remaining : ℕ := 800

def mikes_share : ℕ := mikes_remaining + mikes_shirt_cost

def ratio_numerator : ℕ := 2
def ratio_denominator : ℕ := 5

theorem profit_sharing_ratio :
  (mikes_share : ℚ) / johnsons_share = ratio_numerator / ratio_denominator :=
by sorry

end profit_sharing_ratio_l3547_354755


namespace negative_three_plus_nine_equals_six_l3547_354724

theorem negative_three_plus_nine_equals_six : (-3) + 9 = 6 := by
  sorry

end negative_three_plus_nine_equals_six_l3547_354724


namespace square_mirror_side_length_l3547_354797

theorem square_mirror_side_length 
  (wall_width : ℝ) 
  (wall_length : ℝ) 
  (mirror_area_ratio : ℝ) :
  wall_width = 42 →
  wall_length = 27.428571428571427 →
  mirror_area_ratio = 1 / 2 →
  ∃ (mirror_side : ℝ), 
    mirror_side = 24 ∧ 
    mirror_side^2 = mirror_area_ratio * wall_width * wall_length :=
by sorry

end square_mirror_side_length_l3547_354797


namespace max_value_of_fraction_l3547_354738

theorem max_value_of_fraction (x : ℝ) (h : x ≠ 0) :
  x^2 / (x^6 - 2*x^5 - 2*x^4 + 4*x^3 + 4*x^2 + 16) ≤ 1/8 := by
sorry

end max_value_of_fraction_l3547_354738


namespace car_distance_theorem_l3547_354766

/-- Given a car traveling at 160 km/h for 5 hours, the distance covered is 800 km. -/
theorem car_distance_theorem (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 160 ∧ time = 5 → distance = speed * time → distance = 800 := by
  sorry

end car_distance_theorem_l3547_354766


namespace satellite_height_scientific_notation_l3547_354736

/-- The height of a medium-high orbit satellite in China's Beidou navigation system. -/
def satellite_height : ℝ := 21500000

/-- Scientific notation representation of the satellite height. -/
def satellite_height_scientific : ℝ := 2.15 * (10 ^ 7)

/-- Theorem stating that the satellite height is equal to its scientific notation representation. -/
theorem satellite_height_scientific_notation :
  satellite_height = satellite_height_scientific := by sorry

end satellite_height_scientific_notation_l3547_354736


namespace given_square_is_magic_l3547_354786

/-- Represents a 3x3 magic square -/
def MagicSquare : Type := Fin 3 → Fin 3 → ℕ

/-- Converts a number from base 5 to base 10 -/
def toBase10 (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 5 => 5
  | 6 => 6
  | 10 => 5
  | 20 => 10
  | 21 => 11
  | 22 => 12
  | 23 => 13
  | _ => n  -- Default case

/-- The given magic square -/
def givenSquare : MagicSquare :=
  fun i j => match i, j with
    | 0, 0 => 22
    | 0, 1 => 2
    | 0, 2 => 20
    | 1, 0 => 5
    | 1, 1 => 10
    | 1, 2 => 21
    | 2, 0 => 6
    | 2, 1 => 23
    | 2, 2 => 3

/-- Sum of a row in the magic square -/
def rowSum (s : MagicSquare) (i : Fin 3) : ℕ :=
  (toBase10 (s i 0)) + (toBase10 (s i 1)) + (toBase10 (s i 2))

/-- Sum of a column in the magic square -/
def colSum (s : MagicSquare) (j : Fin 3) : ℕ :=
  (toBase10 (s 0 j)) + (toBase10 (s 1 j)) + (toBase10 (s 2 j))

/-- Sum of the main diagonal of the magic square -/
def mainDiagSum (s : MagicSquare) : ℕ :=
  (toBase10 (s 0 0)) + (toBase10 (s 1 1)) + (toBase10 (s 2 2))

/-- Sum of the other diagonal of the magic square -/
def otherDiagSum (s : MagicSquare) : ℕ :=
  (toBase10 (s 0 2)) + (toBase10 (s 1 1)) + (toBase10 (s 2 0))

/-- Theorem: The given square is a magic square when interpreted in base 5 -/
theorem given_square_is_magic : 
  (∀ i : Fin 3, rowSum givenSquare i = 21) ∧ 
  (∀ j : Fin 3, colSum givenSquare j = 21) ∧ 
  mainDiagSum givenSquare = 21 ∧ 
  otherDiagSum givenSquare = 21 := by
  sorry

end given_square_is_magic_l3547_354786


namespace unique_perpendicular_tangent_perpendicular_tangent_equation_slope_angle_range_l3547_354741

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Statement for the unique perpendicular tangent line
theorem unique_perpendicular_tangent :
  ∃! a : ℝ, ∃! x : ℝ, f' a x = -1 ∧ a = 3 := by sorry

-- Statement for the equation of the perpendicular tangent line
theorem perpendicular_tangent_equation (a : ℝ) (h : a = 3) :
  ∃ x y : ℝ, 3*x + 3*y - 8 = 0 ∧ y = f a x ∧ f' a x = -1 := by sorry

-- Statement for the range of the slope angle
theorem slope_angle_range (a : ℝ) (h : a = 3) :
  ∀ x : ℝ, -π/4 ≤ Real.arctan (f' a x) ∧ Real.arctan (f' a x) < π/2 := by sorry

end

end unique_perpendicular_tangent_perpendicular_tangent_equation_slope_angle_range_l3547_354741


namespace function_property_l3547_354710

theorem function_property (f : ℕ → ℕ) :
  (∀ m n : ℕ, (m^2 + f n) ∣ (m * f m + n)) →
  (∀ n : ℕ, f n = n) :=
by sorry

end function_property_l3547_354710


namespace greatest_power_of_two_l3547_354734

theorem greatest_power_of_two (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (10^1006 - 6^503) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1006 - 6^503) → m ≤ k) → 
  (∃ k : ℕ, k = 503 ∧ 2^k ∣ (10^1006 - 6^503) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1006 - 6^503) → m ≤ k) :=
by sorry

end greatest_power_of_two_l3547_354734


namespace volume_of_extended_box_l3547_354796

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def extendedVolume (b : Box) : ℝ :=
  sorry

/-- The specific box in the problem -/
def problemBox : Box :=
  { length := 4, width := 5, height := 6 }

/-- The theorem to be proved -/
theorem volume_of_extended_box :
  extendedVolume problemBox = (804 + 139 * Real.pi) / 3 := by
  sorry

end volume_of_extended_box_l3547_354796


namespace class_average_problem_l3547_354725

theorem class_average_problem (avg_class1 avg_combined : ℝ) (n1 n2 : ℕ) 
  (h1 : avg_class1 = 40)
  (h2 : n1 = 24)
  (h3 : n2 = 50)
  (h4 : avg_combined = 53.513513513513516)
  (h5 : (n1 : ℝ) * avg_class1 + (n2 : ℝ) * (((n1 + n2 : ℕ) : ℝ) * avg_combined - (n1 : ℝ) * avg_class1) / (n2 : ℝ) = 
        (n1 + n2 : ℕ) * avg_combined) :
  (((n1 + n2 : ℕ) : ℝ) * avg_combined - (n1 : ℝ) * avg_class1) / (n2 : ℝ) = 60 := by
  sorry

#check class_average_problem

end class_average_problem_l3547_354725


namespace function_monotonicity_l3547_354784

theorem function_monotonicity (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π) 
  (f : Real → Real) 
  (hf : ∀ x, f x = Real.sqrt 3 * Real.sin (2 * x + θ) + Real.cos (2 * x + θ))
  (h3 : f (π / 2) = 0) :
  StrictMonoOn f (Set.Ioo (π / 4) (3 * π / 4)) := by
sorry

end function_monotonicity_l3547_354784


namespace parallel_vectors_m_value_l3547_354785

theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (2, 1) →
  b = (1, m) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  m = 1/2 := by
  sorry

end parallel_vectors_m_value_l3547_354785


namespace arithmetic_sequence_property_l3547_354774

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the properties of the sequence
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 7 = 2 * (a 4)^2 →
  a 3 = 1 →
  a 2 = Real.sqrt 2 / 2 :=
by
  sorry


end arithmetic_sequence_property_l3547_354774


namespace shortest_distance_ln_to_line_l3547_354709

open Real

theorem shortest_distance_ln_to_line (x : ℝ) : 
  let g (x : ℝ) := log x
  let P : ℝ × ℝ := (x, g x)
  let d (p : ℝ × ℝ) := |p.1 - p.2| / sqrt 2
  ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → d P ≥ d (x₀, g x₀) ∧ d (x₀, g x₀) = 1 / sqrt 2 :=
sorry

end shortest_distance_ln_to_line_l3547_354709


namespace quadratic_roots_distance_bounds_l3547_354717

theorem quadratic_roots_distance_bounds (z₁ z₂ m : ℂ) (α β : ℂ) :
  (∀ x : ℂ, x^2 + z₁*x + z₂ + m = 0 ↔ x = α ∨ x = β) →
  z₁^2 - 4*z₂ = 16 + 20*I →
  Complex.abs (α - β) = 2 * Real.sqrt 7 →
  (Complex.abs m ≤ 7 + Real.sqrt 41 ∧ Complex.abs m ≥ 7 - Real.sqrt 41) ∧
  (∃ m₁ m₂ : ℂ, Complex.abs m₁ = 7 + Real.sqrt 41 ∧ Complex.abs m₂ = 7 - Real.sqrt 41) :=
by sorry

end quadratic_roots_distance_bounds_l3547_354717


namespace quadratic_intersection_and_vertex_l3547_354726

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*(m+1)*x - m + 1

-- Define the discriminant of the quadratic function
def discriminant (m : ℝ) : ℝ := 4*(m^2 + 3*m)

-- Define the x-coordinate of the vertex
def vertex_x (m : ℝ) : ℝ := -(m + 1)

-- Define the y-coordinate of the vertex
def vertex_y (m : ℝ) : ℝ := -(m^2 + 3*m)

theorem quadratic_intersection_and_vertex (m : ℝ) :
  -- Part 1: The number of intersection points with the x-axis is 0, 1, or 2
  (∃ x : ℝ, f m x = 0 ∧ 
    (∀ y : ℝ, f m y = 0 → y = x ∨ 
    (∃ z : ℝ, z ≠ x ∧ z ≠ y ∧ f m z = 0))) ∨
  (∀ x : ℝ, f m x ≠ 0) ∧
  -- Part 2: If the line y = x + 1 passes through the vertex, then m = -2 or m = 0
  (vertex_y m = vertex_x m + 1 → m = -2 ∨ m = 0) :=
sorry

end quadratic_intersection_and_vertex_l3547_354726


namespace class_b_wins_l3547_354716

/-- Represents the grades in a class --/
structure ClassGrades where
  excellent : ℕ
  good : ℕ
  average : ℕ
  satisfactory : ℕ

/-- Calculates the average grade for a class --/
def averageGrade (cg : ClassGrades) (totalStudents : ℕ) : ℚ :=
  (5 * cg.excellent + 4 * cg.good + 3 * cg.average + 2 * cg.satisfactory) / totalStudents

theorem class_b_wins (classA classB : ClassGrades) : 
  classA.excellent = 6 ∧
  classA.good = 16 ∧
  classA.average = 10 ∧
  classA.satisfactory = 8 ∧
  classB.excellent = 5 ∧
  classB.good = 15 ∧
  classB.average = 15 ∧
  classB.satisfactory = 3 →
  averageGrade classB 38 > averageGrade classA 40 := by
  sorry

#eval averageGrade ⟨6, 16, 10, 8⟩ 40
#eval averageGrade ⟨5, 15, 15, 3⟩ 38

end class_b_wins_l3547_354716


namespace min_value_of_w_l3547_354765

def w (x y : ℝ) : ℝ := 2 * x^2 + 3 * y^2 + 8 * x - 6 * y + 30

theorem min_value_of_w :
  ∀ x y : ℝ, w x y ≥ 19 ∧ ∃ x₀ y₀ : ℝ, w x₀ y₀ = 19 := by
  sorry

end min_value_of_w_l3547_354765


namespace bouquet_combinations_l3547_354707

theorem bouquet_combinations (total : ℕ) (rose_cost : ℕ) (carnation_cost : ℕ) 
  (h_total : total = 50)
  (h_rose : rose_cost = 3)
  (h_carnation : carnation_cost = 2) :
  (∃ (solutions : Finset (ℕ × ℕ)), 
    solutions.card = 9 ∧ 
    ∀ (r c : ℕ), (r, c) ∈ solutions ↔ rose_cost * r + carnation_cost * c = total) :=
sorry

end bouquet_combinations_l3547_354707


namespace not_in_first_quadrant_l3547_354756

/-- Proves that the complex number z = (m-2i)/(1+2i) cannot be in the first quadrant for any real m -/
theorem not_in_first_quadrant (m : ℝ) : 
  let z : ℂ := (m - 2*Complex.I) / (1 + 2*Complex.I)
  ¬ (z.re > 0 ∧ z.im > 0) := by
  sorry


end not_in_first_quadrant_l3547_354756


namespace complex_computations_l3547_354719

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_computations :
  (Complex.abs (3 - i) = Real.sqrt 10) ∧
  ((10 * i) / (3 - i) = -1 + 3 * i) :=
by sorry

end complex_computations_l3547_354719


namespace percentage_apartments_with_two_residents_l3547_354762

theorem percentage_apartments_with_two_residents
  (total_apartments : ℕ)
  (percentage_with_at_least_one : ℚ)
  (apartments_with_one : ℕ)
  (h1 : total_apartments = 120)
  (h2 : percentage_with_at_least_one = 85 / 100)
  (h3 : apartments_with_one = 30) :
  (((percentage_with_at_least_one * total_apartments) - apartments_with_one) / total_apartments) * 100 = 60 := by
sorry

end percentage_apartments_with_two_residents_l3547_354762


namespace solve_star_equation_l3547_354705

-- Define the ★ operation
def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

-- State the theorem
theorem solve_star_equation : 
  ∃ (a : ℝ), star a 3 = 15 ∧ a = 11 := by
  sorry

end solve_star_equation_l3547_354705


namespace second_account_interest_rate_l3547_354764

/-- Proves that the interest rate of the second account is 5% given the problem conditions -/
theorem second_account_interest_rate 
  (total_investment : ℝ) 
  (first_account_investment : ℝ) 
  (first_account_rate : ℝ) 
  (total_interest : ℝ) 
  (h1 : total_investment = 8000)
  (h2 : first_account_investment = 3000)
  (h3 : first_account_rate = 0.08)
  (h4 : total_interest = 490) :
  let second_account_investment := total_investment - first_account_investment
  let first_account_interest := first_account_investment * first_account_rate
  let second_account_interest := total_interest - first_account_interest
  let second_account_rate := second_account_interest / second_account_investment
  second_account_rate = 0.05 := by
sorry


end second_account_interest_rate_l3547_354764


namespace annie_hamburgers_l3547_354703

/-- Proves that Annie bought 8 hamburgers given the problem conditions -/
theorem annie_hamburgers :
  ∀ (initial_money : ℕ) (hamburger_cost : ℕ) (milkshake_cost : ℕ) 
    (milkshakes_bought : ℕ) (money_left : ℕ),
  initial_money = 132 →
  hamburger_cost = 4 →
  milkshake_cost = 5 →
  milkshakes_bought = 6 →
  money_left = 70 →
  ∃ (hamburgers_bought : ℕ),
    hamburgers_bought * hamburger_cost + milkshakes_bought * milkshake_cost = initial_money - money_left ∧
    hamburgers_bought = 8 :=
by sorry

end annie_hamburgers_l3547_354703


namespace initial_men_count_initial_men_count_correct_l3547_354749

/-- The number of days the initial food supply lasts for the initial group -/
def initial_days : ℕ := 22

/-- The number of days that pass before new men join -/
def days_before_joining : ℕ := 2

/-- The number of new men that join -/
def new_men : ℕ := 1140

/-- The number of additional days the food lasts after new men join -/
def additional_days : ℕ := 8

/-- Proves that the initial number of men is 760 -/
theorem initial_men_count : ℕ :=
  760

/-- Theorem stating that the initial_men_count satisfies the given conditions -/
theorem initial_men_count_correct :
  initial_men_count * initial_days =
  (initial_men_count + new_men) * additional_days +
  initial_men_count * days_before_joining :=
by
  sorry

end initial_men_count_initial_men_count_correct_l3547_354749


namespace quadratic_root_value_l3547_354702

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 2 * x + k = 0 ↔ x = (1 + Complex.I * Real.sqrt 39) / 10 ∨ x = (1 - Complex.I * Real.sqrt 39) / 10) →
  k = 2.15 := by
sorry

end quadratic_root_value_l3547_354702


namespace race_result_kilometer_race_result_l3547_354732

/-- Represents a runner in the race -/
structure Runner where
  time : ℝ  -- Time taken to complete the race in seconds
  distance : ℝ  -- Distance covered in meters

/-- The race scenario -/
def race_scenario (race_distance : ℝ) (a b : Runner) : Prop :=
  a.distance = race_distance ∧
  b.distance = race_distance ∧
  a.time + 10 = b.time ∧
  a.time = 390

/-- The theorem to be proved -/
theorem race_result (race_distance : ℝ) (a b : Runner) 
  (h : race_scenario race_distance a b) : 
  a.distance - b.distance * (a.time / b.time) = 25 := by
  sorry

/-- Main theorem stating the race result -/
theorem kilometer_race_result :
  ∃ (a b : Runner), race_scenario 1000 a b ∧ 
  a.distance - b.distance * (a.time / b.time) = 25 := by
  sorry

end race_result_kilometer_race_result_l3547_354732


namespace restaurant_bill_entree_cost_l3547_354753

/-- Given the conditions of a restaurant bill, prove the cost of each entree -/
theorem restaurant_bill_entree_cost 
  (appetizer_cost : ℝ)
  (tip_percentage : ℝ)
  (total_spent : ℝ)
  (num_entrees : ℕ)
  (h_appetizer : appetizer_cost = 10)
  (h_tip : tip_percentage = 0.2)
  (h_total : total_spent = 108)
  (h_num_entrees : num_entrees = 4) :
  ∃ (entree_cost : ℝ), 
    entree_cost * num_entrees + appetizer_cost + 
    (entree_cost * num_entrees + appetizer_cost) * tip_percentage = total_spent ∧
    entree_cost = 20 := by
  sorry

end restaurant_bill_entree_cost_l3547_354753


namespace partial_fraction_decomposition_l3547_354771

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (4 * x + 8) / (x^2 - 3*x - 18) = 32 / (9 * (x - 6)) + 4 / (9 * (x + 3)) := by
  sorry

end partial_fraction_decomposition_l3547_354771


namespace square_area_error_l3547_354744

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * 1.1
  let actual_area := s ^ 2
  let calculated_area := measured_side ^ 2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.21 := by
  sorry

end square_area_error_l3547_354744


namespace exists_class_with_at_least_35_students_l3547_354783

/-- Proves that in a school with 33 classes and 1150 students, there exists at least one class with 35 or more students. -/
theorem exists_class_with_at_least_35_students 
  (num_classes : ℕ) 
  (total_students : ℕ) 
  (h1 : num_classes = 33) 
  (h2 : total_students = 1150) : 
  ∃ (class_size : ℕ), class_size ≥ 35 ∧ class_size ≤ total_students := by
  sorry

#check exists_class_with_at_least_35_students

end exists_class_with_at_least_35_students_l3547_354783
