import Mathlib

namespace square_root_of_one_is_one_l3445_344506

theorem square_root_of_one_is_one : Real.sqrt 1 = 1 := by sorry

end square_root_of_one_is_one_l3445_344506


namespace simplify_expressions_l3445_344519

variable (a b : ℝ)

theorem simplify_expressions :
  (4 * a^2 + 3 * b^2 + 2 * a * b - 4 * a^2 - 4 * b = 3 * b^2 + 2 * a * b - 4 * b) ∧
  (2 * (5 * a - 3 * b) - 3 = 10 * a - 6 * b - 3) := by
  sorry

end simplify_expressions_l3445_344519


namespace tan_fifteen_degree_fraction_equals_sqrt_three_over_three_l3445_344545

theorem tan_fifteen_degree_fraction_equals_sqrt_three_over_three :
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end tan_fifteen_degree_fraction_equals_sqrt_three_over_three_l3445_344545


namespace permutations_of_three_letter_word_is_six_l3445_344573

/-- The number of permutations of a 3-letter word with distinct letters -/
def permutations_of_three_letter_word : ℕ :=
  Nat.factorial 3

/-- Proof that the number of permutations of a 3-letter word with distinct letters is 6 -/
theorem permutations_of_three_letter_word_is_six :
  permutations_of_three_letter_word = 6 := by
  sorry

#eval permutations_of_three_letter_word

end permutations_of_three_letter_word_is_six_l3445_344573


namespace orthocenter_of_specific_triangle_l3445_344503

/-- The orthocenter of a triangle ABC in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (5/2, 3, 7/2) -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (5/2, 3, 7/2) :=
sorry

end orthocenter_of_specific_triangle_l3445_344503


namespace square_sum_and_product_l3445_344559

theorem square_sum_and_product (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) : 
  a^2 + b^2 = 5 ∧ a * b = 1 := by
sorry

end square_sum_and_product_l3445_344559


namespace intersection_of_M_and_N_l3445_344504

def M : Set ℝ := {x : ℝ | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | (-4 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x ≤ 7)} := by sorry

end intersection_of_M_and_N_l3445_344504


namespace positive_numbers_l3445_344565

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_numbers_l3445_344565


namespace eggs_per_meal_l3445_344501

def initial_eggs : ℕ := 24
def used_eggs : ℕ := 6
def meals : ℕ := 3

theorem eggs_per_meal :
  let remaining_after_use := initial_eggs - used_eggs
  let remaining_after_sharing := remaining_after_use / 2
  remaining_after_sharing / meals = 3 :=
by sorry

end eggs_per_meal_l3445_344501


namespace polynomial_properties_l3445_344549

def polynomial_expansion (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5

theorem polynomial_properties 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, polynomial_expansion x a₀ a₁ a₂ a₃ a₄ a₅) : 
  (a₀ + a₁ + a₂ + a₃ + a₄ = -31) ∧ 
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) := by
  sorry

end polynomial_properties_l3445_344549


namespace range_of_a_l3445_344513

open Set

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x < a}
def B : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (A a ∪ (Bᶜ) = univ) → a ∈ Set.Ici 2 := by
  sorry

end range_of_a_l3445_344513


namespace max_sum_constraint_l3445_344537

theorem max_sum_constraint (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) :
  ∀ a b : ℝ, 3 * (a^2 + b^2) = a - b → x + y ≤ (1 : ℝ) / 3 :=
by sorry

end max_sum_constraint_l3445_344537


namespace square_triangles_area_bounds_l3445_344580

-- Define the unit square
def UnitSquare : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the right-angled triangles constructed outward
def OutwardTriangles (s : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

-- Define the vertices A, B, C, D
def RightAngleVertices (triangles : Set (Set (ℝ × ℝ))) : Set (ℝ × ℝ) := sorry

-- Define the incircle centers O₁, O₂, O₃, O₄
def IncircleCenters (triangles : Set (Set (ℝ × ℝ))) : Set (ℝ × ℝ) := sorry

-- Define the area of a quadrilateral
def QuadrilateralArea (vertices : Set (ℝ × ℝ)) : ℝ := sorry

theorem square_triangles_area_bounds :
  let s := UnitSquare
  let triangles := OutwardTriangles s
  let abcd := RightAngleVertices triangles
  let o₁o₂o₃o₄ := IncircleCenters triangles
  (QuadrilateralArea abcd ≤ 2) ∧ (QuadrilateralArea o₁o₂o₃o₄ ≤ 1) := by
  sorry

end square_triangles_area_bounds_l3445_344580


namespace function_analysis_l3445_344575

/-- Given a function f and some conditions, prove its analytical expression and range -/
theorem function_analysis (f : ℝ → ℝ) (ω φ : ℝ) :
  (ω > 0) →
  (φ > 0 ∧ φ < Real.pi / 2) →
  (Real.tan φ = 2 * Real.sqrt 3) →
  (∀ x, f x = Real.sqrt 13 * Real.cos (ω * x) * Real.cos (ω * x - φ) - Real.sin (ω * x) ^ 2) →
  (∀ x, f (x + Real.pi / ω) = f x) →
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi / ω) →
  (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧
  (Set.Icc (1 / 13) 2 = { y | ∃ x ∈ Set.Icc (Real.pi / 12) φ, f x = y }) := by
  sorry

end function_analysis_l3445_344575


namespace paint_mixture_ratio_l3445_344556

/-- Given a paint mixture with a ratio of red:yellow:white as 5:3:7,
    if 21 quarts of white paint are used, then 9 quarts of yellow paint should be used. -/
theorem paint_mixture_ratio (red yellow white : ℚ) (total : ℚ) : 
  red / total = 5 / 15 →
  yellow / total = 3 / 15 →
  white / total = 7 / 15 →
  white = 21 →
  yellow = 9 := by
sorry

end paint_mixture_ratio_l3445_344556


namespace triangle_properties_l3445_344593

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  4 * a = Real.sqrt 5 * c →
  Real.cos C = 3 / 5 →
  b = 11 →
  Real.sin A = Real.sqrt 5 / 5 ∧
  1 / 2 * a * b * Real.sin C = 22 :=
by sorry

end triangle_properties_l3445_344593


namespace incorrect_inequality_l3445_344591

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(5 - a > 5 - b) := by
  sorry

end incorrect_inequality_l3445_344591


namespace parallel_vectors_x_value_l3445_344557

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a = (k * b.1, k * b.2)

/-- The theorem states that if vectors (4,2) and (x,3) are parallel, then x = 6 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (4, 2) (x, 3) → x = 6 := by
  sorry

end parallel_vectors_x_value_l3445_344557


namespace hyperbola_asymptote_slope_l3445_344525

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := y^2 / 9 - x^2 / 16 = 1

/-- The asymptote equation -/
def asymptote_eq (m x y : ℝ) : Prop := y = m * x ∨ y = -m * x

/-- Theorem: The value of m for the given hyperbola's asymptotes is 3/4 -/
theorem hyperbola_asymptote_slope :
  ∃ (m : ℝ), (∀ (x y : ℝ), hyperbola_eq x y → asymptote_eq m x y) ∧ m = 3/4 := by
  sorry

end hyperbola_asymptote_slope_l3445_344525


namespace x_with_three_prime_divisors_l3445_344572

theorem x_with_three_prime_divisors (x n : ℕ) 
  (h1 : x = 2^n - 32)
  (h2 : (Nat.factors x).toFinset.card = 3)
  (h3 : 2 ∈ Nat.factors x) :
  x = 2016 ∨ x = 16352 := by
  sorry

end x_with_three_prime_divisors_l3445_344572


namespace hyperbola_real_axis_length_l3445_344589

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, its real axis has length 4 -/
theorem hyperbola_real_axis_length :
  ∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1 → 
  ∃ (a : ℝ), a > 0 ∧ x^2 / a^2 - y^2 / (3*a^2) = 1 ∧ 2 * a = 4 :=
by sorry

end hyperbola_real_axis_length_l3445_344589


namespace cube_halving_l3445_344542

theorem cube_halving (r : ℝ) :
  let a := (2 * r) ^ 3
  let a_half := (2 * (r / 2)) ^ 3
  a_half = (1 / 8) * a := by
  sorry

end cube_halving_l3445_344542


namespace inequality_proof_l3445_344586

theorem inequality_proof (a b c α : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^α + b^α + c^α) ≥ a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ∧
  (a * b * c * (a^α + b^α + c^α) = a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ↔ a = b ∧ b = c) :=
by sorry

end inequality_proof_l3445_344586


namespace darryl_break_even_l3445_344590

/-- Calculates the number of machines needed to break even given costs and selling price -/
def machines_to_break_even (parts_cost patent_cost selling_price : ℕ) : ℕ :=
  (parts_cost + patent_cost) / selling_price

/-- Theorem: Darryl needs to sell 45 machines to break even -/
theorem darryl_break_even :
  machines_to_break_even 3600 4500 180 = 45 := by
  sorry

end darryl_break_even_l3445_344590


namespace min_f_value_l3445_344581

theorem min_f_value (d e f : ℕ+) (h1 : d < e) (h2 : e < f)
  (h3 : ∃! x y : ℝ, 3 * x + y = 3005 ∧ y = |x - d| + |x - e| + |x - f|) :
  1504 ≤ f :=
sorry

end min_f_value_l3445_344581


namespace relationship_implies_function_l3445_344550

-- Define the relationship between x and y
def relationship (x y : ℝ) : Prop :=
  y = 2*x - 1 - Real.sqrt (y^2 - 2*x*y + 3*x - 2)

-- Define the function we want to prove
def function (x : ℝ) : Set ℝ :=
  if x ≠ 1 then {2*x - 1.5}
  else {y : ℝ | y ≤ 1}

-- Theorem statement
theorem relationship_implies_function :
  ∀ x y : ℝ, relationship x y → y ∈ function x :=
sorry

end relationship_implies_function_l3445_344550


namespace two_numbers_difference_l3445_344576

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (square_diff : x^2 - y^2 = 24) : 
  |x - y| = 2.4 := by
sorry

end two_numbers_difference_l3445_344576


namespace least_common_multiple_7_6_4_l3445_344536

theorem least_common_multiple_7_6_4 : ∃ (n : ℕ), n > 0 ∧ 7 ∣ n ∧ 6 ∣ n ∧ 4 ∣ n ∧ ∀ (m : ℕ), m > 0 → 7 ∣ m → 6 ∣ m → 4 ∣ m → n ≤ m :=
by sorry

end least_common_multiple_7_6_4_l3445_344536


namespace lcm_18_10_l3445_344524

theorem lcm_18_10 : Nat.lcm 18 10 = 36 :=
by
  have h1 : Nat.gcd 18 10 = 5 := by sorry
  sorry

end lcm_18_10_l3445_344524


namespace combined_savings_difference_l3445_344507

/-- The cost of a single window -/
def window_cost : ℕ := 100

/-- The number of windows purchased to get one free -/
def windows_for_free : ℕ := 4

/-- The number of windows Dave needs -/
def dave_windows : ℕ := 7

/-- The number of windows Doug needs -/
def doug_windows : ℕ := 8

/-- Calculate the cost of windows with the promotion -/
def cost_with_promotion (n : ℕ) : ℕ :=
  ((n + windows_for_free - 1) / windows_for_free) * windows_for_free * window_cost

/-- Calculate the savings for a given number of windows -/
def savings (n : ℕ) : ℕ :=
  n * window_cost - cost_with_promotion n

/-- The main theorem: combined savings minus individual savings equals $100 -/
theorem combined_savings_difference : 
  savings (dave_windows + doug_windows) - (savings dave_windows + savings doug_windows) = 100 := by
  sorry

end combined_savings_difference_l3445_344507


namespace arkansas_game_profit_calculation_l3445_344541

/-- The amount of money made per t-shirt sold -/
def profit_per_shirt : ℕ := 98

/-- The total number of t-shirts sold during both games -/
def total_shirts_sold : ℕ := 163

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts_sold : ℕ := 89

/-- The money made from selling t-shirts during the Arkansas game -/
def arkansas_game_profit : ℕ := profit_per_shirt * arkansas_shirts_sold

theorem arkansas_game_profit_calculation :
  arkansas_game_profit = 8722 := by sorry

end arkansas_game_profit_calculation_l3445_344541


namespace expression_evaluation_l3445_344529

theorem expression_evaluation (a b c d : ℝ) 
  (ha : a = 11) (hb : b = 13) (hc : c = 17) (hd : d = 19) :
  (a^2 * (1/b - 1/d) + b^2 * (1/d - 1/a) + c^2 * (1/a - 1/c) + d^2 * (1/c - 1/b)) /
  (a * (1/b - 1/d) + b * (1/d - 1/a) + c * (1/a - 1/c) + d * (1/c - 1/b)) = a + b + c + d :=
by sorry

#eval (11 : ℝ) + 13 + 17 + 19  -- To verify the result is indeed 60

end expression_evaluation_l3445_344529


namespace journey_distance_l3445_344555

theorem journey_distance (total_time : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_time = 36 ∧ 
  speed1 = 21 ∧ 
  speed2 = 45 ∧ 
  speed3 = 24 → 
  ∃ (distance : ℝ),
    distance = 972 ∧
    total_time = distance / (3 * speed1) + distance / (3 * speed2) + distance / (3 * speed3) :=
by sorry

end journey_distance_l3445_344555


namespace at_least_three_functional_probability_l3445_344534

def num_lamps : ℕ := 5
def func_prob : ℝ := 0.2

theorem at_least_three_functional_probability :
  let p := func_prob
  let q := 1 - p
  let binom_prob (n k : ℕ) := (Nat.choose n k : ℝ) * p^k * q^(n-k)
  binom_prob num_lamps 3 + binom_prob num_lamps 4 + binom_prob num_lamps 5 = 0.06 := by
  sorry

end at_least_three_functional_probability_l3445_344534


namespace hyperbola_asymptotes_l3445_344597

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = (4/3) * x ∨ y = -(4/3) * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 0) ↔ (y = (4/3) * x ∨ y = -(4/3) * x) :=
by sorry

end hyperbola_asymptotes_l3445_344597


namespace stratified_sample_size_l3445_344552

/-- Represents the number of students in each grade --/
structure StudentCounts where
  freshman : ℕ
  sophomore : ℕ
  senior : ℕ

/-- Calculates the total number of students --/
def total_students (counts : StudentCounts) : ℕ :=
  counts.freshman + counts.sophomore + counts.senior

/-- Calculates the sample size based on the number of sampled freshmen --/
def sample_size (counts : StudentCounts) (sampled_freshmen : ℕ) : ℕ :=
  sampled_freshmen * (total_students counts) / counts.freshman

/-- Theorem stating that for the given student counts and sampled freshmen, the sample size is 30 --/
theorem stratified_sample_size 
  (counts : StudentCounts) 
  (h1 : counts.freshman = 700)
  (h2 : counts.sophomore = 500)
  (h3 : counts.senior = 300)
  (h4 : sampled_freshmen = 14) :
  sample_size counts sampled_freshmen = 30 := by
  sorry

end stratified_sample_size_l3445_344552


namespace cubic_equation_solutions_l3445_344520

theorem cubic_equation_solutions (x : ℝ) : 
  2.21 * (((5 + x)^2)^(1/3)) + 4 * (((5 - x)^2)^(1/3)) = 5 * ((25 - x)^(1/3)) ↔ 
  x = 0 ∨ x = 63/13 :=
by sorry

end cubic_equation_solutions_l3445_344520


namespace cyclist_minimum_speed_l3445_344547

/-- Minimum speed for a cyclist to intercept a car -/
theorem cyclist_minimum_speed (v a b : ℝ) (hv : v > 0) (ha : a > 0) (hb : b > 0) :
  let min_speed := v * b / a
  ∀ (cyclist_speed : ℝ), cyclist_speed ≥ min_speed → 
  ∃ (t : ℝ), t > 0 ∧ 
    cyclist_speed * t = (a^2 + (v*t)^2).sqrt ∧
    cyclist_speed * t ≥ v * t :=
by sorry

end cyclist_minimum_speed_l3445_344547


namespace min_c_value_l3445_344588

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c)
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 2023 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1012 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 1012 ∧
    ∃! (x y : ℝ), 2 * x + y = 2023 ∧ y = |x - a'| + |x - b'| + |x - 1012| :=
by sorry

end min_c_value_l3445_344588


namespace student_marks_theorem_l3445_344599

/-- Calculates the total marks secured by a student in an examination with the given conditions. -/
def total_marks (total_questions : ℕ) (correct_answers : ℕ) (marks_per_correct : ℕ) (marks_per_wrong : ℕ) : ℤ :=
  (correct_answers * marks_per_correct : ℤ) - ((total_questions - correct_answers) * marks_per_wrong)

/-- Theorem stating that under the given conditions, the student secures 150 marks. -/
theorem student_marks_theorem :
  total_marks 60 42 4 1 = 150 := by
  sorry

end student_marks_theorem_l3445_344599


namespace triangle_separation_l3445_344569

/-- A triangle in a 2D plane -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Check if two triangles have no common interior or boundary points -/
def no_common_points (t1 t2 : Triangle) : Prop := sorry

/-- Check if a line separates two triangles -/
def separates (line : ℝ × ℝ → Prop) (t1 t2 : Triangle) : Prop := sorry

/-- Check if a line is a side of a triangle -/
def is_side (line : ℝ × ℝ → Prop) (t : Triangle) : Prop := sorry

/-- Main theorem: For any two triangles with no common points, 
    there exists a side of one triangle that separates them -/
theorem triangle_separation (t1 t2 : Triangle) 
  (h : no_common_points t1 t2) : 
  ∃ (line : ℝ × ℝ → Prop), 
    (is_side line t1 ∨ is_side line t2) ∧ 
    separates line t1 t2 := by sorry

end triangle_separation_l3445_344569


namespace jordans_rectangle_width_l3445_344540

theorem jordans_rectangle_width (carol_length carol_width jordan_length : ℕ) 
  (jordan_width : ℕ) : 
  carol_length = 12 → 
  carol_width = 15 → 
  jordan_length = 9 → 
  carol_length * carol_width = jordan_length * jordan_width → 
  jordan_width = 20 := by
sorry

end jordans_rectangle_width_l3445_344540


namespace triangle_problem_l3445_344522

theorem triangle_problem (a b c A B C : ℝ) :
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  (c = Real.sqrt 3 * a * Real.sin C - c * Real.cos A) →
  (a = 2) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →
  (A = π/3 ∧ b = 2 ∧ c = 2) := by sorry

end triangle_problem_l3445_344522


namespace reflection_composition_l3445_344539

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  let q := (p.1, p.2 + 1)
  let r := (q.2, q.1)
  (r.1, r.2 - 1)

theorem reflection_composition (H : ℝ × ℝ) :
  H = (5, 0) →
  reflect_y_eq_x_minus_1 (reflect_x H) = (1, 4) := by
  sorry

end reflection_composition_l3445_344539


namespace fraction_of_120_l3445_344598

theorem fraction_of_120 : (1 / 6 : ℚ) * (1 / 4 : ℚ) * (1 / 5 : ℚ) * 120 = 1 ∧ 1 = 2 * (1 / 2 : ℚ) := by
  sorry

end fraction_of_120_l3445_344598


namespace power_expressions_l3445_344531

theorem power_expressions (m n : ℤ) (a b : ℝ) 
  (h1 : 4^m = a) (h2 : 8^n = b) : 
  (2^(2*m + 3*n) = a * b) ∧ (2^(4*m - 6*n) = a^2 / b^2) := by
  sorry

end power_expressions_l3445_344531


namespace max_equilateral_triangles_l3445_344584

/-- Represents a matchstick configuration --/
structure MatchstickConfig where
  num_matchsticks : ℕ
  connected_end_to_end : Bool

/-- Represents the number of equilateral triangles in a configuration --/
def num_equilateral_triangles (config : MatchstickConfig) : ℕ := sorry

/-- The theorem stating the maximum number of equilateral triangles --/
theorem max_equilateral_triangles (config : MatchstickConfig) 
  (h1 : config.num_matchsticks = 6) 
  (h2 : config.connected_end_to_end = true) : 
  ∃ (n : ℕ), n ≤ 4 ∧ ∀ (m : ℕ), num_equilateral_triangles config ≤ m → n ≤ m :=
sorry

end max_equilateral_triangles_l3445_344584


namespace f_inequality_l3445_344544

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ 1/3 < x ∧ x < 1 := by
  sorry

end f_inequality_l3445_344544


namespace problem_solution_l3445_344595

def p (x : ℝ) : Prop := x^2 ≤ 5*x - 4

def q (x a : ℝ) : Prop := x^2 - (a + 2)*x + 2*a ≤ 0

theorem problem_solution :
  (∀ x : ℝ, ¬(p x) ↔ (x < 1 ∨ x > 4)) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x a → p x) ∧ (∃ x : ℝ, p x ∧ ¬(q x a)) ↔ (1 ≤ a ∧ a ≤ 4)) :=
sorry

end problem_solution_l3445_344595


namespace remainder_problem_l3445_344530

theorem remainder_problem (N : ℕ) (R : ℕ) :
  (∃ q : ℕ, N = 34 * q + 2) →
  (N = 44 * 432 + R) →
  R = 0 := by
sorry

end remainder_problem_l3445_344530


namespace train_speed_crossing_bridge_l3445_344563

/-- The speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 135)
  (h2 : bridge_length = 240)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_crossing_bridge

end train_speed_crossing_bridge_l3445_344563


namespace inequality_proof_equality_condition_l3445_344578

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ 2 / (1 + a*b) :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) = 2 / (1 + a*b) ↔ a = b :=
by sorry

end inequality_proof_equality_condition_l3445_344578


namespace count_four_digit_integers_eq_six_l3445_344502

def digits : Multiset ℕ := {2, 2, 9, 9}

/-- The number of different positive, four-digit integers that can be formed using the digits 2, 2, 9, and 9 -/
def count_four_digit_integers : ℕ :=
  Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)

theorem count_four_digit_integers_eq_six :
  count_four_digit_integers = 6 := by
  sorry

end count_four_digit_integers_eq_six_l3445_344502


namespace average_weight_problem_l3445_344561

theorem average_weight_problem (rachel_weight jimmy_weight adam_weight : ℝ) :
  rachel_weight = 75 ∧
  rachel_weight = jimmy_weight - 6 ∧
  rachel_weight = adam_weight + 15 →
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 :=
by sorry

end average_weight_problem_l3445_344561


namespace gcd_problem_l3445_344523

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1061) :
  Int.gcd (3 * b^2 + 41 * b + 96) (b + 17) = 17 := by
  sorry

end gcd_problem_l3445_344523


namespace comic_book_arrangements_l3445_344592

def spiderman_comics : ℕ := 6
def archie_comics : ℕ := 5
def garfield_comics : ℕ := 4
def batman_comics : ℕ := 7

def total_arrangements : ℕ := 59536691200

theorem comic_book_arrangements :
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * batman_comics.factorial) *
  (spiderman_comics + archie_comics + garfield_comics + batman_comics - 3).factorial = total_arrangements := by
  sorry

end comic_book_arrangements_l3445_344592


namespace people_per_entrance_l3445_344574

theorem people_per_entrance 
  (total_entrances : ℕ) 
  (total_people : ℕ) 
  (h1 : total_entrances = 5) 
  (h2 : total_people = 1415) :
  total_people / total_entrances = 283 :=
by sorry

end people_per_entrance_l3445_344574


namespace old_selling_price_l3445_344567

/-- Given a product with cost C, prove that if the selling price increased from 110% of C to 115% of C, 
    and the new selling price is $92.00, then the old selling price was $88.00. -/
theorem old_selling_price (C : ℝ) 
  (h1 : C + 0.15 * C = 92) 
  (h2 : C > 0) : 
  C + 0.10 * C = 88 := by
sorry

end old_selling_price_l3445_344567


namespace square_sum_of_three_reals_l3445_344594

theorem square_sum_of_three_reals (x y z : ℝ) 
  (h1 : (x + y + z)^2 = 25)
  (h2 : x*y + x*z + y*z = 8) :
  x^2 + y^2 + z^2 = 9 := by
sorry

end square_sum_of_three_reals_l3445_344594


namespace vector_q_solution_l3445_344512

/-- Custom vector operation ⊗ -/
def vector_op (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 * b.1, a.2 * b.2)

theorem vector_q_solution :
  let p : ℝ × ℝ := (1, 2)
  let q : ℝ × ℝ := (-3, -2)
  vector_op p q = (-3, -4) :=
by sorry

end vector_q_solution_l3445_344512


namespace remainder_18273_mod_9_l3445_344558

theorem remainder_18273_mod_9 : 18273 % 9 = 3 := by
  sorry

end remainder_18273_mod_9_l3445_344558


namespace valentinas_burger_length_l3445_344579

/-- The length of a burger shared equally between two people, given the length of one person's share. -/
def burger_length (share_length : ℝ) : ℝ := 2 * share_length

/-- Proof that Valentina's burger is 12 inches long. -/
theorem valentinas_burger_length : 
  let share_length := 6
  burger_length share_length = 12 := by
  sorry

end valentinas_burger_length_l3445_344579


namespace exists_cycle_not_div_by_three_l3445_344518

/-- A graph is a set of vertices and a set of edges between them. -/
structure Graph (V : Type) where
  edge : V → V → Prop

/-- The degree of a vertex in a graph is the number of edges connected to it. -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- A path in a graph is a sequence of vertices where each consecutive pair is connected by an edge. -/
def is_path (G : Graph V) (path : List V) : Prop := sorry

/-- A cycle is a path that starts and ends at the same vertex. -/
def is_cycle (G : Graph V) (cycle : List V) : Prop := sorry

/-- The main theorem: In any graph where each vertex has degree at least 3,
    there exists a cycle whose length is not divisible by 3. -/
theorem exists_cycle_not_div_by_three {V : Type} (G : Graph V) :
  (∀ v : V, degree G v ≥ 3) →
  ∃ cycle : List V, is_cycle G cycle ∧ (cycle.length % 3 ≠ 0) := by
  sorry

end exists_cycle_not_div_by_three_l3445_344518


namespace arithmetic_sequence_with_prime_factors_l3445_344517

theorem arithmetic_sequence_with_prime_factors
  (n d : ℕ+) :
  ∃ (a : ℕ → ℕ+),
    (∀ i j : ℕ, i < n ∧ j < n → a (i + 1) - a j = d * (i - j)) ∧
    (∀ i : ℕ, i < n → ∃ p : ℕ, p.Prime ∧ p ≥ i + 1 ∧ p ∣ a (i + 1)) :=
by sorry

end arithmetic_sequence_with_prime_factors_l3445_344517


namespace product_a4_a5_a6_l3445_344570

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem product_a4_a5_a6 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 9 = 16 →
  a 1 + a 9 = 10 →
  a 4 * a 5 * a 6 = 64 := by
sorry

end product_a4_a5_a6_l3445_344570


namespace plane_speed_problem_l3445_344505

theorem plane_speed_problem (speed1 : ℝ) (time : ℝ) (total_distance : ℝ) (speed2 : ℝ) : 
  speed1 = 75 →
  time = 4.84848484848 →
  total_distance = 800 →
  (speed1 + speed2) * time = total_distance →
  speed2 = 90 := by
sorry

end plane_speed_problem_l3445_344505


namespace stuffed_animals_problem_l3445_344528

theorem stuffed_animals_problem (num_dogs : ℕ) : 
  (∃ (group_size : ℕ), group_size > 0 ∧ (14 + num_dogs) = 7 * group_size) →
  num_dogs = 7 := by
sorry

end stuffed_animals_problem_l3445_344528


namespace multichoose_eq_choose_l3445_344571

/-- F_n^r represents the number of ways to choose r elements from [1, n] with repetition and disregarding order -/
def F (n : ℕ) (r : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to choose r elements from [1, n] with repetition and disregarding order
    is equal to the number of ways to choose r elements from [1, n+r-1] without repetition -/
theorem multichoose_eq_choose (n : ℕ) (r : ℕ) : F n r = Nat.choose (n + r - 1) r := by sorry

end multichoose_eq_choose_l3445_344571


namespace max_abs_difference_l3445_344583

theorem max_abs_difference (x y : ℝ) 
  (h1 : x^2 + y^2 = 2023)
  (h2 : (x - 2) * (y - 2) = 3) :
  |x - y| ≤ 13 * Real.sqrt 13 ∧ ∃ x y : ℝ, x^2 + y^2 = 2023 ∧ (x - 2) * (y - 2) = 3 ∧ |x - y| = 13 * Real.sqrt 13 := by
  sorry

end max_abs_difference_l3445_344583


namespace monotonic_cubic_function_param_range_l3445_344527

/-- A function f(x) = -x^3 + ax^2 - x - 1 is monotonic on (-∞, +∞) if and only if a ∈ [-√3, √3] -/
theorem monotonic_cubic_function_param_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end monotonic_cubic_function_param_range_l3445_344527


namespace triangle_inequality_l3445_344560

theorem triangle_inequality (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) : 
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ 
  y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end triangle_inequality_l3445_344560


namespace exists_fixed_point_l3445_344509

variable {X : Type u}
variable (μ : Set X → Set X)

axiom μ_union_disjoint {A B : Set X} (h : Disjoint A B) : μ (A ∪ B) = μ A ∪ μ B

theorem exists_fixed_point : ∃ F : Set X, μ F = F := by
  sorry

end exists_fixed_point_l3445_344509


namespace exists_surjective_function_with_property_l3445_344568

-- Define the property of the function
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f (x + y) - f x - f y) ∈ ({0, 1} : Set ℝ)

-- State the theorem
theorem exists_surjective_function_with_property :
  ∃ f : ℝ → ℝ, Function.Surjective f ∧ has_property f :=
sorry

end exists_surjective_function_with_property_l3445_344568


namespace parabola_hyperbola_equations_l3445_344585

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with focus on the x-axis -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a hyperbola with focus on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- The intersection point of the asymptotes -/
def intersectionPoint : Point := { x := 4, y := 8 }

/-- Check if a point satisfies the parabola equation -/
def satisfiesParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Check if a point satisfies the hyperbola equation -/
def satisfiesHyperbola (point : Point) (hyperbola : Hyperbola) : Prop :=
  (point.x^2 / hyperbola.a^2) - (point.y^2 / hyperbola.b^2) = 1

/-- The main theorem -/
theorem parabola_hyperbola_equations :
  ∃ (parabola : Parabola) (hyperbola : Hyperbola),
    satisfiesParabola intersectionPoint parabola ∧
    satisfiesHyperbola intersectionPoint hyperbola ∧
    parabola.p = 8 ∧
    hyperbola.a^2 = 16/5 ∧
    hyperbola.b^2 = 64/5 := by
  sorry

end parabola_hyperbola_equations_l3445_344585


namespace point_on_x_axis_l3445_344577

theorem point_on_x_axis (m : ℚ) :
  (∃ x : ℚ, x = 2 - m ∧ 0 = 3 * m + 1) → m = -1/3 := by
  sorry

end point_on_x_axis_l3445_344577


namespace binomial_307_307_equals_1_l3445_344521

theorem binomial_307_307_equals_1 : Nat.choose 307 307 = 1 := by
  sorry

end binomial_307_307_equals_1_l3445_344521


namespace hyperbola_focus_distance_l3445_344564

/-- Represents a hyperbola with semi-major axis a -/
structure Hyperbola (a : ℝ) where
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / 9 = 1
  asymptote : ℝ → ℝ → Prop := fun x y => 3 * x - 2 * y = 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Left focus of the hyperbola -/
def F1 (h : Hyperbola 2) : Point := sorry

/-- Right focus of the hyperbola -/
def F2 (h : Hyperbola 2) : Point := sorry

/-- A point P on the hyperbola -/
def P (h : Hyperbola 2) : Point := sorry

theorem hyperbola_focus_distance (h : Hyperbola 2) (p : Point) 
  (hp : h.equation p.x p.y) 
  (ha : h.asymptote 3 2) 
  (hd : distance p (F1 h) = 3) : 
  distance p (F2 h) = 7 := by sorry

end hyperbola_focus_distance_l3445_344564


namespace mark_soup_donation_l3445_344514

/-- The number of homeless shelters Mark donates to -/
def num_shelters : ℕ := 6

/-- The number of people served per shelter -/
def people_per_shelter : ℕ := 30

/-- The number of cans of soup per person -/
def cans_per_person : ℕ := 10

/-- The total number of cans Mark donates -/
def total_cans : ℕ := num_shelters * people_per_shelter * cans_per_person

theorem mark_soup_donation : total_cans = 1800 := by
  sorry

end mark_soup_donation_l3445_344514


namespace boys_in_line_l3445_344508

theorem boys_in_line (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, k > 0 ∧ k = 19 ∧ k = n + 1 - 19) → n = 37 := by
  sorry

end boys_in_line_l3445_344508


namespace dance_steps_l3445_344553

theorem dance_steps (nancy_ratio : ℕ) (total_steps : ℕ) (jason_steps : ℕ) : 
  nancy_ratio = 3 →
  total_steps = 32 →
  jason_steps + nancy_ratio * jason_steps = total_steps →
  jason_steps = 8 := by
sorry

end dance_steps_l3445_344553


namespace scott_sales_theorem_scott_total_sales_l3445_344587

/-- Calculates the total money made from selling items at given prices and quantities -/
def total_money_made (smoothie_price cake_price : ℕ) (smoothie_qty cake_qty : ℕ) : ℕ :=
  smoothie_price * smoothie_qty + cake_price * cake_qty

/-- Theorem stating that the total money made is equal to the sum of products of prices and quantities -/
theorem scott_sales_theorem (smoothie_price cake_price : ℕ) (smoothie_qty cake_qty : ℕ) :
  total_money_made smoothie_price cake_price smoothie_qty cake_qty =
  smoothie_price * smoothie_qty + cake_price * cake_qty :=
by
  sorry

/-- Verifies that Scott's total sales match the calculated amount -/
theorem scott_total_sales :
  total_money_made 3 2 40 18 = 156 :=
by
  sorry

end scott_sales_theorem_scott_total_sales_l3445_344587


namespace quadratic_solution_sum_l3445_344582

theorem quadratic_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 14 = 31) → 
  (d^2 - 6*d + 14 = 31) → 
  c ≥ d → 
  c + 2*d = 9 - Real.sqrt 26 := by
sorry

end quadratic_solution_sum_l3445_344582


namespace tenth_term_of_inverse_proportional_sequence_l3445_344515

/-- A sequence where each term after the first is inversely proportional to the preceding term -/
def InverseProportionalSequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = k

theorem tenth_term_of_inverse_proportional_sequence
  (a : ℕ → ℝ)
  (h_seq : InverseProportionalSequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 4) :
  a 10 = 4 := by
sorry

end tenth_term_of_inverse_proportional_sequence_l3445_344515


namespace merchant_profit_l3445_344554

theorem merchant_profit (C S : ℝ) (h : 18 * C = 16 * S) : 
  (S - C) / C * 100 = 12.5 := by
sorry

end merchant_profit_l3445_344554


namespace swallow_theorem_l3445_344546

/-- The number of swallows initially on the wire -/
def initial_swallows : ℕ := 9

/-- The distance between the first and last swallow in centimeters -/
def total_distance : ℕ := 720

/-- The number of additional swallows added between each pair -/
def additional_swallows : ℕ := 3

/-- Theorem stating the distance between neighboring swallows and the total number after adding more -/
theorem swallow_theorem :
  (let gaps := initial_swallows - 1
   let distance_between := total_distance / gaps
   let new_swallows := gaps * additional_swallows
   let total_swallows := initial_swallows + new_swallows
   (distance_between = 90 ∧ total_swallows = 33)) :=
by sorry

end swallow_theorem_l3445_344546


namespace sqrt_difference_equals_negative_two_sqrt_ten_l3445_344510

theorem sqrt_difference_equals_negative_two_sqrt_ten :
  Real.sqrt (25 - 10 * Real.sqrt 6) - Real.sqrt (25 + 10 * Real.sqrt 6) = -2 * Real.sqrt 10 := by
  sorry

end sqrt_difference_equals_negative_two_sqrt_ten_l3445_344510


namespace natural_exp_inequality_l3445_344526

theorem natural_exp_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end natural_exp_inequality_l3445_344526


namespace numerical_puzzle_solution_l3445_344551

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that checks if two digits are different -/
def differentDigits (a b : ℕ) : Prop := a ≠ b ∧ a < 10 ∧ b < 10

/-- The main theorem stating the solution to the numerical puzzle -/
theorem numerical_puzzle_solution :
  ∀ (a b : ℕ), differentDigits a b →
    isTwoDigit (10 * a + b) →
    (10 * a + b = b ^ (10 * a + b)) ↔ 
    ((a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 4)) :=
by sorry

end numerical_puzzle_solution_l3445_344551


namespace special_line_equation_l3445_344596

/-- A line passing through a point and at a fixed distance from the origin -/
structure SpecialLine where
  a : ℝ  -- x-coordinate of the point
  b : ℝ  -- y-coordinate of the point
  d : ℝ  -- distance from the origin

/-- The equation of the special line -/
def lineEquation (l : SpecialLine) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = l.a ∨ 3 * p.1 + 4 * p.2 - 5 = 0}

/-- Theorem: The equation of the line passing through (-1, 2) and at a distance of 1 from the origin -/
theorem special_line_equation :
  let l : SpecialLine := ⟨-1, 2, 1⟩
  lineEquation l = {p : ℝ × ℝ | p.1 = -1 ∨ 3 * p.1 + 4 * p.2 - 5 = 0} := by
  sorry


end special_line_equation_l3445_344596


namespace inverse_sum_product_identity_l3445_344548

theorem inverse_sum_product_identity (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y*z + x*z + x*y) * x⁻¹ * y⁻¹ * z⁻¹ * (x + y + z)⁻¹ := by
  sorry

end inverse_sum_product_identity_l3445_344548


namespace symmetry_implies_b_pow_a_l3445_344516

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite and y-coordinates are equal -/
def symmetric_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetry_implies_b_pow_a (a b : ℝ) :
  symmetric_y_axis (2*a) 2 (-8) (a+b) → b^a = 16 :=
by
  sorry

end symmetry_implies_b_pow_a_l3445_344516


namespace two_digit_number_proof_l3445_344511

theorem two_digit_number_proof :
  ∀ n : ℕ,
  (10 ≤ n ∧ n < 100) →  -- two-digit number
  (n % 2 = 0) →  -- even number
  (n / 10 * (n % 10) = 20) →  -- product of digits is 20
  n = 54 :=
by
  sorry

end two_digit_number_proof_l3445_344511


namespace square_diagonal_cut_l3445_344535

theorem square_diagonal_cut (s : ℝ) (h : s = 10) : 
  let diagonal := s * Real.sqrt 2
  ∃ (a b c : ℝ), a = s ∧ b = s ∧ c = diagonal ∧ 
    a^2 + b^2 = c^2 :=
by sorry

end square_diagonal_cut_l3445_344535


namespace trip_savings_l3445_344566

def evening_ticket_cost : ℚ := 10
def combo_cost : ℚ := 10
def ticket_discount_percent : ℚ := 20
def combo_discount_percent : ℚ := 50

def ticket_savings : ℚ := (ticket_discount_percent / 100) * evening_ticket_cost
def combo_savings : ℚ := (combo_discount_percent / 100) * combo_cost
def total_savings : ℚ := ticket_savings + combo_savings

theorem trip_savings : total_savings = 7 := by sorry

end trip_savings_l3445_344566


namespace magazine_selling_price_l3445_344533

/-- Given the cost price, number of magazines, and total gain, 
    calculate the selling price per magazine. -/
theorem magazine_selling_price 
  (cost_price : ℝ) 
  (num_magazines : ℕ) 
  (total_gain : ℝ) 
  (h1 : cost_price = 3)
  (h2 : num_magazines = 10)
  (h3 : total_gain = 5) :
  (cost_price * num_magazines + total_gain) / num_magazines = 3.5 := by
  sorry

end magazine_selling_price_l3445_344533


namespace train_length_calculation_l3445_344562

/-- Proves that the length of each train is 75 meters given the specified conditions. -/
theorem train_length_calculation (v1 v2 t : ℝ) (h1 : v1 = 46) (h2 : v2 = 36) (h3 : t = 54) :
  let relative_speed := (v1 - v2) * (5 / 18)
  let distance := relative_speed * t
  let train_length := distance / 2
  train_length = 75 := by sorry

end train_length_calculation_l3445_344562


namespace smaller_acute_angle_measure_l3445_344532

-- Define a right triangle with acute angles x and 4x
def right_triangle (x : ℝ) : Prop :=
  x > 0 ∧ x < 90 ∧ x + 4*x = 90

-- Theorem statement
theorem smaller_acute_angle_measure :
  ∃ (x : ℝ), right_triangle x ∧ x = 18 :=
by
  sorry

end smaller_acute_angle_measure_l3445_344532


namespace eighth_grade_class_problem_l3445_344500

theorem eighth_grade_class_problem (total students_math students_foreign : ℕ) 
  (h_total : total = 93)
  (h_math : students_math = 70)
  (h_foreign : students_foreign = 54) :
  students_math - (total - (students_math + students_foreign - total)) = 39 := by
  sorry

end eighth_grade_class_problem_l3445_344500


namespace boat_purchase_l3445_344543

theorem boat_purchase (a b c d : ℝ) 
  (h1 : a + b + c + d = 60)
  (h2 : a = (1/2) * (b + c + d))
  (h3 : b = (1/3) * (a + c + d))
  (h4 : c = (1/4) * (a + b + d))
  (h5 : a ≥ 0) (h6 : b ≥ 0) (h7 : c ≥ 0) (h8 : d ≥ 0) : d = 13 := by
  sorry

end boat_purchase_l3445_344543


namespace abs_minus_self_nonneg_l3445_344538

theorem abs_minus_self_nonneg (m : ℚ) : |m| - m ≥ 0 := by sorry

end abs_minus_self_nonneg_l3445_344538
