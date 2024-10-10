import Mathlib

namespace class_size_l2764_276460

theorem class_size (football : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ)
  (h1 : football = 26)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 7) :
  football + tennis - both + neither = 36 := by
  sorry

end class_size_l2764_276460


namespace three_digit_number_proof_l2764_276496

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def left_append_2 (n : ℕ) : ℕ := 2000 + n

def right_append_2 (n : ℕ) : ℕ := n * 10 + 2

theorem three_digit_number_proof :
  ∃! n : ℕ, is_three_digit n ∧ has_distinct_digits n ∧
  (left_append_2 n - right_append_2 n = 945 ∨ right_append_2 n - left_append_2 n = 945) ∧
  n = 327 :=
sorry

end three_digit_number_proof_l2764_276496


namespace expression_simplification_l2764_276463

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 - a / (a + 1)) / ((a^2 - 1) / (a^2 + 2*a + 1)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l2764_276463


namespace right_triangle_max_ratio_l2764_276424

theorem right_triangle_max_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_a : a = 3) :
  (∀ x y z, x^2 + y^2 = z^2 → x = 3 → (x^2 + y^2 + z^2) / z^2 ≤ 2) ∧
  (∃ x y z, x^2 + y^2 = z^2 ∧ x = 3 ∧ (x^2 + y^2 + z^2) / z^2 = 2) :=
by sorry

end right_triangle_max_ratio_l2764_276424


namespace f_neg_two_eq_neg_three_l2764_276449

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_neg_two_eq_neg_three
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_pos : ∀ x > 0, f x = x^2 - 1) :
  f (-2) = -3 := by
  sorry

end f_neg_two_eq_neg_three_l2764_276449


namespace sqrt_product_equality_l2764_276406

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_product_equality_l2764_276406


namespace polynomial_simplification_and_evaluation_l2764_276490

theorem polynomial_simplification_and_evaluation (a b : ℝ) :
  (-3 * a * (a - 2 * b)^5 + 6 * b * (a - 2 * b)^5 - 5 * (-a + 2 * b)^3 = -3 * (a - 2 * b)^6 + 5 * (a - 2 * b)^3) ∧
  (a - 2 * b = -1 → -3 * (a - 2 * b)^6 + 5 * (a - 2 * b)^3 = -8) :=
by sorry

end polynomial_simplification_and_evaluation_l2764_276490


namespace integer_solutions_of_system_l2764_276429

theorem integer_solutions_of_system (x y : ℤ) : 
  (4 * x^2 = y^2 + 2*y + 1 + 3 ∧ 
   (2*x)^2 - (y + 1)^2 = 3 ∧ 
   (2*x - y - 1) * (2*x + y + 1) = 3) ↔ 
  ((x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -2)) :=
by sorry

end integer_solutions_of_system_l2764_276429


namespace rod_length_l2764_276465

theorem rod_length (pieces : ℝ) (piece_length : ℝ) (h1 : pieces = 118.75) (h2 : piece_length = 0.40) :
  pieces * piece_length = 47.5 := by
  sorry

end rod_length_l2764_276465


namespace solution_set_characterization_l2764_276472

def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - (a + 1) * x + 1 < 0

def solution_set (a : ℝ) : Set ℝ :=
  {x | quadratic_inequality a x}

theorem solution_set_characterization (a : ℝ) (h : a > 0) :
  (a = 2 → solution_set a = Set.Ioo (1/2) 1) ∧
  (0 < a ∧ a < 1 → solution_set a = Set.Ioo 1 (1/a)) ∧
  (a = 1 → solution_set a = ∅) ∧
  (a > 1 → solution_set a = Set.Ioo (1/a) 1) :=
sorry

end solution_set_characterization_l2764_276472


namespace f_max_value_l2764_276408

def f (a b c : Real) : Real :=
  a * (1 - a + a * b) * (1 - a * b + a * b * c) * (1 - c)

theorem f_max_value (a b c : Real) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  f a b c ≤ 8/27 := by
  sorry

end f_max_value_l2764_276408


namespace fraction_evaluation_l2764_276475

theorem fraction_evaluation : (5 * 6 + 4) / 8 = 4.25 := by
  sorry

end fraction_evaluation_l2764_276475


namespace sqrt_11_plus_1_bounds_l2764_276489

-- Define the theorem
theorem sqrt_11_plus_1_bounds : 4 < Real.sqrt 11 + 1 ∧ Real.sqrt 11 + 1 < 5 := by
  sorry

#check sqrt_11_plus_1_bounds

end sqrt_11_plus_1_bounds_l2764_276489


namespace abs_x_bound_inequality_x_y_l2764_276410

-- Part 1
theorem abs_x_bound (x y : ℝ) 
  (h1 : |x - 3*y| < 1/2) (h2 : |x + 2*y| < 1/6) : 
  |x| < 3/10 := by sorry

-- Part 2
theorem inequality_x_y (x y : ℝ) :
  x^4 + 16*y^4 ≥ 2*x^3*y + 8*x*y^3 := by sorry

end abs_x_bound_inequality_x_y_l2764_276410


namespace small_cakes_needed_l2764_276444

/-- Prove that given the conditions, the number of small cakes needed is 630 --/
theorem small_cakes_needed (helpers : ℕ) (large_cakes_needed : ℕ) (hours : ℕ)
  (large_cakes_per_hour : ℕ) (small_cakes_per_hour : ℕ) :
  helpers = 10 →
  large_cakes_needed = 20 →
  hours = 3 →
  large_cakes_per_hour = 2 →
  small_cakes_per_hour = 35 →
  (helpers * hours * small_cakes_per_hour) - 
  (large_cakes_needed * small_cakes_per_hour * hours / large_cakes_per_hour) = 630 := by
  sorry

#check small_cakes_needed

end small_cakes_needed_l2764_276444


namespace range_of_f_l2764_276467

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Define the domain
def D : Set ℝ := {x | -3 ≤ x ∧ x < 2}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ D, f x = y} = {y | 2 ≤ y ∧ y < 11} :=
sorry

end range_of_f_l2764_276467


namespace range_of_a_l2764_276441

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x + 2/x + a ≥ 0) → a ≥ -3 := by
  sorry

end range_of_a_l2764_276441


namespace sufficient_not_necessary_l2764_276487

-- Define the interval [1,2]
def I : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 2 }

-- Define the proposition
def P (a : ℝ) : Prop := ∀ x ∈ I, x^2 - a ≤ 0

-- Define the sufficient condition
def S (a : ℝ) : Prop := a ≥ 5

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ a, S a → P a) ∧ (∃ a, P a ∧ ¬S a) :=
sorry

end sufficient_not_necessary_l2764_276487


namespace similar_triangles_exist_l2764_276413

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorFunction : Point → Color := sorry

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define similarity ratio between two triangles
def similarityRatio (T1 T2 : Triangle) : ℝ := sorry

-- Define a predicate to check if all vertices of a triangle have the same color
def sameColor (T : Triangle) : Prop :=
  colorFunction T.A = colorFunction T.B ∧ colorFunction T.B = colorFunction T.C

-- The main theorem
theorem similar_triangles_exist :
  ∃ (T1 T2 : Triangle), similarityRatio T1 T2 = 1995 ∧ sameColor T1 ∧ sameColor T2 := by
  sorry

end similar_triangles_exist_l2764_276413


namespace car_speed_problem_l2764_276476

theorem car_speed_problem (distance_AB : ℝ) (speed_A : ℝ) (time : ℝ) (final_distance : ℝ) :
  distance_AB = 300 →
  speed_A = 40 →
  time = 2 →
  final_distance = 100 →
  (∃ speed_B : ℝ, speed_B = 140 ∨ speed_B = 60) :=
by sorry

end car_speed_problem_l2764_276476


namespace locus_of_centers_l2764_276484

/-- Circle C1 with center (1,1) and radius 2 -/
def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

/-- Circle C2 with center (4,1) and radius 3 -/
def C2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = 9

/-- A circle with center (a,b) and radius r -/
def Circle (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

/-- External tangency condition -/
def ExternallyTangent (a b r : ℝ) : Prop := (a - 1)^2 + (b - 1)^2 = (r + 2)^2

/-- Internal tangency condition -/
def InternallyTangent (a b r : ℝ) : Prop := (a - 4)^2 + (b - 1)^2 = (3 - r)^2

/-- The locus equation -/
def LocusEquation (a b : ℝ) : Prop := 84*a^2 + 100*b^2 - 336*a - 200*b + 900 = 0

theorem locus_of_centers :
  ∀ a b : ℝ, (∃ r : ℝ, ExternallyTangent a b r ∧ InternallyTangent a b r) ↔ LocusEquation a b :=
sorry

end locus_of_centers_l2764_276484


namespace sum_of_ages_l2764_276426

/-- Proves that the sum of Henry and Jill's present ages is 43 years -/
theorem sum_of_ages (henry_age jill_age : ℕ) : 
  henry_age = 27 →
  jill_age = 16 →
  henry_age - 5 = 2 * (jill_age - 5) →
  henry_age + jill_age = 43 :=
by sorry

end sum_of_ages_l2764_276426


namespace min_sum_given_product_minus_sum_l2764_276446

theorem min_sum_given_product_minus_sum (a b : ℝ) 
  (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) : 
  a + b ≥ 2 + 2 * Real.sqrt 2 := by
sorry

end min_sum_given_product_minus_sum_l2764_276446


namespace function_properties_l2764_276428

noncomputable def f (b c x : ℝ) : ℝ := (2 * x^2 + b * x + c) / (x^2 + 1)

noncomputable def F (b c : ℝ) (x : ℝ) : ℝ := Real.log (f b c x) / Real.log 10

theorem function_properties (b c : ℝ) 
  (h_b : b < 0) 
  (h_range : Set.range (f b c) = Set.Icc 1 3) :
  (b = -2 ∧ c = 2) ∧ 
  (∀ x y : ℝ, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x < y → F b c x < F b c y) ∧
  (∀ t : ℝ, Real.log (7/5) / Real.log 10 ≤ F b c (|t - 1/6| - |t + 1/6|) ∧ 
            F b c (|t - 1/6| - |t + 1/6|) ≤ Real.log (13/5) / Real.log 10) :=
by sorry

end function_properties_l2764_276428


namespace inequality_constraint_l2764_276450

theorem inequality_constraint (a : ℝ) : 
  (∀ x : ℝ, x > 1 → (a + 1) / x + Real.log x > a) → a ≠ 3 := by
  sorry

end inequality_constraint_l2764_276450


namespace min_value_f_neg_reals_l2764_276481

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x - 8/x + 4/x^2 + 5

theorem min_value_f_neg_reals :
  ∃ (x_min : ℝ), x_min < 0 ∧
  ∀ (x : ℝ), x < 0 → f x ≥ f x_min ∧ f x_min = 9 + 8 * Real.sqrt 2 :=
sorry

end min_value_f_neg_reals_l2764_276481


namespace luke_needs_307_stars_l2764_276439

/-- The number of additional stars Luke needs to make -/
def additional_stars_needed (stars_per_jar : ℕ) (jars_to_fill : ℕ) (stars_already_made : ℕ) : ℕ :=
  stars_per_jar * jars_to_fill - stars_already_made

/-- Proof that Luke needs to make 307 more stars -/
theorem luke_needs_307_stars :
  additional_stars_needed 85 4 33 = 307 := by
  sorry

end luke_needs_307_stars_l2764_276439


namespace two_color_similar_ngons_l2764_276492

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points on a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define similarity between two n-gons
def AreSimilarNGons (n : ℕ) (k : ℝ) (ngon1 ngon2 : Fin n → Point) : Prop :=
  ∃ (center : Point), ∀ (i : Fin n),
    let p1 := ngon1 i
    let p2 := ngon2 i
    (p2.x - center.x)^2 + (p2.y - center.y)^2 = k^2 * ((p1.x - center.x)^2 + (p1.y - center.y)^2)

theorem two_color_similar_ngons 
  (n : ℕ) 
  (h_n : n ≥ 3) 
  (k : ℝ) 
  (h_k : k > 0 ∧ k ≠ 1) 
  (coloring : Coloring) :
  ∃ (ngon1 ngon2 : Fin n → Point),
    AreSimilarNGons n k ngon1 ngon2 ∧
    (∃ (c : Color), (∀ (i : Fin n), coloring (ngon1 i) = c)) ∧
    (∃ (c : Color), (∀ (i : Fin n), coloring (ngon2 i) = c)) :=
by sorry

end two_color_similar_ngons_l2764_276492


namespace system_solution_l2764_276494

theorem system_solution (x y m : ℝ) : 
  (3 * x + 2 * y = 4 * m - 5 ∧ 
   2 * x + 3 * y = m ∧ 
   x + y = 2) → 
  m = 3 := by
sorry

end system_solution_l2764_276494


namespace equation_solution_l2764_276436

theorem equation_solution (x : ℝ) : (x - 3)^2 = x^2 - 9 → x = 3 := by
  sorry

end equation_solution_l2764_276436


namespace arithmetic_mean_product_l2764_276402

theorem arithmetic_mean_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 ∧ 
  a = 14 ∧ 
  b = 25 ∧ 
  c + 3 = d → 
  c * d = 418 := by sorry

end arithmetic_mean_product_l2764_276402


namespace polynomial_identity_result_l2764_276486

theorem polynomial_identity_result : 
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ),
  (∀ x : ℝ, (x^2 - x + 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                               a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12) →
  (a₀ + a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂)^2 - (a₁ + a₃ + a₅ + a₇ + a₉ + a₁₁)^2 = 729 :=
by sorry

end polynomial_identity_result_l2764_276486


namespace max_value_inequality_max_value_attainable_l2764_276412

theorem max_value_inequality (x y : ℝ) :
  (2 * x + 3 * y + 5) / Real.sqrt (2 * x^2 + 3 * y^2 + 7) ≤ Real.sqrt 38 :=
by sorry

theorem max_value_attainable :
  ∃ x y : ℝ, (2 * x + 3 * y + 5) / Real.sqrt (2 * x^2 + 3 * y^2 + 7) = Real.sqrt 38 :=
by sorry

end max_value_inequality_max_value_attainable_l2764_276412


namespace geometric_sequence_sum_l2764_276482

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 1365/16384 := by
sorry

end geometric_sequence_sum_l2764_276482


namespace scientific_notation_equals_original_l2764_276488

/-- Scientific notation representation of 470,000,000 -/
def scientific_notation : ℝ := 4.7 * (10 ^ 8)

/-- The original number -/
def original_number : ℕ := 470000000

theorem scientific_notation_equals_original : 
  (scientific_notation : ℝ) = original_number := by sorry

end scientific_notation_equals_original_l2764_276488


namespace cubic_third_root_l2764_276437

theorem cubic_third_root (a b : ℚ) :
  (∀ x : ℚ, a * x^3 + (a + 4*b) * x^2 + (b - 5*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 8/3) →
  (a * (-1)^3 + (a + 4*b) * (-1)^2 + (b - 5*a) * (-1) + (10 - a) = 0) →
  (a * 4^3 + (a + 4*b) * 4^2 + (b - 5*a) * 4 + (10 - a) = 0) →
  ∃ x : ℚ, x = 8/3 ∧ a * x^3 + (a + 4*b) * x^2 + (b - 5*a) * x + (10 - a) = 0 :=
by sorry

end cubic_third_root_l2764_276437


namespace equal_utility_days_l2764_276419

/-- Utility function --/
def utility (math reading painting : ℝ) : ℝ := math^2 + reading * painting

/-- The problem statement --/
theorem equal_utility_days (t : ℝ) : 
  utility 4 t (12 - t) = utility 3 (t + 1) (11 - t) → t = 2 := by
  sorry

end equal_utility_days_l2764_276419


namespace sams_new_crime_books_l2764_276479

theorem sams_new_crime_books 
  (used_adventure : ℝ) 
  (used_mystery : ℝ) 
  (total_books : ℝ) 
  (h1 : used_adventure = 13.0)
  (h2 : used_mystery = 17.0)
  (h3 : total_books = 45.0) :
  total_books - (used_adventure + used_mystery) = 15.0 := by
  sorry

end sams_new_crime_books_l2764_276479


namespace alpha_minus_beta_range_l2764_276430

theorem alpha_minus_beta_range (α β : Real) (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π/2) :
  ∃ (x : Real), x = α - β ∧ -3*π/2 ≤ x ∧ x ≤ 0 ∧
  ∀ (y : Real), (-3*π/2 ≤ y ∧ y ≤ 0) → ∃ (α' β' : Real), 
    -π ≤ α' ∧ α' ≤ β' ∧ β' ≤ π/2 ∧ y = α' - β' :=
by
  sorry

end alpha_minus_beta_range_l2764_276430


namespace aftershave_alcohol_percentage_l2764_276499

/-- Proves that the initial alcohol percentage in an after-shave lotion is 30% -/
theorem aftershave_alcohol_percentage
  (initial_volume : ℝ)
  (water_volume : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 50)
  (h2 : water_volume = 30)
  (h3 : final_percentage = 18.75)
  (h4 : (initial_volume * x / 100) = ((initial_volume + water_volume) * final_percentage / 100)) :
  x = 30 := by
  sorry

end aftershave_alcohol_percentage_l2764_276499


namespace smallest_marble_count_l2764_276405

/-- Represents the number of marbles of each color -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles -/
def totalMarbles (m : MarbleCount) : ℕ :=
  m.red + m.white + m.blue + m.green + m.yellow

/-- Represents the probability of selecting a specific combination of marbles -/
def selectProbability (m : MarbleCount) (r w b g : ℕ) : ℚ :=
  (m.red.choose r * m.white.choose w * m.blue.choose b * m.green.choose g : ℚ) /
  (totalMarbles m).choose 5

/-- The conditions for the marble selection probabilities to be equal -/
def equalProbabilities (m : MarbleCount) : Prop :=
  selectProbability m 5 0 0 0 = selectProbability m 4 1 0 0 ∧
  selectProbability m 5 0 0 0 = selectProbability m 3 1 1 0 ∧
  selectProbability m 5 0 0 0 = selectProbability m 2 1 1 1 ∧
  selectProbability m 5 0 0 0 = selectProbability m 1 1 1 1

/-- The theorem stating the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : 
  ∃ (m : MarbleCount), 
    m.yellow = 4 ∧ 
    equalProbabilities m ∧
    totalMarbles m = 27 ∧
    (∀ (m' : MarbleCount), m'.yellow = 4 → equalProbabilities m' → totalMarbles m' ≥ 27) :=
sorry

end smallest_marble_count_l2764_276405


namespace cube_inscribed_in_sphere_l2764_276455

theorem cube_inscribed_in_sphere (edge_length : ℝ) (sphere_area : ℝ) : 
  edge_length = Real.sqrt 2 →
  sphere_area = 6 * Real.pi :=
by
  sorry

end cube_inscribed_in_sphere_l2764_276455


namespace cosine_sum_special_angle_l2764_276448

/-- 
If the terminal side of angle θ passes through the point (3, -4), 
then cos(θ + π/4) = 7√2/10.
-/
theorem cosine_sum_special_angle (θ : ℝ) : 
  (3 : ℝ) * Real.cos θ = 3 ∧ (3 : ℝ) * Real.sin θ = -4 → 
  Real.cos (θ + π/4) = 7 * Real.sqrt 2 / 10 := by
  sorry

end cosine_sum_special_angle_l2764_276448


namespace A_intersect_B_l2764_276447

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < 2 - x ∧ 2 - x < 3}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end A_intersect_B_l2764_276447


namespace complex_equation_solution_l2764_276471

theorem complex_equation_solution (a b c : ℕ+) 
  (h : (a - b * Complex.I) ^ 2 + c = 13 - 8 * Complex.I) :
  a = 2 ∧ b = 2 ∧ c = 13 := by
  sorry

end complex_equation_solution_l2764_276471


namespace parallel_planes_through_two_points_l2764_276435

-- Define a plane
def Plane : Type := sorry

-- Define a point
def Point : Type := sorry

-- Define a function to check if a point is outside a plane
def isOutside (p : Point) (pl : Plane) : Prop := sorry

-- Define a function to check if a plane is parallel to another plane
def isParallel (pl1 : Plane) (pl2 : Plane) : Prop := sorry

-- Define a function to count the number of planes that can be drawn through two points and parallel to a given plane
def countParallelPlanes (p1 p2 : Point) (pl : Plane) : Nat := sorry

-- Theorem statement
theorem parallel_planes_through_two_points 
  (p1 p2 : Point) (pl : Plane) 
  (h1 : isOutside p1 pl) 
  (h2 : isOutside p2 pl) : 
  countParallelPlanes p1 p2 pl = 0 ∨ countParallelPlanes p1 p2 pl = 1 := by
  sorry

end parallel_planes_through_two_points_l2764_276435


namespace pi_between_three_and_four_l2764_276459

theorem pi_between_three_and_four : 
  Irrational Real.pi ∧ 3 < Real.pi ∧ Real.pi < 4 := by sorry

end pi_between_three_and_four_l2764_276459


namespace shortest_ant_path_equals_slant_edge_l2764_276445

/-- Represents a regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  slantEdgeLength : ℝ
  dihedralAngle : ℝ

/-- The shortest path for an ant to visit all slant edges and return to the starting point -/
def shortestAntPath (pyramid : RegularHexagonalPyramid) : ℝ :=
  pyramid.slantEdgeLength

theorem shortest_ant_path_equals_slant_edge 
  (pyramid : RegularHexagonalPyramid) 
  (h1 : pyramid.dihedralAngle = 10) : 
  shortestAntPath pyramid = pyramid.slantEdgeLength :=
sorry

end shortest_ant_path_equals_slant_edge_l2764_276445


namespace amusement_park_spending_l2764_276485

theorem amusement_park_spending (initial_amount snack_cost : ℕ) : 
  initial_amount = 100 →
  snack_cost = 20 →
  initial_amount - (snack_cost + 3 * snack_cost) = 20 := by
  sorry

end amusement_park_spending_l2764_276485


namespace solve_equation_l2764_276497

theorem solve_equation (x : ℝ) :
  Real.sqrt ((3 / x) + 3 * x) = 3 →
  x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2 := by
  sorry

end solve_equation_l2764_276497


namespace complex_number_real_imag_equal_l2764_276473

theorem complex_number_real_imag_equal (a : ℝ) : 
  let z : ℂ := (6 + a * Complex.I) / (3 - Complex.I)
  (z.re = z.im) → a = 3 := by
  sorry

end complex_number_real_imag_equal_l2764_276473


namespace fiona_hoodies_l2764_276478

theorem fiona_hoodies (total : ℕ) (casey_extra : ℕ) : 
  total = 8 → casey_extra = 2 → ∃ (fiona : ℕ), 
    fiona + (fiona + casey_extra) = total ∧ fiona = 3 := by
  sorry

end fiona_hoodies_l2764_276478


namespace monge_circle_theorem_monge_circle_tangent_point_l2764_276420

/-- The Monge circle theorem for an ellipse with semi-major axis a and semi-minor axis b --/
theorem monge_circle_theorem (a b : ℝ) (h : 0 < b ∧ b < a) :
  ∃ (r : ℝ), r^2 = a^2 + b^2 ∧ 
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 → 
    ∃ (t s : ℝ), (t * x + s * y = 0 ∧ t^2 + s^2 ≠ 0) → 
      x^2 + y^2 = r^2) :=
sorry

/-- The main theorem about the value of b --/
theorem monge_circle_tangent_point (b : ℝ) : 
  (∃ (x y : ℝ), x^2/3 + y^2 = 1) →  -- Ellipse exists
  (∃ (x y : ℝ), (x-3)^2 + (y-b)^2 = 9) →  -- Circle exists
  (∃! (x y : ℝ), x^2 + y^2 = 4 ∧ (x-3)^2 + (y-b)^2 = 9) →  -- Exactly one common point
  b = 4 :=
sorry

end monge_circle_theorem_monge_circle_tangent_point_l2764_276420


namespace sum_of_squares_representation_l2764_276434

theorem sum_of_squares_representation (n m : ℕ) :
  ∃ (x y : ℕ), (2014^2 + 2016^2) / 2 = x^2 + y^2 ∧
  ∃ (a b : ℕ), (4*n^2 + 4*m^2) / 2 = a^2 + b^2 :=
by sorry

end sum_of_squares_representation_l2764_276434


namespace smallest_integer_with_remainders_l2764_276431

theorem smallest_integer_with_remainders : ∃ n : ℕ, n > 0 ∧
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 6 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_integer_with_remainders_l2764_276431


namespace blackboard_remainder_l2764_276416

theorem blackboard_remainder (a : ℕ) : 
  a < 10 → (a + 100) % 7 = 5 → a = 5 := by sorry

end blackboard_remainder_l2764_276416


namespace coin_toss_sequence_count_l2764_276415

/-- Represents a coin toss sequence. -/
def CoinSequence := List Bool

/-- Counts the number of occurrences of a given subsequence in a coin sequence. -/
def countSubsequence (seq : CoinSequence) (subseq : List Bool) : Nat :=
  sorry

/-- Checks if a coin sequence satisfies the given conditions. -/
def isValidSequence (seq : CoinSequence) : Prop :=
  seq.length = 20 ∧
  countSubsequence seq [true, true] = 3 ∧
  countSubsequence seq [true, false] = 4 ∧
  countSubsequence seq [false, true] = 5 ∧
  countSubsequence seq [false, false] = 7

/-- The number of valid coin toss sequences. -/
def validSequenceCount : Nat :=
  sorry

theorem coin_toss_sequence_count :
  validSequenceCount = 11550 :=
sorry

end coin_toss_sequence_count_l2764_276415


namespace intersection_complement_theorem_l2764_276477

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 4}
def B : Set Nat := {2, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1} := by sorry

end intersection_complement_theorem_l2764_276477


namespace project_completion_equation_l2764_276407

/-- Represents the number of days required for a person to complete the project alone -/
structure ProjectTime where
  person_a : ℝ
  person_b : ℝ

/-- Represents the work schedule for the project -/
structure WorkSchedule where
  solo_days : ℝ
  total_days : ℝ

/-- Theorem stating the equation for the total number of days required to complete the project -/
theorem project_completion_equation (pt : ProjectTime) (ws : WorkSchedule) :
  pt.person_a = 12 →
  pt.person_b = 8 →
  ws.solo_days = 3 →
  ws.total_days / pt.person_a + (ws.total_days - ws.solo_days) / pt.person_b = 1 := by
  sorry

end project_completion_equation_l2764_276407


namespace team_formation_count_l2764_276425

theorem team_formation_count (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  (Nat.choose (n - 1) (k - 1)) = 406 :=
by sorry

end team_formation_count_l2764_276425


namespace matrix_determinant_l2764_276469

theorem matrix_determinant : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 0, 1; -5, 5, -4; 3, 3, 6]
  Matrix.det A = 96 := by sorry

end matrix_determinant_l2764_276469


namespace specific_cube_surface_area_l2764_276457

/-- Calculates the total surface area of a cube with holes -/
def cubeWithHolesSurfaceArea (cubeEdge : ℝ) (holeEdge : ℝ) (numHoles : ℕ) : ℝ :=
  let originalSurfaceArea := 6 * cubeEdge^2
  let holeArea := numHoles * holeEdge^2
  let newExposedArea := numHoles * 4 * cubeEdge * holeEdge
  originalSurfaceArea - holeArea + newExposedArea

/-- Theorem stating the surface area of a specific cube with holes -/
theorem specific_cube_surface_area :
  cubeWithHolesSurfaceArea 5 2 3 = 258 := by
  sorry

#eval cubeWithHolesSurfaceArea 5 2 3

end specific_cube_surface_area_l2764_276457


namespace no_perfect_square_9999_xxxx_l2764_276411

theorem no_perfect_square_9999_xxxx : 
  ¬ ∃ x : ℕ, 99990000 ≤ x ∧ x ≤ 99999999 ∧ ∃ y : ℕ, x = y^2 := by
  sorry

end no_perfect_square_9999_xxxx_l2764_276411


namespace triangle_uniqueness_l2764_276491

/-- Given two excircle radii and an altitude of a triangle, 
    the triangle is uniquely determined iff the altitude is not 
    equal to the harmonic mean of the two radii. -/
theorem triangle_uniqueness (ρa ρb mc : ℝ) (h_pos : ρa > 0 ∧ ρb > 0 ∧ mc > 0) :
  ∃! (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (a + b > c ∧ b + c > a ∧ c + a > b) ∧
    (ρa = (a + b + c) / (2 * (b + c))) ∧
    (ρb = (a + b + c) / (2 * (c + a))) ∧
    (mc = 2 * (a * b * c) / ((a + b + c) * c)) ↔ 
  mc ≠ 2 * ρa * ρb / (ρa + ρb) := by
sorry

end triangle_uniqueness_l2764_276491


namespace closest_ratio_is_one_to_one_l2764_276417

def admission_fee_adult : ℕ := 30
def admission_fee_child : ℕ := 15
def total_collected : ℕ := 2250

def is_valid_combination (adults children : ℕ) : Prop :=
  adults ≥ 1 ∧ children ≥ 1 ∧
  adults * admission_fee_adult + children * admission_fee_child = total_collected

def ratio_difference_from_one (adults children : ℕ) : ℚ :=
  |((adults : ℚ) / (children : ℚ)) - 1|

theorem closest_ratio_is_one_to_one :
  ∃ (a c : ℕ), is_valid_combination a c ∧
    ∀ (x y : ℕ), is_valid_combination x y →
      ratio_difference_from_one a c ≤ ratio_difference_from_one x y :=
by sorry

end closest_ratio_is_one_to_one_l2764_276417


namespace floor_sqrt_80_l2764_276427

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end floor_sqrt_80_l2764_276427


namespace triangle_area_l2764_276423

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x)^2 + Real.sin x * Real.cos x

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  f (A / 2) = Real.sqrt 3 ∧
  a = 4 ∧
  b + c = 5 →
  (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 4 :=
by sorry

end triangle_area_l2764_276423


namespace tan_alpha_value_l2764_276432

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 5 * Real.cos (2 * α) = Real.sqrt 2 * Real.sin (π / 4 - α)) : 
  Real.tan α = -4/3 := by
sorry

end tan_alpha_value_l2764_276432


namespace contrapositive_equivalence_l2764_276404

theorem contrapositive_equivalence :
  (∀ x y : ℝ, (x = 3 ∧ y = 5) → x + y = 8) ↔
  (∀ x y : ℝ, x + y ≠ 8 → (x ≠ 3 ∨ y ≠ 5)) :=
by sorry

end contrapositive_equivalence_l2764_276404


namespace triangle_problem_l2764_276440

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.b = Real.sqrt 3)
  (h2 : t.a + t.c = 4)
  (h3 : Real.cos t.B / Real.cos t.C = -t.b / (2 * t.a + t.c)) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B : ℝ) = 13 * Real.sqrt 3 / 4 := by
  sorry

end triangle_problem_l2764_276440


namespace blackboard_sum_l2764_276403

def Operation : Type := List ℕ → List ℕ → (List ℕ × List ℕ)

def performOperations (initialBoard : List ℕ) (n : ℕ) (op : Operation) : (List ℕ × List ℕ) :=
  sorry

theorem blackboard_sum (initialBoard : List ℕ) (finalBoard : List ℕ) (paperNumbers : List ℕ) :
  initialBoard = [1, 3, 5, 7, 9] →
  (∃ op : Operation, performOperations initialBoard 4 op = (finalBoard, paperNumbers)) →
  finalBoard.length = 1 →
  paperNumbers.length = 4 →
  paperNumbers.sum = 230 :=
  sorry

end blackboard_sum_l2764_276403


namespace increasing_function_condition_l2764_276453

theorem increasing_function_condition (f : ℝ → ℝ) (h : Monotone f) :
  ∀ a b : ℝ, a + b > 0 ↔ f a + f b > f (-a) + f (-b) :=
by sorry

end increasing_function_condition_l2764_276453


namespace gold_bar_value_proof_l2764_276474

/-- The value of one bar of gold -/
def gold_bar_value : ℝ := 2200

/-- The number of gold bars Legacy has -/
def legacy_bars : ℕ := 5

/-- The number of gold bars Aleena has -/
def aleena_bars : ℕ := legacy_bars - 2

/-- The total value of gold Legacy and Aleena have together -/
def total_value : ℝ := 17600

theorem gold_bar_value_proof : 
  gold_bar_value * (legacy_bars + aleena_bars : ℝ) = total_value := by
  sorry

end gold_bar_value_proof_l2764_276474


namespace base6_addition_puzzle_l2764_276466

/-- Converts a base 6 number to base 10 -/
def base6_to_base10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 36 + tens * 6 + ones

/-- Converts a base 10 number to base 6 -/
def base10_to_base6 (n : Nat) : Nat × Nat × Nat :=
  let hundreds := n / 36
  let tens := (n % 36) / 6
  let ones := n % 6
  (hundreds, tens, ones)

theorem base6_addition_puzzle :
  ∃ (S H E : Nat),
    S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 ∧
    S < 6 ∧ H < 6 ∧ E < 6 ∧
    S ≠ H ∧ S ≠ E ∧ H ≠ E ∧
    base6_to_base10 S H E + base6_to_base10 0 H E = base6_to_base10 S E S ∧
    S = 4 ∧ H = 1 ∧ E = 2 ∧
    base10_to_base6 (S + H + E) = (0, 1, 1) := by
  sorry

#eval base10_to_base6 7  -- Expected output: (0, 1, 1)

end base6_addition_puzzle_l2764_276466


namespace max_visible_cubes_12x12x12_l2764_276451

/-- Represents a cubic structure composed of unit cubes -/
structure CubicStructure where
  size : ℕ
  deriving Repr

/-- Calculates the maximum number of visible unit cubes from a single point outside the cube -/
def maxVisibleUnitCubes (c : CubicStructure) : ℕ :=
  3 * c.size^2 - 3 * (c.size - 1) + 1

/-- Theorem stating that for a 12 × 12 × 12 cube, the maximum number of visible unit cubes is 400 -/
theorem max_visible_cubes_12x12x12 :
  maxVisibleUnitCubes ⟨12⟩ = 400 := by
  sorry

#eval maxVisibleUnitCubes ⟨12⟩

end max_visible_cubes_12x12x12_l2764_276451


namespace boat_speed_in_still_water_l2764_276468

/-- Proves that the speed of a boat in still water is 16 km/hr, given the rate of the stream and the time and distance traveled downstream. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 8)
  (h3 : downstream_distance = 168) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 16 ∧ 
    downstream_distance = (still_water_speed + stream_speed) * downstream_time :=
by sorry

end boat_speed_in_still_water_l2764_276468


namespace negation_of_universal_proposition_l2764_276464

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 3 > 0) ↔ (∃ x : ℝ, x^2 + x + 3 ≤ 0) := by sorry

end negation_of_universal_proposition_l2764_276464


namespace overall_class_average_l2764_276461

-- Define the percentages of each group
def group1_percent : Real := 0.20
def group2_percent : Real := 0.50
def group3_percent : Real := 1 - group1_percent - group2_percent

-- Define the test averages for each group
def group1_average : Real := 80
def group2_average : Real := 60
def group3_average : Real := 40

-- Define the overall class average
def class_average : Real :=
  group1_percent * group1_average +
  group2_percent * group2_average +
  group3_percent * group3_average

-- Theorem statement
theorem overall_class_average :
  class_average = 58 := by
  sorry

end overall_class_average_l2764_276461


namespace part_one_part_two_l2764_276421

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x + a|
def g (x : ℝ) : ℝ := |x + 3| - x

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x < g x}

-- Statement for part (1)
theorem part_one (a : ℝ) : (a - 3) ∈ M a → a ∈ Set.Ioo 0 3 := by sorry

-- Statement for part (2)
theorem part_two (a : ℝ) : Set.Icc (-1) 1 ⊆ M a → a ∈ Set.Ioo (-2) 2 := by sorry

end part_one_part_two_l2764_276421


namespace equation_one_solution_l2764_276438

theorem equation_one_solution (x : ℝ) : 
  (3 * x + 2)^2 = 25 ↔ x = 1 ∨ x = -7/3 := by sorry

end equation_one_solution_l2764_276438


namespace cos_sin_expression_in_terms_of_p_q_l2764_276442

open Real

theorem cos_sin_expression_in_terms_of_p_q (x : ℝ) 
  (p : ℝ) (hp : p = (1 - cos x) * (1 + sin x))
  (q : ℝ) (hq : q = (1 + cos x) * (1 - sin x)) :
  cos x ^ 2 - cos x ^ 4 - sin (2 * x) + 2 = p * q - (p + q) := by
  sorry

end cos_sin_expression_in_terms_of_p_q_l2764_276442


namespace exists_valid_nail_configuration_l2764_276493

/-- Represents a nail configuration for hanging a painting -/
structure NailConfiguration where
  nails : Fin 4 → Unit

/-- Represents the state of the painting (hanging or fallen) -/
inductive PaintingState
  | Hanging
  | Fallen

/-- Determines the state of the painting given a nail configuration and a set of removed nails -/
def paintingState (config : NailConfiguration) (removed : Set (Fin 4)) : PaintingState :=
  sorry

/-- Theorem stating the existence of a nail configuration satisfying the given conditions -/
theorem exists_valid_nail_configuration :
  ∃ (config : NailConfiguration),
    (∀ (i : Fin 4), paintingState config {i} = PaintingState.Hanging) ∧
    (∀ (i j : Fin 4), i ≠ j → paintingState config {i, j} = PaintingState.Fallen) :=
  sorry

end exists_valid_nail_configuration_l2764_276493


namespace power_calculation_l2764_276422

theorem power_calculation (n : ℝ) : 
  (3/5 : ℝ) * (14.500000000000002 : ℝ)^n = 126.15 → n = 2 := by
  sorry

end power_calculation_l2764_276422


namespace locus_of_centers_l2764_276452

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ with equation (x - 1)² + (y - 1)² = 81 -/
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 81

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 1)^2 + (b - 1)^2 = (9 - r)^2

/-- The locus of centers (a, b) of circles externally tangent to C₁ and internally tangent to C₂ -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) → 
  a^2 + b^2 - (2*a*b)/63 - (66*a)/63 - (66*b)/63 + 17 = 0 := by sorry

end locus_of_centers_l2764_276452


namespace range_of_a_l2764_276480

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 := by
  sorry

end range_of_a_l2764_276480


namespace polygon_exterior_interior_angles_equal_l2764_276400

theorem polygon_exterior_interior_angles_equal (n : ℕ) : 
  (n ≥ 3) → (360 = (n - 2) * 180) → n = 4 := by
  sorry

end polygon_exterior_interior_angles_equal_l2764_276400


namespace units_digit_expression_l2764_276456

def units_digit (n : ℤ) : ℕ :=
  (n % 10).toNat

theorem units_digit_expression : units_digit ((8 * 23 * 1982 - 8^3) + 8) = 4 := by
  sorry

end units_digit_expression_l2764_276456


namespace train_speed_l2764_276414

/-- The speed of a train given its length, time to cross a walking man, and the man's speed. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 500 →
  crossing_time = 29.997600191984642 →
  man_speed_kmh = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63) < 0.1 := by
  sorry


end train_speed_l2764_276414


namespace investment_average_rate_l2764_276470

theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) 
  (h1 : total = 6000)
  (h2 : rate1 = 0.035)
  (h3 : rate2 = 0.055)
  (h4 : ∃ x : ℝ, x > 0 ∧ x < total ∧ rate1 * (total - x) = rate2 * x) :
  ∃ avg_rate : ℝ, abs (avg_rate - 0.043) < 0.0001 ∧ 
  avg_rate * total = rate1 * (total - x) + rate2 * x :=
sorry

end investment_average_rate_l2764_276470


namespace arithmetic_geometric_sequence_l2764_276409

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 = 1 →                     -- first term condition
  d ≠ 0 →                       -- non-zero common difference
  (a 2) ^ 2 = a 1 * a 5 →       -- geometric sequence condition
  d = 2 := by
sorry

end arithmetic_geometric_sequence_l2764_276409


namespace probability_of_arrangement_l2764_276454

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

theorem probability_of_arrangement (total_children num_girls num_boys : ℕ) 
  (h1 : total_children = 20)
  (h2 : num_girls = 11)
  (h3 : num_boys = 9)
  (h4 : total_children = num_girls + num_boys) :
  (Nat.choose 14 9 + 6 * Nat.choose 13 8) / Nat.choose 20 9 = 
    (Nat.choose 14 9 + 6 * Nat.choose 13 8) / Nat.choose total_children num_boys := by
  sorry

end probability_of_arrangement_l2764_276454


namespace equidistant_point_y_coordinate_l2764_276443

/-- The y-coordinate of the point on the y-axis equidistant from C(-3, 0) and D(-2, 5) is 2 -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, ((-3 : ℝ)^2 + 0^2 + 0^2 + y^2 = (-2 : ℝ)^2 + 5^2 + 0^2 + (y - 5)^2) ∧ y = 2 := by
  sorry

end equidistant_point_y_coordinate_l2764_276443


namespace lcm_count_theorem_exists_valid_k_upper_bound_valid_k_l2764_276498

-- Define the function to count valid k values
def count_valid_k : ℕ :=
  -- Count the number of a values from 0 to 18 (inclusive)
  -- where k = 2^a * 3^36 satisfies the LCM condition
  (Finset.range 19).card

-- State the theorem
theorem lcm_count_theorem : 
  count_valid_k = 19 :=
sorry

-- Define the LCM condition
def is_valid_k (k : ℕ) : Prop :=
  Nat.lcm (Nat.lcm (9^9) (16^16)) k = 18^18

-- State the existence of valid k values
theorem exists_valid_k :
  ∃ k : ℕ, k > 0 ∧ is_valid_k k :=
sorry

-- State the upper bound of valid k values
theorem upper_bound_valid_k :
  ∀ k : ℕ, is_valid_k k → k ≤ 18^18 :=
sorry

end lcm_count_theorem_exists_valid_k_upper_bound_valid_k_l2764_276498


namespace point_in_first_quadrant_l2764_276401

/-- Given a complex equation, prove that the point is in the first quadrant -/
theorem point_in_first_quadrant (x y : ℝ) (h : x + y + (x - y) * Complex.I = 3 - Complex.I) :
  x > 0 ∧ y > 0 := by
  sorry

end point_in_first_quadrant_l2764_276401


namespace milan_bill_cost_l2764_276462

/-- The total cost of a long distance phone bill given the monthly fee, per-minute cost, and minutes used. -/
def total_cost (monthly_fee : ℚ) (per_minute_cost : ℚ) (minutes_used : ℕ) : ℚ :=
  monthly_fee + per_minute_cost * minutes_used

/-- Proof that Milan's long distance bill is $23.36 -/
theorem milan_bill_cost :
  let monthly_fee : ℚ := 2
  let per_minute_cost : ℚ := 12 / 100
  let minutes_used : ℕ := 178
  total_cost monthly_fee per_minute_cost minutes_used = 2336 / 100 := by
  sorry

#eval total_cost 2 (12/100) 178

end milan_bill_cost_l2764_276462


namespace added_balls_relationship_l2764_276433

/-- Proves the relationship between added white and black balls to maintain a specific probability -/
theorem added_balls_relationship (x y : ℕ) : 
  (x + 2 : ℚ) / (5 + x + y : ℚ) = 1/3 → y = 2*x + 1 := by
  sorry

end added_balls_relationship_l2764_276433


namespace no_scalene_equilateral_triangle_no_right_equilateral_triangle_l2764_276458

-- Define a triangle
structure Triangle where
  sides : Fin 3 → ℝ
  angles : Fin 3 → ℝ

-- Define properties of triangles
def Triangle.isScalene (t : Triangle) : Prop :=
  ∀ i j : Fin 3, i ≠ j → t.sides i ≠ t.sides j

def Triangle.isEquilateral (t : Triangle) : Prop :=
  ∀ i j : Fin 3, t.sides i = t.sides j

def Triangle.isRight (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i = 90

-- Theorem: A scalene equilateral triangle is impossible
theorem no_scalene_equilateral_triangle :
  ¬∃ t : Triangle, t.isScalene ∧ t.isEquilateral :=
sorry

-- Theorem: A right equilateral triangle is impossible
theorem no_right_equilateral_triangle :
  ¬∃ t : Triangle, t.isRight ∧ t.isEquilateral :=
sorry

end no_scalene_equilateral_triangle_no_right_equilateral_triangle_l2764_276458


namespace gcd_10010_20020_l2764_276418

theorem gcd_10010_20020 : Nat.gcd 10010 20020 = 10010 := by
  sorry

end gcd_10010_20020_l2764_276418


namespace infinite_solutions_diophantine_equation_l2764_276495

theorem infinite_solutions_diophantine_equation (n : ℤ) :
  ∃ (S : Set (ℕ × ℕ × ℕ)), Set.Infinite S ∧
    ∀ (x y z : ℕ), (x, y, z) ∈ S → (x^2 : ℤ) + y^2 - z^2 = n :=
sorry

end infinite_solutions_diophantine_equation_l2764_276495


namespace power_of_128_fourth_sevenths_l2764_276483

theorem power_of_128_fourth_sevenths (h : 128 = 2^7) : (128 : ℝ)^(4/7) = 16 := by
  sorry

end power_of_128_fourth_sevenths_l2764_276483
