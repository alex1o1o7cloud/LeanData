import Mathlib

namespace pool_capacity_is_12000_l742_74202

/-- Represents the capacity of a pool and its filling rates. -/
structure PoolSystem where
  capacity : ℝ
  bothValvesTime : ℝ
  firstValveTime : ℝ
  secondValveExtraRate : ℝ

/-- Theorem stating that under given conditions, the pool capacity is 12000 cubic meters. -/
theorem pool_capacity_is_12000 (p : PoolSystem)
  (h1 : p.bothValvesTime = 48)
  (h2 : p.firstValveTime = 120)
  (h3 : p.secondValveExtraRate = 50)
  (h4 : p.capacity / p.firstValveTime + (p.capacity / p.firstValveTime + p.secondValveExtraRate) = p.capacity / p.bothValvesTime) :
  p.capacity = 12000 := by
  sorry

#check pool_capacity_is_12000

end pool_capacity_is_12000_l742_74202


namespace money_distribution_l742_74270

theorem money_distribution (x : ℝ) (x_pos : x > 0) : 
  let total_money := 6*x + 5*x + 4*x + 3*x
  let ott_money := x + x + x + x
  ott_money / total_money = 2 / 9 := by
sorry


end money_distribution_l742_74270


namespace smallest_angle_sine_cosine_equality_l742_74262

theorem smallest_angle_sine_cosine_equality : 
  ∃ x : ℝ, x > 0 ∧ x < (2 * Real.pi / 360) * 11 ∧
    Real.sin (4 * x) * Real.sin (5 * x) = Real.cos (4 * x) * Real.cos (5 * x) ∧
    ∀ y : ℝ, 0 < y ∧ y < x → 
      Real.sin (4 * y) * Real.sin (5 * y) ≠ Real.cos (4 * y) * Real.cos (5 * y) ∧
    x = (Real.pi / 18) := by
  sorry

end smallest_angle_sine_cosine_equality_l742_74262


namespace julia_video_games_fraction_l742_74269

/-- Given the number of video games owned by Theresa, Julia, and Tory,
    prove that Julia has 1/3 as many video games as Tory. -/
theorem julia_video_games_fraction (theresa julia tory : ℕ) : 
  theresa = 3 * julia + 5 →
  tory = 6 →
  theresa = 11 →
  julia * 3 = tory := by
  sorry

end julia_video_games_fraction_l742_74269


namespace quadratic_function_properties_l742_74281

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- Define the theorem
theorem quadratic_function_properties
  (b c : ℝ)
  (h1 : f b c 2 = f b c (-2))
  (h2 : f b c 1 = 0) :
  (∀ x, f b c x = x^2 - 1) ∧
  (∀ m : ℝ, (∀ x ≥ (1/2 : ℝ), ∃ y, 4*m*(f b c y) + f b c (y-1) = 4-4*m) →
    -1/4 < m ∧ m ≤ 19/4) :=
by sorry

end quadratic_function_properties_l742_74281


namespace consecutive_integers_around_sqrt11_l742_74236

theorem consecutive_integers_around_sqrt11 (m n : ℤ) :
  (n = m + 1) →
  (m < Real.sqrt 11) →
  (Real.sqrt 11 < n) →
  m + n = 7 := by
sorry

end consecutive_integers_around_sqrt11_l742_74236


namespace inequality_system_solution_l742_74232

theorem inequality_system_solution (a : ℝ) :
  (∃ x : ℝ, (1 + x > a) ∧ (2 * x - 4 ≤ 0)) ↔ (a < 3) := by
  sorry

end inequality_system_solution_l742_74232


namespace handshakes_count_l742_74295

/-- Represents a social gathering with specific group interactions -/
structure SocialGathering where
  total_people : Nat
  group1_size : Nat
  subgroup_size : Nat
  group2_size : Nat
  outsiders : Nat

/-- Calculates the number of handshakes in a social gathering -/
def handshakes (sg : SocialGathering) : Nat :=
  sg.subgroup_size * (sg.group2_size + sg.outsiders) +
  (sg.group1_size - sg.subgroup_size) * sg.outsiders +
  sg.group2_size * sg.outsiders

/-- Theorem stating the number of handshakes in the specific social gathering -/
theorem handshakes_count :
  let sg : SocialGathering := {
    total_people := 36,
    group1_size := 25,
    subgroup_size := 15,
    group2_size := 6,
    outsiders := 5
  }
  handshakes sg = 245 := by sorry

end handshakes_count_l742_74295


namespace purely_imaginary_complex_number_l742_74283

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (((2 : ℂ) - a * Complex.I) / ((1 : ℂ) + Complex.I)).re = 0 →
  a = 2 :=
by sorry

end purely_imaginary_complex_number_l742_74283


namespace similar_triangle_perimeter_l742_74203

theorem similar_triangle_perimeter (a b c d e : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for smaller triangle
  (d/a)^2 + (e/b)^2 = 1 →  -- Similar triangles condition
  2*c = 30 →  -- Hypotenuse of larger triangle
  d + e + 30 = 72 := by
  sorry

end similar_triangle_perimeter_l742_74203


namespace dan_remaining_money_l742_74275

/-- Given an initial amount and a spent amount, calculate the remaining amount --/
def remaining_amount (initial : ℚ) (spent : ℚ) : ℚ :=
  initial - spent

/-- Proof that Dan has $1 left --/
theorem dan_remaining_money :
  let initial_amount : ℚ := 4
  let spent_amount : ℚ := 3
  remaining_amount initial_amount spent_amount = 1 := by
  sorry

end dan_remaining_money_l742_74275


namespace coin_count_l742_74274

theorem coin_count (total_value : ℕ) (nickel_value dime_value quarter_value : ℕ) :
  total_value = 360 →
  nickel_value = 5 →
  dime_value = 10 →
  quarter_value = 25 →
  ∃ (x : ℕ), x * (nickel_value + dime_value + quarter_value) = total_value ∧
              3 * x = 27 :=
by
  sorry

#check coin_count

end coin_count_l742_74274


namespace soap_bubble_radius_l742_74246

noncomputable def final_radius (α : ℝ) (r₀ : ℝ) (U : ℝ) (k : ℝ) : ℝ :=
  (U^2 * r₀^2 / (32 * k * Real.pi * α))^(1/3)

theorem soap_bubble_radius (α : ℝ) (r₀ : ℝ) (U : ℝ) (k : ℝ) 
  (h1 : α > 0) (h2 : r₀ > 0) (h3 : U ≠ 0) (h4 : k > 0) :
  ∃ (r : ℝ), r = final_radius α r₀ U k ∧ 
  r = (U^2 * r₀^2 / (32 * k * Real.pi * α))^(1/3) :=
by sorry

end soap_bubble_radius_l742_74246


namespace intersection_of_complex_equations_l742_74240

open Complex

theorem intersection_of_complex_equations (k : ℝ) : 
  (∃! z : ℂ, (Complex.abs (z - 3) = 3 * Complex.abs (z + 3)) ∧ (Complex.abs z = k)) ↔ k = 3 :=
by sorry

end intersection_of_complex_equations_l742_74240


namespace egg_grouping_l742_74220

theorem egg_grouping (total_eggs : ℕ) (group_size : ℕ) (h1 : total_eggs = 16) (h2 : group_size = 2) :
  total_eggs / group_size = 8 := by
  sorry

end egg_grouping_l742_74220


namespace intersection_of_A_and_B_l742_74205

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by sorry

end intersection_of_A_and_B_l742_74205


namespace minimum_students_l742_74229

theorem minimum_students (b g : ℕ) : 
  b > 0 → g > 0 → 
  (b / 2 : ℚ) = 2 * (2 * g / 3 : ℚ) → 
  b + g ≥ 11 := by
sorry

end minimum_students_l742_74229


namespace intersection_product_l742_74231

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 6*y + 9 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 6*y + 21 = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | circle1 p.1 p.2 ∧ circle2 p.1 p.2}

-- Theorem statement
theorem intersection_product : 
  ∀ p ∈ intersection_points, p.1 * p.2 = 12 := by sorry

end intersection_product_l742_74231


namespace students_without_vision_assistance_l742_74233

/-- Given a group of 40 students where 25% wear glasses and 40% wear contact lenses,
    prove that 14 students do not wear any vision assistance wear. -/
theorem students_without_vision_assistance (total_students : ℕ) (glasses_percent : ℚ) (contacts_percent : ℚ) :
  total_students = 40 →
  glasses_percent = 25 / 100 →
  contacts_percent = 40 / 100 →
  total_students - (glasses_percent * total_students + contacts_percent * total_students) = 14 := by
  sorry

end students_without_vision_assistance_l742_74233


namespace no_positive_integer_solutions_l742_74263

theorem no_positive_integer_solutions :
  ¬ ∃ (n : ℕ+) (p : ℕ), Prime p ∧ n.val^2 - 47*n.val + 660 = p := by
  sorry

end no_positive_integer_solutions_l742_74263


namespace correct_calculation_l742_74252

theorem correct_calculation (a : ℝ) : -2*a + (2*a - 1) = -1 := by
  sorry

end correct_calculation_l742_74252


namespace M_subset_P_l742_74221

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x > 1}

def P : Set ℝ := {x : ℝ | x^2 > 1}

theorem M_subset_P : M ⊆ P := by sorry

end M_subset_P_l742_74221


namespace distinct_triangles_in_3x3_grid_l742_74225

/-- The number of points in a row or column of the grid -/
def gridSize : Nat := 3

/-- The total number of points in the grid -/
def totalPoints : Nat := gridSize * gridSize

/-- The number of ways to choose 3 points from the total points -/
def totalCombinations : Nat := Nat.choose totalPoints 3

/-- The number of sets of collinear points in the grid -/
def collinearSets : Nat := 2 * gridSize + 2

/-- The number of distinct triangles in a 3x3 grid -/
def distinctTriangles : Nat := totalCombinations - collinearSets

theorem distinct_triangles_in_3x3_grid :
  distinctTriangles = 76 := by sorry

end distinct_triangles_in_3x3_grid_l742_74225


namespace existence_of_sequences_l742_74235

theorem existence_of_sequences : ∃ (a b : ℕ → ℝ), 
  (∀ i : ℕ, 3 * Real.pi / 2 ≤ a i ∧ a i ≤ b i) ∧
  (∀ i : ℕ, ∀ x : ℝ, 0 < x ∧ x < 1 → Real.cos (a i * x) - Real.cos (b i * x) ≥ -1 / i) := by
  sorry

end existence_of_sequences_l742_74235


namespace laptop_price_l742_74219

theorem laptop_price : ∃ (x : ℝ), 
  (0.855 * x - 50) = (0.88 * x - 20) - 30 ∧ x = 2400 := by
  sorry

end laptop_price_l742_74219


namespace derivative_zero_implies_x_equals_plus_minus_a_l742_74297

theorem derivative_zero_implies_x_equals_plus_minus_a (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := fun x ↦ (x^2 + a^2) / x
  let f' : ℝ → ℝ := fun x ↦ (x^2 - a^2) / x^2
  ∀ x₀ : ℝ, x₀ ≠ 0 → f' x₀ = 0 → x₀ = a ∨ x₀ = -a := by
  sorry

end derivative_zero_implies_x_equals_plus_minus_a_l742_74297


namespace prob_same_color_equal_one_l742_74268

/-- Procedure A: Choose one card from k cards with equal probability 1/k and replace it with a different color card. -/
def procedureA (k : ℕ) : Unit := sorry

/-- The probability of reaching a state where all cards are of the same color after n repetitions of procedure A. -/
def probSameColor (k n : ℕ) : ℝ := sorry

theorem prob_same_color_equal_one (k n : ℕ) (h1 : k > 0) (h2 : n > 0) (h3 : k % 2 = 0) :
  probSameColor k n = 1 := by sorry

end prob_same_color_equal_one_l742_74268


namespace same_grade_percentage_is_50_l742_74292

/-- Represents the number of students who got the same grade on both tests for each grade -/
structure GradeCount where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- Calculates the percentage of students who got the same grade on both tests -/
def sameGradePercentage (totalStudents : ℕ) (gradeCount : GradeCount) : ℚ :=
  (gradeCount.a + gradeCount.b + gradeCount.c + gradeCount.d : ℚ) / totalStudents * 100

/-- The main theorem stating that 50% of students received the same grade on both tests -/
theorem same_grade_percentage_is_50 :
  let totalStudents : ℕ := 40
  let gradeCount : GradeCount := { a := 3, b := 6, c := 7, d := 4 }
  sameGradePercentage totalStudents gradeCount = 50 := by
  sorry


end same_grade_percentage_is_50_l742_74292


namespace combined_rectangle_perimeter_l742_74278

/-- The perimeter of a rectangle formed by combining a square of side 8 cm
    with a rectangle of dimensions 8 cm x 4 cm is 48 cm. -/
theorem combined_rectangle_perimeter :
  let square_side : ℝ := 8
  let rect_length : ℝ := 8
  let rect_width : ℝ := 4
  let new_rect_length : ℝ := square_side + rect_length
  let new_rect_width : ℝ := square_side
  let perimeter : ℝ := 2 * (new_rect_length + new_rect_width)
  perimeter = 48 := by sorry

end combined_rectangle_perimeter_l742_74278


namespace intersection_of_A_and_B_l742_74253

def setA : Set ℤ := {x | |x| < 3}
def setB : Set ℤ := {x | |x| > 1}

theorem intersection_of_A_and_B :
  setA ∩ setB = {-2, 2} := by
  sorry

end intersection_of_A_and_B_l742_74253


namespace sanctuary_bird_pairs_l742_74254

/-- The number of endangered bird species in Tyler's sanctuary -/
def num_species : ℕ := 29

/-- The number of pairs of birds per species -/
def pairs_per_species : ℕ := 7

/-- The total number of pairs of birds in Tyler's sanctuary -/
def total_pairs : ℕ := num_species * pairs_per_species

theorem sanctuary_bird_pairs : total_pairs = 203 := by
  sorry

end sanctuary_bird_pairs_l742_74254


namespace quadratic_roots_sum_l742_74288

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, x^2 - 6*x + 11 = 23 ↔ x = a ∨ x = b) →
  a ≥ b →
  3*a + 2*b = 15 + Real.sqrt 21 := by
sorry

end quadratic_roots_sum_l742_74288


namespace intersection_of_A_and_B_l742_74206

def A : Set ℤ := {0, 2}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end intersection_of_A_and_B_l742_74206


namespace least_common_multiple_first_ten_l742_74247

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ i : ℕ, i > 0 ∧ i ≤ 10 → n % i = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (∀ i : ℕ, i > 0 ∧ i ≤ 10 → m % i = 0) → m ≥ n) ∧
  n = 2520 := by
  sorry

end least_common_multiple_first_ten_l742_74247


namespace moon_speed_in_km_per_hour_l742_74207

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.04

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Converts speed from kilometers per second to kilometers per hour -/
def km_per_sec_to_km_per_hour (speed_km_per_sec : ℝ) : ℝ :=
  speed_km_per_sec * (seconds_per_hour : ℝ)

theorem moon_speed_in_km_per_hour :
  km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3744 := by
  sorry

end moon_speed_in_km_per_hour_l742_74207


namespace area_circle_outside_square_l742_74226

/-- The area inside a circle of radius 1 but outside a square of side length 2, when both share the same center, is equal to π - 2. -/
theorem area_circle_outside_square :
  let circle_radius : ℝ := 1
  let square_side : ℝ := 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let square_area : ℝ := square_side ^ 2
  let area_difference : ℝ := circle_area - square_area
  area_difference = π - 2 := by sorry

end area_circle_outside_square_l742_74226


namespace equal_reciprocal_sum_l742_74213

theorem equal_reciprocal_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 2 + 1 / y) (h2 : y = 2 + 1 / x) : y = x := by
  sorry

end equal_reciprocal_sum_l742_74213


namespace inequality_proof_l742_74217

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((a*b*c + a*b*d + a*c*d + b*c*d) / 4) ^ (1/3)) := by
  sorry

end inequality_proof_l742_74217


namespace training_hours_per_day_l742_74264

/-- 
Given a person who trains for a constant number of hours per day over a period of time,
this theorem proves that if the total training period is 42 days and the total training time
is 210 hours, then the person trains for 5 hours every day.
-/
theorem training_hours_per_day 
  (total_days : ℕ) 
  (total_hours : ℕ) 
  (hours_per_day : ℕ) 
  (h1 : total_days = 42) 
  (h2 : total_hours = 210) 
  (h3 : total_hours = total_days * hours_per_day) : 
  hours_per_day = 5 := by
  sorry

end training_hours_per_day_l742_74264


namespace sum_ab_over_c_squared_plus_one_le_one_l742_74228

theorem sum_ab_over_c_squared_plus_one_le_one 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (sum_eq_two : a + b + c = 2) :
  (a * b) / (c^2 + 1) + (b * c) / (a^2 + 1) + (c * a) / (b^2 + 1) ≤ 1 := by
  sorry

end sum_ab_over_c_squared_plus_one_le_one_l742_74228


namespace boy_scouts_permission_slips_l742_74256

theorem boy_scouts_permission_slips 
  (total_permission : Real) 
  (boy_scouts_percentage : Real) 
  (girl_scouts_permission : Real) :
  total_permission = 0.7 →
  boy_scouts_percentage = 0.6 →
  girl_scouts_permission = 0.625 →
  (total_permission - ((1 - boy_scouts_percentage) * girl_scouts_permission)) / boy_scouts_percentage = 0.75 := by
sorry

end boy_scouts_permission_slips_l742_74256


namespace exists_n_ratio_f_g_eq_2012_l742_74209

/-- The number of divisors of n which are perfect squares -/
def f (n : ℕ+) : ℕ := sorry

/-- The number of divisors of n which are perfect cubes -/
def g (n : ℕ+) : ℕ := sorry

/-- There exists a positive integer n such that f(n) / g(n) = 2012 -/
theorem exists_n_ratio_f_g_eq_2012 : ∃ n : ℕ+, (f n : ℚ) / (g n : ℚ) = 2012 := by sorry

end exists_n_ratio_f_g_eq_2012_l742_74209


namespace product_of_solutions_abs_equation_l742_74222

theorem product_of_solutions_abs_equation : 
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3*(|y₁| - 2)) ∧ (|y₂| = 3*(|y₂| - 2)) ∧ (y₁ ≠ y₂) ∧ (y₁ * y₂ = -9) :=
sorry

end product_of_solutions_abs_equation_l742_74222


namespace first_investment_interest_rate_l742_74204

/-- Prove that the annual simple interest rate of the first investment is 8.5% --/
theorem first_investment_interest_rate 
  (total_income : ℝ) 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_rate : ℝ) 
  (h1 : total_income = 575)
  (h2 : total_investment = 8000)
  (h3 : first_investment = 3000)
  (h4 : second_rate = 0.064)
  (h5 : total_income = first_investment * x + (total_investment - first_investment) * second_rate) :
  x = 0.085 := by
sorry

end first_investment_interest_rate_l742_74204


namespace cat_age_proof_l742_74284

theorem cat_age_proof (cat_age rabbit_age dog_age : ℕ) : 
  rabbit_age = cat_age / 2 →
  dog_age = 3 * rabbit_age →
  dog_age = 12 →
  cat_age = 8 := by
sorry

end cat_age_proof_l742_74284


namespace arithmetic_sum_10_terms_l742_74238

def arithmetic_sum (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sum_10_terms : arithmetic_sum (-2) 7 10 = 295 := by
  sorry

end arithmetic_sum_10_terms_l742_74238


namespace complex_modulus_product_l742_74208

theorem complex_modulus_product : Complex.abs (5 - 3*Complex.I) * Complex.abs (5 + 3*Complex.I) = 34 := by
  sorry

end complex_modulus_product_l742_74208


namespace total_ladybugs_l742_74287

theorem total_ladybugs (num_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
  (h1 : num_leaves = 84) 
  (h2 : ladybugs_per_leaf = 139) : 
  num_leaves * ladybugs_per_leaf = 11676 := by
  sorry

end total_ladybugs_l742_74287


namespace quadratic_equation_result_l742_74244

theorem quadratic_equation_result (y : ℂ) : 
  3 * y^2 + 2 * y + 1 = 0 → (6 * y + 5)^2 = -7 + 12 * Complex.I * Real.sqrt 2 ∨ 
                              (6 * y + 5)^2 = -7 - 12 * Complex.I * Real.sqrt 2 := by
  sorry

end quadratic_equation_result_l742_74244


namespace cube_face_perimeter_l742_74250

-- Define the volume of the cube
def cube_volume : ℝ := 216

-- Define the function to calculate the side length of a cube given its volume
def side_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Define the function to calculate the perimeter of a square given its side length
def square_perimeter (side : ℝ) : ℝ := 4 * side

-- Theorem statement
theorem cube_face_perimeter :
  square_perimeter (side_length cube_volume) = 24 := by
  sorry

end cube_face_perimeter_l742_74250


namespace abs_diff_given_prod_and_sum_l742_74216

theorem abs_diff_given_prod_and_sum (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 7) : |a - b| = 5 := by
  sorry

end abs_diff_given_prod_and_sum_l742_74216


namespace restaurant_ratio_change_l742_74201

theorem restaurant_ratio_change (initial_cooks : ℕ) (initial_waiters : ℕ) 
  (additional_waiters : ℕ) :
  initial_cooks = 9 →
  initial_cooks * 10 = initial_waiters * 3 →
  additional_waiters = 12 →
  (initial_cooks : ℚ) / (initial_waiters + additional_waiters : ℚ) = 3 / 14 :=
by sorry

end restaurant_ratio_change_l742_74201


namespace fraction_evaluation_l742_74218

theorem fraction_evaluation (x : ℝ) (h : x = 6) : 3 / (2 - 3 / x) = 2 := by
  sorry

end fraction_evaluation_l742_74218


namespace dice_probability_theorem_l742_74282

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The total number of possible outcomes when rolling 6 dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (at least one pair but not a four-of-a-kind) -/
def favorableOutcomes : ℕ := 28800

/-- The probability of getting at least one pair but not a four-of-a-kind when rolling 6 dice -/
def probabilityPairNotFourOfAKind : ℚ := favorableOutcomes / totalOutcomes

theorem dice_probability_theorem : probabilityPairNotFourOfAKind = 25 / 81 := by
  sorry

end dice_probability_theorem_l742_74282


namespace johns_pictures_l742_74248

/-- The number of pictures John drew and colored -/
def num_pictures : ℕ := 10

/-- The time it takes John to draw one picture (in hours) -/
def drawing_time : ℝ := 2

/-- The time it takes John to color one picture (in hours) -/
def coloring_time : ℝ := drawing_time * 0.7

/-- The total time John spent on all pictures (in hours) -/
def total_time : ℝ := 34

theorem johns_pictures :
  (drawing_time + coloring_time) * num_pictures = total_time := by sorry

end johns_pictures_l742_74248


namespace solution_correct_l742_74277

def M : Matrix (Fin 2) (Fin 2) ℚ := !![5, 2; 4, 1]
def N : Matrix (Fin 2) (Fin 1) ℚ := !![5; 8]
def X : Matrix (Fin 2) (Fin 1) ℚ := !![11/3; -20/3]

theorem solution_correct : M * X = N := by sorry

end solution_correct_l742_74277


namespace sphere_volume_equal_surface_area_l742_74280

theorem sphere_volume_equal_surface_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
sorry

end sphere_volume_equal_surface_area_l742_74280


namespace mode_of_scores_l742_74249

def Scores := List Nat

def count (n : Nat) (scores : Scores) : Nat :=
  scores.filter (· = n) |>.length

def isMode (n : Nat) (scores : Scores) : Prop :=
  ∀ m, count n scores ≥ count m scores

theorem mode_of_scores (scores : Scores) 
  (h1 : scores.all (· ≤ 120))
  (h2 : count 91 scores = 5)
  (h3 : ∀ n, n ≠ 91 → count n scores ≤ 5) :
  isMode 91 scores :=
sorry

end mode_of_scores_l742_74249


namespace john_uber_earnings_l742_74261

/-- Calculates the total money made from Uber before considering depreciation -/
def total_money_before_depreciation (initial_car_value trade_in_value profit_after_depreciation : ℕ) : ℕ :=
  profit_after_depreciation + (initial_car_value - trade_in_value)

/-- Theorem stating that John's total money made from Uber before depreciation is $30,000 -/
theorem john_uber_earnings :
  let initial_car_value : ℕ := 18000
  let trade_in_value : ℕ := 6000
  let profit_after_depreciation : ℕ := 18000
  total_money_before_depreciation initial_car_value trade_in_value profit_after_depreciation = 30000 := by
  sorry

end john_uber_earnings_l742_74261


namespace triangle_perimeter_in_divided_square_l742_74239

/-- Given a square of side length z divided into a central rectangle and four congruent right-angled triangles,
    where the shorter side of the rectangle is x, the perimeter of one of the triangles is 3z/2. -/
theorem triangle_perimeter_in_divided_square (z x : ℝ) (hz : z > 0) (hx : 0 < x ∧ x < z) :
  let triangle_perimeter := (z - x) / 2 + (z + x) / 2 + z / 2
  triangle_perimeter = 3 * z / 2 :=
by sorry

end triangle_perimeter_in_divided_square_l742_74239


namespace square_of_triple_l742_74272

theorem square_of_triple (a : ℝ) : (3 * a)^2 = 9 * a^2 := by
  sorry

end square_of_triple_l742_74272


namespace vertex_D_coordinates_l742_74289

/-- A parallelogram with vertices A, B, C, and D in 2D space. -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The given parallelogram ABCD with specified coordinates for A, B, and C. -/
def givenParallelogram : Parallelogram where
  A := (0, 0)
  B := (1, 2)
  C := (3, 1)
  D := (2, -1)  -- We include D here, but will prove it's correct

/-- Theorem stating that the coordinates of vertex D in the given parallelogram are (2, -1). -/
theorem vertex_D_coordinates (p : Parallelogram) (h : p = givenParallelogram) :
  p.D = (2, -1) := by
  sorry

end vertex_D_coordinates_l742_74289


namespace vacation_cost_l742_74294

/-- 
If dividing a total cost among 5 people results in a per-person cost that is $120 more than 
dividing the same total cost among 8 people, then the total cost is $1600.
-/
theorem vacation_cost (total_cost : ℝ) : 
  (total_cost / 5 - total_cost / 8 = 120) → total_cost = 1600 := by
  sorry

end vacation_cost_l742_74294


namespace price_reduction_percentage_l742_74223

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x < 1 ∧ 
    initial_price * (1 - x)^2 = final_price ∧ 
    x = 1/5 := by
  sorry

end price_reduction_percentage_l742_74223


namespace fraction_problem_l742_74215

theorem fraction_problem (x : ℝ) : 
  (0.60 * x * 100 = 36) → x = 3/5 := by
  sorry

end fraction_problem_l742_74215


namespace zoo_animal_count_l742_74258

/-- The number of tiger enclosures in the zoo -/
def tiger_enclosures : ℕ := 4

/-- The number of zebra enclosures behind each tiger enclosure -/
def zebras_per_tiger : ℕ := 2

/-- The number of tigers in each tiger enclosure -/
def tigers_per_enclosure : ℕ := 4

/-- The number of zebras in each zebra enclosure -/
def zebras_per_enclosure : ℕ := 10

/-- The number of giraffes in each giraffe enclosure -/
def giraffes_per_enclosure : ℕ := 2

/-- The ratio of giraffe enclosures to zebra enclosures -/
def giraffe_to_zebra_ratio : ℕ := 3

/-- The total number of zebra enclosures in the zoo -/
def total_zebra_enclosures : ℕ := tiger_enclosures * zebras_per_tiger

/-- The total number of giraffe enclosures in the zoo -/
def total_giraffe_enclosures : ℕ := total_zebra_enclosures * giraffe_to_zebra_ratio

/-- The total number of animals in the zoo -/
def total_animals : ℕ := 
  tiger_enclosures * tigers_per_enclosure + 
  total_zebra_enclosures * zebras_per_enclosure + 
  total_giraffe_enclosures * giraffes_per_enclosure

theorem zoo_animal_count : total_animals = 144 := by
  sorry

end zoo_animal_count_l742_74258


namespace range_of_z_l742_74291

theorem range_of_z (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 1) :
  let z := 2 * x - 3 * y
  ∃ (a b : ℝ), a = -5 ∧ b = 4 ∧ ∀ w, w ∈ Set.Icc a b ↔ ∃ (x' y' : ℝ), 
    -1 ≤ x' ∧ x' ≤ 2 ∧ 0 ≤ y' ∧ y' ≤ 1 ∧ w = 2 * x' - 3 * y' :=
by sorry

end range_of_z_l742_74291


namespace f_minimum_and_inequality_l742_74211

def f (x : ℝ) := |2*x - 1| + |x - 3|

theorem f_minimum_and_inequality (x y : ℝ) :
  (∀ x, f x ≥ 5/2) ∧
  (∀ m, (∀ x y, f x > m * (|y + 1| - |y - 1|)) ↔ -5/4 < m ∧ m < 5/4) :=
by sorry

end f_minimum_and_inequality_l742_74211


namespace ellipse_min_area_l742_74259

/-- An ellipse containing two specific circles has a minimum area of π -/
theorem ellipse_min_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
    ((x - 2)^2 + y^2 ≤ 4 ∧ (x + 2)^2 + y^2 ≤ 4)) → 
  π * a * b ≥ π := by sorry

end ellipse_min_area_l742_74259


namespace jerrys_action_figures_l742_74266

theorem jerrys_action_figures :
  ∀ (initial_figures initial_books added_figures : ℕ),
    initial_figures = 2 →
    initial_books = 10 →
    initial_books = (initial_figures + added_figures) + 4 →
    added_figures = 4 :=
by
  sorry

end jerrys_action_figures_l742_74266


namespace original_number_problem_l742_74273

theorem original_number_problem : ∃ x : ℕ, x / 3 = 42 ∧ x = 126 := by
  sorry

end original_number_problem_l742_74273


namespace fourth_student_in_sample_l742_74267

/-- Represents a systematic sample from a class of students. -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_student : ℕ

/-- Checks if a student number is part of the systematic sample. -/
def is_in_sample (s : SystematicSample) (student : ℕ) : Prop :=
  ∃ k : ℕ, student = s.first_student + k * s.interval

/-- The main theorem to be proved. -/
theorem fourth_student_in_sample
  (s : SystematicSample)
  (h_class_size : s.class_size = 48)
  (h_sample_size : s.sample_size = 4)
  (h_interval : s.interval = s.class_size / s.sample_size)
  (h_6_in_sample : is_in_sample s 6)
  (h_30_in_sample : is_in_sample s 30)
  (h_42_in_sample : is_in_sample s 42)
  : is_in_sample s 18 :=
sorry

end fourth_student_in_sample_l742_74267


namespace system_solution_l742_74271

theorem system_solution :
  let f (x y : ℝ) := x * Real.sqrt (1 - y^2) = (Real.sqrt 3 + 1) / 4
  let g (x y : ℝ) := y * Real.sqrt (1 - x^2) = (Real.sqrt 3 - 1) / 4
  ∀ x y : ℝ, (f x y ∧ g x y) ↔ 
    ((x = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ y = Real.sqrt 2 / 2) ∨
     (x = Real.sqrt 2 / 2 ∧ y = (Real.sqrt 6 - Real.sqrt 2) / 4)) :=
by sorry

end system_solution_l742_74271


namespace digit_distribution_proof_l742_74210

theorem digit_distribution_proof (n : ℕ) 
  (h1 : n / 2 = n * (1 / 2 : ℚ))  -- 1/2 of all digits are 1
  (h2 : n / 5 = n * (1 / 5 : ℚ))  -- proportion of 2 and 5 are 1/5 each
  (h3 : n / 10 = n * (1 / 10 : ℚ))  -- proportion of other digits is 1/10
  (h4 : (1 / 2 : ℚ) + (1 / 5 : ℚ) + (1 / 5 : ℚ) + (1 / 10 : ℚ) = 1)  -- sum of all proportions is 1
  : n = 10 := by
  sorry

end digit_distribution_proof_l742_74210


namespace sum_and_ratio_implies_difference_l742_74299

theorem sum_and_ratio_implies_difference (x y : ℝ) 
  (sum_eq : x + y = 540)
  (ratio_eq : x / y = 4 / 5) :
  y - x = 60 := by
sorry

end sum_and_ratio_implies_difference_l742_74299


namespace tan_pi_minus_alpha_neg_two_l742_74243

theorem tan_pi_minus_alpha_neg_two (α : ℝ) (h : Real.tan (π - α) = -2) :
  (Real.cos (2 * π - α) + 2 * Real.cos ((3 * π) / 2 - α)) /
  (Real.sin (π - α) - Real.sin (-π / 2 - α)) = -1 := by
  sorry

end tan_pi_minus_alpha_neg_two_l742_74243


namespace intersection_complement_equality_l742_74245

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_equality : M ∩ (U \ N) = {0, 1} := by sorry

end intersection_complement_equality_l742_74245


namespace remaining_oranges_l742_74251

/-- The number of oranges Michaela needs to get full -/
def michaela_oranges : ℕ := 45

/-- The number of oranges Cassandra needs to get full -/
def cassandra_oranges : ℕ := 5 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 520

/-- The number of oranges remaining after Michaela and Cassandra have eaten until full -/
theorem remaining_oranges : total_oranges - (michaela_oranges + cassandra_oranges) = 250 := by
  sorry

end remaining_oranges_l742_74251


namespace bicycle_cost_price_l742_74276

/-- Proves that the initial cost price of a bicycle is 112.5 given specific profit margins and final selling price -/
theorem bicycle_cost_price 
  (profit_a profit_b final_price : ℝ) 
  (h1 : profit_a = 0.6) 
  (h2 : profit_b = 0.25) 
  (h3 : final_price = 225) : 
  ∃ (initial_price : ℝ), 
    initial_price * (1 + profit_a) * (1 + profit_b) = final_price ∧ 
    initial_price = 112.5 := by
  sorry

#check bicycle_cost_price

end bicycle_cost_price_l742_74276


namespace crossroads_four_roads_routes_l742_74286

/-- Represents a crossroads with a given number of roads -/
structure Crossroads :=
  (num_roads : ℕ)

/-- Calculates the number of possible driving routes at a crossroads -/
def driving_routes (c : Crossroads) : ℕ :=
  c.num_roads * (c.num_roads - 1)

/-- Theorem: At a crossroads with 4 roads, where vehicles are not allowed to turn back,
    the total number of possible driving routes is 12 -/
theorem crossroads_four_roads_routes :
  ∃ (c : Crossroads), c.num_roads = 4 ∧ driving_routes c = 12 :=
sorry

end crossroads_four_roads_routes_l742_74286


namespace line_through_midpoint_l742_74255

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := x - 3*y + 10 = 0
def l2 (x y : ℝ) : Prop := 2*x + y - 8 = 0

-- Define point P
def P : ℝ × ℝ := (0, 1)

-- Define the line l
def l (x y : ℝ) : Prop := x + 4*y - 4 = 0

-- Theorem statement
theorem line_through_midpoint (A B : ℝ × ℝ) :
  l A.1 A.2 →
  l B.1 B.2 →
  l1 A.1 A.2 →
  l2 B.1 B.2 →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∀ x y, l x y ↔ x + 4*y - 4 = 0 :=
sorry

end line_through_midpoint_l742_74255


namespace quadratic_properties_l742_74296

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots of the equation
def are_roots (a b c x₁ x₂ : ℝ) : Prop :=
  quadratic_equation a b c x₁ ∧ quadratic_equation a b c x₂

theorem quadratic_properties
  (a b c x₁ x₂ : ℝ) (ha : a ≠ 0) (h_roots : are_roots a b c x₁ x₂) :
  (¬ (∃ z : ℂ, x₁ = z ∧ x₂ = z ∧ z.im ≠ 0)) ∧
  (∀ x, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) ∧
  (x₁^2 * x₂ + x₁ * x₂^2 = -b * c / a^2) ∧
  (b^2 - 4*a*c < 0 → ∃ y : ℝ, x₁ - x₂ = Complex.I * y) :=
sorry

end quadratic_properties_l742_74296


namespace sum_first_25_odd_numbers_l742_74257

/-- The sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ :=
  n * n

/-- The 25th odd number -/
def last_odd_number (n : ℕ) : ℕ :=
  2 * n - 1

theorem sum_first_25_odd_numbers :
  sum_odd_numbers 25 = 625 :=
sorry

end sum_first_25_odd_numbers_l742_74257


namespace magnitude_of_z_l742_74214

theorem magnitude_of_z (z : ℂ) : z + Complex.I = 3 → Complex.abs z = Real.sqrt 10 := by
  sorry

end magnitude_of_z_l742_74214


namespace discount_markup_percentage_l742_74224

theorem discount_markup_percentage (original_price : ℝ) (discount_rate : ℝ) (h1 : discount_rate = 0.2) :
  let discounted_price := original_price * (1 - discount_rate)
  let markup_rate := (original_price - discounted_price) / discounted_price
  markup_rate = 0.25 := by
  sorry

end discount_markup_percentage_l742_74224


namespace order_cost_proof_l742_74237

def english_book_cost : ℝ := 7.50
def geography_book_cost : ℝ := 10.50
def num_books : ℕ := 35

def total_cost : ℝ := num_books * english_book_cost + num_books * geography_book_cost

theorem order_cost_proof : total_cost = 630 := by
  sorry

end order_cost_proof_l742_74237


namespace trigonometric_identities_l742_74279

theorem trigonometric_identities (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 = 1 - 2 * Real.cos A * Real.cos B * Real.cos C ∧
  (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 2 * Real.cos A * Real.cos B * Real.cos C :=
by sorry

end trigonometric_identities_l742_74279


namespace trajectory_equation_l742_74241

theorem trajectory_equation (x y : ℝ) (h : x ≠ 0) :
  (y + Real.sqrt 2) / x * (y - Real.sqrt 2) / x = -2 →
  y^2 / 2 + x^2 = 1 := by
  sorry

end trajectory_equation_l742_74241


namespace congruent_sufficient_not_necessary_for_equal_area_l742_74293

-- Define a triangle type
structure Triangle where
  -- You might define a triangle using its vertices or side lengths
  -- For simplicity, we'll just assume such a type exists
  mk :: (area : ℝ)

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop :=
  -- In reality, this would involve comparing all sides and angles
  -- For our purposes, we'll leave it as an abstract property
  sorry

-- Theorem statement
theorem congruent_sufficient_not_necessary_for_equal_area :
  (∀ t1 t2 : Triangle, congruent t1 t2 → t1.area = t2.area) ∧
  (∃ t1 t2 : Triangle, t1.area = t2.area ∧ ¬congruent t1 t2) :=
sorry

end congruent_sufficient_not_necessary_for_equal_area_l742_74293


namespace jane_work_days_l742_74298

theorem jane_work_days (john_rate : ℚ) (total_days : ℕ) (jane_stop_days : ℕ) :
  john_rate = 1/20 →
  total_days = 10 →
  jane_stop_days = 5 →
  ∃ jane_rate : ℚ,
    (5 * (john_rate + jane_rate) + 5 * john_rate = 1) ∧
    (jane_rate = 1/10) :=
by sorry

end jane_work_days_l742_74298


namespace joao_salary_height_l742_74285

/-- Conversion rate from real to cruzado -/
def real_to_cruzado : ℝ := 2750000000

/-- João's monthly salary in reais -/
def joao_salary : ℝ := 640

/-- Height of 100 cruzado notes in centimeters -/
def stack_height : ℝ := 1.5

/-- Number of cruzado notes in a stack -/
def notes_per_stack : ℝ := 100

/-- Conversion factor from centimeters to kilometers -/
def cm_to_km : ℝ := 100000

theorem joao_salary_height : 
  (joao_salary * real_to_cruzado / notes_per_stack * stack_height) / cm_to_km = 264000 := by
  sorry

end joao_salary_height_l742_74285


namespace pirate_treasure_l742_74212

theorem pirate_treasure (m : ℕ) (n : ℕ) (u : ℕ) : 
  (2/3 * (2/3 * (2/3 * (m - 1) - 1) - 1) = 3 * n) →
  (110 ≤ 81 * u + 25) →
  (81 * u + 25 ≤ 200) →
  (m = 187 ∧ 
   1 + (187 - 1) / 3 + 18 = 81 ∧
   1 + (187 - (1 + (187 - 1) / 3) - 1) / 3 + 18 = 60 ∧
   1 + (187 - (1 + (187 - 1) / 3) - (1 + (187 - (1 + (187 - 1) / 3) - 1) / 3) - 1) / 3 + 18 = 46) :=
by sorry

end pirate_treasure_l742_74212


namespace exponent_multiplication_l742_74227

theorem exponent_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by sorry

end exponent_multiplication_l742_74227


namespace abs_sum_nonzero_iff_either_nonzero_l742_74234

theorem abs_sum_nonzero_iff_either_nonzero (x y : ℝ) :
  (abs x + abs y ≠ 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end abs_sum_nonzero_iff_either_nonzero_l742_74234


namespace quadratic_sum_zero_l742_74290

-- Define the quadratic function P(x)
def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_sum_zero 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_Pa : P a b c a = 2021 * b * c)
  (h_Pb : P a b c b = 2021 * c * a)
  (h_Pc : P a b c c = 2021 * a * b) :
  a + 2021 * b + c = 0 := by
  sorry

end quadratic_sum_zero_l742_74290


namespace valerie_light_bulb_shortage_l742_74200

structure LightBulb where
  price : Float
  quantity : Nat

def small_bulb : LightBulb := { price := 8.75, quantity := 3 }
def medium_bulb : LightBulb := { price := 11.25, quantity := 4 }
def large_bulb : LightBulb := { price := 15.50, quantity := 3 }
def extra_small_bulb : LightBulb := { price := 6.10, quantity := 4 }

def budget : Float := 120.00

def total_cost : Float :=
  small_bulb.price * small_bulb.quantity.toFloat +
  medium_bulb.price * medium_bulb.quantity.toFloat +
  large_bulb.price * large_bulb.quantity.toFloat +
  extra_small_bulb.price * extra_small_bulb.quantity.toFloat

theorem valerie_light_bulb_shortage :
  total_cost - budget = 22.15 := by
  sorry


end valerie_light_bulb_shortage_l742_74200


namespace stream_speed_l742_74260

/-- Proves that the speed of the stream is 4 km/hr given the boat's speed in still water and its downstream travel details -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  boat_speed = 24 →
  distance = 140 →
  time = 5 →
  (boat_speed + (distance / time - boat_speed)) = 4 := by
sorry


end stream_speed_l742_74260


namespace smallest_shift_l742_74242

-- Define the function f with the given property
def f_periodic (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 12) = f x

-- Define the property for the shifted function
def shifted_f_equal (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f ((x - a) / 3) = f (x / 3)

-- Theorem statement
theorem smallest_shift (f : ℝ → ℝ) (h : f_periodic f) :
  (∃ a : ℝ, a > 0 ∧ shifted_f_equal f a ∧
    ∀ b : ℝ, b > 0 ∧ shifted_f_equal f b → a ≤ b) →
  ∃ a : ℝ, a = 36 ∧ shifted_f_equal f a ∧
    ∀ b : ℝ, b > 0 ∧ shifted_f_equal f b → a ≤ b :=
sorry

end smallest_shift_l742_74242


namespace cube_cutting_surface_area_l742_74230

/-- Calculates the total surface area of pieces after cutting a cube -/
def total_surface_area_after_cutting (edge_length : ℝ) (horizontal_cuts : ℕ) (vertical_cuts : ℕ) : ℝ :=
  let original_surface_area := 6 * edge_length^2
  let new_horizontal_faces := 2 * edge_length^2 * (2 * horizontal_cuts : ℝ)
  let new_vertical_faces := 2 * edge_length^2 * (2 * vertical_cuts : ℝ)
  original_surface_area + new_horizontal_faces + new_vertical_faces

/-- Theorem: The total surface area of pieces after cutting a 2-decimeter cube 4 times horizontally and 5 times vertically is 96 square decimeters -/
theorem cube_cutting_surface_area :
  total_surface_area_after_cutting 2 4 5 = 96 := by
  sorry

end cube_cutting_surface_area_l742_74230


namespace circle_equation_radius_l742_74265

theorem circle_equation_radius (k : ℝ) :
  (∃ (h : ℝ) (v : ℝ),
    ∀ (x y : ℝ),
      x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x - h)^2 + (y - v)^2 = 10^2) ↔
  k = 35 := by
sorry

end circle_equation_radius_l742_74265
