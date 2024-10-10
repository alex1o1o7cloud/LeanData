import Mathlib

namespace min_distance_complex_circles_l1611_161126

theorem min_distance_complex_circles (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 4*I) = 2)
  (hw : Complex.abs (w - 5 - 6*I) = 4) :
  ∃ (m : ℝ), m = Real.sqrt 149 - 6 ∧ ∀ (z' w' : ℂ), 
    Complex.abs (z' + 2 + 4*I) = 2 → 
    Complex.abs (w' - 5 - 6*I) = 4 → 
    Complex.abs (z' - w') ≥ m :=
by sorry

end min_distance_complex_circles_l1611_161126


namespace banana_bread_recipe_l1611_161122

/-- Banana bread recipe problem -/
theorem banana_bread_recipe 
  (bananas_per_mush : ℕ) 
  (total_bananas : ℕ) 
  (total_flour : ℕ) 
  (h1 : bananas_per_mush = 4)
  (h2 : total_bananas = 20)
  (h3 : total_flour = 15) :
  (total_flour : ℚ) / ((total_bananas : ℚ) / (bananas_per_mush : ℚ)) = 3 := by
  sorry

end banana_bread_recipe_l1611_161122


namespace imaginary_part_of_z_l1611_161168

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = 5) : z.im = 4/5 := by
  sorry

end imaginary_part_of_z_l1611_161168


namespace root_equation_value_l1611_161101

theorem root_equation_value (b c : ℝ) : 
  (2 : ℝ)^2 - b * 2 + c = 0 → 4 * b - 2 * c + 1 = 9 := by
  sorry

end root_equation_value_l1611_161101


namespace interior_triangle_area_l1611_161158

/-- Given three squares with areas 36, 64, and 100, where the largest square is diagonal to the other two squares, the area of the interior triangle is 24. -/
theorem interior_triangle_area (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100)
  (h_diagonal : c = max a b) : (1/2) * a * b = 24 := by
  sorry

end interior_triangle_area_l1611_161158


namespace exists_t_shape_l1611_161146

/-- Represents a grid of squares -/
structure Grid :=
  (size : ℕ)
  (removed : ℕ)

/-- Function that measures the connectivity of the grid -/
def f (g : Grid) : ℤ :=
  2 * g.size^2 - 4 * g.size - 10 * g.removed

/-- Theorem stating that after removing 1950 rectangles, 
    there always exists a square with at least three adjacent squares -/
theorem exists_t_shape (g : Grid) 
  (h1 : g.size = 100) 
  (h2 : g.removed = 1950) : 
  ∃ (square : Unit), f g > 0 :=
sorry

end exists_t_shape_l1611_161146


namespace y_min_max_sum_l1611_161179

theorem y_min_max_sum (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ y', (∃ x' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11) → m ≤ y' ∧ y' ≤ M) ∧
  m + M = 8/3 :=
sorry

end y_min_max_sum_l1611_161179


namespace monochromatic_rectangle_exists_l1611_161172

def Strip (n : ℤ) := {p : ℝ × ℝ | n ≤ p.1 ∧ p.1 < n + 1}

def ColoredStrip (n : ℤ) := Strip n → Bool

structure ColoredPlane :=
  (coloring : ℤ → Bool)

def hasMonochromaticRectangle (cp : ColoredPlane) (a b : ℕ) : Prop :=
  ∃ (x y : ℝ), 
    cp.coloring ⌊x⌋ = cp.coloring ⌊x + a⌋ ∧
    cp.coloring ⌊x⌋ = cp.coloring ⌊y⌋ ∧
    cp.coloring ⌊x⌋ = cp.coloring ⌊y + b⌋

theorem monochromatic_rectangle_exists (a b : ℕ) (h : a ≠ b) :
  ∀ cp : ColoredPlane, hasMonochromaticRectangle cp a b :=
sorry

end monochromatic_rectangle_exists_l1611_161172


namespace part_one_part_two_l1611_161165

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | x^2 - a*x + a^2 - 19 = 0}
def B := {x : ℝ | x^2 - 5*x + 6 = 0}
def C := {x : ℝ | x^2 + 2*x - 8 = 0}

-- Theorem for part (1)
theorem part_one (a : ℝ) : A a ∩ B = A a ∪ B → a = 5 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (∅ ⊂ A a ∩ B) ∧ (A a ∩ C = ∅) → a = -2 := by sorry

end part_one_part_two_l1611_161165


namespace remainder_theorem_application_l1611_161187

theorem remainder_theorem_application (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^6 + E * x^4 + F * x^2 + 6
  (q 2 = 16) → (q (-2) = 16) := by
sorry

end remainder_theorem_application_l1611_161187


namespace compare_negative_two_and_three_l1611_161142

theorem compare_negative_two_and_three : -2 > -3 := by
  sorry

end compare_negative_two_and_three_l1611_161142


namespace production_rates_and_minimum_machines_l1611_161151

/-- Represents the production rate of machine A in kg per hour -/
def machine_a_rate : ℝ := 60

/-- Represents the production rate of machine B in kg per hour -/
def machine_b_rate : ℝ := 50

/-- The difference in production rate between machine A and B -/
def rate_difference : ℝ := 10

/-- The total number of machines used -/
def total_machines : ℕ := 18

/-- The minimum required production in kg per hour -/
def min_production : ℝ := 1000

theorem production_rates_and_minimum_machines :
  (machine_a_rate = machine_b_rate + rate_difference) ∧
  (600 / machine_a_rate = 500 / machine_b_rate) ∧
  (∃ (m : ℕ), m ≤ total_machines ∧ 
    machine_a_rate * m + machine_b_rate * (total_machines - m) ≥ min_production ∧
    ∀ (n : ℕ), n < m → 
      machine_a_rate * n + machine_b_rate * (total_machines - n) < min_production) :=
by sorry

end production_rates_and_minimum_machines_l1611_161151


namespace marble_probability_l1611_161167

/-- The probability of drawing a red, blue, or green marble from a bag -/
theorem marble_probability (red blue green yellow : ℕ) : 
  red = 5 → blue = 4 → green = 3 → yellow = 6 →
  (red + blue + green : ℚ) / (red + blue + green + yellow) = 2/3 := by
  sorry

end marble_probability_l1611_161167


namespace max_soap_boxes_in_carton_l1611_161127

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end max_soap_boxes_in_carton_l1611_161127


namespace player_A_win_probability_l1611_161199

/-- The probability of winning a single game for either player -/
def win_prob : ℚ := 1/2

/-- The number of games player A needs to win to become the final winner -/
def games_needed_A : ℕ := 2

/-- The number of games player B needs to win to become the final winner -/
def games_needed_B : ℕ := 3

/-- The probability of player A becoming the final winner -/
def prob_A_wins : ℚ := 11/16

theorem player_A_win_probability :
  prob_A_wins = 11/16 := by sorry

end player_A_win_probability_l1611_161199


namespace min_cups_to_fill_container_l1611_161164

def container_capacity : ℝ := 640
def cup_capacity : ℝ := 120

theorem min_cups_to_fill_container : 
  ∃ n : ℕ, (n : ℝ) * cup_capacity ≥ container_capacity ∧ 
  ∀ m : ℕ, (m : ℝ) * cup_capacity ≥ container_capacity → n ≤ m ∧ 
  n = 6 :=
sorry

end min_cups_to_fill_container_l1611_161164


namespace total_students_is_63_l1611_161185

/-- The number of tables in the classroom -/
def num_tables : ℕ := 6

/-- The number of students currently sitting at each table -/
def students_per_table : ℕ := 3

/-- The number of girls who went to the bathroom -/
def girls_in_bathroom : ℕ := 4

/-- The number of students in new group 1 -/
def new_group1 : ℕ := 4

/-- The number of students in new group 2 -/
def new_group2 : ℕ := 5

/-- The number of students in new group 3 -/
def new_group3 : ℕ := 6

/-- The number of foreign exchange students from Germany -/
def german_students : ℕ := 2

/-- The number of foreign exchange students from France -/
def french_students : ℕ := 4

/-- The number of foreign exchange students from Norway -/
def norwegian_students : ℕ := 3

/-- The number of foreign exchange students from Italy -/
def italian_students : ℕ := 1

/-- The total number of students supposed to be in the class -/
def total_students : ℕ := 
  num_tables * students_per_table + 
  girls_in_bathroom + 
  4 * girls_in_bathroom + 
  new_group1 + new_group2 + new_group3 + 
  german_students + french_students + norwegian_students + italian_students

theorem total_students_is_63 : total_students = 63 := by
  sorry

end total_students_is_63_l1611_161185


namespace lcm_12_18_l1611_161133

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l1611_161133


namespace hyperbola_equation_l1611_161156

/-- The equation of a hyperbola with given properties -/
theorem hyperbola_equation (m n : ℝ) (h : m < 0) :
  (∀ x y : ℝ, x^2 / m + y^2 / n = 1) →  -- Given hyperbola equation
  (n = 1) →                            -- Derived from eccentricity = 2 and a = 1
  (m = -3) →                           -- Derived from b^2 = 3
  (∀ x y : ℝ, y^2 - x^2 / 3 = 1) :=    -- Equation to prove
by sorry

end hyperbola_equation_l1611_161156


namespace simplify_expression_range_of_values_find_values_l1611_161169

-- Question 1
theorem simplify_expression (a : ℝ) (h : 3 ≤ a ∧ a ≤ 7) :
  Real.sqrt ((3 - a)^2) + Real.sqrt ((a - 7)^2) = 4 :=
sorry

-- Question 2
theorem range_of_values (a : ℝ) :
  Real.sqrt ((a - 1)^2) + Real.sqrt ((a - 6)^2) = 5 ↔ 1 ≤ a ∧ a ≤ 6 :=
sorry

-- Question 3
theorem find_values (a : ℝ) :
  Real.sqrt ((a + 1)^2) + Real.sqrt ((a - 3)^2) = 6 ↔ a = -2 ∨ a = 4 :=
sorry

end simplify_expression_range_of_values_find_values_l1611_161169


namespace vessel_width_proof_l1611_161178

/-- The width of a rectangular vessel's base when a cube is immersed in it -/
theorem vessel_width_proof (cube_edge : ℝ) (vessel_length : ℝ) (water_rise : ℝ) 
  (h_cube_edge : cube_edge = 16)
  (h_vessel_length : vessel_length = 20)
  (h_water_rise : water_rise = 13.653333333333334) : 
  (cube_edge ^ 3) / (vessel_length * water_rise) = 15 := by
  sorry

end vessel_width_proof_l1611_161178


namespace three_numbers_sum_l1611_161132

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ordered numbers
  b = 10 ∧  -- Median is 10
  (a + b + c) / 3 = a + 15 ∧  -- Mean is 15 more than least
  (a + b + c) / 3 = c - 20  -- Mean is 20 less than greatest
  → a + b + c = 45 := by
sorry

end three_numbers_sum_l1611_161132


namespace angleBisectorRatioNotDeterminesShape_twoAnglesAndSideDeterminesShape_angleBisectorRatiosDetermineShape_sideLengthRatiosDetermineShape_threeAnglesDetermineShape_l1611_161188

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of an angle bisector to its corresponding opposite side --/
def angleBisectorToOppositeSideRatio (t : Triangle) : ℝ := sorry

/-- Determines if two triangles have the same shape (are similar) --/
def sameShape (t1 t2 : Triangle) : Prop := sorry

/-- The theorem stating that the ratio of an angle bisector to its corresponding opposite side
    does not uniquely determine the shape of a triangle --/
theorem angleBisectorRatioNotDeterminesShape :
  ∃ t1 t2 : Triangle, 
    angleBisectorToOppositeSideRatio t1 = angleBisectorToOppositeSideRatio t2 ∧
    ¬ sameShape t1 t2 := by sorry

/-- The theorem stating that the ratio of two angles and the included side
    uniquely determines the shape of a triangle --/
theorem twoAnglesAndSideDeterminesShape (α β : ℝ) (s : ℝ) :
  ∀ t1 t2 : Triangle,
    (α = sorry) ∧ (β = sorry) ∧ (s = sorry) →
    sameShape t1 t2 := by sorry

/-- The theorem stating that the ratios of the three angle bisectors
    uniquely determine the shape of a triangle --/
theorem angleBisectorRatiosDetermineShape (r1 r2 r3 : ℝ) :
  ∀ t1 t2 : Triangle,
    (r1 = sorry) ∧ (r2 = sorry) ∧ (r3 = sorry) →
    sameShape t1 t2 := by sorry

/-- The theorem stating that the ratios of the three side lengths
    uniquely determine the shape of a triangle --/
theorem sideLengthRatiosDetermineShape (r1 r2 r3 : ℝ) :
  ∀ t1 t2 : Triangle,
    (r1 = sorry) ∧ (r2 = sorry) ∧ (r3 = sorry) →
    sameShape t1 t2 := by sorry

/-- The theorem stating that three angles
    uniquely determine the shape of a triangle --/
theorem threeAnglesDetermineShape (α β γ : ℝ) :
  ∀ t1 t2 : Triangle,
    (α = sorry) ∧ (β = sorry) ∧ (γ = sorry) →
    sameShape t1 t2 := by sorry

end angleBisectorRatioNotDeterminesShape_twoAnglesAndSideDeterminesShape_angleBisectorRatiosDetermineShape_sideLengthRatiosDetermineShape_threeAnglesDetermineShape_l1611_161188


namespace rational_numbers_closed_l1611_161190

-- Define the set of rational numbers
def RationalNumbers : Set ℚ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

-- State the theorem
theorem rational_numbers_closed :
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → (a + b) ∈ RationalNumbers) ∧
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → (a - b) ∈ RationalNumbers) ∧
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → (a * b) ∈ RationalNumbers) ∧
  (∀ (a b : ℚ), a ∈ RationalNumbers → b ∈ RationalNumbers → b ≠ 0 → (a / b) ∈ RationalNumbers) :=
by sorry

end rational_numbers_closed_l1611_161190


namespace all_positive_rationals_in_X_l1611_161138

theorem all_positive_rationals_in_X (X : Set ℚ) 
  (h1 : ∀ x : ℚ, 2021 ≤ x ∧ x ≤ 2022 → x ∈ X) 
  (h2 : ∀ x y : ℚ, x ∈ X → y ∈ X → (x / y) ∈ X) :
  ∀ q : ℚ, 0 < q → q ∈ X := by
  sorry

end all_positive_rationals_in_X_l1611_161138


namespace quadratic_roots_real_and_equal_l1611_161106

theorem quadratic_roots_real_and_equal : ∃ x : ℝ, 
  x^2 - 4*x*Real.sqrt 5 + 20 = 0 ∧ 
  (∀ y : ℝ, y^2 - 4*y*Real.sqrt 5 + 20 = 0 → y = x) :=
by sorry

end quadratic_roots_real_and_equal_l1611_161106


namespace largest_three_digit_divisible_by_6_5_8_9_l1611_161173

theorem largest_three_digit_divisible_by_6_5_8_9 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 6 ∣ n ∧ 5 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≤ 720 :=
by sorry

end largest_three_digit_divisible_by_6_5_8_9_l1611_161173


namespace scientific_notation_86000000_l1611_161162

theorem scientific_notation_86000000 : 
  86000000 = 8.6 * (10 : ℝ)^7 := by sorry

end scientific_notation_86000000_l1611_161162


namespace cubic_function_property_l1611_161140

/-- Given a cubic function f(x) = ax³ + bx - 4 where a and b are constants,
    if f(-2) = 2, then f(2) = -10 -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^3 + b * x - 4)
    (h2 : f (-2) = 2) : 
  f 2 = -10 := by
  sorry

end cubic_function_property_l1611_161140


namespace coffee_package_size_l1611_161192

theorem coffee_package_size (total_coffee : ℝ) (known_size : ℝ) (extra_known : ℕ) (unknown_count : ℕ) :
  total_coffee = 85 ∧ 
  known_size = 5 ∧ 
  extra_known = 2 ∧ 
  unknown_count = 5 → 
  ∃ (unknown_size : ℝ), 
    unknown_size * unknown_count + known_size * (unknown_count + extra_known) = total_coffee ∧ 
    unknown_size = 10 := by
  sorry

end coffee_package_size_l1611_161192


namespace possible_values_of_a_l1611_161141

def A : Set ℝ := {x | x^2 ≠ 1}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem possible_values_of_a (a : ℝ) (h : A ∪ B a = A) : a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end possible_values_of_a_l1611_161141


namespace part_to_whole_ratio_l1611_161137

theorem part_to_whole_ratio (N : ℝ) (P : ℝ) : 
  (1 / 4 : ℝ) * P = 10 →
  (40 / 100 : ℝ) * N = 120 →
  P / ((2 / 5 : ℝ) * N) = 1 / 3 := by
sorry

end part_to_whole_ratio_l1611_161137


namespace dollar_cube_difference_l1611_161107

-- Define the $ operation for real numbers
def dollar (a b : ℝ) : ℝ := (a - b)^3

-- Theorem statement
theorem dollar_cube_difference (x y : ℝ) :
  dollar ((x - y)^3) ((y - x)^3) = -8 * (y - x)^9 := by
  sorry

end dollar_cube_difference_l1611_161107


namespace power_function_through_point_l1611_161115

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the condition that f passes through (9, 3)
def passesThroughPoint (f : ℝ → ℝ) : Prop :=
  f 9 = 3

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) (h2 : passesThroughPoint f) : f 25 = 5 := by
  sorry

end power_function_through_point_l1611_161115


namespace karthiks_weight_upper_bound_l1611_161183

-- Define the variables
def lower_bound : ℝ := 55
def upper_bound : ℝ := 58
def average_weight : ℝ := 56.5

-- Define the theorem
theorem karthiks_weight_upper_bound (X : ℝ) 
  (h1 : X > 50)  -- Karthik's brother's lower bound
  (h2 : X ≤ 62)  -- Karthik's upper bound
  (h3 : X ≤ 58)  -- Karthik's father's upper bound
  (h4 : (lower_bound + X) / 2 = average_weight)  -- Average condition
  : X = upper_bound := by
  sorry

end karthiks_weight_upper_bound_l1611_161183


namespace triangle_focus_property_l1611_161150

/-- Given a triangle ABC with vertices corresponding to complex numbers z₁, z₂, and z₃,
    and a point F corresponding to complex number z, prove that:
    (z - z₁)(z - z₂) + (z - z₂)(z - z₃) + (z - z₃)(z - z₁) = 0 -/
theorem triangle_focus_property (z z₁ z₂ z₃ : ℂ) : 
  (z - z₁) * (z - z₂) + (z - z₂) * (z - z₃) + (z - z₃) * (z - z₁) = 0 := by
  sorry

end triangle_focus_property_l1611_161150


namespace at_least_one_not_greater_than_one_l1611_161117

theorem at_least_one_not_greater_than_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b ≤ 1) ∨ (b / c ≤ 1) ∨ (c / a ≤ 1) := by
  sorry

end at_least_one_not_greater_than_one_l1611_161117


namespace sum_abc_equals_13_l1611_161160

theorem sum_abc_equals_13 (a b c : ℕ+) 
  (h : (a : ℝ)^2 + (b : ℝ)^2 + (c : ℝ)^2 + 43 ≤ (a : ℝ) * (b : ℝ) + 9 * (b : ℝ) + 8 * (c : ℝ)) :
  (a : ℕ) + b + c = 13 := by
sorry

end sum_abc_equals_13_l1611_161160


namespace distance_calculation_l1611_161119

theorem distance_calculation (A B C D : ℝ) 
  (h1 : A = 350)
  (h2 : A + B = 600)
  (h3 : A + B + C + D = 1500)
  (h4 : D = 275) :
  C = 625 ∧ B + C = 875 := by
  sorry

end distance_calculation_l1611_161119


namespace solve_equation_l1611_161152

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := by
  sorry

end solve_equation_l1611_161152


namespace hydrogen_atoms_in_compound_l1611_161177

def atomic_weight_Al : ℝ := 27
def atomic_weight_O : ℝ := 16
def atomic_weight_H : ℝ := 1

def num_Al : ℕ := 1
def num_O : ℕ := 3

def molecular_weight : ℝ := 78

theorem hydrogen_atoms_in_compound :
  ∃ (num_H : ℕ), 
    (num_Al : ℝ) * atomic_weight_Al + 
    (num_O : ℝ) * atomic_weight_O + 
    (num_H : ℝ) * atomic_weight_H = molecular_weight ∧
    num_H = 3 := by sorry

end hydrogen_atoms_in_compound_l1611_161177


namespace correct_sample_ids_l1611_161100

/-- A function that generates the sample IDs based on the given conditions -/
def generateSampleIDs (populationSize : Nat) (sampleSize : Nat) : List Nat :=
  (List.range sampleSize).map (fun i => 6 * i + 3)

/-- The theorem stating that the generated sample IDs match the expected result -/
theorem correct_sample_ids :
  generateSampleIDs 60 10 = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57] := by
  sorry

#eval generateSampleIDs 60 10

end correct_sample_ids_l1611_161100


namespace car_speed_l1611_161129

/-- Given a car that travels 495 km in 5 hours, its speed is 99 km/h. -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 495 ∧ time = 5 ∧ speed = distance / time → speed = 99 :=
by sorry

end car_speed_l1611_161129


namespace cannot_afford_both_phones_l1611_161105

/-- Represents the financial situation of Alexander and Natalia --/
structure FinancialSituation where
  alexander_salary : ℕ
  natalia_salary : ℕ
  utilities_expenses : ℕ
  loan_expenses : ℕ
  cultural_expenses : ℕ
  vacation_savings : ℕ
  dining_expenses : ℕ
  phone_a_cost : ℕ
  phone_b_cost : ℕ

/-- Theorem stating that Alexander and Natalia cannot afford both phones --/
theorem cannot_afford_both_phones (fs : FinancialSituation) 
  (h1 : fs.alexander_salary = 125000)
  (h2 : fs.natalia_salary = 61000)
  (h3 : fs.utilities_expenses = 17000)
  (h4 : fs.loan_expenses = 15000)
  (h5 : fs.cultural_expenses = 7000)
  (h6 : fs.vacation_savings = 20000)
  (h7 : fs.dining_expenses = 60000)
  (h8 : fs.phone_a_cost = 57000)
  (h9 : fs.phone_b_cost = 37000) :
  fs.alexander_salary + fs.natalia_salary - 
  (fs.utilities_expenses + fs.loan_expenses + fs.cultural_expenses + 
   fs.vacation_savings + fs.dining_expenses) < 
  fs.phone_a_cost + fs.phone_b_cost :=
by sorry

end cannot_afford_both_phones_l1611_161105


namespace quadratic_product_equals_quadratic_l1611_161118

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial (a b : ℤ) : ℤ → ℤ := fun x ↦ x^2 + a * x + b

theorem quadratic_product_equals_quadratic (a b n : ℤ) :
  ∃ M : ℤ, (QuadraticPolynomial a b n) * (QuadraticPolynomial a b (n + 1)) =
    QuadraticPolynomial a b M := by
  sorry

end quadratic_product_equals_quadratic_l1611_161118


namespace movie_theater_ticket_sales_l1611_161153

/-- Theorem: Movie Theater Ticket Sales
Given the prices and quantities of different types of movie tickets,
prove that the number of evening tickets sold is 300. -/
theorem movie_theater_ticket_sales
  (matinee_price : ℕ) (evening_price : ℕ) (threeD_price : ℕ)
  (matinee_quantity : ℕ) (threeD_quantity : ℕ)
  (total_revenue : ℕ) :
  matinee_price = 5 →
  evening_price = 12 →
  threeD_price = 20 →
  matinee_quantity = 200 →
  threeD_quantity = 100 →
  total_revenue = 6600 →
  ∃ evening_quantity : ℕ,
    evening_quantity = 300 ∧
    total_revenue = matinee_price * matinee_quantity +
                    evening_price * evening_quantity +
                    threeD_price * threeD_quantity :=
by sorry

end movie_theater_ticket_sales_l1611_161153


namespace M_intersect_N_equals_zero_l1611_161128

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | |x - 1| < 2}

theorem M_intersect_N_equals_zero : M ∩ N = {0} := by
  sorry

end M_intersect_N_equals_zero_l1611_161128


namespace solution_set_for_negative_two_minimum_value_for_one_range_of_m_l1611_161134

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2*x - 1

-- Part 1
theorem solution_set_for_negative_two (x : ℝ) :
  f (-2) x ≤ 0 ↔ x ≥ 1 :=
sorry

-- Part 2
theorem minimum_value_for_one (x : ℝ) :
  f 1 x + |x + 2| ≥ 0 :=
sorry

-- Range of m
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f 1 x + |x + 2| ≤ m) ↔ m ≥ 0 :=
sorry

end solution_set_for_negative_two_minimum_value_for_one_range_of_m_l1611_161134


namespace circle_radius_l1611_161163

theorem circle_radius (x y : ℝ) (h : x + y = 150 * Real.pi) : 
  ∃ (r : ℝ), r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = Real.sqrt 151 - 1 := by
  sorry

end circle_radius_l1611_161163


namespace invalid_votes_percentage_l1611_161197

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (vote_difference_percentage : ℝ)
  (candidate_b_votes : ℕ)
  (h1 : total_votes = 6720)
  (h2 : vote_difference_percentage = 0.15)
  (h3 : candidate_b_votes = 2184) :
  (total_votes - (2 * candidate_b_votes + vote_difference_percentage * total_votes)) / total_votes = 0.2 :=
by sorry

end invalid_votes_percentage_l1611_161197


namespace removed_terms_product_l1611_161155

theorem removed_terms_product (s : Finset ℚ) : 
  s ⊆ {1/2, 1/4, 1/6, 1/8, 1/10, 1/12} →
  s.sum id = 1 →
  (({1/2, 1/4, 1/6, 1/8, 1/10, 1/12} \ s).prod id) = 1/80 := by
  sorry

end removed_terms_product_l1611_161155


namespace rectangle_max_area_l1611_161180

theorem rectangle_max_area (l w : ℝ) : 
  l + w = 30 →  -- Perimeter condition (half of 60)
  l - w = 10 →  -- Difference between length and width
  l * w ≤ 200   -- Maximum area
  := by sorry

end rectangle_max_area_l1611_161180


namespace quadratic_inequality_solution_range_l1611_161176

theorem quadratic_inequality_solution_range (q : ℝ) : 
  (q > 0) → 
  (∃ x : ℝ, x^2 - 8*x + q < 0) ↔ 
  (q > 0 ∧ q < 16) :=
sorry

end quadratic_inequality_solution_range_l1611_161176


namespace arithmetic_sequence_common_difference_l1611_161112

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem: The common difference of an arithmetic sequence is 1, 
    given S_3 = 6 and a_1 = 1 -/
theorem arithmetic_sequence_common_difference :
  ∀ d : ℚ, S 3 1 d = 6 → d = 1 := by
  sorry

end arithmetic_sequence_common_difference_l1611_161112


namespace square_division_l1611_161144

theorem square_division (original_side : ℝ) (n : ℕ) (smaller_side : ℝ) : 
  original_side = 12 →
  n = 4 →
  smaller_side^2 * n = original_side^2 →
  smaller_side = 6 := by
sorry

end square_division_l1611_161144


namespace P_not_subset_Q_l1611_161171

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | |x| > 0}

-- Statement to prove
theorem P_not_subset_Q : ¬(P ⊆ Q) := by
  sorry

end P_not_subset_Q_l1611_161171


namespace power_of_81_four_thirds_l1611_161186

theorem power_of_81_four_thirds :
  (81 : ℝ) ^ (4/3) = 243 * (3 : ℝ) ^ (1/3) := by sorry

end power_of_81_four_thirds_l1611_161186


namespace isosceles_triangle_largest_angle_l1611_161148

/-- 
Given an isosceles triangle where one of the angles opposite an equal side is 40°,
prove that the largest angle measures 100°.
-/
theorem isosceles_triangle_largest_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α = β →            -- The triangle is isosceles (two angles are equal)
  α = 40 →           -- One of the angles opposite an equal side is 40°
  max α (max β γ) = 100 := by  -- The largest angle measures 100°
sorry

end isosceles_triangle_largest_angle_l1611_161148


namespace product_of_largest_primes_l1611_161102

def largest_one_digit_primes : Finset Nat := {5, 7}
def largest_two_digit_prime : Nat := 97

theorem product_of_largest_primes : 
  (Finset.prod largest_one_digit_primes id) * largest_two_digit_prime = 3395 := by
  sorry

end product_of_largest_primes_l1611_161102


namespace cos_sin_inequalities_l1611_161159

theorem cos_sin_inequalities (x : ℝ) (h : 0 < x ∧ x < π/4) :
  (Real.cos x) ^ ((Real.cos x)^2) > (Real.sin x) ^ ((Real.sin x)^2) ∧
  (Real.cos x) ^ ((Real.cos x)^4) < (Real.sin x) ^ ((Real.sin x)^4) := by
  sorry

end cos_sin_inequalities_l1611_161159


namespace triangle_area_triangle_area_is_18_l1611_161193

/-- The area of a triangle with vertices at (1, 2), (7, 6), and (1, 8) is 18 square units. -/
theorem triangle_area : ℝ :=
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (7, 6)
  let C : ℝ × ℝ := (1, 8)
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  18

/-- The theorem statement. -/
theorem triangle_area_is_18 : triangle_area = 18 := by
  sorry

end triangle_area_triangle_area_is_18_l1611_161193


namespace work_completion_time_l1611_161161

/-- Represents the time taken to complete a work -/
structure WorkTime where
  days : ℚ
  hours : ℚ

/-- Represents a worker's capacity to complete work -/
structure Worker where
  completion_time : ℚ

/-- Represents a work period with multiple workers -/
structure WorkPeriod where
  duration : ℚ
  workers : List Worker

/-- Calculates the fraction of work completed in a day by a worker -/
def Worker.daily_work (w : Worker) : ℚ :=
  1 / w.completion_time

/-- Calculates the total work completed in a period -/
def WorkPeriod.work_completed (wp : WorkPeriod) : ℚ :=
  wp.duration * (wp.workers.map Worker.daily_work).sum

/-- Converts days to a WorkTime structure -/
def days_to_work_time (d : ℚ) : WorkTime :=
  ⟨d.floor, (d - d.floor) * 24⟩

theorem work_completion_time 
  (worker_a worker_b worker_c : Worker)
  (period1 period2 period3 : WorkPeriod)
  (h1 : worker_a.completion_time = 15)
  (h2 : worker_b.completion_time = 10)
  (h3 : worker_c.completion_time = 12)
  (h4 : period1 = ⟨2, [worker_a, worker_b, worker_c]⟩)
  (h5 : period2 = ⟨3, [worker_a, worker_c]⟩)
  (h6 : period3 = ⟨(1 - period1.work_completed - period2.work_completed) / worker_a.daily_work, [worker_a]⟩) :
  days_to_work_time (period1.duration + period2.duration + period3.duration) = ⟨5, 18⟩ :=
sorry

end work_completion_time_l1611_161161


namespace bucket_water_problem_l1611_161184

/-- Given two equations representing the weight of a bucket with water,
    prove that the original amount of water is 3 kg and the bucket weighs 4 kg. -/
theorem bucket_water_problem (x y : ℝ) 
  (eq1 : 4 * x + y = 16)
  (eq2 : 6 * x + y = 22) :
  x = 3 ∧ y = 4 := by
  sorry

end bucket_water_problem_l1611_161184


namespace inequality_solution_set_l1611_161103

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ ((m - 1) * x < Real.sqrt (4 * x) - x^2)) → 
  m = 1 :=
sorry

end inequality_solution_set_l1611_161103


namespace expression_value_l1611_161116

theorem expression_value (a b c d x : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |x| = 3) : 
  3 * (a + b) - (-c * d) ^ 2021 + x = 4 ∨ 3 * (a + b) - (-c * d) ^ 2021 + x = -2 :=
by sorry

end expression_value_l1611_161116


namespace andrei_valentin_distance_at_finish_l1611_161136

/-- Represents the race scenario with three runners -/
structure RaceScenario where
  race_distance : ℝ
  andrei_boris_gap : ℝ
  boris_valentin_gap : ℝ

/-- Calculates the distance between Andrei and Valentin at Andrei's finish -/
def distance_andrei_valentin (scenario : RaceScenario) : ℝ :=
  scenario.race_distance - (scenario.race_distance - scenario.andrei_boris_gap - scenario.boris_valentin_gap)

/-- Theorem stating the distance between Andrei and Valentin when Andrei finishes -/
theorem andrei_valentin_distance_at_finish (scenario : RaceScenario) 
  (h1 : scenario.race_distance = 1000)
  (h2 : scenario.andrei_boris_gap = 100)
  (h3 : scenario.boris_valentin_gap = 50) :
  distance_andrei_valentin scenario = 145 := by
  sorry

#eval distance_andrei_valentin ⟨1000, 100, 50⟩

end andrei_valentin_distance_at_finish_l1611_161136


namespace max_value_of_m_l1611_161108

theorem max_value_of_m :
  (∀ x : ℝ, x < m → x^2 - 2*x - 8 > 0) →
  (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m) →
  (∀ ε > 0, ∃ x : ℝ, x < -2 + ε ∧ x ≥ m) →
  m ≤ -2 :=
by sorry

end max_value_of_m_l1611_161108


namespace root_quadratic_implies_value_l1611_161196

theorem root_quadratic_implies_value (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → 2*m^2 - 4*m = 6 := by
  sorry

end root_quadratic_implies_value_l1611_161196


namespace distance_to_double_reflection_distance_C_to_C_l1611_161124

/-- The distance between a point and its reflection over both x and y axes --/
theorem distance_to_double_reflection (x y : ℝ) : 
  let C : ℝ × ℝ := (x, y)
  let C' : ℝ × ℝ := (-x, -y)
  Real.sqrt ((C'.1 - C.1)^2 + (C'.2 - C.2)^2) = Real.sqrt (4 * (x^2 + y^2)) :=
by sorry

/-- The specific case for point C(-3, 2) --/
theorem distance_C_to_C'_is_sqrt_52 : 
  let C : ℝ × ℝ := (-3, 2)
  let C' : ℝ × ℝ := (3, -2)
  Real.sqrt ((C'.1 - C.1)^2 + (C'.2 - C.2)^2) = Real.sqrt 52 :=
by sorry

end distance_to_double_reflection_distance_C_to_C_l1611_161124


namespace f_10_eq_756_l1611_161143

/-- The polynomial function f(x) = x^3 - 2x^2 - 5x + 6 -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

/-- Theorem: f(10) = 756 -/
theorem f_10_eq_756 : f 10 = 756 := by
  sorry

end f_10_eq_756_l1611_161143


namespace point_M_coordinates_l1611_161114

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Theorem statement
theorem point_M_coordinates :
  ∃ (x y : ℝ), 
    curve y = curve x ∧ 
    curve_derivative x = -4 ∧ 
    x = -1 ∧ 
    y = 3 := by
  sorry

end point_M_coordinates_l1611_161114


namespace interest_calculation_l1611_161113

/-- Simple interest calculation function -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ := 1) : ℝ :=
  principal * rate * time

theorem interest_calculation 
  (r : ℝ) -- Interest rate as a decimal
  (h1 : simpleInterest 5000 r = 250) -- Condition for the initial investment
  : simpleInterest 20000 r = 1000 := by
  sorry

#check interest_calculation

end interest_calculation_l1611_161113


namespace binary_101101_equals_45_l1611_161125

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end binary_101101_equals_45_l1611_161125


namespace common_tangent_lines_C₁_C₂_l1611_161130

/-- Circle C₁ with equation x² + y² - 2x = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 = 0}

/-- Circle C₂ with equation x² + (y - √3)² = 4 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - Real.sqrt 3)^2 = 4}

/-- The number of common tangent lines between two circles -/
def commonTangentLines (c1 c2 : Set (ℝ × ℝ)) : ℕ :=
  sorry

/-- Theorem stating that the number of common tangent lines between C₁ and C₂ is 2 -/
theorem common_tangent_lines_C₁_C₂ :
  commonTangentLines C₁ C₂ = 2 :=
sorry

end common_tangent_lines_C₁_C₂_l1611_161130


namespace cost_calculation_l1611_161170

theorem cost_calculation (pencil_cost pen_cost eraser_cost : ℝ) 
  (eq1 : 8 * pencil_cost + 2 * pen_cost + eraser_cost = 4.60)
  (eq2 : 2 * pencil_cost + 5 * pen_cost + eraser_cost = 3.90)
  (eq3 : pencil_cost + pen_cost + 3 * eraser_cost = 2.75) :
  4 * pencil_cost + 3 * pen_cost + 2 * eraser_cost = 7.4135 := by
sorry

end cost_calculation_l1611_161170


namespace arithmetic_sequence_sum_inequality_l1611_161149

-- Define the arithmetic sequence and its sum
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d
def S (a₁ d : ℚ) (n : ℕ) : ℚ := n * a₁ + n * (n - 1 : ℚ) / 2 * d

-- State the theorem
theorem arithmetic_sequence_sum_inequality 
  (p q : ℕ) (a₁ d : ℚ) (hp : p ≠ q) (hSp : S a₁ d p = p / q) (hSq : S a₁ d q = q / p) :
  S a₁ d (p + q) > 4 :=
sorry

end arithmetic_sequence_sum_inequality_l1611_161149


namespace largest_gcd_of_sum_1008_l1611_161175

theorem largest_gcd_of_sum_1008 :
  ∃ (max_gcd : ℕ), ∀ (a b : ℕ), 
    a > 0 → b > 0 → a + b = 1008 →
    gcd a b ≤ max_gcd ∧
    ∃ (a' b' : ℕ), a' > 0 ∧ b' > 0 ∧ a' + b' = 1008 ∧ gcd a' b' = max_gcd :=
by
  -- The proof goes here
  sorry

end largest_gcd_of_sum_1008_l1611_161175


namespace consecutive_cubes_l1611_161139

theorem consecutive_cubes (a b c d : ℤ) (y z w x v : ℤ) : 
  (d = c + 1 ∧ c = b + 1 ∧ b = a + 1) → 
  (v = x + 1 ∧ x = w + 1 ∧ w = z + 1 ∧ z = y + 1) →
  ((a^3 + b^3 + c^3 = d^3) ↔ (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6)) ∧
  (y^3 + z^3 + w^3 + x^3 ≠ v^3) := by
sorry

end consecutive_cubes_l1611_161139


namespace apple_cost_l1611_161194

/-- The cost of apples under specific pricing conditions -/
theorem apple_cost (l q : ℚ) : 
  (30 * l + 3 * q = 333) →  -- Price for 33 kg
  (30 * l + 6 * q = 366) →  -- Price for 36 kg
  (15 * l = 150)            -- Price for 15 kg
:= by sorry

end apple_cost_l1611_161194


namespace time_period_is_seven_days_l1611_161145

/-- The number of horses Minnie mounts per day -/
def minnie_daily_mounts : ℕ := sorry

/-- The number of days in the time period -/
def time_period : ℕ := sorry

/-- The number of horses Mickey mounts per day -/
def mickey_daily_mounts : ℕ := sorry

/-- Mickey mounts six less than twice as many horses per day as Minnie -/
axiom mickey_minnie_relation : mickey_daily_mounts = 2 * minnie_daily_mounts - 6

/-- Minnie mounts three more horses per day than there are days in the time period -/
axiom minnie_time_relation : minnie_daily_mounts = time_period + 3

/-- Mickey mounts 98 horses per week -/
axiom mickey_weekly_mounts : mickey_daily_mounts * 7 = 98

/-- The main theorem: The time period is 7 days -/
theorem time_period_is_seven_days : time_period = 7 := by sorry

end time_period_is_seven_days_l1611_161145


namespace square_root_div_five_l1611_161135

theorem square_root_div_five : Real.sqrt 625 / 5 = 5 := by
  sorry

end square_root_div_five_l1611_161135


namespace pure_imaginary_condition_l1611_161121

theorem pure_imaginary_condition (a b : ℝ) : 
  (∀ x y : ℝ, x + y * Complex.I = Complex.I * y → x = 0) ∧
  (∃ x y : ℝ, x = 0 ∧ x + y * Complex.I ≠ Complex.I * y) :=
sorry

end pure_imaginary_condition_l1611_161121


namespace impossible_30_gon_numbering_l1611_161191

theorem impossible_30_gon_numbering : ¬ ∃ (f : Fin 30 → Nat),
  (∀ i, f i ∈ Finset.range 30) ∧
  (∀ i, f i ≠ 0) ∧
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  (∀ i : Fin 30, ∃ k : Nat, (f i + f ((i + 1) % 30) : Nat) = k^2) := by
  sorry

end impossible_30_gon_numbering_l1611_161191


namespace trapezoid_triangle_area_l1611_161189

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Checks if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a point is on a line segment -/
def isOnSegment (p : Point) (p1 p2 : Point) : Prop := sorry

/-- Checks if two line segments are parallel -/
def isParallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem trapezoid_triangle_area
  (ABCD : Trapezoid)
  (E : Point)
  (h1 : isPerpendicular ABCD.A ABCD.D ABCD.D ABCD.C)
  (h2 : ABCD.A.x - ABCD.D.x = 5)
  (h3 : ABCD.A.y - ABCD.B.y = 5)
  (h4 : ABCD.D.x - ABCD.C.x = 10)
  (h5 : isOnSegment E ABCD.D ABCD.C)
  (h6 : E.x - ABCD.D.x = 4)
  (h7 : isParallel ABCD.B E ABCD.A ABCD.D)
  : triangleArea ABCD.A ABCD.D E = 10 := by
  sorry

end trapezoid_triangle_area_l1611_161189


namespace direction_vector_of_line_l1611_161174

/-- Given a line 2x - 3y + 1 = 0, prove that (3, 2) is a direction vector --/
theorem direction_vector_of_line (x y : ℝ) : 
  (2 * x - 3 * y + 1 = 0) → (∃ (t : ℝ), x = 3 * t ∧ y = 2 * t) := by
  sorry

end direction_vector_of_line_l1611_161174


namespace right_angle_in_clerts_l1611_161154

/-- In a system where a full circle is measured as 500 units, a right angle is 125 units. -/
theorem right_angle_in_clerts (full_circle : ℕ) (right_angle : ℕ) 
  (h1 : full_circle = 500) 
  (h2 : right_angle = full_circle / 4) : 
  right_angle = 125 := by
  sorry

end right_angle_in_clerts_l1611_161154


namespace fraction_equivalence_l1611_161147

theorem fraction_equivalence : (16 : ℝ) / (8 * 17) = 1.6 / (0.8 * 17) := by
  sorry

end fraction_equivalence_l1611_161147


namespace divide_by_six_multiply_by_twelve_l1611_161111

theorem divide_by_six_multiply_by_twelve (x : ℝ) : (x / 6) * 12 = 2 * x := by
  sorry

end divide_by_six_multiply_by_twelve_l1611_161111


namespace problem_one_problem_two_l1611_161195

-- Problem 1
theorem problem_one : (-1/3)⁻¹ + (Real.pi - 3.14)^0 = -2 := by sorry

-- Problem 2
theorem problem_two : ∀ x : ℝ, (2*x - 3)^2 - 2*x*(2*x - 6) = 9 := by sorry

end problem_one_problem_two_l1611_161195


namespace logarithm_difference_equals_two_l1611_161123

theorem logarithm_difference_equals_two :
  (Real.log 80 / Real.log 2) / (Real.log 40 / Real.log 2) -
  (Real.log 160 / Real.log 2) / (Real.log 20 / Real.log 2) = 2 := by
  sorry

end logarithm_difference_equals_two_l1611_161123


namespace minimize_z_l1611_161166

-- Define the function z
def z (x a b c d : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + c*(x - a) + d*(x - b)

-- Theorem statement
theorem minimize_z (a b c d : ℝ) :
  ∃ x : ℝ, ∀ y : ℝ, z x a b c d ≤ z y a b c d ∧ x = (2*(a+b) - (c+d)) / 4 :=
sorry

end minimize_z_l1611_161166


namespace polynomial_division_theorem_l1611_161131

-- Define the polynomial and divisor
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 9*x - 6
def g (x : ℝ) : ℝ := x^2 - x + 4

-- Define the quotient and remainder
def q (x : ℝ) : ℝ := x - 3
def r (x : ℝ) : ℝ := 2*x + 6

-- Theorem statement
theorem polynomial_division_theorem :
  ∀ x : ℝ, f x = g x * q x + r x :=
by sorry

end polynomial_division_theorem_l1611_161131


namespace investment_growth_l1611_161110

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof that $30,697 grows to at least $50,000 in 10 years at 5% interest -/
theorem investment_growth :
  let initial_deposit : ℝ := 30697
  let interest_rate : ℝ := 0.05
  let years : ℕ := 10
  let target_amount : ℝ := 50000
  compound_interest initial_deposit interest_rate years ≥ target_amount :=
by
  sorry

#check investment_growth

end investment_growth_l1611_161110


namespace integer_with_specific_cube_root_l1611_161157

theorem integer_with_specific_cube_root : ∃ n : ℕ+,
  (↑n : ℝ) > 0 ∧
  ∃ k : ℕ, n = 24 * k ∧
  (9 : ℝ) < (↑n : ℝ) ^ (1/3) ∧ (↑n : ℝ) ^ (1/3) < 9.1 :=
by
  use 744
  sorry

end integer_with_specific_cube_root_l1611_161157


namespace x_value_proof_l1611_161182

theorem x_value_proof (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 7)
  (eq2 : x^2 + 3 * y = 17) : 
  x = 3.5 := by
sorry

end x_value_proof_l1611_161182


namespace parallel_perpendicular_implication_l1611_161109

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (l m : Line) (α : Plane) :
  parallel l m → perpendicular l α → perpendicular m α :=
sorry

end parallel_perpendicular_implication_l1611_161109


namespace smallest_number_l1611_161198

def numbers : Set ℤ := {0, -2, -1, 3}

theorem smallest_number (n : ℤ) (hn : n ∈ numbers) : -2 ≤ n := by
  sorry

end smallest_number_l1611_161198


namespace mary_balloons_l1611_161104

-- Define the number of Nancy's balloons
def nancy_balloons : ℕ := 7

-- Define the ratio of Mary's balloons to Nancy's
def mary_ratio : ℕ := 4

-- Theorem to prove
theorem mary_balloons : nancy_balloons * mary_ratio = 28 := by
  sorry

end mary_balloons_l1611_161104


namespace average_weight_increase_l1611_161181

theorem average_weight_increase (group_size : ℕ) (original_weight new_weight : ℝ) :
  group_size = 6 →
  original_weight = 65 →
  new_weight = 74 →
  (new_weight - original_weight) / group_size = 1.5 := by
sorry

end average_weight_increase_l1611_161181


namespace bales_in_shed_l1611_161120

theorem bales_in_shed (initial_barn : ℕ) (added : ℕ) (final_barn : ℕ) : 
  initial_barn = 47 → added = 35 → final_barn = 82 → 
  final_barn = initial_barn + added → initial_barn + added = 82 → 0 = final_barn - (initial_barn + added) :=
by
  sorry

end bales_in_shed_l1611_161120
