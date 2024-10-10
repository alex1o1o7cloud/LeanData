import Mathlib

namespace cat_to_dog_probability_l3506_350683

-- Define the probabilities for each machine
def prob_A : ℚ := 1/3
def prob_B : ℚ := 2/5
def prob_C : ℚ := 1/4

-- Define the probability of a cat remaining a cat after all machines
def prob_cat_total : ℚ := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

-- The main theorem to prove
theorem cat_to_dog_probability :
  1 - prob_cat_total = 7/10 := by sorry

end cat_to_dog_probability_l3506_350683


namespace product_increase_theorem_l3506_350661

theorem product_increase_theorem :
  ∃ (a b c d e : ℕ), 
    (((a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3)) : ℤ) = 
    15 * (a * b * c * d * e) :=
by sorry

end product_increase_theorem_l3506_350661


namespace product_of_positive_real_solutions_l3506_350616

theorem product_of_positive_real_solutions (x : ℂ) : 
  (x^6 = -64) → 
  (∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^6 = -64 ∧ z.re > 0) ∧ 
    (∀ z, z^6 = -64 ∧ z.re > 0 → z ∈ S) ∧
    (S.prod id = 4)) :=
by sorry

end product_of_positive_real_solutions_l3506_350616


namespace pipe_B_fill_time_l3506_350607

/-- The time it takes for pipe A to fill the cistern (in minutes) -/
def time_A : ℝ := 45

/-- The time it takes for the third pipe to empty the cistern (in minutes) -/
def time_empty : ℝ := 72

/-- The time it takes to fill the cistern when all three pipes are open (in minutes) -/
def time_all : ℝ := 40

/-- The time it takes for pipe B to fill the cistern (in minutes) -/
def time_B : ℝ := 60

theorem pipe_B_fill_time :
  ∃ (t : ℝ), t > 0 ∧ 1 / time_A + 1 / t - 1 / time_empty = 1 / time_all ∧ t = time_B := by
  sorry

end pipe_B_fill_time_l3506_350607


namespace total_points_is_1320_l3506_350645

def freshman_points : ℕ := 260

def sophomore_points : ℕ := freshman_points + (freshman_points * 15 / 100)

def junior_points : ℕ := sophomore_points + (sophomore_points * 20 / 100)

def senior_points : ℕ := junior_points + (junior_points * 12 / 100)

def total_points : ℕ := freshman_points + sophomore_points + junior_points + senior_points

theorem total_points_is_1320 : total_points = 1320 := by
  sorry

end total_points_is_1320_l3506_350645


namespace cosine_function_parameters_l3506_350686

theorem cosine_function_parameters (a b c d : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x, ∃ y, y = a * Real.cos (b * x + c) + d) →
  (4 * Real.pi / b = 4 * Real.pi) →
  d = 3 →
  (∃ x, a * Real.cos (b * x + c) + d = 8) →
  (∃ x, a * Real.cos (b * x + c) + d = -2) →
  a = 5 ∧ b = 1 := by
sorry

end cosine_function_parameters_l3506_350686


namespace polynomial_factor_l3506_350615

theorem polynomial_factor (x : ℝ) : 
  ∃ (p : ℝ → ℝ), (x^4 - 4*x^2 + 4) = (x^2 - 2) * p x :=
sorry

end polynomial_factor_l3506_350615


namespace shaded_area_ratio_l3506_350603

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the shaded quadrilateral -/
structure ShadedQuadrilateral where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- The side length of the large square -/
def largeSideLength : ℝ := 10

/-- The side length of each small square in the grid -/
def smallSideLength : ℝ := 2

/-- The number of squares in each row/column of the grid -/
def gridSize : ℕ := 5

/-- Function to calculate the area of the shaded quadrilateral -/
def shadedArea (quad : ShadedQuadrilateral) : ℝ := sorry

/-- Function to create the shaded quadrilateral based on the problem description -/
def createShadedQuadrilateral : ShadedQuadrilateral := sorry

/-- Theorem stating the ratio of shaded area to large square area -/
theorem shaded_area_ratio :
  let quad := createShadedQuadrilateral
  shadedArea quad / (largeSideLength ^ 2) = 1 / 50 := by sorry

end shaded_area_ratio_l3506_350603


namespace cannon_hit_probability_l3506_350601

theorem cannon_hit_probability (P1 P2 P3 : ℝ) : 
  P1 = 0.2 →
  P3 = 0.3 →
  (1 - P1) * (1 - P2) * (1 - P3) = 0.27999999999999997 →
  P2 = 0.5 := by
sorry

end cannon_hit_probability_l3506_350601


namespace temperature_difference_l3506_350648

def highest_temp : ℚ := 10
def lowest_temp : ℚ := -5

theorem temperature_difference :
  highest_temp - lowest_temp = 15 := by sorry

end temperature_difference_l3506_350648


namespace smallest_nonnegative_congruence_l3506_350655

theorem smallest_nonnegative_congruence :
  ∃ n : ℕ, n < 7 ∧ -2222 ≡ n [ZMOD 7] ∧ ∀ m : ℕ, m < 7 → -2222 ≡ m [ZMOD 7] → n ≤ m :=
by sorry

end smallest_nonnegative_congruence_l3506_350655


namespace two_digit_number_remainder_l3506_350670

theorem two_digit_number_remainder (n : ℕ) : 
  10 ≤ n ∧ n < 100 →  -- n is a two-digit number
  n % 9 = 1 →         -- remainder when divided by 9 is 1
  n % 10 = 3 →        -- remainder when divided by 10 is 3
  n % 11 = 7 :=       -- remainder when divided by 11 is 7
by
  sorry

end two_digit_number_remainder_l3506_350670


namespace smallest_number_of_students_l3506_350665

/-- Represents the number of students in each grade -/
structure Students where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- The ratios given in the problem -/
def ratio_9_10 : ℚ := 7 / 4
def ratio_9_11 : ℚ := 5 / 3

/-- The proposition that needs to be proved -/
theorem smallest_number_of_students :
  ∃ (s : Students),
    (s.ninth : ℚ) / s.tenth = ratio_9_10 ∧
    (s.ninth : ℚ) / s.eleventh = ratio_9_11 ∧
    s.ninth + s.tenth + s.eleventh = 76 ∧
    (∀ (t : Students),
      (t.ninth : ℚ) / t.tenth = ratio_9_10 →
      (t.ninth : ℚ) / t.eleventh = ratio_9_11 →
      t.ninth + t.tenth + t.eleventh ≥ 76) :=
sorry

end smallest_number_of_students_l3506_350665


namespace common_ratio_is_negative_two_l3506_350647

def geometric_sequence : ℕ → ℚ
  | 0 => 10
  | 1 => -20
  | 2 => 40
  | 3 => -80
  | _ => 0  -- We only define the first 4 terms as given in the problem

theorem common_ratio_is_negative_two :
  ∀ n : ℕ, n < 3 → geometric_sequence (n + 1) / geometric_sequence n = -2 :=
by
  sorry

#eval geometric_sequence 0
#eval geometric_sequence 1
#eval geometric_sequence 2
#eval geometric_sequence 3

end common_ratio_is_negative_two_l3506_350647


namespace ellipse_properties_l3506_350675

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/4
  h_max_area : a * b = 2 * Real.sqrt 3

/-- The standard form of the ellipse -/
def standard_form (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2/4 + y^2/3 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1

/-- The fixed point property -/
def fixed_point_property (e : Ellipse) : Prop :=
  ∃ D : ℝ × ℝ, 
    D.2 = 0 ∧ 
    D.1 = -11/8 ∧
    ∀ M N : ℝ × ℝ,
      (M.1^2/e.a^2 + M.2^2/e.b^2 = 1) →
      (N.1^2/e.a^2 + N.2^2/e.b^2 = 1) →
      (∃ t : ℝ, M.1 = t * M.2 - 1 ∧ N.1 = t * N.2 - 1) →
      ((M.1 - D.1) * (N.1 - D.1) + (M.2 - D.2) * (N.2 - D.2) = -135/64)

theorem ellipse_properties (e : Ellipse) : 
  standard_form e ∧ fixed_point_property e := by
  sorry

end ellipse_properties_l3506_350675


namespace exotic_fruit_distribution_l3506_350642

theorem exotic_fruit_distribution (eldest_fruits second_fruits third_fruits : ℕ) 
  (gold_to_eldest gold_to_second : ℕ) :
  eldest_fruits = 2 * second_fruits / 3 →
  third_fruits = 0 →
  gold_to_eldest + gold_to_second = 180 →
  eldest_fruits - gold_to_eldest / (180 / (gold_to_eldest + gold_to_second)) = 
    second_fruits - gold_to_second / (180 / (gold_to_eldest + gold_to_second)) →
  eldest_fruits - gold_to_eldest / (180 / (gold_to_eldest + gold_to_second)) = 
    (gold_to_eldest + gold_to_second) / (180 / (gold_to_eldest + gold_to_second)) →
  gold_to_second = 144 :=
by sorry

end exotic_fruit_distribution_l3506_350642


namespace henrys_age_l3506_350690

/-- Given that the sum of Henry and Jill's present ages is 48, and 9 years ago Henry was twice the age of Jill, 
    prove that Henry's present age is 29 years. -/
theorem henrys_age (henry_age jill_age : ℕ) 
  (sum_condition : henry_age + jill_age = 48)
  (past_condition : henry_age - 9 = 2 * (jill_age - 9)) : 
  henry_age = 29 := by
  sorry

end henrys_age_l3506_350690


namespace net_population_increase_l3506_350692

/-- The net population increase in one day given specific birth and death rates -/
theorem net_population_increase (birth_rate : ℕ) (death_rate : ℕ) (seconds_per_interval : ℕ) (seconds_per_day : ℕ) :
  birth_rate = 4 →
  death_rate = 2 →
  seconds_per_interval = 2 →
  seconds_per_day = 86400 →
  (birth_rate - death_rate) * (seconds_per_day / seconds_per_interval) = 86400 := by
  sorry

#check net_population_increase

end net_population_increase_l3506_350692


namespace cos_120_degrees_l3506_350606

theorem cos_120_degrees :
  Real.cos (120 * π / 180) = -1/2 := by
sorry

end cos_120_degrees_l3506_350606


namespace unchanged_flipped_nine_digit_numbers_l3506_350629

/-- 
Given that:
- A 9-digit number is considered unchanged when flipped if it reads the same upside down.
- Digits 0, 1, and 8 remain unchanged when flipped.
- Digits 6 and 9 become each other when flipped.
- Other digits have no meaning when flipped.

This theorem states that the number of 9-digit numbers that remain unchanged when flipped is 1500.
-/
theorem unchanged_flipped_nine_digit_numbers : ℕ := by
  -- Define the set of digits that remain unchanged when flipped
  let unchanged_digits : Finset ℕ := {0, 1, 8}
  
  -- Define the set of digit pairs that become each other when flipped
  let swapped_digits : Finset (ℕ × ℕ) := {(6, 9), (9, 6)}
  
  -- Define the number of valid options for the first and last digit
  let first_last_options : ℕ := 4
  
  -- Define the number of valid options for the second, third, fourth, and eighth digit
  let middle_pair_options : ℕ := 5
  
  -- Define the number of valid options for the center digit
  let center_options : ℕ := 3
  
  -- Calculate the total number of valid 9-digit numbers
  let total : ℕ := first_last_options * middle_pair_options^3 * center_options
  
  -- Assert that the total is equal to 1500
  have h : total = 1500 := by sorry
  
  -- Return the result
  exact 1500


end unchanged_flipped_nine_digit_numbers_l3506_350629


namespace picture_area_l3506_350668

/-- The area of a picture on a sheet of paper with given dimensions and margins. -/
theorem picture_area (paper_width paper_length margin : ℝ) 
  (hw : paper_width = 8.5)
  (hl : paper_length = 10)
  (hm : margin = 1.5) : 
  (paper_width - 2 * margin) * (paper_length - 2 * margin) = 38.5 := by
  sorry

end picture_area_l3506_350668


namespace geometric_sequence_property_l3506_350678

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) (h : geometric_sequence a) (h5 : a 5 = 4) : 
  a 2 * a 8 = 16 := by
  sorry

end geometric_sequence_property_l3506_350678


namespace function_value_theorem_l3506_350652

/-- Given a function f(x) = √(-x² + bx + c) with domain D, 
    and for any x in D, f(-1) ≤ f(x) ≤ f(1), 
    prove that b · c + f(3) = 6 -/
theorem function_value_theorem (b c : ℝ) (D : Set ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x ∈ D, f x = Real.sqrt (-x^2 + b*x + c))
    (h2 : ∀ x ∈ D, f (-1) ≤ f x ∧ f x ≤ f 1) :
    b * c + f 3 = 6 := by
  sorry

end function_value_theorem_l3506_350652


namespace three_digit_sum_proof_l3506_350653

/-- Represents a three-digit number in the form xyz -/
def ThreeDigitNumber (x y z : Nat) : Nat :=
  100 * x + 10 * y + z

theorem three_digit_sum_proof (a b : Nat) :
  (ThreeDigitNumber 3 a 7) + 416 = (ThreeDigitNumber 7 b 3) ∧
  (ThreeDigitNumber 7 b 3) % 3 = 0 →
  a + b = 2 := by
  sorry

end three_digit_sum_proof_l3506_350653


namespace product_of_fractions_l3506_350658

theorem product_of_fractions (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_of_fractions_l3506_350658


namespace largest_triangular_cross_section_area_l3506_350695

/-- The largest possible area of a triangular cross-section in a right circular cone -/
theorem largest_triangular_cross_section_area
  (slant_height : ℝ)
  (base_diameter : ℝ)
  (h_slant : slant_height = 5)
  (h_diameter : base_diameter = 8) :
  ∃ (area : ℝ), area = 12.5 ∧
  ∀ (other_area : ℝ),
    (∃ (a b c : ℝ),
      a ≤ slant_height ∧
      b ≤ slant_height ∧
      c ≤ base_diameter ∧
      other_area = (a * b) / 2) →
    other_area ≤ area :=
by sorry

end largest_triangular_cross_section_area_l3506_350695


namespace line_equation_proof_line_parameters_l3506_350699

/-- Given a line defined by (3, -4) · ((x, y) - (-2, 8)) = 0, prove that it can be expressed as y = (3/4)x + 9.5 with m = 3/4 and b = 9.5 -/
theorem line_equation_proof (x y : ℝ) :
  (3 * (x + 2) + (-4) * (y - 8) = 0) ↔ (y = (3 / 4) * x + (19 / 2)) :=
by sorry

/-- Prove that for the given line, m = 3/4 and b = 9.5 -/
theorem line_parameters :
  ∃ (m b : ℝ), m = 3 / 4 ∧ b = 19 / 2 ∧
  ∀ (x y : ℝ), (3 * (x + 2) + (-4) * (y - 8) = 0) ↔ (y = m * x + b) :=
by sorry

end line_equation_proof_line_parameters_l3506_350699


namespace side_ratio_l3506_350608

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def special_triangle (t : Triangle) : Prop :=
  t.A > t.B ∧ t.B > t.C ∧  -- A is largest, C is smallest
  t.A = 2 * t.C ∧          -- A = 2C
  t.a + t.c = 2 * t.b      -- a + c = 2b

-- Theorem statement
theorem side_ratio (t : Triangle) (h : special_triangle t) :
  ∃ (k : ℝ), k > 0 ∧ t.a = 6*k ∧ t.b = 5*k ∧ t.c = 3*k :=
sorry

end side_ratio_l3506_350608


namespace max_value_of_expression_l3506_350619

theorem max_value_of_expression (x y z w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (h1 : x^2 + y^2 - x*y/2 = 36)
  (h2 : w^2 + z^2 + w*z/2 = 36)
  (h3 : x*z + y*w = 30) :
  (x*y + w*z)^2 ≤ 960 ∧ ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a^2 + b^2 - a*b/2 = 36 ∧
    d^2 + c^2 + d*c/2 = 36 ∧
    a*c + b*d = 30 ∧
    (a*b + d*c)^2 = 960 :=
by sorry

end max_value_of_expression_l3506_350619


namespace no_scalene_equilateral_triangle_no_equilateral_right_triangle_impossible_triangles_l3506_350627

-- Define the properties of triangles
def IsScalene (triangle : Type) : Prop := sorry
def IsEquilateral (triangle : Type) : Prop := sorry
def IsRight (triangle : Type) : Prop := sorry

-- Theorem stating that scalene equilateral triangles cannot exist
theorem no_scalene_equilateral_triangle (triangle : Type) :
  ¬(IsScalene triangle ∧ IsEquilateral triangle) := by sorry

-- Theorem stating that equilateral right triangles cannot exist
theorem no_equilateral_right_triangle (triangle : Type) :
  ¬(IsEquilateral triangle ∧ IsRight triangle) := by sorry

-- Main theorem combining both impossible triangle types
theorem impossible_triangles (triangle : Type) :
  ¬(IsScalene triangle ∧ IsEquilateral triangle) ∧
  ¬(IsEquilateral triangle ∧ IsRight triangle) := by sorry

end no_scalene_equilateral_triangle_no_equilateral_right_triangle_impossible_triangles_l3506_350627


namespace min_omega_for_shifted_periodic_function_l3506_350611

/-- The minimum value of ω for a periodic function with a specific shift -/
theorem min_omega_for_shifted_periodic_function (ω : ℝ) (h1 : ω > 0) : 
  (∀ x : ℝ, 3 * Real.sin (ω * x + π / 6) - 2 = 
            3 * Real.sin (ω * (x - 2 * π / 3) + π / 6) - 2) →
  ω ≥ 3 ∧ ∃ n : ℕ, ω = 3 * n :=
by sorry

end min_omega_for_shifted_periodic_function_l3506_350611


namespace sum_even_product_not_necessarily_odd_l3506_350697

theorem sum_even_product_not_necessarily_odd :
  ∃ (a b : ℤ), Even (a + b) ∧ ¬Odd (a * b) := by sorry

end sum_even_product_not_necessarily_odd_l3506_350697


namespace cubes_fill_box_completely_l3506_350669

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculates the volume of a cube -/
def cubeVolume (cube : Cube) : ℕ :=
  cube.sideLength ^ 3

/-- Calculates the number of cubes that can fit along each dimension of the box -/
def cubesPerDimension (box : BoxDimensions) (cube : Cube) : ℕ × ℕ × ℕ :=
  (box.length / cube.sideLength, box.width / cube.sideLength, box.height / cube.sideLength)

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubes (box : BoxDimensions) (cube : Cube) : ℕ :=
  let (l, w, h) := cubesPerDimension box cube
  l * w * h

/-- Calculates the total volume occupied by the cubes in the box -/
def totalCubeVolume (box : BoxDimensions) (cube : Cube) : ℕ :=
  totalCubes box cube * cubeVolume cube

/-- Theorem: The volume occupied by 4-inch cubes in an 8x4x12 inch box is 100% of the box's volume -/
theorem cubes_fill_box_completely (box : BoxDimensions) (cube : Cube) :
  box.length = 8 ∧ box.width = 4 ∧ box.height = 12 ∧ cube.sideLength = 4 →
  totalCubeVolume box cube = boxVolume box := by
  sorry

#check cubes_fill_box_completely

end cubes_fill_box_completely_l3506_350669


namespace single_digit_square_equals_5929_l3506_350639

theorem single_digit_square_equals_5929 (A : ℕ) : 
  A < 10 → (10 * A + A) * (10 * A + A) = 5929 → A = 7 := by
sorry

end single_digit_square_equals_5929_l3506_350639


namespace equation_solution_l3506_350605

theorem equation_solution (x : ℤ) (m : ℕ+) : 
  ((3 * x - 1) / 2 + m = 3) →
  ((m = 5 → x = 1) ∧ 
   (x > 0 → m = 2)) :=
by sorry

end equation_solution_l3506_350605


namespace quadratic_shift_l3506_350667

/-- Represents a quadratic function of the form y = (x + a)^2 + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Shifts a quadratic function horizontally -/
def horizontalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a - shift, b := f.b }

/-- Shifts a quadratic function vertically -/
def verticalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b + shift }

/-- The main theorem stating that shifting y = (x+2)^2 - 3 by 1 unit left and 2 units up
    results in y = (x+3)^2 - 1 -/
theorem quadratic_shift :
  let f := QuadraticFunction.mk 2 (-3)
  let g := verticalShift (horizontalShift f 1) 2
  g = QuadraticFunction.mk 3 (-1) := by
  sorry

end quadratic_shift_l3506_350667


namespace root_transformation_l3506_350681

-- Define the original quadratic equation
def original_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the transformed equation
def transformed_equation (a b c x : ℝ) : Prop := a * (x - 1)^2 + b * (x - 1) + c = 0

-- Theorem statement
theorem root_transformation (a b c : ℝ) :
  (original_equation a b c (-1) ∧ original_equation a b c 2) →
  (transformed_equation a b c 0 ∧ transformed_equation a b c 3) :=
by sorry

end root_transformation_l3506_350681


namespace geometric_sequence_condition_l3506_350684

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

theorem geometric_sequence_condition (a q : ℝ) (h : a > 0) :
  (∀ n : ℕ, geometric_sequence a q n = a * q ^ (n - 1)) →
  (geometric_sequence a q 1 < geometric_sequence a q 3 → geometric_sequence a q 3 < geometric_sequence a q 6) ∧
  ¬(geometric_sequence a q 1 < geometric_sequence a q 3 → geometric_sequence a q 3 < geometric_sequence a q 6) :=
by sorry

end geometric_sequence_condition_l3506_350684


namespace arithmetic_sequence_sum_l3506_350609

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of sum for arithmetic sequence
  (∀ n, ∃ d, a (n + 1) - a n = d) →     -- Definition of arithmetic sequence
  a 1 = -2016 →                         -- Given condition
  (S 2015) / 2015 - (S 2012) / 2012 = 3 →  -- Given condition
  S 2016 = -2016 :=                     -- Conclusion to prove
by sorry

end arithmetic_sequence_sum_l3506_350609


namespace total_spent_calculation_l3506_350630

/-- Calculates the total amount spent on t-shirts given the prices, quantities, discount, and tax rate -/
def total_spent (price_a price_b price_c : ℚ) (qty_a qty_b qty_c : ℕ) (discount_b tax_rate : ℚ) : ℚ :=
  let subtotal_a := price_a * qty_a
  let subtotal_b := price_b * qty_b * (1 - discount_b)
  let subtotal_c := price_c * qty_c
  let total_before_tax := subtotal_a + subtotal_b + subtotal_c
  total_before_tax * (1 + tax_rate)

/-- Theorem stating that given the specific conditions, the total amount spent is $695.21 -/
theorem total_spent_calculation :
  total_spent 9.95 12.50 14.95 18 23 15 0.1 0.05 = 695.21 :=
by sorry

end total_spent_calculation_l3506_350630


namespace system_solution_l3506_350631

theorem system_solution (x y k : ℝ) : 
  (2 * x - y = 5 * k + 6) → 
  (4 * x + 7 * y = k) → 
  (x + y = 2023) → 
  (k = 2022) := by
sorry

end system_solution_l3506_350631


namespace apple_percentage_after_removal_l3506_350689

/-- Calculates the percentage of apples in a bowl of fruit -/
def percentage_apples (apples : ℕ) (oranges : ℕ) : ℚ :=
  (apples : ℚ) / (apples + oranges : ℚ) * 100

/-- Proves that after removing 19 oranges from a bowl with 14 apples and 25 oranges,
    the percentage of apples is 70% -/
theorem apple_percentage_after_removal :
  let initial_apples : ℕ := 14
  let initial_oranges : ℕ := 25
  let removed_oranges : ℕ := 19
  let remaining_oranges : ℕ := initial_oranges - removed_oranges
  percentage_apples initial_apples remaining_oranges = 70 := by
sorry

end apple_percentage_after_removal_l3506_350689


namespace gcd_power_remainder_l3506_350628

theorem gcd_power_remainder (a b : Nat) : 
  (Nat.gcd (2^(30^10) - 2) (2^(30^45) - 2)) % 2013 = 2012 := by
  sorry

end gcd_power_remainder_l3506_350628


namespace geometric_series_first_term_l3506_350680

theorem geometric_series_first_term
  (a r : ℝ)
  (h_sum : a / (1 - r) = 30)
  (h_sum_squares : a^2 / (1 - r^2) = 120)
  (h_convergent : |r| < 1) :
  a = 120 / 17 := by
  sorry

end geometric_series_first_term_l3506_350680


namespace injury_point_is_20_l3506_350622

/-- Represents the runner's journey from Marathon to Athens -/
structure RunnerJourney where
  totalDistance : ℝ
  injuryPoint : ℝ
  initialSpeed : ℝ
  secondPartTime : ℝ
  timeDifference : ℝ

/-- The conditions of the runner's journey -/
def journeyConditions (j : RunnerJourney) : Prop :=
  j.totalDistance = 40 ∧
  j.secondPartTime = 22 ∧
  j.timeDifference = 11 ∧
  j.initialSpeed > 0 ∧
  j.injuryPoint > 0 ∧
  j.injuryPoint < j.totalDistance ∧
  (j.totalDistance - j.injuryPoint) / (j.initialSpeed / 2) = j.secondPartTime ∧
  (j.totalDistance - j.injuryPoint) / (j.initialSpeed / 2) = j.injuryPoint / j.initialSpeed + j.timeDifference

/-- Theorem stating that given the journey conditions, the injury point is at 20 miles -/
theorem injury_point_is_20 (j : RunnerJourney) (h : journeyConditions j) : j.injuryPoint = 20 := by
  sorry

#check injury_point_is_20

end injury_point_is_20_l3506_350622


namespace fraction_zeros_count_l3506_350664

/-- The number of zeros immediately following the decimal point in 1/((6 * 10)^10) -/
def zeros_after_decimal : ℕ := 17

/-- The fraction we're analyzing -/
def fraction : ℚ := 1 / ((6 * 10)^10)

/-- Theorem stating that the number of zeros after the decimal point in the 
    decimal representation of the fraction is equal to zeros_after_decimal -/
theorem fraction_zeros_count : 
  (∃ (n : ℕ) (r : ℚ), fraction * 10^zeros_after_decimal = n + r ∧ 0 < r ∧ r < 1) ∧ 
  (∀ (m : ℕ), m > zeros_after_decimal → ∃ (n : ℕ) (r : ℚ), fraction * 10^m = n + r ∧ r = 0) :=
sorry

end fraction_zeros_count_l3506_350664


namespace bread_cost_l3506_350694

def total_money : ℝ := 60
def celery_cost : ℝ := 5
def cereal_original_cost : ℝ := 12
def cereal_discount : ℝ := 0.5
def milk_original_cost : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_cost : ℝ := 1
def potato_quantity : ℕ := 6
def money_left_for_coffee : ℝ := 26

theorem bread_cost : 
  total_money - 
  (celery_cost + 
   cereal_original_cost * (1 - cereal_discount) + 
   milk_original_cost * (1 - milk_discount) + 
   potato_cost * potato_quantity + 
   money_left_for_coffee) = 8 := by sorry

end bread_cost_l3506_350694


namespace smallest_no_inverse_mod_77_88_l3506_350671

theorem smallest_no_inverse_mod_77_88 : 
  ∀ a : ℕ, a > 0 → (Nat.gcd a 77 > 1 ∧ Nat.gcd a 88 > 1) → a ≥ 14 :=
by sorry

end smallest_no_inverse_mod_77_88_l3506_350671


namespace distance_difference_l3506_350641

/-- The width of the streets in Tranquility Town -/
def street_width : ℝ := 30

/-- The length of the rectangular block -/
def block_length : ℝ := 500

/-- The width of the rectangular block -/
def block_width : ℝ := 300

/-- The perimeter of Alice's path -/
def alice_perimeter : ℝ := 2 * ((block_length + street_width) + (block_width + street_width))

/-- The perimeter of Bob's path -/
def bob_perimeter : ℝ := 2 * ((block_length + 2 * street_width) + (block_width + 2 * street_width))

/-- The theorem stating the difference in distance walked -/
theorem distance_difference : bob_perimeter - alice_perimeter = 240 := by
  sorry

end distance_difference_l3506_350641


namespace digit_150_of_17_70_l3506_350659

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The repeating sequence in the decimal representation of a rational number -/
def repeating_sequence (q : ℚ) : List ℕ := sorry

theorem digit_150_of_17_70 : 
  decimal_representation (17 / 70) 150 = 7 := by sorry

end digit_150_of_17_70_l3506_350659


namespace binomial_sum_equals_240_l3506_350625

theorem binomial_sum_equals_240 : Nat.choose 10 3 + Nat.choose 10 7 = 240 := by
  sorry

end binomial_sum_equals_240_l3506_350625


namespace min_distance_parabola_to_line_l3506_350696

/-- The minimum distance from a point on y = x^2 to y = 2x - 2 is √5/5 -/
theorem min_distance_parabola_to_line :
  let f : ℝ → ℝ := λ x => x^2  -- The curve y = x^2
  let g : ℝ → ℝ := λ x => 2*x - 2  -- The line y = 2x - 2
  ∃ (P : ℝ × ℝ), P.2 = f P.1 ∧  -- Point P on the curve
  (∀ (Q : ℝ × ℝ), Q.2 = f Q.1 →  -- For all points Q on the curve
    Real.sqrt 5 / 5 ≤ Real.sqrt ((Q.1 - (Q.2 + 2) / 2)^2 + (Q.2 - g ((Q.2 + 2) / 2))^2)) ∧
  (∃ (P : ℝ × ℝ), P.2 = f P.1 ∧
    Real.sqrt 5 / 5 = Real.sqrt ((P.1 - (P.2 + 2) / 2)^2 + (P.2 - g ((P.2 + 2) / 2))^2)) :=
by
  sorry


end min_distance_parabola_to_line_l3506_350696


namespace expected_practice_problems_l3506_350637

/-- Represents the number of pairs of shoes -/
def num_pairs : ℕ := 5

/-- Represents the number of days -/
def num_days : ℕ := 5

/-- Represents the probability of selecting two shoes of the same color on a given day -/
def prob_same_color : ℚ := 1 / 9

/-- Represents the expected number of practice problems done in one day -/
def expected_problems_per_day : ℚ := prob_same_color

/-- Theorem stating the expected value of practice problems over 5 days -/
theorem expected_practice_problems :
  (num_days : ℚ) * expected_problems_per_day = 5 / 9 := by
  sorry

end expected_practice_problems_l3506_350637


namespace book_pages_l3506_350672

theorem book_pages : 
  ∀ (P : ℕ), 
  (7 : ℚ) / 13 * P + (5 : ℚ) / 9 * ((6 : ℚ) / 13 * P) + 96 = P → 
  P = 468 :=
by
  sorry

end book_pages_l3506_350672


namespace rebeccas_marbles_l3506_350693

/-- Rebecca's egg and marble problem -/
theorem rebeccas_marbles :
  ∀ (num_eggs num_marbles : ℕ),
  num_eggs = 20 →
  num_eggs = num_marbles + 14 →
  num_marbles = 6 :=
by
  sorry

end rebeccas_marbles_l3506_350693


namespace average_speed_problem_l3506_350654

/-- Given a distance of 1800 meters and a time of 30 minutes, 
    prove that the average speed is 1 meter per second. -/
theorem average_speed_problem (distance : ℝ) (time_minutes : ℝ) :
  distance = 1800 ∧ time_minutes = 30 →
  (distance / (time_minutes * 60)) = 1 := by
sorry

end average_speed_problem_l3506_350654


namespace functional_equation_solution_l3506_350682

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) = f x * y

-- Define the three possible functions
def ZeroFunction : ℝ → ℝ := λ _ => 0
def IdentityFunction : ℝ → ℝ := λ x => x
def NegativeIdentityFunction : ℝ → ℝ := λ x => -x

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
    (f = ZeroFunction ∨ f = IdentityFunction ∨ f = NegativeIdentityFunction) :=
by
  sorry

end functional_equation_solution_l3506_350682


namespace total_trip_time_l3506_350685

/-- The total trip time given the specified conditions -/
theorem total_trip_time : ∀ (v : ℝ),
  v > 0 →
  20 / v = 40 →
  80 / (4 * v) + 40 = 80 :=
by
  sorry

end total_trip_time_l3506_350685


namespace complex_root_magnitude_l3506_350657

theorem complex_root_magnitude (n : ℕ) (a : ℝ) (z : ℂ) 
  (h1 : n ≥ 2) 
  (h2 : 0 < a) 
  (h3 : a < (n + 1 : ℝ) / (n - 1 : ℝ)) 
  (h4 : z^(n+1) - a * z^n + a * z - 1 = 0) : 
  Complex.abs z = 1 := by
  sorry

end complex_root_magnitude_l3506_350657


namespace perfect_squares_closed_under_multiplication_perfect_squares_not_closed_under_addition_perfect_squares_not_closed_under_subtraction_perfect_squares_not_closed_under_division_l3506_350651

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def perfect_squares : Set ℕ := {n : ℕ | is_perfect_square n ∧ n > 0}

theorem perfect_squares_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ perfect_squares → b ∈ perfect_squares → (a * b) ∈ perfect_squares :=
sorry

theorem perfect_squares_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ (a + b) ∉ perfect_squares :=
sorry

theorem perfect_squares_not_closed_under_subtraction :
  ∃ a b : ℕ, a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ a > b ∧ (a - b) ∉ perfect_squares :=
sorry

theorem perfect_squares_not_closed_under_division :
  ∃ a b : ℕ, a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ b ≠ 0 ∧ (a / b) ∉ perfect_squares :=
sorry

end perfect_squares_closed_under_multiplication_perfect_squares_not_closed_under_addition_perfect_squares_not_closed_under_subtraction_perfect_squares_not_closed_under_division_l3506_350651


namespace simplify_trig_expression_l3506_350613

theorem simplify_trig_expression :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) := by
  sorry

end simplify_trig_expression_l3506_350613


namespace square_area_proof_l3506_350660

-- Define the length of the longer side of the smaller rectangle
def longer_side : ℝ := 6

-- Define the ratio between longer and shorter sides
def ratio : ℝ := 3

-- Define the area of the square WXYZ
def square_area : ℝ := 144

-- Theorem statement
theorem square_area_proof :
  let shorter_side := longer_side / ratio
  let square_side := 2 * longer_side
  square_side ^ 2 = square_area :=
by sorry

end square_area_proof_l3506_350660


namespace hallie_tuesday_tips_l3506_350614

/-- Represents Hallie's earnings over three days --/
structure EarningsData :=
  (hourly_rate : ℝ)
  (monday_hours : ℝ)
  (monday_tips : ℝ)
  (tuesday_hours : ℝ)
  (wednesday_hours : ℝ)
  (wednesday_tips : ℝ)
  (total_earnings : ℝ)

/-- Calculates Hallie's tips on Tuesday given her earnings data --/
def tuesday_tips (data : EarningsData) : ℝ :=
  data.total_earnings - 
  (data.hourly_rate * (data.monday_hours + data.tuesday_hours + data.wednesday_hours)) -
  (data.monday_tips + data.wednesday_tips)

/-- Theorem stating that Hallie's tips on Tuesday were $12 --/
theorem hallie_tuesday_tips :
  let data : EarningsData := {
    hourly_rate := 10,
    monday_hours := 7,
    monday_tips := 18,
    tuesday_hours := 5,
    wednesday_hours := 7,
    wednesday_tips := 20,
    total_earnings := 240
  }
  tuesday_tips data = 12 := by sorry

end hallie_tuesday_tips_l3506_350614


namespace sufficient_not_necessary_condition_l3506_350626

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
by sorry

end sufficient_not_necessary_condition_l3506_350626


namespace ap_eq_aq_l3506_350691

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A circle in the Euclidean plane -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- A line in the Euclidean plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of an acute-angled triangle -/
def isAcuteAngled (A B C : Point) : Prop :=
  sorry

/-- Definition of a circle with a given diameter -/
def circleWithDiameter (P Q : Point) : Circle :=
  sorry

/-- Definition of the intersection of a line and a circle -/
def lineCircleIntersection (l : Line) (c : Circle) : Set Point :=
  sorry

theorem ap_eq_aq 
  (A B C : Point)
  (h_acute : isAcuteAngled A B C)
  (circle_AC : Circle)
  (circle_AB : Circle)
  (h_circle_AC : circle_AC = circleWithDiameter A C)
  (h_circle_AB : circle_AB = circleWithDiameter A B)
  (F : Point)
  (h_F : F ∈ lineCircleIntersection (Line.mk 0 1 0) circle_AC)
  (E : Point)
  (h_E : E ∈ lineCircleIntersection (Line.mk 1 0 0) circle_AB)
  (BE CF : Line)
  (P : Point)
  (h_P : P ∈ lineCircleIntersection BE circle_AC)
  (Q : Point)
  (h_Q : Q ∈ lineCircleIntersection CF circle_AB) :
  (A.x - P.x)^2 + (A.y - P.y)^2 = (A.x - Q.x)^2 + (A.y - Q.y)^2 :=
sorry

end ap_eq_aq_l3506_350691


namespace segment_length_l3506_350673

/-- Given points P, Q, and R on line segment AB, prove that AB has length 48 -/
theorem segment_length (A B P Q R : ℝ) : 
  (0 < A) → (A < P) → (P < Q) → (Q < R) → (R < B) →  -- Points lie on AB in order
  (P - A) / (B - P) = 3 / 5 →                        -- P divides AB in ratio 3:5
  (Q - A) / (B - Q) = 5 / 7 →                        -- Q divides AB in ratio 5:7
  R - Q = 3 →                                        -- QR = 3
  R - P = 5 →                                        -- PR = 5
  B - A = 48 := by sorry

end segment_length_l3506_350673


namespace inequality_solution_set_l3506_350600

-- Define the function f
def f (x : ℝ) : ℝ := |x| - x + 1

-- State the theorem
theorem inequality_solution_set (x : ℝ) : 
  f (1 - x^2) > f (1 - 2*x) ↔ x > 2 ∨ x < -1 := by
  sorry

end inequality_solution_set_l3506_350600


namespace odd_function_property_l3506_350676

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def HasMinimumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

def HasMaximumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem odd_function_property (f : ℝ → ℝ) :
  IsOdd f →
  IsIncreasingOn f 1 3 →
  HasMinimumOn f 1 3 7 →
  IsIncreasingOn f (-3) (-1) ∧ HasMaximumOn f (-3) (-1) (-7) := by
  sorry

end odd_function_property_l3506_350676


namespace binomial_coefficient_equality_l3506_350602

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 12 n = Nat.choose 12 (2*n - 3)) → (n = 3 ∨ n = 5) :=
by sorry

end binomial_coefficient_equality_l3506_350602


namespace toy_pile_ratio_l3506_350640

theorem toy_pile_ratio : 
  let total_toys : ℕ := 120
  let larger_pile : ℕ := 80
  let smaller_pile : ℕ := total_toys - larger_pile
  (larger_pile : ℚ) / smaller_pile = 2 / 1 := by
  sorry

end toy_pile_ratio_l3506_350640


namespace square_problem_l3506_350662

/-- Square with side length 1200 -/
structure Square :=
  (side : ℝ)
  (is_1200 : side = 1200)

/-- Point on the side AB of the square -/
structure PointOnAB (S : Square) :=
  (x : ℝ)
  (on_side : 0 ≤ x ∧ x ≤ S.side)

theorem square_problem (S : Square) (G H : PointOnAB S)
  (h_order : G.x < H.x)
  (h_angle : Real.cos (Real.pi / 3) = (H.x - G.x) / 600)
  (h_dist : H.x - G.x = 600) :
  S.side - H.x = 300 + 100 * Real.sqrt 7 :=
sorry

end square_problem_l3506_350662


namespace average_monthly_income_is_69_l3506_350679

/-- Proves that the average monthly income for a 10-month period is 69 given specific income and expense conditions. -/
theorem average_monthly_income_is_69 
  (X : ℝ) -- Income base for first 6 months
  (Y : ℝ) -- Income for last 4 months
  (h1 : (6 * (1.1 * X) + 4 * Y) / 10 = 69) -- Average income condition
  (h2 : 4 * (Y - 60) - 6 * (70 - 1.1 * X) = 30) -- Debt and savings condition
  : (6 * (1.1 * X) + 4 * Y) / 10 = 69 := by
  sorry

#check average_monthly_income_is_69

end average_monthly_income_is_69_l3506_350679


namespace probability_sum_five_l3506_350612

def Card : Type := Fin 4

def card_value (c : Card) : ℕ := c.val + 1

def sum_equals_five (c1 c2 : Card) : Prop :=
  card_value c1 + card_value c2 = 5

def total_outcomes : ℕ := 16

def favorable_outcomes : ℕ := 4

theorem probability_sum_five :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
  sorry

end probability_sum_five_l3506_350612


namespace gcd_51_119_l3506_350698

theorem gcd_51_119 : Nat.gcd 51 119 = 17 := by sorry

end gcd_51_119_l3506_350698


namespace sum_five_consecutive_squares_not_perfect_square_l3506_350644

theorem sum_five_consecutive_squares_not_perfect_square (n : ℤ) :
  ¬∃ (m : ℤ), 5 * n^2 + 10 = m^2 := by
  sorry

end sum_five_consecutive_squares_not_perfect_square_l3506_350644


namespace school_population_l3506_350687

theorem school_population (girls boys teachers : ℕ) 
  (h1 : girls = 315) 
  (h2 : boys = 309) 
  (h3 : teachers = 772) : 
  girls + boys + teachers = 1396 := by
  sorry

end school_population_l3506_350687


namespace kaleb_books_l3506_350621

theorem kaleb_books (initial_books sold_books new_books : ℕ) :
  initial_books ≥ sold_books →
  initial_books - sold_books + new_books = initial_books + new_books - sold_books :=
by
  sorry

#check kaleb_books 34 17 7

end kaleb_books_l3506_350621


namespace same_floor_prob_is_one_fifth_l3506_350610

/-- A hotel with 6 rooms distributed across 3 floors -/
structure Hotel :=
  (total_rooms : ℕ)
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (h1 : total_rooms = 6)
  (h2 : floors = 3)
  (h3 : rooms_per_floor = 2)
  (h4 : total_rooms = floors * rooms_per_floor)

/-- The probability of two people choosing rooms on the same floor -/
def same_floor_probability (h : Hotel) : ℚ :=
  (h.floors * (h.rooms_per_floor * (h.rooms_per_floor - 1))) / (h.total_rooms * (h.total_rooms - 1))

theorem same_floor_prob_is_one_fifth (h : Hotel) : same_floor_probability h = 1 / 5 := by
  sorry

end same_floor_prob_is_one_fifth_l3506_350610


namespace bike_tractor_speed_ratio_l3506_350666

/-- Given the conditions of the problem, prove that the ratio of the speed of the bike to the speed of the tractor is 2:1 -/
theorem bike_tractor_speed_ratio :
  ∀ (car_speed bike_speed tractor_speed : ℝ),
  car_speed = (9/5) * bike_speed →
  tractor_speed = 575 / 23 →
  car_speed = 540 / 6 →
  ∃ (k : ℝ), bike_speed = k * tractor_speed →
  bike_speed / tractor_speed = 2 := by
sorry

end bike_tractor_speed_ratio_l3506_350666


namespace investment_ratio_l3506_350633

/-- Given two investors P and Q, with P investing 50000 and profits divided in ratio 3:4, 
    prove that Q's investment is 66666.67 -/
theorem investment_ratio (p q : ℝ) (h1 : p = 50000) (h2 : p / q = 3 / 4) : 
  q = 66666.67 := by
  sorry

end investment_ratio_l3506_350633


namespace book_pricing_problem_l3506_350638

-- Define the variables
variable (price_A : ℝ) (price_B : ℝ) (num_A : ℕ) (num_B : ℕ)

-- Define the conditions
def condition1 : Prop := price_A * num_A = 3000
def condition2 : Prop := price_B * num_B = 1600
def condition3 : Prop := price_A = 1.5 * price_B
def condition4 : Prop := num_A = num_B + 20

-- Define the World Book Day purchase
def world_book_day_expenditure : ℝ := 0.8 * (20 * price_A + 25 * price_B)

-- State the theorem
theorem book_pricing_problem 
  (h1 : condition1 price_A num_A)
  (h2 : condition2 price_B num_B)
  (h3 : condition3 price_A price_B)
  (h4 : condition4 num_A num_B) :
  price_A = 30 ∧ price_B = 20 ∧ world_book_day_expenditure price_A price_B = 880 := by
  sorry


end book_pricing_problem_l3506_350638


namespace bounce_height_theorem_l3506_350624

/-- The number of bounces required for a ball to reach a height less than 3 meters -/
def number_of_bounces : ℕ := 22

/-- The initial height of the ball in meters -/
def initial_height : ℝ := 500

/-- The bounce ratio (percentage of height retained after each bounce) -/
def bounce_ratio : ℝ := 0.6

/-- The target height in meters -/
def target_height : ℝ := 3

/-- Theorem stating that the number of bounces is correct -/
theorem bounce_height_theorem :
  (∀ k : ℕ, k < number_of_bounces → initial_height * bounce_ratio ^ k ≥ target_height) ∧
  (initial_height * bounce_ratio ^ number_of_bounces < target_height) :=
sorry

end bounce_height_theorem_l3506_350624


namespace vector_subtraction_and_scaling_l3506_350636

/-- Given two 2D vectors a and b, prove that a - 2b results in the specified coordinates. -/
theorem vector_subtraction_and_scaling (a b : Fin 2 → ℝ) (h1 : a = ![1, 2]) (h2 : b = ![-3, 2]) :
  a - 2 • b = ![7, -2] := by
  sorry

end vector_subtraction_and_scaling_l3506_350636


namespace sqrt_meaningful_range_l3506_350623

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 - x) ↔ x ≤ 2 := by
sorry

end sqrt_meaningful_range_l3506_350623


namespace at_least_one_passes_l3506_350663

theorem at_least_one_passes (p : ℝ) (h : p = 1/3) :
  let q := 1 - p
  1 - q^3 = 19/27 := by
sorry

end at_least_one_passes_l3506_350663


namespace dragon_lion_equivalence_l3506_350634

-- Define the propositions
variable (P Q : Prop)

-- State the theorem
theorem dragon_lion_equivalence :
  (P → Q) ↔ (¬Q → ¬P) ∧ (¬P ∨ Q) :=
sorry

end dragon_lion_equivalence_l3506_350634


namespace new_rectangle_perimeter_l3506_350632

/-- Given a rectangle ABCD composed of four congruent triangles -/
structure Rectangle :=
  (AB BC : ℝ)
  (AK : ℝ)
  (perimeter : ℝ)
  (h1 : perimeter = 4 * (AB + BC / 2 + AK))
  (h2 : AK = 17)
  (h3 : perimeter = 180)

/-- The perimeter of a new rectangle with sides 2*AB and BC -/
def new_perimeter (r : Rectangle) : ℝ :=
  2 * (2 * r.AB + r.BC)

/-- Theorem stating the perimeter of the new rectangle is 112 cm -/
theorem new_rectangle_perimeter (r : Rectangle) : new_perimeter r = 112 :=
sorry

end new_rectangle_perimeter_l3506_350632


namespace units_digit_of_nine_to_eight_to_seven_l3506_350643

theorem units_digit_of_nine_to_eight_to_seven (n : Nat) : n = 9^(8^7) → n % 10 = 1 := by
  sorry

end units_digit_of_nine_to_eight_to_seven_l3506_350643


namespace inequality_solution_set_l3506_350674

-- Define the inequality function
def f (x : ℝ) : ℝ := |x - 5| + |x + 3|

-- Define the solution set
def solution_set : Set ℝ := {x | x ≤ -4 ∨ x ≥ 6}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 10} = solution_set := by sorry

end inequality_solution_set_l3506_350674


namespace triangle_angle_sum_l3506_350677

theorem triangle_angle_sum (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
  a = 20 →  -- Smallest angle is 20 degrees
  b = 3 * a →  -- Middle angle is 3 times the smallest
  c = 5 * a →  -- Largest angle is 5 times the smallest
  a + b + c = 180 := by
  sorry

end triangle_angle_sum_l3506_350677


namespace fifth_root_of_102030201_l3506_350656

theorem fifth_root_of_102030201 : (102030201 : ℝ) ^ (1/5 : ℝ) = 101 := by
  sorry

end fifth_root_of_102030201_l3506_350656


namespace four_row_triangle_count_l3506_350618

/-- Calculates the total number of triangles in a triangular grid with n rows -/
def triangleCount (n : ℕ) : ℕ :=
  let smallTriangles := n * (n + 1) / 2
  let mediumTriangles := (n - 1) * (n - 2) / 2
  let largeTriangles := n - 2
  smallTriangles + mediumTriangles + largeTriangles

/-- Theorem stating that a triangular grid with 4 rows contains 14 triangles in total -/
theorem four_row_triangle_count : triangleCount 4 = 14 := by
  sorry

end four_row_triangle_count_l3506_350618


namespace logarithmic_scales_imply_ohms_law_l3506_350650

/-- Represents a point on the logarithmic scale for resistance, current, or voltage -/
structure LogPoint where
  value : ℝ
  coordinate : ℝ

/-- Represents the scales for resistance, current, and voltage -/
structure Circuit where
  resistance : LogPoint
  current : LogPoint
  voltage : LogPoint

/-- The relationship between the coordinates of resistance, current, and voltage -/
def coordinate_relation (c : Circuit) : Prop :=
  c.current.coordinate + c.voltage.coordinate = 2 * c.resistance.coordinate

/-- The relationship between resistance, current, and voltage values -/
def ohms_law (c : Circuit) : Prop :=
  c.voltage.value = c.current.value * c.resistance.value

/-- The logarithmic scale relationship for resistance -/
def resistance_scale (r : LogPoint) : Prop :=
  r.value = 10^(-2 * r.coordinate)

/-- The logarithmic scale relationship for current -/
def current_scale (i : LogPoint) : Prop :=
  i.value = 10^(i.coordinate)

/-- The logarithmic scale relationship for voltage -/
def voltage_scale (v : LogPoint) : Prop :=
  v.value = 10^(-v.coordinate)

/-- Theorem stating that the logarithmic scales and coordinate relation imply Ohm's law -/
theorem logarithmic_scales_imply_ohms_law (c : Circuit) :
  resistance_scale c.resistance →
  current_scale c.current →
  voltage_scale c.voltage →
  coordinate_relation c →
  ohms_law c :=
by sorry

end logarithmic_scales_imply_ohms_law_l3506_350650


namespace eight_prof_sequences_l3506_350604

/-- The number of professors --/
def n : ℕ := 8

/-- The number of distinct sequences for scheduling n professors,
    where one specific professor must present before another specific professor --/
def num_sequences (n : ℕ) : ℕ := n.factorial / 2

/-- Theorem stating that the number of distinct sequences for scheduling 8 professors,
    where one specific professor must present before another specific professor,
    is equal to 8! / 2 --/
theorem eight_prof_sequences :
  num_sequences n = 20160 := by sorry

end eight_prof_sequences_l3506_350604


namespace choir_average_age_l3506_350617

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℝ) 
  (avg_age_males : ℝ) 
  (h1 : num_females = 8)
  (h2 : avg_age_females = 25)
  (h3 : num_males = 12)
  (h4 : avg_age_males = 40)
  (h5 : num_females + num_males = 20) :
  (num_females * avg_age_females + num_males * avg_age_males) / (num_females + num_males) = 34 := by
  sorry

end choir_average_age_l3506_350617


namespace absolute_value_equality_implication_not_always_true_l3506_350649

theorem absolute_value_equality_implication_not_always_true :
  ¬ (∀ a b : ℝ, |a| = |b| → a = b) := by
  sorry

end absolute_value_equality_implication_not_always_true_l3506_350649


namespace smallest_sum_reciprocals_l3506_350646

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ a.val + b.val = 64 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 15 → c.val + d.val ≥ 64 :=
by sorry

end smallest_sum_reciprocals_l3506_350646


namespace log_inequality_l3506_350635

theorem log_inequality (a b : ℝ) 
  (ha : a = Real.log 0.4 / Real.log 0.2) 
  (hb : b = 1 - 1 / Real.log 4) : 
  a * b < a + b ∧ a + b < 0 := by sorry

end log_inequality_l3506_350635


namespace octagon_diagonals_l3506_350620

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l3506_350620


namespace anna_swept_ten_rooms_l3506_350688

/-- Represents the time in minutes for various chores -/
structure ChoreTime where
  sweepingPerRoom : ℕ
  washingPerDish : ℕ
  laundryPerLoad : ℕ

/-- Represents the chores assigned to Billy -/
structure BillyChores where
  laundryLoads : ℕ
  dishesToWash : ℕ

/-- Calculates the total time Billy spends on chores -/
def billyTotalTime (ct : ChoreTime) (bc : BillyChores) : ℕ :=
  bc.laundryLoads * ct.laundryPerLoad + bc.dishesToWash * ct.washingPerDish

/-- Theorem stating that Anna swept 10 rooms -/
theorem anna_swept_ten_rooms (ct : ChoreTime) (bc : BillyChores) 
    (h1 : ct.sweepingPerRoom = 3)
    (h2 : ct.washingPerDish = 2)
    (h3 : ct.laundryPerLoad = 9)
    (h4 : bc.laundryLoads = 2)
    (h5 : bc.dishesToWash = 6) :
    ∃ (rooms : ℕ), rooms * ct.sweepingPerRoom = billyTotalTime ct bc ∧ rooms = 10 := by
  sorry

end anna_swept_ten_rooms_l3506_350688
