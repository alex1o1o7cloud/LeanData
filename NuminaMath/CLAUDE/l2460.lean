import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_curve_is_correct_l2460_246015

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the symmetric curve
def symmetric_curve (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 1

-- Theorem statement
theorem symmetric_curve_is_correct : 
  ∀ (x y x' y' : ℝ), 
    given_circle x y → 
    symmetry_line ((x + x') / 2) ((y + y') / 2) → 
    symmetric_curve x' y' :=
sorry

end NUMINAMATH_CALUDE_symmetric_curve_is_correct_l2460_246015


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2460_246058

/-- Given that 3 + √5 is a root of x³ + cx² + dx + 15 = 0 where c and d are rational,
    prove that d = -18.5 -/
theorem cubic_equation_root (c d : ℚ) 
  (h : (3 + Real.sqrt 5)^3 + c * (3 + Real.sqrt 5)^2 + d * (3 + Real.sqrt 5) + 15 = 0) :
  d = -37/2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2460_246058


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2460_246084

theorem triangle_perimeter (a b c : ℝ) (h1 : a = 4000) (h2 : b = 3500) 
  (h3 : c^2 = a^2 - b^2) : a + b + c = 9437 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2460_246084


namespace NUMINAMATH_CALUDE_inverse_functions_l2460_246023

-- Define the type for our functions
def Function : Type := ℝ → ℝ

-- Define the property of having an inverse
def has_inverse (f : Function) : Prop := ∃ g : Function, ∀ x, g (f x) = x ∧ f (g x) = x

-- Define our functions based on their graphical properties
def F : Function := sorry
def G : Function := sorry
def H : Function := sorry
def I : Function := sorry

-- State the theorem
theorem inverse_functions :
  (has_inverse F) ∧ (has_inverse H) ∧ (has_inverse I) ∧ ¬(has_inverse G) := by sorry

end NUMINAMATH_CALUDE_inverse_functions_l2460_246023


namespace NUMINAMATH_CALUDE_box_volume_and_area_l2460_246055

/-- A rectangular box with given dimensions -/
structure RectangularBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a rectangular box -/
def volume (box : RectangularBox) : ℝ :=
  box.length * box.width * box.height

/-- Calculate the maximum ground area of a rectangular box -/
def maxGroundArea (box : RectangularBox) : ℝ :=
  box.length * box.width

/-- Theorem about the volume and maximum ground area of a specific rectangular box -/
theorem box_volume_and_area (box : RectangularBox)
    (h1 : box.length = 20)
    (h2 : box.width = 15)
    (h3 : box.height = 5) :
    volume box = 1500 ∧ maxGroundArea box = 300 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_and_area_l2460_246055


namespace NUMINAMATH_CALUDE_larger_number_problem_l2460_246021

theorem larger_number_problem (x y : ℝ) (h_sum : x + y = 30) (h_diff : x - y = 14) :
  max x y = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2460_246021


namespace NUMINAMATH_CALUDE_odd_increasing_function_inequality_l2460_246037

-- Define the properties of function f
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- State the theorem
theorem odd_increasing_function_inequality (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_incr : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f (2*x - 1) < f (x^2 - x + 1)} = 
  Set.union (Set.Iio 1) (Set.Ioi 2) := by sorry

end NUMINAMATH_CALUDE_odd_increasing_function_inequality_l2460_246037


namespace NUMINAMATH_CALUDE_sin_greater_than_cos_l2460_246034

theorem sin_greater_than_cos (x : Real) (h : -7*Real.pi/4 < x ∧ x < -3*Real.pi/2) :
  Real.sin (x + 9*Real.pi/4) > Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_greater_than_cos_l2460_246034


namespace NUMINAMATH_CALUDE_number_value_l2460_246007

theorem number_value (N : ℝ) (h : (1/2) * N = 1) : N = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_value_l2460_246007


namespace NUMINAMATH_CALUDE_sqrt_19_minus_1_squared_plus_2x_plus_2_l2460_246014

theorem sqrt_19_minus_1_squared_plus_2x_plus_2 :
  let x : ℝ := Real.sqrt 19 - 1
  x^2 + 2*x + 2 = 20 := by
sorry

end NUMINAMATH_CALUDE_sqrt_19_minus_1_squared_plus_2x_plus_2_l2460_246014


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l2460_246077

theorem jelly_bean_probability (p_red p_orange p_yellow p_green : ℝ) :
  p_red = 0.25 →
  p_orange = 0.35 →
  p_red + p_orange + p_yellow + p_green = 1 →
  p_red ≥ 0 ∧ p_orange ≥ 0 ∧ p_yellow ≥ 0 ∧ p_green ≥ 0 →
  p_yellow = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l2460_246077


namespace NUMINAMATH_CALUDE_total_students_l2460_246000

/-- The total number of students in five classes given specific conditions -/
theorem total_students (finley johnson garcia smith patel : ℕ) : 
  finley = 24 →
  johnson = finley / 2 + 10 →
  garcia = 2 * johnson →
  smith = finley / 3 →
  patel = (3 * (finley + johnson + garcia)) / 4 →
  finley + johnson + garcia + smith + patel = 166 := by
sorry

end NUMINAMATH_CALUDE_total_students_l2460_246000


namespace NUMINAMATH_CALUDE_eight_hash_six_l2460_246050

/-- Definition of the # operation -/
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

/-- First condition: r # 0 = r + 1 -/
axiom hash_zero (r : ℝ) : hash r 0 = r + 1

/-- Second condition: r # s = s # r -/
axiom hash_comm (r s : ℝ) : hash r s = hash s r

/-- Third condition: (r + 1) # s = (r # s) + s + 2 -/
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 2

/-- The main theorem to prove -/
theorem eight_hash_six : hash 8 6 = 69 :=
  sorry

end NUMINAMATH_CALUDE_eight_hash_six_l2460_246050


namespace NUMINAMATH_CALUDE_expression_value_l2460_246017

theorem expression_value (x y z w : ℝ) 
  (eq1 : 4 * x * z + y * w = 4) 
  (eq2 : x * w + y * z = 8) : 
  (2 * x + y) * (2 * z + w) = 20 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2460_246017


namespace NUMINAMATH_CALUDE_min_students_theorem_l2460_246069

/-- Given a class of students, returns the minimum number of students
    who have brown eyes, a lunch box, and do not wear glasses. -/
def min_students_with_characteristics (total : ℕ) (brown_eyes : ℕ) (lunch_box : ℕ) (glasses : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of students with the given characteristics -/
theorem min_students_theorem :
  min_students_with_characteristics 40 18 25 16 = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_students_theorem_l2460_246069


namespace NUMINAMATH_CALUDE_sarah_apple_ratio_l2460_246093

theorem sarah_apple_ratio : 
  let sarah_apples : ℝ := 45.0
  let brother_apples : ℝ := 9.0
  sarah_apples / brother_apples = 5 := by
sorry

end NUMINAMATH_CALUDE_sarah_apple_ratio_l2460_246093


namespace NUMINAMATH_CALUDE_development_inheritance_relationship_false_l2460_246006

/-- Development is a prerequisite for inheritance -/
def development_prerequisite_for_inheritance : Prop := sorry

/-- Inheritance is a requirement for development -/
def inheritance_requirement_for_development : Prop := sorry

/-- The statement that development is a prerequisite for inheritance
    and inheritance is a requirement for development is false -/
theorem development_inheritance_relationship_false :
  ¬(development_prerequisite_for_inheritance ∧ inheritance_requirement_for_development) :=
sorry

end NUMINAMATH_CALUDE_development_inheritance_relationship_false_l2460_246006


namespace NUMINAMATH_CALUDE_lily_received_35_books_l2460_246040

/-- The number of books Lily received -/
def books_lily_received (mike_books_tuesday : ℕ) (corey_books_tuesday : ℕ) (mike_gave : ℕ) (corey_gave_extra : ℕ) : ℕ :=
  mike_gave + (mike_gave + corey_gave_extra)

/-- Theorem stating that Lily received 35 books -/
theorem lily_received_35_books :
  ∀ (mike_books_tuesday corey_books_tuesday mike_gave corey_gave_extra : ℕ),
    mike_books_tuesday = 45 →
    corey_books_tuesday = 2 * mike_books_tuesday →
    mike_gave = 10 →
    corey_gave_extra = 15 →
    books_lily_received mike_books_tuesday corey_books_tuesday mike_gave corey_gave_extra = 35 := by
  sorry

#eval books_lily_received 45 90 10 15

end NUMINAMATH_CALUDE_lily_received_35_books_l2460_246040


namespace NUMINAMATH_CALUDE_painted_cube_problem_l2460_246057

theorem painted_cube_problem (n : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 →
  n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l2460_246057


namespace NUMINAMATH_CALUDE_successive_numbers_product_l2460_246090

theorem successive_numbers_product (n : ℕ) : 
  n * (n + 1) = 2652 → n = 51 := by
  sorry

end NUMINAMATH_CALUDE_successive_numbers_product_l2460_246090


namespace NUMINAMATH_CALUDE_tree_house_wood_theorem_l2460_246045

/-- The total amount of wood needed for John's tree house -/
def total_wood_needed : ℝ :=
  let pillar_short := 4
  let pillar_long := 5 * pillar_short
  let wall_long := 6
  let wall_short := wall_long - 3
  let floor_wood := 5.5
  let roof_long := 2 * floor_wood
  let roof_short := 1.5 * floor_wood
  4 * pillar_short + 4 * pillar_long +
  10 * wall_long + 10 * wall_short +
  8 * floor_wood +
  6 * roof_long + 6 * roof_short

/-- Theorem stating the total amount of wood needed for John's tree house -/
theorem tree_house_wood_theorem : total_wood_needed = 345.5 := by
  sorry

end NUMINAMATH_CALUDE_tree_house_wood_theorem_l2460_246045


namespace NUMINAMATH_CALUDE_volume_of_one_gram_volume_of_one_gram_substance_l2460_246010

-- Define the constants from the problem
def mass_per_cubic_meter : ℝ := 300
def grams_per_kilogram : ℝ := 1000
def cubic_cm_per_cubic_meter : ℝ := 1000000

-- Define the theorem
theorem volume_of_one_gram (mass_per_cubic_meter : ℝ) (grams_per_kilogram : ℝ) (cubic_cm_per_cubic_meter : ℝ) :
  mass_per_cubic_meter * grams_per_kilogram > 0 →
  cubic_cm_per_cubic_meter / (mass_per_cubic_meter * grams_per_kilogram) = 10 / 3 := by
  sorry

-- Apply the theorem to our specific values
theorem volume_of_one_gram_substance :
  cubic_cm_per_cubic_meter / (mass_per_cubic_meter * grams_per_kilogram) = 10 / 3 := by
  apply volume_of_one_gram mass_per_cubic_meter grams_per_kilogram cubic_cm_per_cubic_meter
  -- Prove that mass_per_cubic_meter * grams_per_kilogram > 0
  sorry

end NUMINAMATH_CALUDE_volume_of_one_gram_volume_of_one_gram_substance_l2460_246010


namespace NUMINAMATH_CALUDE_box_length_l2460_246097

/-- Given a box with width 16 units and height 13 units, which can contain 3120 unit cubes (1 x 1 x 1), prove that the length of the box is 15 units. -/
theorem box_length (width : ℕ) (height : ℕ) (volume : ℕ) (length : ℕ) : 
  width = 16 → height = 13 → volume = 3120 → volume = length * width * height → length = 15 := by
  sorry

end NUMINAMATH_CALUDE_box_length_l2460_246097


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2460_246066

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 50 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2460_246066


namespace NUMINAMATH_CALUDE_find_set_M_l2460_246029

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}

def complement_M : Finset ℕ := {1, 2, 4}

theorem find_set_M : 
  ∀ M : Finset ℕ, (∀ x : ℕ, x ∈ U → (x ∈ M ↔ x ∉ complement_M)) → 
  M = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_find_set_M_l2460_246029


namespace NUMINAMATH_CALUDE_a_less_than_b_less_than_c_l2460_246002

theorem a_less_than_b_less_than_c : ∀ a b c : ℝ,
  a = Real.log (1/2) →
  b = Real.sin (1/2) →
  c = 2^(-1/2 : ℝ) →
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_a_less_than_b_less_than_c_l2460_246002


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2460_246036

theorem rectangle_dimensions (x : ℝ) : 
  (x + 1 > 0) → 
  (3*x - 4 > 0) → 
  (x + 1) * (3*x - 4) = 12*x - 19 → 
  x = (13 + Real.sqrt 349) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2460_246036


namespace NUMINAMATH_CALUDE_wire_length_ratio_l2460_246076

/-- Represents the length of a single piece of wire used by Bonnie -/
def bonnie_wire_length : ℝ := 4

/-- Represents the number of wire pieces used by Bonnie to construct her cube -/
def bonnie_wire_count : ℕ := 12

/-- Represents the length of a single piece of wire used by Roark -/
def roark_wire_length : ℝ := 1

/-- Represents the side length of Bonnie's cube -/
def bonnie_cube_side : ℝ := bonnie_wire_length

/-- Represents the volume of a single unit cube constructed by Roark -/
def unit_cube_volume : ℝ := 1

/-- Theorem stating that the ratio of Bonnie's total wire length to Roark's total wire length is 1/16 -/
theorem wire_length_ratio :
  (bonnie_wire_length * bonnie_wire_count) / 
  (roark_wire_length * (12 * (bonnie_cube_side ^ 3))) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l2460_246076


namespace NUMINAMATH_CALUDE_no_integer_solution_l2460_246089

theorem no_integer_solution : ∀ x y : ℤ, 2 * x^2 - 5 * y^2 ≠ 7 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2460_246089


namespace NUMINAMATH_CALUDE_bianca_drawing_time_l2460_246070

/-- The total time Bianca spent drawing is equal to 41 minutes, given that she spent 22 minutes drawing at school and 19 minutes drawing at home. -/
theorem bianca_drawing_time (time_at_school time_at_home : ℕ) 
  (h1 : time_at_school = 22)
  (h2 : time_at_home = 19) :
  time_at_school + time_at_home = 41 := by
  sorry

end NUMINAMATH_CALUDE_bianca_drawing_time_l2460_246070


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2460_246060

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧ 
  Real.tan (150 * π / 180 - x * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 120 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2460_246060


namespace NUMINAMATH_CALUDE_equation_solution_l2460_246065

theorem equation_solution : ∃ x : ℚ, x + 5/8 = 7/24 + 1/4 ∧ x = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2460_246065


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2460_246061

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := (1 + i) * z = 2 * i

-- Theorem statement
theorem complex_equation_solution :
  ∀ z : ℂ, equation z → z = 1 + i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2460_246061


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2460_246048

/-- Given an arithmetic sequence {a_n} where a_5 = 3 and a_9 = 6, prove that a_13 = 9 -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) 
  (h_a5 : a 5 = 3) 
  (h_a9 : a 9 = 6) : 
  a 13 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2460_246048


namespace NUMINAMATH_CALUDE_equation_solution_l2460_246013

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -1 ∧ x₂ = 5 ∧ 
  (∀ x : ℝ, (x + 1)^2 = 6*x + 6 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2460_246013


namespace NUMINAMATH_CALUDE_function_properties_l2460_246027

-- Define the function f(x) and its derivative
def f (x : ℝ) (c : ℝ) : ℝ := x^3 - 3*x^2 + c
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem function_properties :
  -- f'(x) passes through (0,0) and (2,0)
  f_derivative 0 = 0 ∧ f_derivative 2 = 0 ∧
  -- f(x) attains its minimum at x = 2
  (∀ x : ℝ, f x (-1) ≥ f 2 (-1)) ∧
  -- The minimum value is -5
  f 2 (-1) = -5 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2460_246027


namespace NUMINAMATH_CALUDE_sum_set_cardinality_l2460_246094

/-- A function that generates an arithmetic sequence with a given first term, common difference, and length. -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : Fin n → ℝ :=
  fun i => a₁ + d * (i : ℝ)

/-- The set of all sums of pairs of elements from the arithmetic sequence. -/
def sumSet (a₁ : ℝ) (d : ℝ) (n : ℕ) : Set ℝ :=
  {x | ∃ (i j : Fin n), i ≤ j ∧ x = arithmeticSequence a₁ d n i + arithmeticSequence a₁ d n j}

/-- The theorem stating that the number of elements in the sum set is 2n - 3. -/
theorem sum_set_cardinality (a₁ : ℝ) (d : ℝ) (n : ℕ) (h₁ : n ≥ 3) (h₂ : d > 0) :
  Nat.card (sumSet a₁ d n) = 2 * n - 3 :=
sorry

end NUMINAMATH_CALUDE_sum_set_cardinality_l2460_246094


namespace NUMINAMATH_CALUDE_stating_chess_tournament_players_l2460_246087

/-- The number of players in a chess tournament -/
def num_players : ℕ := 17

/-- The total number of games played in the tournament -/
def total_games : ℕ := 272

/-- 
Theorem stating that the number of players in the chess tournament is correct,
given the conditions of the problem.
-/
theorem chess_tournament_players :
  (2 * num_players * (num_players - 1) = total_games) ∧ 
  (∀ n : ℕ, 2 * n * (n - 1) = total_games → n = num_players) := by
  sorry

#check chess_tournament_players

end NUMINAMATH_CALUDE_stating_chess_tournament_players_l2460_246087


namespace NUMINAMATH_CALUDE_beaker_liquid_distribution_l2460_246095

/-- Proves that if 5 ml of liquid is removed from a beaker and 35 ml remains, 
    then the original amount of liquid would have been 8 ml per cup if equally 
    distributed among 5 cups. -/
theorem beaker_liquid_distribution (initial_volume : ℝ) : 
  initial_volume - 5 = 35 → initial_volume / 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_beaker_liquid_distribution_l2460_246095


namespace NUMINAMATH_CALUDE_amp_five_two_l2460_246086

-- Define the & operation
def amp (a b : ℤ) : ℤ := ((a + b) * (a - b))^2

-- Theorem statement
theorem amp_five_two : amp 5 2 = 441 := by
  sorry

end NUMINAMATH_CALUDE_amp_five_two_l2460_246086


namespace NUMINAMATH_CALUDE_max_rabbits_with_traits_l2460_246063

theorem max_rabbits_with_traits :
  ∃ (N : ℕ), N = 27 ∧
  (∀ (n : ℕ), n > N →
    ¬(∃ (long_ears jump_far both : Finset (Fin n)),
      long_ears.card = 13 ∧
      jump_far.card = 17 ∧
      (long_ears ∩ jump_far).card ≥ 3)) ∧
  (∃ (long_ears jump_far both : Finset (Fin N)),
    long_ears.card = 13 ∧
    jump_far.card = 17 ∧
    (long_ears ∩ jump_far).card ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_max_rabbits_with_traits_l2460_246063


namespace NUMINAMATH_CALUDE_definite_integral_cos_zero_l2460_246092

theorem definite_integral_cos_zero : 
  ∫ x in (π/4)..(9*π/4), Real.sqrt 2 * Real.cos (2*x + π/4) = 0 := by sorry

end NUMINAMATH_CALUDE_definite_integral_cos_zero_l2460_246092


namespace NUMINAMATH_CALUDE_solution_range_l2460_246059

theorem solution_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2/x + 1/y = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2/x + 1/y = 1 ∧ 2*x + y < m^2 - 8*m) ↔ 
  (m < -1 ∨ m > 9) :=
sorry

end NUMINAMATH_CALUDE_solution_range_l2460_246059


namespace NUMINAMATH_CALUDE_total_players_l2460_246031

theorem total_players (kabaddi : ℕ) (kho_kho_only : ℕ) (both : ℕ) 
  (h1 : kabaddi = 10) 
  (h2 : kho_kho_only = 25) 
  (h3 : both = 5) : 
  kabaddi + kho_kho_only - both = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l2460_246031


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l2460_246056

/-- Given that x = -1 is a solution of the quadratic equation ax^2 + bx + 23 = 0,
    prove that -a + b + 2000 = 2023 -/
theorem quadratic_solution_implies_sum (a b : ℝ) 
  (h : a * (-1)^2 + b * (-1) + 23 = 0) : 
  -a + b + 2000 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l2460_246056


namespace NUMINAMATH_CALUDE_chess_game_probability_l2460_246005

theorem chess_game_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3)
  (h_not_lose : p_not_lose = 0.8) :
  p_not_lose - p_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l2460_246005


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_eq_one_or_neg_one_l2460_246096

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x ≤ 1 ∧ x - y = -a

-- Define what it means for the system to have a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x y, system x y a

-- Theorem statement
theorem unique_solution_iff_a_eq_one_or_neg_one :
  ∀ a : ℝ, has_unique_solution a ↔ (a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_eq_one_or_neg_one_l2460_246096


namespace NUMINAMATH_CALUDE_probability_is_one_seventh_l2460_246011

/-- The number of students in the group -/
def total_students : ℕ := 7

/-- The number of students who can speak a foreign language -/
def foreign_language_speakers : ℕ := 3

/-- The number of students being selected -/
def selected_students : ℕ := 2

/-- The probability of selecting two students who both speak a foreign language -/
def probability_both_speak_foreign : ℚ :=
  (foreign_language_speakers.choose selected_students) / (total_students.choose selected_students)

theorem probability_is_one_seventh :
  probability_both_speak_foreign = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_seventh_l2460_246011


namespace NUMINAMATH_CALUDE_manager_employee_ratio_l2460_246099

theorem manager_employee_ratio (total_employees : ℕ) (female_managers : ℕ) 
  (h1 : total_employees = 750) (h2 : female_managers = 300) :
  (female_managers : ℚ) / total_employees = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_manager_employee_ratio_l2460_246099


namespace NUMINAMATH_CALUDE_all_sides_equal_l2460_246072

/-- A convex n-gon with equal interior angles and ordered sides -/
structure ConvexNGon (n : ℕ) where
  -- The sides of the n-gon
  sides : Fin n → ℝ
  -- All sides are non-negative
  sides_nonneg : ∀ i, 0 ≤ sides i
  -- The sides are ordered in descending order
  sides_ordered : ∀ i j, i ≤ j → sides j ≤ sides i
  -- The n-gon is convex
  convex : True
  -- All interior angles are equal
  equal_angles : True

/-- Theorem: In a convex n-gon with equal interior angles and ordered sides, all sides are equal -/
theorem all_sides_equal (n : ℕ) (ngon : ConvexNGon n) :
  ∀ i j : Fin n, ngon.sides i = ngon.sides j :=
sorry

end NUMINAMATH_CALUDE_all_sides_equal_l2460_246072


namespace NUMINAMATH_CALUDE_problem_solution_l2460_246019

theorem problem_solution (a b c d e : ℕ+) 
  (eq1 : a * b + a + b = 182)
  (eq2 : b * c + b + c = 306)
  (eq3 : c * d + c + d = 210)
  (eq4 : d * e + d + e = 156)
  (prod : a * b * c * d * e = Nat.factorial 10) :
  (a : ℤ) - (e : ℤ) = -154 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2460_246019


namespace NUMINAMATH_CALUDE_shaded_fraction_is_seven_sixteenths_l2460_246082

/-- Represents a square divided into smaller squares and triangles -/
structure DividedSquare where
  /-- The number of smaller squares the large square is divided into -/
  num_small_squares : ℕ
  /-- The number of triangles each smaller square is divided into -/
  triangles_per_small_square : ℕ
  /-- The total number of shaded triangles -/
  shaded_triangles : ℕ

/-- Calculates the fraction of the square that is shaded -/
def shaded_fraction (s : DividedSquare) : ℚ :=
  s.shaded_triangles / (s.num_small_squares * s.triangles_per_small_square)

/-- Theorem stating that the shaded fraction of the given square is 7/16 -/
theorem shaded_fraction_is_seven_sixteenths (s : DividedSquare) 
  (h1 : s.num_small_squares = 4)
  (h2 : s.triangles_per_small_square = 4)
  (h3 : s.shaded_triangles = 7) : 
  shaded_fraction s = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_is_seven_sixteenths_l2460_246082


namespace NUMINAMATH_CALUDE_toothpick_20th_stage_l2460_246009

def toothpick_sequence (n : ℕ) : ℕ :=
  3 + 3 * (n - 1)

theorem toothpick_20th_stage :
  toothpick_sequence 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_20th_stage_l2460_246009


namespace NUMINAMATH_CALUDE_furniture_uncountable_l2460_246064

-- Define what an uncountable noun is
def UncountableNoun (word : String) : Prop := sorry

-- Define the property of an uncountable noun not changing form
def DoesNotChangeForm (word : String) : Prop := 
  UncountableNoun word → word = word

-- Theorem statement
theorem furniture_uncountable : 
  UncountableNoun "furniture" → DoesNotChangeForm "furniture" := by
  sorry

end NUMINAMATH_CALUDE_furniture_uncountable_l2460_246064


namespace NUMINAMATH_CALUDE_central_symmetry_intersection_condition_l2460_246071

/-- Two functions are centrally symmetric and intersect at one point -/
def centrally_symmetric_one_intersection (a b c d : ℝ) : Prop :=
  let f := fun x => 2 * a + 1 / (x - b)
  let g := fun x => 2 * c + 1 / (x - d)
  let center := ((b + d) / 2, a + c)
  ∃! x, f x = g x ∧ 
    ∀ y, f ((b + d) - y) = g y ∧ 
         g ((b + d) - y) = f y

/-- The main theorem -/
theorem central_symmetry_intersection_condition (a b c d : ℝ) :
  centrally_symmetric_one_intersection a b c d ↔ (a - c) * (b - d) = 2 :=
by sorry


end NUMINAMATH_CALUDE_central_symmetry_intersection_condition_l2460_246071


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l2460_246026

theorem grasshopper_jump_distance 
  (frog_jump : ℕ) 
  (grasshopper_extra : ℕ) 
  (h1 : frog_jump = 11) 
  (h2 : grasshopper_extra = 2) : 
  frog_jump + grasshopper_extra = 13 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l2460_246026


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2460_246016

def A : Set ℝ := {x | 3 * x + 2 > 0}
def B : Set ℝ := {x | (x + 1) * (x - 3) > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | x > 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2460_246016


namespace NUMINAMATH_CALUDE_intersection_distance_l2460_246004

theorem intersection_distance : 
  ∃ (p1 p2 : ℝ × ℝ),
    (p1.1^2 + p1.2^2 = 13) ∧ 
    (p1.1 + p1.2 = 4) ∧
    (p2.1^2 + p2.2^2 = 13) ∧ 
    (p2.1 + p2.2 = 4) ∧
    (p1 ≠ p2) ∧
    ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = 80) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l2460_246004


namespace NUMINAMATH_CALUDE_budget_allocation_home_electronics_l2460_246053

theorem budget_allocation_home_electronics : 
  ∀ (total_budget : ℝ) (microphotonics food_additives gmo industrial_lubricants astrophysics home_electronics : ℝ),
  total_budget > 0 →
  microphotonics = 0.14 * total_budget →
  food_additives = 0.15 * total_budget →
  gmo = 0.19 * total_budget →
  industrial_lubricants = 0.08 * total_budget →
  astrophysics = (72 / 360) * total_budget →
  home_electronics + microphotonics + food_additives + gmo + industrial_lubricants + astrophysics = total_budget →
  home_electronics = 0.24 * total_budget :=
by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_home_electronics_l2460_246053


namespace NUMINAMATH_CALUDE_cubic_inequality_l2460_246068

theorem cubic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 ≥ a^2*b + b^2*c + c^2*a ∧
  (a^3 + b^3 + c^3 = a^2*b + b^2*c + c^2*a ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2460_246068


namespace NUMINAMATH_CALUDE_average_sales_per_month_l2460_246020

def sales_data : List ℝ := [120, 90, 50, 110, 80, 100]

theorem average_sales_per_month :
  (List.sum sales_data) / (List.length sales_data) = 91.67 := by
  sorry

end NUMINAMATH_CALUDE_average_sales_per_month_l2460_246020


namespace NUMINAMATH_CALUDE_u_floor_formula_l2460_246008

def u : ℕ → ℚ
  | 0 => 2
  | 1 => 5/2
  | (n+2) => u (n+1) * (u n ^ 2 - 2) - u 1

theorem u_floor_formula (n : ℕ) (h : n ≥ 1) :
  ⌊u n⌋ = (2 * (2^n - (-1)^n)) / 3 :=
sorry

end NUMINAMATH_CALUDE_u_floor_formula_l2460_246008


namespace NUMINAMATH_CALUDE_square_diff_over_square_sum_l2460_246012

theorem square_diff_over_square_sum (a b : ℝ) (h : a * b / (a^2 + b^2) = 1/4) :
  |a^2 - b^2| / (a^2 + b^2) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_over_square_sum_l2460_246012


namespace NUMINAMATH_CALUDE_two_digit_congruent_to_two_mod_four_count_l2460_246049

theorem two_digit_congruent_to_two_mod_four_count : 
  (Finset.filter (fun n => n ≥ 10 ∧ n ≤ 99 ∧ n % 4 = 2) (Finset.range 100)).card = 23 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_congruent_to_two_mod_four_count_l2460_246049


namespace NUMINAMATH_CALUDE_sum_of_a_values_l2460_246091

/-- The equation for which we need to find the values of 'a' -/
def equation (a x : ℝ) : Prop := 4 * x^2 + a * x + 8 * x + 9 = 0

/-- The condition for the equation to have only one solution -/
def has_one_solution (a : ℝ) : Prop :=
  ∃! x, equation a x

/-- The theorem stating that the sum of 'a' values is -16 -/
theorem sum_of_a_values :
  ∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧ has_one_solution a₁ ∧ has_one_solution a₂ ∧ a₁ + a₂ = -16 :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_values_l2460_246091


namespace NUMINAMATH_CALUDE_tshirt_sale_ratio_l2460_246079

/-- Prove that the ratio of black shirts to white shirts is 1:1 given the conditions -/
theorem tshirt_sale_ratio :
  ∀ (black white : ℕ),
  black + white = 200 →
  30 * black + 25 * white = 5500 →
  black = white :=
by sorry

end NUMINAMATH_CALUDE_tshirt_sale_ratio_l2460_246079


namespace NUMINAMATH_CALUDE_chair_difference_l2460_246052

theorem chair_difference (initial_chairs left_chairs : ℕ) : 
  initial_chairs = 15 → left_chairs = 3 → initial_chairs - left_chairs = 12 := by
  sorry

end NUMINAMATH_CALUDE_chair_difference_l2460_246052


namespace NUMINAMATH_CALUDE_speeding_fine_calculation_l2460_246028

/-- Calculates the base fine for speeding given the total amount owed and other fees --/
theorem speeding_fine_calculation 
  (speed_limit : ℕ) 
  (actual_speed : ℕ) 
  (fine_increase_per_mph : ℕ) 
  (court_costs : ℕ) 
  (lawyer_fee_per_hour : ℕ) 
  (lawyer_hours : ℕ) 
  (total_owed : ℕ) : 
  speed_limit = 30 →
  actual_speed = 75 →
  fine_increase_per_mph = 2 →
  court_costs = 300 →
  lawyer_fee_per_hour = 80 →
  lawyer_hours = 3 →
  total_owed = 820 →
  ∃ (base_fine : ℕ),
    base_fine = 190 ∧
    total_owed = base_fine + 
      2 * (actual_speed - speed_limit) * 2 + 
      court_costs + 
      lawyer_fee_per_hour * lawyer_hours :=
by sorry

end NUMINAMATH_CALUDE_speeding_fine_calculation_l2460_246028


namespace NUMINAMATH_CALUDE_phone_number_prime_factorization_l2460_246054

theorem phone_number_prime_factorization :
  ∃ (p q r s : ℕ), 
    (Nat.Prime p) ∧ 
    (Nat.Prime q) ∧ 
    (Nat.Prime r) ∧ 
    (Nat.Prime s) ∧
    (q = p + 2) ∧ 
    (r = q + 2) ∧ 
    (s = r + 2) ∧
    (p * q * r * s = 27433619) ∧
    (p + q + r + s = 290) := by
  sorry

end NUMINAMATH_CALUDE_phone_number_prime_factorization_l2460_246054


namespace NUMINAMATH_CALUDE_exists_k_no_carry_l2460_246083

/-- 
There exists a positive integer k such that 3993·k is a number 
consisting only of the digit 9.
-/
theorem exists_k_no_carry : ∃ k : ℕ+, 
  ∃ n : ℕ+, (3993 * k.val : ℕ) = (10^n.val - 1) := by sorry

end NUMINAMATH_CALUDE_exists_k_no_carry_l2460_246083


namespace NUMINAMATH_CALUDE_largest_x_value_l2460_246080

theorem largest_x_value : ∃ (x_max : ℝ), 
  (∀ x : ℝ, (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x = 8 * x - 2 → x ≤ x_max) ∧
  ((15 * x_max^2 - 40 * x_max + 18) / (4 * x_max - 3) + 7 * x_max = 8 * x_max - 2) ∧
  x_max = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l2460_246080


namespace NUMINAMATH_CALUDE_cars_meet_time_l2460_246018

/-- Represents a rectangle ABCD -/
structure Rectangle where
  BC : ℝ
  CD : ℝ

/-- Represents a car with a constant speed -/
structure Car where
  speed : ℝ

/-- Time for cars to meet on diagonal BD -/
def meetingTime (rect : Rectangle) (car1 car2 : Car) : ℝ :=
  40 -- in minutes

/-- Theorem stating that cars meet after 40 minutes -/
theorem cars_meet_time (rect : Rectangle) (car1 car2 : Car) :
  meetingTime rect car1 car2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cars_meet_time_l2460_246018


namespace NUMINAMATH_CALUDE_schedule_arrangements_eq_192_l2460_246081

/-- Represents the number of periods in a day -/
def num_periods : ℕ := 6

/-- Represents the number of subjects to be scheduled -/
def num_subjects : ℕ := 6

/-- Represents the number of morning periods -/
def num_morning_periods : ℕ := 4

/-- Represents the number of afternoon periods -/
def num_afternoon_periods : ℕ := 2

/-- Calculates the number of ways to arrange the schedule given the constraints -/
def schedule_arrangements : ℕ :=
  (num_morning_periods.choose 1) * (num_afternoon_periods.choose 1) * (num_subjects - 2).factorial

theorem schedule_arrangements_eq_192 : schedule_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_schedule_arrangements_eq_192_l2460_246081


namespace NUMINAMATH_CALUDE_exactly_three_valid_sets_l2460_246051

/-- A set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  length_ge_3 : length ≥ 3

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set (sum equals 150) -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  sum_consecutive s = 150

theorem exactly_three_valid_sets :
  ∃! (sets : Finset ConsecutiveSet), 
    (∀ s ∈ sets, is_valid_set s) ∧ 
    Finset.card sets = 3 := by sorry

end NUMINAMATH_CALUDE_exactly_three_valid_sets_l2460_246051


namespace NUMINAMATH_CALUDE_rectangular_plot_width_l2460_246085

theorem rectangular_plot_width
  (length : ℝ)
  (num_poles : ℕ)
  (pole_distance : ℝ)
  (h1 : length = 90)
  (h2 : num_poles = 70)
  (h3 : pole_distance = 4)
  : ∃ width : ℝ, width = 48 ∧ 2 * (length + width) = (num_poles - 1 : ℝ) * pole_distance :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_width_l2460_246085


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l2460_246035

/-- Given sets A and B, prove the range of a when A ∩ B = B -/
theorem intersection_equality_implies_a_range
  (A : Set ℝ)
  (B : Set ℝ)
  (a : ℝ)
  (h_A : A = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (h_B : B = {x : ℝ | a < x ∧ x < a + 1})
  (h_int : A ∩ B = B) :
  -2 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l2460_246035


namespace NUMINAMATH_CALUDE_first_five_terms_sequence_1_l2460_246074

def a (n : ℕ+) : ℚ := 1 / (4 * n - 1)

theorem first_five_terms_sequence_1 :
  [a 1, a 2, a 3, a 4, a 5] = [1/3, 1/7, 1/11, 1/15, 1/19] := by sorry

end NUMINAMATH_CALUDE_first_five_terms_sequence_1_l2460_246074


namespace NUMINAMATH_CALUDE_modular_inverse_seven_mod_thirtysix_l2460_246067

theorem modular_inverse_seven_mod_thirtysix : 
  ∃ x : ℤ, 0 ≤ x ∧ x < 36 ∧ (7 * x) % 36 = 1 :=
by
  use 31
  sorry

end NUMINAMATH_CALUDE_modular_inverse_seven_mod_thirtysix_l2460_246067


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l2460_246025

theorem same_remainder_divisor : ∃ (r : ℕ), 
  1108 % 23 = r ∧ 
  1453 % 23 = r ∧ 
  1844 % 23 = r ∧ 
  2281 % 23 = r :=
by sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l2460_246025


namespace NUMINAMATH_CALUDE_zeros_equality_l2460_246038

/-- 
  f(n) represents the number of 0's in the binary representation of a positive integer n
-/
def f (n : ℕ+) : ℕ := sorry

/-- 
  Theorem: For all positive integers n, 
  the number of 0's in the binary representation of 8n+7 
  is equal to the number of 0's in the binary representation of 4n+3
-/
theorem zeros_equality (n : ℕ+) : f (8*n+7) = f (4*n+3) := by sorry

end NUMINAMATH_CALUDE_zeros_equality_l2460_246038


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2460_246078

theorem inequality_equivalence (x : ℝ) :
  (x + 1) / (x - 5) ≥ 3 ↔ x ≥ 8 ∧ x ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2460_246078


namespace NUMINAMATH_CALUDE_stream_speed_l2460_246043

/-- 
Given a canoe that rows upstream at 8 km/hr and downstream at 12 km/hr, 
this theorem proves that the speed of the stream is 2 km/hr.
-/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 8)
  (h_downstream : downstream_speed = 12) :
  let canoe_speed := (upstream_speed + downstream_speed) / 2
  let stream_speed := (downstream_speed - upstream_speed) / 2
  stream_speed = 2 := by sorry

end NUMINAMATH_CALUDE_stream_speed_l2460_246043


namespace NUMINAMATH_CALUDE_perfect_square_expression_l2460_246075

theorem perfect_square_expression (x : ℝ) : ∃ y : ℝ, x^2 - x + (1/4 : ℝ) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l2460_246075


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2460_246073

-- Define the sets A and B
def A : Set ℝ := {x | x - 3 > 0}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2460_246073


namespace NUMINAMATH_CALUDE_difference_of_squares_528_529_l2460_246033

theorem difference_of_squares_528_529 : (528 * 528) - (527 * 529) = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_528_529_l2460_246033


namespace NUMINAMATH_CALUDE_missy_additional_capacity_l2460_246088

/-- Proves that Missy can handle 15 more claims than John given the conditions -/
theorem missy_additional_capacity (jan_capacity : ℕ) (john_capacity : ℕ) (missy_capacity : ℕ) :
  jan_capacity = 20 →
  john_capacity = jan_capacity + (jan_capacity * 3 / 10) →
  missy_capacity = 41 →
  missy_capacity - john_capacity = 15 := by
  sorry

#check missy_additional_capacity

end NUMINAMATH_CALUDE_missy_additional_capacity_l2460_246088


namespace NUMINAMATH_CALUDE_project_scores_mode_l2460_246030

def project_scores : List ℕ := [7, 10, 9, 8, 7, 9, 9, 8]

def mode (l : List ℕ) : ℕ := 
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem project_scores_mode :
  mode project_scores = 9 := by sorry

end NUMINAMATH_CALUDE_project_scores_mode_l2460_246030


namespace NUMINAMATH_CALUDE_wire_length_l2460_246098

/-- Represents the lengths of five wire pieces in a specific ratio --/
structure WirePieces where
  ratio : Fin 5 → ℕ
  shortest : ℝ
  total : ℝ

/-- The wire pieces satisfy the given conditions --/
def satisfies_conditions (w : WirePieces) : Prop :=
  w.ratio 0 = 4 ∧
  w.ratio 1 = 5 ∧
  w.ratio 2 = 7 ∧
  w.ratio 3 = 3 ∧
  w.ratio 4 = 2 ∧
  w.shortest = 16

/-- Theorem stating the total length of the wire --/
theorem wire_length (w : WirePieces) (h : satisfies_conditions w) : w.total = 84 := by
  sorry


end NUMINAMATH_CALUDE_wire_length_l2460_246098


namespace NUMINAMATH_CALUDE_quadratic_condition_necessary_not_sufficient_l2460_246039

theorem quadratic_condition_necessary_not_sufficient :
  (∀ b : ℝ, (∀ x : ℝ, x^2 - b*x + 1 > 0) → b ∈ Set.Icc 0 1) ∧
  ¬(∀ b : ℝ, b ∈ Set.Icc 0 1 → (∀ x : ℝ, x^2 - b*x + 1 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_necessary_not_sufficient_l2460_246039


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2460_246032

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 2*x - 3 < 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2460_246032


namespace NUMINAMATH_CALUDE_function_value_comparison_l2460_246044

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem function_value_comparison
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : ∀ x, f a b c (x + 1) = f a b c (1 - x)) :
  f a b c (Real.arcsin (1/3)) > f a b c (Real.arcsin (2/3)) :=
by sorry

end NUMINAMATH_CALUDE_function_value_comparison_l2460_246044


namespace NUMINAMATH_CALUDE_probability_theorem_l2460_246062

def total_balls : ℕ := 9
def red_balls : ℕ := 2
def black_balls : ℕ := 3
def white_balls : ℕ := 4

def prob_black_then_white : ℚ := 1/6

def prob_red_within_three : ℚ := 7/12

theorem probability_theorem :
  (black_balls / total_balls * white_balls / (total_balls - 1) = prob_black_then_white) ∧
  (red_balls / total_balls + 
   (total_balls - red_balls) / total_balls * red_balls / (total_balls - 1) + 
   (total_balls - red_balls) / total_balls * (total_balls - red_balls - 1) / (total_balls - 1) * red_balls / (total_balls - 2) = prob_red_within_three) :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2460_246062


namespace NUMINAMATH_CALUDE_gcd_90_405_l2460_246046

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_405_l2460_246046


namespace NUMINAMATH_CALUDE_cosine_translation_symmetry_l2460_246047

theorem cosine_translation_symmetry (x : ℝ) (k : ℤ) :
  let f : ℝ → ℝ := λ x => Real.cos (2 * (x + π / 12))
  let axis : ℝ := k * π / 2 - π / 12
  (∀ t, f (axis + t) = f (axis - t)) := by sorry

end NUMINAMATH_CALUDE_cosine_translation_symmetry_l2460_246047


namespace NUMINAMATH_CALUDE_sum_equality_l2460_246042

theorem sum_equality (a b : Fin 2016 → ℝ) 
  (h1 : ∀ n ∈ Finset.range 2015, a (n + 1) = (1 / 65) * Real.sqrt (2 * (n + 1) + 2) + a n)
  (h2 : ∀ n ∈ Finset.range 2015, b (n + 1) = (1 / 1009) * Real.sqrt (2 * (n + 1) + 2) - b n)
  (h3 : a 0 = b 2015)
  (h4 : b 0 = a 2015) :
  (Finset.range 2015).sum (λ k => a (k + 1) * b k - a k * b (k + 1)) = 62 := by
sorry

end NUMINAMATH_CALUDE_sum_equality_l2460_246042


namespace NUMINAMATH_CALUDE_function_max_min_difference_l2460_246041

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a^x else -x + a

-- State the theorem
theorem function_max_min_difference (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 2, f a x ≤ max) ∧
    (∀ x ∈ Set.Icc 0 2, f a x ≥ min) ∧
    (max - min = 5/2)) →
  (a = 1/2 ∨ a = 7/2) :=
by sorry

end NUMINAMATH_CALUDE_function_max_min_difference_l2460_246041


namespace NUMINAMATH_CALUDE_least_satisfying_number_l2460_246001

def satisfies_conditions (n : ℕ) : Prop :=
  n % 10 = 9 ∧ n % 11 = 10 ∧ n % 12 = 11 ∧ n % 13 = 12

theorem least_satisfying_number : 
  satisfies_conditions 8579 ∧ 
  ∀ m : ℕ, m < 8579 → ¬(satisfies_conditions m) :=
by sorry

end NUMINAMATH_CALUDE_least_satisfying_number_l2460_246001


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l2460_246003

/-- The position function of a particle -/
def S (t : ℝ) : ℝ := 2 * t^3 + t

/-- The velocity function of a particle -/
def V (t : ℝ) : ℝ := 6 * t^2 + 1

theorem instantaneous_velocity_at_3s :
  V 3 = 55 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l2460_246003


namespace NUMINAMATH_CALUDE_susie_savings_account_l2460_246024

/-- The compound interest formula for yearly compounding -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The problem statement -/
theorem susie_savings_account :
  let principal : ℝ := 2500
  let rate : ℝ := 0.06
  let years : ℕ := 21
  let result := compound_interest principal rate years
  ∃ ε > 0, |result - 8017.84| < ε :=
sorry

end NUMINAMATH_CALUDE_susie_savings_account_l2460_246024


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l2460_246022

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  populationSize : ℕ
  sampleSize : ℕ
  firstElement : ℕ
  commonDifference : ℕ

/-- Checks if a given list is a valid systematic sample -/
def isValidSample (s : SystematicSample) (sample : List ℕ) : Prop :=
  sample.length = s.sampleSize ∧
  sample.head! = s.firstElement ∧
  ∀ i, 0 < i → i < s.sampleSize → 
    sample[i]! = sample[i-1]! + s.commonDifference ∧
    sample[i]! ≤ s.populationSize

/-- The main theorem to prove -/
theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.populationSize = 60)
  (h2 : s.sampleSize = 5)
  (h3 : 4 ∈ (List.range s.sampleSize).map (fun i => s.firstElement + i * s.commonDifference)) :
  isValidSample s [4, 16, 28, 40, 52] :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l2460_246022
