import Mathlib

namespace alex_trips_l3761_376163

def savings : ℝ := 14500
def car_cost : ℝ := 14600
def trip_charge : ℝ := 1.5
def grocery_percentage : ℝ := 0.05
def grocery_value : ℝ := 800

def earnings_per_trip : ℝ := trip_charge + grocery_percentage * grocery_value

theorem alex_trips : 
  ∃ n : ℕ, (n : ℝ) * earnings_per_trip ≥ car_cost - savings ∧ 
  ∀ m : ℕ, (m : ℝ) * earnings_per_trip ≥ car_cost - savings → n ≤ m :=
by sorry

end alex_trips_l3761_376163


namespace four_digit_difference_l3761_376122

def original_number : ℕ := 201312210840

def is_valid_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ (d1 d2 d3 d4 : ℕ), 
    d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n ∧
    (d1 = 2 ∨ d1 = 0 ∨ d1 = 1 ∨ d1 = 3 ∨ d1 = 8 ∨ d1 = 4) ∧
    (d2 = 2 ∨ d2 = 0 ∨ d2 = 1 ∨ d2 = 3 ∨ d2 = 8 ∨ d2 = 4) ∧
    (d3 = 2 ∨ d3 = 0 ∨ d3 = 1 ∨ d3 = 3 ∨ d3 = 8 ∨ d3 = 4) ∧
    (d4 = 2 ∨ d4 = 0 ∨ d4 = 1 ∨ d4 = 3 ∨ d4 = 8 ∨ d4 = 4))

theorem four_digit_difference :
  ∃ (max min : ℕ), 
    is_valid_four_digit max ∧
    is_valid_four_digit min ∧
    (∀ n, is_valid_four_digit n → n ≤ max) ∧
    (∀ n, is_valid_four_digit n → min ≤ n) ∧
    max - min = 2800 := by sorry

end four_digit_difference_l3761_376122


namespace rectangular_solid_surface_area_l3761_376105

/-- The surface area of a rectangular solid. -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: The surface area of a rectangular solid with length 10 meters, width 9 meters, 
    and depth 6 meters is 408 square meters. -/
theorem rectangular_solid_surface_area :
  surface_area 10 9 6 = 408 := by
  sorry

end rectangular_solid_surface_area_l3761_376105


namespace smallest_n_satisfying_conditions_l3761_376171

def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^4

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_n_satisfying_conditions : 
  (∀ n : ℕ, n > 0 ∧ n < 2000 → ¬(is_perfect_fourth_power (5*n) ∧ is_perfect_cube (4*n))) ∧
  (is_perfect_fourth_power (5*2000) ∧ is_perfect_cube (4*2000)) :=
sorry

end smallest_n_satisfying_conditions_l3761_376171


namespace apartment_office_sale_net_effect_l3761_376133

theorem apartment_office_sale_net_effect :
  ∀ (apartment_cost office_cost : ℝ),
  apartment_cost * (1 - 0.25) = 15000 →
  office_cost * (1 + 0.25) = 15000 →
  apartment_cost + office_cost - 2 * 15000 = 2000 :=
by
  sorry

end apartment_office_sale_net_effect_l3761_376133


namespace quadratic_root_scaling_l3761_376187

theorem quadratic_root_scaling (a b c n : ℝ) (h : a ≠ 0) :
  let original_eq := fun x : ℝ => a * x^2 + b * x + c
  let scaled_eq := fun x : ℝ => a * x^2 + n * b * x + n^2 * c
  let roots := { x : ℝ | original_eq x = 0 }
  let scaled_roots := { x : ℝ | ∃ y ∈ roots, x = n * y }
  scaled_roots = { x : ℝ | scaled_eq x = 0 } :=
by sorry

end quadratic_root_scaling_l3761_376187


namespace uncrossed_numbers_count_l3761_376145

theorem uncrossed_numbers_count : 
  let total_numbers := 1000
  let gcd_value := Nat.gcd 1000 15
  let crossed_out := (total_numbers - 1) / gcd_value + 1
  total_numbers - crossed_out = 800 := by
  sorry

end uncrossed_numbers_count_l3761_376145


namespace log_equality_implies_product_one_l3761_376164

theorem log_equality_implies_product_one (M N : ℝ) 
  (h1 : (Real.log N / Real.log M)^2 = (Real.log M / Real.log N)^2)
  (h2 : M ≠ N)
  (h3 : M * N > 0)
  (h4 : M ≠ 1)
  (h5 : N ≠ 1) :
  M * N = 1 := by
sorry

end log_equality_implies_product_one_l3761_376164


namespace area_of_EFGH_l3761_376148

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The configuration of rectangles forming EFGH -/
structure Configuration where
  small_rectangle : Rectangle
  large_rectangle : Rectangle

/-- The properties of the configuration as described in the problem -/
def valid_configuration (c : Configuration) : Prop :=
  c.small_rectangle.height = 6 ∧
  c.large_rectangle.width = 2 * c.small_rectangle.width ∧
  c.large_rectangle.height = 2 * c.small_rectangle.height

theorem area_of_EFGH (c : Configuration) (h : valid_configuration c) :
  c.large_rectangle.area = 144 :=
sorry

end area_of_EFGH_l3761_376148


namespace binary_1101_equals_base5_23_l3761_376194

/-- Converts a binary number to decimal --/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal number to base-5 --/
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The binary representation of 1101 --/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_equals_base5_23 :
  decimal_to_base5 (binary_to_decimal binary_1101) = [2, 3] := by
  sorry

#eval binary_to_decimal binary_1101
#eval decimal_to_base5 (binary_to_decimal binary_1101)

end binary_1101_equals_base5_23_l3761_376194


namespace intersection_implies_sum_l3761_376117

-- Define the functions
def f (a b x : ℝ) : ℝ := -|x - a|^2 + b
def g (c d x : ℝ) : ℝ := |x - c|^2 + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) :
  f a b 1 = 4 ∧ g c d 1 = 4 ∧ f a b 7 = 2 ∧ g c d 7 = 2 → a + c = 8 := by
  sorry

end intersection_implies_sum_l3761_376117


namespace parabola_function_expression_l3761_376184

-- Define the parabola function
def parabola (a : ℝ) (x : ℝ) : ℝ := a * (x + 3)^2 + 2

-- State the theorem
theorem parabola_function_expression :
  ∃ a : ℝ, 
    (parabola a (-3) = 2) ∧ 
    (parabola a 1 = -14) ∧
    (∀ x : ℝ, parabola a x = -(x + 3)^2 + 2) := by
  sorry


end parabola_function_expression_l3761_376184


namespace brownie_pieces_count_l3761_376135

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from the pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  (area pan.panDimensions) / (area pan.pieceDimensions)

theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end brownie_pieces_count_l3761_376135


namespace even_sum_condition_l3761_376110

-- Define what it means for a number to be even
def IsEven (n : Int) : Prop := ∃ k : Int, n = 2 * k

-- Statement of the theorem
theorem even_sum_condition :
  (∀ a b : Int, IsEven a ∧ IsEven b → IsEven (a + b)) ∧
  (∃ a b : Int, IsEven (a + b) ∧ (¬IsEven a ∨ ¬IsEven b)) := by
  sorry

end even_sum_condition_l3761_376110


namespace expression_evaluation_l3761_376130

theorem expression_evaluation : (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end expression_evaluation_l3761_376130


namespace bracelet_count_l3761_376144

/-- Calculates the number of sets that can be made from a given number of beads -/
def sets_from_beads (beads : ℕ) : ℕ := beads / 2

/-- Represents the number of beads Nancy and Rose have -/
structure BeadCounts where
  metal : ℕ
  pearl : ℕ
  crystal : ℕ
  stone : ℕ

/-- Calculates the maximum number of bracelets that can be made -/
def max_bracelets (counts : BeadCounts) : ℕ :=
  min (min (sets_from_beads counts.metal) (sets_from_beads counts.pearl))
      (min (sets_from_beads counts.crystal) (sets_from_beads counts.stone))

theorem bracelet_count (counts : BeadCounts)
  (h1 : counts.metal = 40)
  (h2 : counts.pearl = 60)
  (h3 : counts.crystal = 20)
  (h4 : counts.stone = 40) :
  max_bracelets counts = 10 := by
  sorry

end bracelet_count_l3761_376144


namespace complex_number_problem_l3761_376180

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  z₁ * (2 + Complex.I) = 5 * Complex.I →
  (∃ (r : ℝ), z₁ + z₂ = r) →
  (∃ (y : ℝ), y ≠ 0 ∧ z₁ * z₂ = y * Complex.I) →
  z₂ = -4 - 2 * Complex.I :=
by sorry

end complex_number_problem_l3761_376180


namespace power_and_division_equality_l3761_376139

theorem power_and_division_equality : (12 : ℕ)^3 * 6^4 / 432 = 5184 := by sorry

end power_and_division_equality_l3761_376139


namespace no_tetrahedron_with_given_edges_l3761_376183

/-- Represents a tetrahedron with three pairs of opposite edges --/
structure Tetrahedron where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ

/-- Checks if a tetrahedron with given edge lengths can exist --/
def tetrahedronExists (t : Tetrahedron) : Prop :=
  t.edge1 > 0 ∧ t.edge2 > 0 ∧ t.edge3 > 0 ∧
  t.edge1^2 + t.edge2^2 > t.edge3^2 ∧
  t.edge1^2 + t.edge3^2 > t.edge2^2 ∧
  t.edge2^2 + t.edge3^2 > t.edge1^2

/-- Theorem stating that a tetrahedron with the given edge lengths does not exist --/
theorem no_tetrahedron_with_given_edges :
  ¬ ∃ (t : Tetrahedron), t.edge1 = 12 ∧ t.edge2 = 12.5 ∧ t.edge3 = 13 ∧ tetrahedronExists t :=
by sorry


end no_tetrahedron_with_given_edges_l3761_376183


namespace calculate_expression_l3761_376150

theorem calculate_expression : 75 * 1313 - 25 * 1313 = 65750 := by
  sorry

end calculate_expression_l3761_376150


namespace triangle_inequality_l3761_376121

/-- Given a triangle with side lengths a, b, c, and semiperimeter p, 
    prove that 2√((p-b)(p-c)) ≤ a. -/
theorem triangle_inequality (a b c p : ℝ) 
    (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_semiperimeter : p = (a + b + c) / 2) : 
  2 * Real.sqrt ((p - b) * (p - c)) ≤ a := by
  sorry

end triangle_inequality_l3761_376121


namespace unique_a_value_l3761_376131

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem unique_a_value : ∃! a : ℝ, (9 ∈ (A a ∩ B a)) ∧ ({9} = A a ∩ B a) := by
  sorry

end unique_a_value_l3761_376131


namespace tangent_parallel_to_given_line_l3761_376161

-- Define the curve
def f (x : ℝ) := x^4

-- Define the derivative of the curve
def f' (x : ℝ) := 4 * x^3

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the given line
def givenLine (x y : ℝ) : Prop := 4 * x - y + 1 = 0

-- Define parallel lines
def parallel (m₁ b₁ m₂ b₂ : ℝ) : Prop := m₁ = m₂ ∧ b₁ ≠ b₂

theorem tangent_parallel_to_given_line :
  let m := f' P.1  -- Slope of tangent line
  let b := P.2 - m * P.1  -- y-intercept of tangent line
  parallel m b 4 (-1) := by sorry

end tangent_parallel_to_given_line_l3761_376161


namespace white_coincide_pairs_l3761_376104

-- Define the structure of our figure
structure Figure where
  red_triangles : ℕ
  blue_triangles : ℕ
  white_triangles : ℕ
  red_coincide : ℕ
  blue_coincide : ℕ
  red_white_pairs : ℕ

-- Define our specific figure
def our_figure : Figure :=
  { red_triangles := 4
  , blue_triangles := 6
  , white_triangles := 10
  , red_coincide := 3
  , blue_coincide := 4
  , red_white_pairs := 3 }

-- Theorem statement
theorem white_coincide_pairs (f : Figure) (h : f = our_figure) : 
  ∃ (white_coincide : ℕ), white_coincide = 3 := by
  sorry

end white_coincide_pairs_l3761_376104


namespace sphere_diameter_from_cylinder_l3761_376190

/-- Given a cylinder with diameter 6 cm and height 6 cm, if spheres of equal volume are made from the same material, the diameter of each sphere is equal to the cube root of (162 * π) cm. -/
theorem sphere_diameter_from_cylinder (π : ℝ) (h : π > 0) :
  let cylinder_diameter : ℝ := 6
  let cylinder_height : ℝ := 6
  let cylinder_volume : ℝ := π * (cylinder_diameter / 2)^2 * cylinder_height
  let sphere_volume : ℝ := cylinder_volume
  let sphere_diameter : ℝ := 2 * (3 * sphere_volume / (4 * π))^(1/3)
  sphere_diameter = (162 * π)^(1/3) :=
by sorry

end sphere_diameter_from_cylinder_l3761_376190


namespace max_total_marks_is_1127_l3761_376100

/-- Represents the pass requirements and scores for a student's exam -/
structure ExamResults where
  math_pass_percent : ℚ
  physics_pass_percent : ℚ
  chem_pass_percent : ℚ
  math_score : ℕ
  math_fail_margin : ℕ
  physics_score : ℕ
  physics_fail_margin : ℕ
  chem_score : ℕ
  chem_fail_margin : ℕ

/-- Calculates the maximum total marks obtainable across all subjects -/
def maxTotalMarks (results : ExamResults) : ℕ :=
  sorry

/-- Theorem stating that given the exam results, the maximum total marks is 1127 -/
theorem max_total_marks_is_1127 (results : ExamResults) 
  (h1 : results.math_pass_percent = 36/100)
  (h2 : results.physics_pass_percent = 40/100)
  (h3 : results.chem_pass_percent = 45/100)
  (h4 : results.math_score = 130)
  (h5 : results.math_fail_margin = 14)
  (h6 : results.physics_score = 120)
  (h7 : results.physics_fail_margin = 20)
  (h8 : results.chem_score = 160)
  (h9 : results.chem_fail_margin = 10) :
  maxTotalMarks results = 1127 :=
  sorry

end max_total_marks_is_1127_l3761_376100


namespace work_increase_with_absences_l3761_376177

/-- Given a total amount of work W and p persons, prove that when 1/3 of the persons are absent,
    the increase in work for each remaining person is W/(2p). -/
theorem work_increase_with_absences (W p : ℝ) (h₁ : W > 0) (h₂ : p > 0) :
  let initial_work_per_person := W / p
  let remaining_persons := (2 / 3) * p
  let new_work_per_person := W / remaining_persons
  new_work_per_person - initial_work_per_person = W / (2 * p) :=
by sorry

end work_increase_with_absences_l3761_376177


namespace parallel_lines_a_value_l3761_376149

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, a * x + y - 1 - a = 0 ↔ x - (1/2) * y = 0) → a = -2 := by
  sorry

end parallel_lines_a_value_l3761_376149


namespace simplify_and_evaluate_expression_l3761_376154

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 3 + 1
  (x + 1) / x / (x - 1 / x) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_expression_l3761_376154


namespace integral_sum_equals_pi_over_four_plus_ln_two_l3761_376140

theorem integral_sum_equals_pi_over_four_plus_ln_two :
  ∫ (x : ℝ) in (0)..(1), Real.sqrt (1 - x^2) + ∫ (x : ℝ) in (1)..(2), 1/x = π/4 + Real.log 2 := by
  sorry

end integral_sum_equals_pi_over_four_plus_ln_two_l3761_376140


namespace smallest_n_for_P_less_than_threshold_l3761_376120

/-- Represents the number of boxes in the game --/
def total_boxes : ℕ := 2023

/-- Represents the number of boxes with 2 red marbles --/
def boxes_with_two_red : ℕ := 1012

/-- Calculates the probability of drawing a red marble from a box --/
def prob_red (box_number : ℕ) : ℚ :=
  if box_number ≤ boxes_with_two_red then
    2 / (box_number + 2)
  else
    1 / (box_number + 1)

/-- Calculates the probability of drawing a white marble from a box --/
def prob_white (box_number : ℕ) : ℚ :=
  1 - prob_red box_number

/-- Represents the probability of Isabella stopping after drawing exactly n marbles --/
noncomputable def P (n : ℕ) : ℚ :=
  sorry -- Definition of P(n) based on the game rules

/-- Theorem stating that 51 is the smallest n for which P(n) < 1/2023 --/
theorem smallest_n_for_P_less_than_threshold :
  (∀ k < 51, P k ≥ 1 / total_boxes) ∧
  P 51 < 1 / total_boxes :=
sorry

#check smallest_n_for_P_less_than_threshold

end smallest_n_for_P_less_than_threshold_l3761_376120


namespace slope_of_solutions_l3761_376160

theorem slope_of_solutions (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : (5 / x₁) + (4 / y₁) = 0) (h₃ : (5 / x₂) + (4 / y₂) = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -4/5 := by
  sorry

end slope_of_solutions_l3761_376160


namespace focus_to_latus_rectum_distance_l3761_376109

/-- A parabola with equation y^2 = 2px (p > 0) whose latus rectum is tangent to the circle (x-3)^2 + y^2 = 16 -/
structure TangentParabola where
  p : ℝ
  p_pos : p > 0
  latus_rectum_tangent : ∃ (x y : ℝ), y^2 = 2*p*x ∧ (x-3)^2 + y^2 = 16

/-- The distance from the focus of the parabola to the latus rectum is 2 -/
theorem focus_to_latus_rectum_distance (tp : TangentParabola) : tp.p = 2 := by
  sorry

end focus_to_latus_rectum_distance_l3761_376109


namespace consecutive_digits_count_l3761_376108

theorem consecutive_digits_count : ∃ (m n : ℕ), 
  (10^(m-1) < 2^2020 ∧ 2^2020 < 10^m) ∧
  (10^(n-1) < 5^2020 ∧ 5^2020 < 10^n) ∧
  m + n = 2021 := by
  sorry

end consecutive_digits_count_l3761_376108


namespace children_not_enrolled_l3761_376165

theorem children_not_enrolled (total children_basketball children_robotics children_both : ℕ) 
  (h_total : total = 150)
  (h_basketball : children_basketball = 85)
  (h_robotics : children_robotics = 58)
  (h_both : children_both = 18) :
  total - (children_basketball + children_robotics - children_both) = 25 := by
  sorry

end children_not_enrolled_l3761_376165


namespace max_abs_value_l3761_376181

theorem max_abs_value (x y : ℝ) 
  (h1 : x + y - 2 ≤ 0) 
  (h2 : x - y + 4 ≥ 0) 
  (h3 : y ≥ 0) : 
  ∃ (z : ℝ), z = |x - 2*y + 2| ∧ z ≤ 5 ∧ ∀ (w : ℝ), w = |x - 2*y + 2| → w ≤ z :=
by sorry

end max_abs_value_l3761_376181


namespace oz_language_lost_words_l3761_376173

/-- Represents the number of letters in the Oz alphabet -/
def alphabet_size : ℕ := 65

/-- Represents the number of letters in a word (either 1 or 2) -/
def word_length : Fin 2 → ℕ
| 0 => 1
| 1 => 2

/-- Calculates the number of words lost when one letter is forbidden -/
def words_lost (n : ℕ) : ℕ :=
  1 + n + n

/-- Theorem stating that forbidding one letter in the Oz language results in 131 lost words -/
theorem oz_language_lost_words :
  words_lost alphabet_size = 131 := by
  sorry

end oz_language_lost_words_l3761_376173


namespace cost_price_articles_l3761_376153

/-- Given that the cost price of N articles equals the selling price of 50 articles,
    and the profit percentage is 10.000000000000004%, prove that N = 55. -/
theorem cost_price_articles (N : ℕ) (C S : ℝ) : 
  N * C = 50 * S →
  (S - C) / C * 100 = 10.000000000000004 →
  N = 55 := by
  sorry

end cost_price_articles_l3761_376153


namespace f_even_and_increasing_l3761_376193

def f (x : ℝ) : ℝ := |x| + 1

theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_even_and_increasing_l3761_376193


namespace always_odd_l3761_376119

theorem always_odd (n : ℤ) : ∃ k : ℤ, 2017 + 2*n = 2*k + 1 := by
  sorry

end always_odd_l3761_376119


namespace smallest_sum_B_plus_c_l3761_376169

def base_5_to_10 (b : ℕ) : ℕ := 780 * b

def base_c_to_10 (c : ℕ) : ℕ := 4 * (c + 1)

def valid_base_5_digit (b : ℕ) : Prop := 1 ≤ b ∧ b ≤ 4

def valid_base_c (c : ℕ) : Prop := c > 6

theorem smallest_sum_B_plus_c :
  ∃ (B c : ℕ),
    valid_base_5_digit B ∧
    valid_base_c c ∧
    base_5_to_10 B = base_c_to_10 c ∧
    (∀ (B' c' : ℕ),
      valid_base_5_digit B' →
      valid_base_c c' →
      base_5_to_10 B' = base_c_to_10 c' →
      B + c ≤ B' + c') ∧
    B + c = 195 :=
sorry

end smallest_sum_B_plus_c_l3761_376169


namespace parabola_coefficient_l3761_376107

/-- A quadratic function with vertex form (x - h)^2 where h is the x-coordinate of the vertex -/
def quadratic_vertex_form (a : ℝ) (h : ℝ) (x : ℝ) : ℝ := a * (x - h)^2

theorem parabola_coefficient (f : ℝ → ℝ) (h : ℝ) (a : ℝ) :
  (∀ x, f x = quadratic_vertex_form a h x) →
  f 5 = -36 →
  h = 2 →
  a = -4 := by
sorry

end parabola_coefficient_l3761_376107


namespace solution_y_b_percentage_l3761_376137

-- Define the solutions and their compositions
def solution_x_a : ℝ := 0.3
def solution_x_b : ℝ := 0.7
def solution_y_a : ℝ := 0.4

-- Define the mixture composition
def mixture_x : ℝ := 0.8
def mixture_y : ℝ := 0.2
def mixture_a : ℝ := 0.32

-- Theorem to prove
theorem solution_y_b_percentage : 
  solution_x_a + solution_x_b = 1 →
  mixture_x + mixture_y = 1 →
  mixture_x * solution_x_a + mixture_y * solution_y_a = mixture_a →
  1 - solution_y_a = 0.6 :=
by sorry

end solution_y_b_percentage_l3761_376137


namespace staplers_left_after_stapling_l3761_376132

/-- The number of staplers left after stapling some reports -/
def staplers_left (initial_staplers : ℕ) (dozen_reports_stapled : ℕ) : ℕ :=
  initial_staplers - dozen_reports_stapled * 12

/-- Theorem: Given 50 initial staplers and 3 dozen reports stapled, 14 staplers are left -/
theorem staplers_left_after_stapling :
  staplers_left 50 3 = 14 := by
  sorry

end staplers_left_after_stapling_l3761_376132


namespace cn_length_l3761_376113

/-- Right-angled triangle with squares on legs -/
structure RightTriangleWithSquares where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  -- Conditions
  right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 4
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 1
  square_acde : (D.1 - A.1) = (E.1 - C.1) ∧ (D.2 - A.2) = (E.2 - C.2) ∧
                 (D.1 - A.1) * (E.1 - C.1) + (D.2 - A.2) * (E.2 - C.2) = 0
  square_bcfg : (F.1 - B.1) = (G.1 - C.1) ∧ (F.2 - B.2) = (G.2 - C.2) ∧
                 (F.1 - B.1) * (G.1 - C.1) + (F.2 - B.2) * (G.2 - C.2) = 0
  m_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  n_on_cm : (N.2 - C.2) * (M.1 - C.1) = (N.1 - C.1) * (M.2 - C.2)
  n_on_df : (N.2 - D.2) * (F.1 - D.1) = (N.1 - D.1) * (F.2 - D.2)

/-- The length of CN is √17 -/
theorem cn_length (t : RightTriangleWithSquares) : 
  Real.sqrt ((t.N.1 - t.C.1)^2 + (t.N.2 - t.C.2)^2) = Real.sqrt 17 := by
  sorry

end cn_length_l3761_376113


namespace greatest_distance_between_circle_centers_l3761_376198

/-- The greatest distance between centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 16)
  (h_height : rectangle_height = 20)
  (h_diameter : circle_diameter = 8)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = 2 * Real.sqrt 52 ∧
    ∀ (d' : ℝ), d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        circle_diameter / 2 ≤ x₁ ∧ x₁ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₁ ∧ y₁ ≤ rectangle_height - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ x₂ ∧ x₂ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₂ ∧ y₂ ≤ rectangle_height - circle_diameter / 2 ∧
        d' = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by
  sorry

end greatest_distance_between_circle_centers_l3761_376198


namespace problem_solution_l3761_376146

theorem problem_solution : 
  let x := 0.47 * 1442 - 0.36 * 1412
  ∃ y, x + y = 3 ∧ y = -166.42 := by
sorry

end problem_solution_l3761_376146


namespace diophantus_age_problem_l3761_376170

theorem diophantus_age_problem :
  ∀ (x : ℕ),
    (x / 6 : ℚ) + (x / 12 : ℚ) + (x / 7 : ℚ) + 5 + (x / 2 : ℚ) + 4 = x →
    x = 84 :=
by
  sorry

end diophantus_age_problem_l3761_376170


namespace max_profit_l3761_376156

noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 300 * x
  else if x ≥ 40 then (901 * x^2 - 9450 * x + 10000) / x
  else 0

noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then -10 * x^2 + 600 * x - 260
  else if x ≥ 40 then (-x^2 + 9190 * x - 10000) / x
  else 0

theorem max_profit (x : ℝ) :
  (∀ y, y > 0 → W y ≤ W 100) ∧ W 100 = 8990 := by sorry

end max_profit_l3761_376156


namespace square_sum_divisible_by_product_l3761_376128

def is_valid_triple (k : ℤ) (x y z : ℤ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≤ 1000 ∧ y ≤ 1000 ∧ z ≤ 1000 ∧
  x^2 + y^2 + z^2 = k * x * y * z

def valid_triples_for_k (k : ℤ) : List (ℤ × ℤ × ℤ) :=
  if k = 3 then
    [(1, 1, 1), (1, 1, 2), (1, 2, 5), (1, 5, 13), (2, 5, 29), (5, 29, 169)]
  else if k = 1 then
    [(3, 3, 3), (3, 3, 6), (3, 6, 15), (6, 15, 39), (6, 15, 87)]
  else
    []

theorem square_sum_divisible_by_product :
  ∀ k x y z : ℤ,
    is_valid_triple k x y z ↔
      (k = 1 ∨ k = 3) ∧
      (x, y, z) ∈ valid_triples_for_k k :=
sorry

end square_sum_divisible_by_product_l3761_376128


namespace root_quadratic_equation_l3761_376176

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - m - 1 = 0 → m^2 - m = 1 := by
sorry

end root_quadratic_equation_l3761_376176


namespace remainder_theorem_l3761_376179

-- Define the polynomial p(x) = x^4 - 2x^2 + 4x - 5
def p (x : ℝ) : ℝ := x^4 - 2*x^2 + 4*x - 5

-- State the theorem
theorem remainder_theorem : 
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ), p x = (x - 1) * q x + (-2) := by
  sorry

end remainder_theorem_l3761_376179


namespace banana_cost_l3761_376147

/-- The cost of a bunch of bananas can be expressed as $5 minus the cost of a dozen apples -/
theorem banana_cost (apple_cost banana_cost : ℝ) : 
  apple_cost + banana_cost = 5 → banana_cost = 5 - apple_cost := by
  sorry

end banana_cost_l3761_376147


namespace operation_result_l3761_376174

theorem operation_result (x : ℝ) : 40 + 5 * x / (180 / 3) = 41 → x = 12 := by
  sorry

end operation_result_l3761_376174


namespace sqrt_square_not_always_equal_to_a_l3761_376125

theorem sqrt_square_not_always_equal_to_a : ¬ ∀ a : ℝ, Real.sqrt (a^2) = a := by
  sorry

end sqrt_square_not_always_equal_to_a_l3761_376125


namespace min_sum_of_squares_l3761_376158

theorem min_sum_of_squares (x y z : ℝ) : 
  (x + 5) * (y - 5) = 0 →
  (y + 5) * (z - 5) = 0 →
  (z + 5) * (x - 5) = 0 →
  x^2 + y^2 + z^2 ≥ 75 ∧ ∃ (x' y' z' : ℝ), 
    (x' + 5) * (y' - 5) = 0 ∧
    (y' + 5) * (z' - 5) = 0 ∧
    (z' + 5) * (x' - 5) = 0 ∧
    x'^2 + y'^2 + z'^2 = 75 :=
by sorry

end min_sum_of_squares_l3761_376158


namespace janice_overtime_shifts_l3761_376159

/-- Proves that Janice worked 3 overtime shifts given her work schedule and earnings --/
theorem janice_overtime_shifts :
  let regular_days : ℕ := 5
  let regular_daily_pay : ℕ := 30
  let overtime_pay : ℕ := 15
  let total_earnings : ℕ := 195
  let regular_earnings := regular_days * regular_daily_pay
  let overtime_earnings := total_earnings - regular_earnings
  overtime_earnings / overtime_pay = 3 := by sorry

end janice_overtime_shifts_l3761_376159


namespace solution_set_equality_l3761_376199

/-- A function f: ℝ → ℝ that is odd, monotonically increasing on (0, +∞), and f(-1) = 2 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y) ∧
  (f (-1) = 2)

/-- The theorem statement -/
theorem solution_set_equality (f : ℝ → ℝ) (h : special_function f) :
  {x : ℝ | x > 0 ∧ f (x - 1) + 2 ≤ 0} = Set.Ioo 1 2 := by
  sorry

end solution_set_equality_l3761_376199


namespace complex_sum_modulus_l3761_376186

theorem complex_sum_modulus (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs (z₁ - z₂) = 1) : 
  Complex.abs (z₁ + z₂) = Real.sqrt 3 := by
  sorry

end complex_sum_modulus_l3761_376186


namespace path_count_equals_binomial_coefficient_l3761_376115

/-- The number of paths composed of n rises and n descents of the same amplitude -/
def pathCount (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- Theorem: The number of paths composed of n rises and n descents of the same amplitude
    is equal to the binomial coefficient (2n choose n) -/
theorem path_count_equals_binomial_coefficient (n : ℕ) :
  pathCount n = Nat.choose (2 * n) n := by sorry

end path_count_equals_binomial_coefficient_l3761_376115


namespace steinburg_marching_band_max_size_l3761_376192

theorem steinburg_marching_band_max_size :
  ∀ n : ℕ,
  (30 * n) % 34 = 6 →
  30 * n < 1200 →
  (∀ m : ℕ, (30 * m) % 34 = 6 → 30 * m < 1200 → 30 * m ≤ 30 * n) →
  30 * n = 720 := by
sorry

end steinburg_marching_band_max_size_l3761_376192


namespace simplify_fraction_l3761_376175

theorem simplify_fraction : 5 * (14 / 3) * (9 / -42) = -5 := by sorry

end simplify_fraction_l3761_376175


namespace ceiling_floor_sum_l3761_376123

theorem ceiling_floor_sum (x : ℝ) : 
  ⌈x⌉ - ⌊x⌋ = 0 → ⌈x⌉ + ⌊x⌋ = 2*x := by
  sorry

end ceiling_floor_sum_l3761_376123


namespace no_five_digit_flippy_divisible_by_33_l3761_376162

-- Define a flippy number
def is_flippy (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 
    (n = a * 10000 + b * 1000 + a * 100 + b * 10 + a ∨
     n = b * 10000 + a * 1000 + b * 100 + a * 10 + b)

-- Define a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

-- Theorem statement
theorem no_five_digit_flippy_divisible_by_33 :
  ¬ ∃ (n : ℕ), is_five_digit n ∧ is_flippy n ∧ n % 33 = 0 :=
sorry

end no_five_digit_flippy_divisible_by_33_l3761_376162


namespace partial_fraction_decomposition_l3761_376112

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 73/15 ∧ B = 17/15 ∧
  ∀ (x : ℚ), x ≠ 12 → x ≠ -3 →
    (6*x + 1) / (x^2 - 9*x - 36) = A / (x - 12) + B / (x + 3) :=
by sorry

end partial_fraction_decomposition_l3761_376112


namespace no_solution_exists_l3761_376152

/-- A polynomial with roots -p, -p-1, -p-2, -p-3 -/
def g (p : ℕ+) (x : ℝ) : ℝ :=
  (x + p) * (x + p + 1) * (x + p + 2) * (x + p + 3)

/-- Coefficients of the expanded polynomial g -/
def a (p : ℕ+) : ℝ := 4 * p + 6
def b (p : ℕ+) : ℝ := 10 * p^2 + 15 * p + 11
def c (p : ℕ+) : ℝ := 12 * p^3 + 18 * p^2 + 22 * p + 6
def d (p : ℕ+) : ℝ := 6 * p^4 + 9 * p^3 + 20 * p^2 + 15 * p + 6

/-- Theorem stating that there is no positive integer p satisfying the given condition -/
theorem no_solution_exists : ¬ ∃ (p : ℕ+), a p + b p + c p + d p = 2056 := by
  sorry

end no_solution_exists_l3761_376152


namespace repair_shop_earnings_l3761_376178

/-- Calculates the total earnings for a repair shop given the number of repairs and their costs. -/
def total_earnings (phone_repairs laptop_repairs computer_repairs : ℕ) 
  (phone_cost laptop_cost computer_cost : ℕ) : ℕ :=
  phone_repairs * phone_cost + laptop_repairs * laptop_cost + computer_repairs * computer_cost

/-- Theorem stating that the total earnings for the given repairs and costs is $121. -/
theorem repair_shop_earnings : 
  total_earnings 5 2 2 11 15 18 = 121 := by
  sorry

end repair_shop_earnings_l3761_376178


namespace union_equals_interval_l3761_376166

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 5*x - 6 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}

-- Define the open interval (-6, 6)
def openInterval : Set ℝ := Ioo (-6) 6

-- Theorem statement
theorem union_equals_interval : A ∪ B = openInterval := by
  sorry

end union_equals_interval_l3761_376166


namespace smallest_x_for_perfect_cube_l3761_376103

/-- 
A perfect cube is a number that is the result of multiplying an integer by itself twice.
-/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

/-- 
The smallest positive integer x such that 1152x is a perfect cube is 36.
-/
theorem smallest_x_for_perfect_cube : 
  (∀ y : ℕ+, is_perfect_cube (1152 * y) → y ≥ 36) ∧ 
  is_perfect_cube (1152 * 36) := by
  sorry


end smallest_x_for_perfect_cube_l3761_376103


namespace quadratic_minimum_value_l3761_376143

theorem quadratic_minimum_value (x y : ℝ) :
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 ≥ -13/5 ∧
  ∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * x₀ * y₀ + 2 * y₀^2 - 6 * x₀ + 8 * y₀ + 10 = -13/5 :=
by sorry

end quadratic_minimum_value_l3761_376143


namespace peters_horses_feeding_days_l3761_376141

theorem peters_horses_feeding_days :
  let num_horses : ℕ := 4
  let oats_per_meal : ℕ := 4
  let oats_meals_per_day : ℕ := 2
  let grain_per_day : ℕ := 3
  let total_food : ℕ := 132
  
  let food_per_horse_per_day : ℕ := oats_per_meal * oats_meals_per_day + grain_per_day
  let total_food_per_day : ℕ := num_horses * food_per_horse_per_day
  
  total_food / total_food_per_day = 3 :=
by sorry

end peters_horses_feeding_days_l3761_376141


namespace expected_attempts_proof_l3761_376197

/-- The expected number of attempts to open a safe with n keys -/
def expected_attempts (n : ℕ) : ℚ :=
  (n + 1 : ℚ) / 2

/-- Theorem stating that the expected number of attempts to open a safe
    with n keys distributed sequentially to n students is (n+1)/2 -/
theorem expected_attempts_proof (n : ℕ) :
  expected_attempts n = (n + 1 : ℚ) / 2 := by
  sorry

end expected_attempts_proof_l3761_376197


namespace brick_length_calculation_l3761_376157

theorem brick_length_calculation (courtyard_length : Real) (courtyard_width : Real)
  (brick_width : Real) (total_bricks : Nat) :
  courtyard_length = 18 ∧ 
  courtyard_width = 12 ∧
  brick_width = 0.06 ∧
  total_bricks = 30000 →
  ∃ brick_length : Real,
    brick_length = 0.12 ∧
    courtyard_length * courtyard_width * 10000 = total_bricks * brick_length * brick_width :=
by sorry

end brick_length_calculation_l3761_376157


namespace expression_equality_l3761_376118

theorem expression_equality : 
  |Real.sqrt 3 - 3| - Real.sqrt 16 + Real.cos (30 * π / 180) + (1/3)^0 = -Real.sqrt 3 / 2 := by
  sorry

end expression_equality_l3761_376118


namespace unique_function_existence_l3761_376196

def is_valid_function (f : ℕ → ℝ) : Prop :=
  (∀ x : ℕ, f x > 0) ∧
  (∀ a b : ℕ, f (a + b) = f a * f b) ∧
  (f 2 = 4)

theorem unique_function_existence : 
  ∃! f : ℕ → ℝ, is_valid_function f ∧ ∀ x : ℕ, f x = 2^x :=
sorry

end unique_function_existence_l3761_376196


namespace r_amount_l3761_376114

theorem r_amount (total : ℝ) (r_fraction : ℝ) (h1 : total = 5000) (h2 : r_fraction = 2/3) :
  r_fraction * (total / (1 + r_fraction)) = 2000 := by
  sorry

end r_amount_l3761_376114


namespace wolves_hunt_in_five_days_l3761_376106

/-- Calculates the number of days before wolves need to hunt again -/
def days_before_next_hunt (hunting_wolves : ℕ) (additional_wolves : ℕ) 
  (meat_per_wolf_per_day : ℕ) (meat_per_deer : ℕ) : ℕ :=
  let total_wolves := hunting_wolves + additional_wolves
  let daily_meat_requirement := total_wolves * meat_per_wolf_per_day
  let total_meat_from_hunt := hunting_wolves * meat_per_deer
  total_meat_from_hunt / daily_meat_requirement

theorem wolves_hunt_in_five_days : 
  days_before_next_hunt 4 16 8 200 = 5 := by sorry

end wolves_hunt_in_five_days_l3761_376106


namespace kevin_cards_l3761_376168

theorem kevin_cards (initial_cards lost_cards : ℝ) 
  (h1 : initial_cards = 47.0)
  (h2 : lost_cards = 7.0) :
  initial_cards - lost_cards = 40.0 := by
  sorry

end kevin_cards_l3761_376168


namespace power_of_power_l3761_376151

theorem power_of_power (a : ℝ) : (a^3)^4 = a^12 := by
  sorry

end power_of_power_l3761_376151


namespace leibniz_recursive_relation_leibniz_boundary_condition_pascal_leibniz_relation_l3761_376195

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Element in Leibniz's Triangle -/
def leibniz (n k : ℕ) : ℚ := 1 / ((n + 1 : ℚ) * (binomial n k))

/-- Theorem stating the recursive relationship in Leibniz's Triangle -/
theorem leibniz_recursive_relation (n k : ℕ) (h : 0 < k ∧ k ≤ n) :
  leibniz n (k - 1) + leibniz n k = leibniz (n - 1) (k - 1) := by sorry

/-- Theorem stating that the formula for Leibniz's Triangle satisfies its boundary condition -/
theorem leibniz_boundary_condition (n : ℕ) :
  leibniz n 0 = 1 / (n + 1 : ℚ) ∧ leibniz n n = 1 / (n + 1 : ℚ) := by sorry

/-- Main theorem relating Pascal's Triangle to Leibniz's Triangle -/
theorem pascal_leibniz_relation (n k : ℕ) (h : k ≤ n) :
  leibniz n k = 1 / ((n + 1 : ℚ) * (binomial n k : ℚ)) := by sorry

end leibniz_recursive_relation_leibniz_boundary_condition_pascal_leibniz_relation_l3761_376195


namespace max_cookies_per_student_l3761_376185

/-- Proves the maximum number of cookies a single student can take in a class -/
theorem max_cookies_per_student
  (num_students : ℕ)
  (mean_cookies : ℕ)
  (h_num_students : num_students = 25)
  (h_mean_cookies : mean_cookies = 4)
  (h_min_cookie : ∀ student, student ≥ 1) :
  (num_students * mean_cookies) - (num_students - 1) = 76 := by
sorry

end max_cookies_per_student_l3761_376185


namespace calculate_second_month_sale_second_month_sale_l3761_376127

/-- Given sales figures for 5 out of 6 months and the average sale, 
    prove the sales figure for the remaining month. -/
theorem calculate_second_month_sale 
  (sale1 sale3 sale4 sale5 sale6 average_sale : ℕ) : ℕ :=
  let total_sales := average_sale * 6
  let known_sales := sale1 + sale3 + sale4 + sale5 + sale6
  let sale2 := total_sales - known_sales
  sale2

/-- The sales figure for the second month is 9000. -/
theorem second_month_sale : 
  calculate_second_month_sale 5400 6300 7200 4500 1200 5600 = 9000 := by
  sorry

end calculate_second_month_sale_second_month_sale_l3761_376127


namespace yellow_marbles_count_l3761_376155

/-- Given a jar with blue, red, and yellow marbles, this theorem proves
    the number of yellow marbles, given the number of blue and red marbles
    and the probability of picking a yellow marble. -/
theorem yellow_marbles_count
  (blue : ℕ) (red : ℕ) (prob_yellow : ℚ)
  (h_blue : blue = 7)
  (h_red : red = 11)
  (h_prob : prob_yellow = 1/4) :
  ∃ (yellow : ℕ), yellow = 6 ∧
    prob_yellow = yellow / (blue + red + yellow) :=
by sorry

end yellow_marbles_count_l3761_376155


namespace log_cutting_problem_l3761_376189

/-- Represents the number of cuts needed to divide logs into 1-meter pieces -/
def num_cuts (x y : ℕ) : ℕ := 2 * x + 3 * y

theorem log_cutting_problem :
  ∃ (x y : ℕ),
    x + y = 30 ∧
    3 * x + 4 * y = 100 ∧
    num_cuts x y = 70 :=
by sorry

end log_cutting_problem_l3761_376189


namespace broken_bamboo_equation_l3761_376191

theorem broken_bamboo_equation (x : ℝ) : 
  (0 ≤ x) ∧ (x ≤ 10) →
  x^2 + 3^2 = (10 - x)^2 :=
by sorry

/- Explanation of the Lean 4 statement:
   - We import Mathlib to access necessary mathematical definitions and theorems.
   - We define a theorem named 'broken_bamboo_equation'.
   - The theorem takes a real number 'x' as input, representing the height of the broken part.
   - The condition (0 ≤ x) ∧ (x ≤ 10) ensures that x is between 0 and 10 chi.
   - The equation x^2 + 3^2 = (10 - x)^2 represents the Pythagorean theorem applied to the scenario.
   - We use 'by sorry' to skip the proof, as requested.
-/

end broken_bamboo_equation_l3761_376191


namespace fraction_equality_l3761_376188

theorem fraction_equality (q r s t v : ℝ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : v / t = 4)
  (h4 : s / v = 1 / 3) :
  t / q = 1 / 2 := by
  sorry

end fraction_equality_l3761_376188


namespace cardinal_transitivity_l3761_376126

-- Define the theorem
theorem cardinal_transitivity (α β γ : Cardinal) 
  (h1 : α < β) (h2 : β < γ) : α < γ := by
  sorry

end cardinal_transitivity_l3761_376126


namespace complete_square_sum_l3761_376116

theorem complete_square_sum (x : ℝ) : ∃ (a b c : ℤ), 
  (25 * x^2 + 30 * x - 75 = 0 ↔ (a * x + b)^2 = c) ∧ 
  a > 0 ∧ 
  a + b + c = -58 := by
  sorry

end complete_square_sum_l3761_376116


namespace rounding_estimate_less_than_exact_l3761_376102

theorem rounding_estimate_less_than_exact (x y z : ℝ) 
  (hx : 1.5 < x ∧ x < 2) 
  (hy : 9 < y ∧ y < 9.5) 
  (hz : 3 < z ∧ z < 3.5) : 
  1 * 9 + 4 < x * y + z := by
  sorry

end rounding_estimate_less_than_exact_l3761_376102


namespace number_of_white_balls_l3761_376142

/-- Given the number of red and blue balls, and the relationship between red balls and the sum of blue and white balls, prove the number of white balls. -/
theorem number_of_white_balls (red blue : ℕ) (h1 : red = 60) (h2 : blue = 30) 
  (h3 : red = blue + white + 5) : white = 25 :=
by
  sorry

#check number_of_white_balls

end number_of_white_balls_l3761_376142


namespace students_checked_out_early_l3761_376129

theorem students_checked_out_early (initial_students remaining_students : ℕ) 
  (h1 : initial_students = 16)
  (h2 : remaining_students = 9) :
  initial_students - remaining_students = 7 :=
by sorry

end students_checked_out_early_l3761_376129


namespace painting_selection_ways_l3761_376138

theorem painting_selection_ways (oil_paintings : ℕ) (chinese_paintings : ℕ) (watercolor_paintings : ℕ)
  (h1 : oil_paintings = 3)
  (h2 : chinese_paintings = 4)
  (h3 : watercolor_paintings = 5) :
  oil_paintings + chinese_paintings + watercolor_paintings = 12 := by
  sorry

end painting_selection_ways_l3761_376138


namespace max_value_of_expression_l3761_376111

theorem max_value_of_expression (a b c d e f g h k : Int) 
  (ha : a = 1 ∨ a = -1) (hb : b = 1 ∨ b = -1) (hc : c = 1 ∨ c = -1)
  (hd : d = 1 ∨ d = -1) (he : e = 1 ∨ e = -1) (hf : f = 1 ∨ f = -1)
  (hg : g = 1 ∨ g = -1) (hh : h = 1 ∨ h = -1) (hk : k = 1 ∨ k = -1) :
  (∃ (a' b' c' d' e' f' g' h' k' : Int),
    (a' = 1 ∨ a' = -1) ∧ (b' = 1 ∨ b' = -1) ∧ (c' = 1 ∨ c' = -1) ∧
    (d' = 1 ∨ d' = -1) ∧ (e' = 1 ∨ e' = -1) ∧ (f' = 1 ∨ f' = -1) ∧
    (g' = 1 ∨ g' = -1) ∧ (h' = 1 ∨ h' = -1) ∧ (k' = 1 ∨ k' = -1) ∧
    a'*e'*k' - a'*f'*h' + b'*f'*g' - b'*d'*k' + c'*d'*h' - c'*e'*g' = 4) ∧
  (∀ (a' b' c' d' e' f' g' h' k' : Int),
    (a' = 1 ∨ a' = -1) → (b' = 1 ∨ b' = -1) → (c' = 1 ∨ c' = -1) →
    (d' = 1 ∨ d' = -1) → (e' = 1 ∨ e' = -1) → (f' = 1 ∨ f' = -1) →
    (g' = 1 ∨ g' = -1) → (h' = 1 ∨ h' = -1) → (k' = 1 ∨ k' = -1) →
    a'*e'*k' - a'*f'*h' + b'*f'*g' - b'*d'*k' + c'*d'*h' - c'*e'*g' ≤ 4) :=
by sorry

end max_value_of_expression_l3761_376111


namespace smallest_fraction_greater_than_five_sevenths_l3761_376167

theorem smallest_fraction_greater_than_five_sevenths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (5 : ℚ) / 7 < (a : ℚ) / b →
    (68 : ℚ) / 95 ≤ (a : ℚ) / b :=
by sorry

end smallest_fraction_greater_than_five_sevenths_l3761_376167


namespace population_change_l3761_376136

theorem population_change (x : ℝ) : 
  let initial_population : ℝ := 10000
  let first_year_population : ℝ := initial_population * (1 + x / 100)
  let second_year_population : ℝ := first_year_population * (1 - 5 / 100)
  second_year_population = 9975 → x = 5 := by
sorry

end population_change_l3761_376136


namespace poles_inside_base_l3761_376172

/-- A non-convex polygon representing the fence -/
structure Fence where
  isNonConvex : Bool

/-- A power line with poles -/
structure PowerLine where
  totalPoles : Nat

/-- A spy walking around the fence -/
structure Spy where
  totalCount : Nat

/-- The secret base surrounded by the fence -/
structure Base where
  polesInside : Nat

/-- Theorem stating the number of poles inside the base -/
theorem poles_inside_base 
  (fence : Fence) 
  (powerLine : PowerLine)
  (spy : Spy) :
  fence.isNonConvex = true →
  powerLine.totalPoles = 36 →
  spy.totalCount = 2015 →
  ∃ (base : Base), base.polesInside = 1 := by
  sorry

end poles_inside_base_l3761_376172


namespace symmetric_point_theorem_l3761_376134

/-- The coordinates of a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Given a point P, returns its symmetric point with respect to the origin -/
def symmetricPoint (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

/-- Theorem: The symmetric point of P(1, 3, -5) with respect to the origin is (-1, -3, 5) -/
theorem symmetric_point_theorem :
  let P : Point3D := { x := 1, y := 3, z := -5 }
  symmetricPoint P = { x := -1, y := -3, z := 5 } := by
  sorry


end symmetric_point_theorem_l3761_376134


namespace bernoulli_inequality_l3761_376124

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (h : x ≥ -1/3) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end bernoulli_inequality_l3761_376124


namespace cube_preserves_order_l3761_376101

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end cube_preserves_order_l3761_376101


namespace complex_power_modulus_l3761_376182

theorem complex_power_modulus : Complex.abs ((5 : ℂ) + (2 * Complex.I * Real.sqrt 3))^4 = 1369 := by
  sorry

end complex_power_modulus_l3761_376182
