import Mathlib

namespace three_students_left_l75_7516

/-- Calculates the number of students who left during the year. -/
def students_left (initial : ℕ) (new : ℕ) (final : ℕ) : ℕ :=
  initial + new - final

/-- Proves that 3 students left during the year given the initial, new, and final student counts. -/
theorem three_students_left : students_left 4 42 43 = 3 := by
  sorry

end three_students_left_l75_7516


namespace odd_spaced_stone_selections_count_l75_7550

/-- The number of ways to select 5 stones from 15 stones in a line, 
    such that there are an odd number of stones between any two selected stones. -/
def oddSpacedStoneSelections : ℕ := 77

/-- The total number of stones in the line. -/
def totalStones : ℕ := 15

/-- The number of stones to be selected. -/
def stonesToSelect : ℕ := 5

/-- The number of odd-numbered stones in the line. -/
def oddNumberedStones : ℕ := 8

/-- The number of even-numbered stones in the line. -/
def evenNumberedStones : ℕ := 7

theorem odd_spaced_stone_selections_count :
  oddSpacedStoneSelections = Nat.choose oddNumberedStones stonesToSelect + Nat.choose evenNumberedStones stonesToSelect :=
by sorry

end odd_spaced_stone_selections_count_l75_7550


namespace equation_system_solution_l75_7597

theorem equation_system_solution (a b : ℝ) : 
  ((a / 4 - 1) + 2 * (b / 3 + 2) = 4 ∧ 2 * (a / 4 - 1) + (b / 3 + 2) = 5) → 
  (a = 12 ∧ b = -3) := by sorry

end equation_system_solution_l75_7597


namespace quadratic_function_property_l75_7517

/-- A quadratic function y = x^2 + bx + c -/
def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_function_property (b c m n : ℝ) :
  (∀ x ≤ 2, quadratic_function b c (x + 0.01) < quadratic_function b c x) →
  quadratic_function b c m = n →
  quadratic_function b c (m + 1) = n →
  m ≥ 3/2 := by
  sorry

end quadratic_function_property_l75_7517


namespace map_distance_calculation_l75_7543

/-- Given a map with a scale of 1:1000000 and two points A and B that are 8 cm apart on the map,
    the actual distance between A and B is 80 km. -/
theorem map_distance_calculation (scale : ℚ) (map_distance : ℚ) (actual_distance : ℚ) :
  scale = 1 / 1000000 →
  map_distance = 8 →
  actual_distance = map_distance / scale →
  actual_distance = 80 * 100000 := by
  sorry


end map_distance_calculation_l75_7543


namespace prob_at_least_two_women_l75_7565

/-- The probability of selecting at least 2 women from a group of 8 men and 4 women when choosing 4 people at random -/
theorem prob_at_least_two_women (total : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) : 
  total = men + women →
  men = 8 →
  women = 4 →
  selected = 4 →
  (1 : ℚ) - (Nat.choose men selected / Nat.choose total selected + 
    (Nat.choose women 1 * Nat.choose men (selected - 1)) / Nat.choose total selected) = 67 / 165 := by
  sorry

end prob_at_least_two_women_l75_7565


namespace spherical_coordinates_conversion_l75_7584

/-- Converts non-standard spherical coordinates to standard form -/
def standardize_spherical (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Checks if spherical coordinates are in standard form -/
def is_standard_form (ρ θ φ : ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

theorem spherical_coordinates_conversion :
  let original := (5, 5 * Real.pi / 6, 9 * Real.pi / 4)
  let standard := (5, 11 * Real.pi / 6, 3 * Real.pi / 4)
  standardize_spherical original.1 original.2.1 original.2.2 = standard ∧
  is_standard_form standard.1 standard.2.1 standard.2.2 :=
sorry

end spherical_coordinates_conversion_l75_7584


namespace charlie_golden_delicious_l75_7513

/-- The number of bags of Golden Delicious apples Charlie picked -/
def golden_delicious : ℝ :=
  0.67 - (0.17 + 0.33)

theorem charlie_golden_delicious :
  golden_delicious = 0.17 := by
  sorry

end charlie_golden_delicious_l75_7513


namespace fruit_cost_price_l75_7535

/-- Calculates the total cost price of fruits sold given their selling prices, loss ratios, and quantities. -/
def total_cost_price (apple_sp orange_sp banana_sp : ℚ) 
                     (apple_loss orange_loss banana_loss : ℚ) 
                     (apple_qty orange_qty banana_qty : ℕ) : ℚ :=
  let apple_cp := apple_sp / (1 - apple_loss)
  let orange_cp := orange_sp / (1 - orange_loss)
  let banana_cp := banana_sp / (1 - banana_loss)
  apple_cp * apple_qty + orange_cp * orange_qty + banana_cp * banana_qty

/-- The total cost price of fruits sold is 947.45 given the specified conditions. -/
theorem fruit_cost_price : 
  total_cost_price 18 24 12 (1/6) (1/8) (1/4) 10 15 20 = 947.45 := by
  sorry

#eval total_cost_price 18 24 12 (1/6) (1/8) (1/4) 10 15 20

end fruit_cost_price_l75_7535


namespace intersection_of_A_and_B_l75_7572

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end intersection_of_A_and_B_l75_7572


namespace abc_sum_equals_888_l75_7533

/-- Given ABC + ABC + ABC = 888, where A, B, and C are all different single digit numbers, prove A = 2 -/
theorem abc_sum_equals_888 (A B C : ℕ) : 
  (100 * A + 10 * B + C) * 3 = 888 →
  A < 10 ∧ B < 10 ∧ C < 10 →
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A = 2 := by
sorry

end abc_sum_equals_888_l75_7533


namespace inequality_solution_and_a_range_l75_7548

def f (x : ℝ) := |3*x + 2|

theorem inequality_solution_and_a_range :
  (∃ S : Set ℝ, S = {x : ℝ | -5/4 < x ∧ x < 1/2} ∧
    ∀ x, x ∈ S ↔ f x < 4 - |x - 1|) ∧
  ∀ m n : ℝ, m > 0 → n > 0 → m + n = 1 →
    (∀ a : ℝ, a > 0 →
      (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) →
        0 < a ∧ a ≤ 10/3) :=
by sorry

end inequality_solution_and_a_range_l75_7548


namespace arithmetic_sequence_ratio_l75_7595

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d) (h3 : a 3 = 2 * a 1) :
  (a 1 + a 3) / (a 2 + a 4) = 3 / 4 := by
  sorry

end arithmetic_sequence_ratio_l75_7595


namespace fraction_ordering_l75_7529

theorem fraction_ordering : (4 : ℚ) / 13 < 12 / 37 ∧ 12 / 37 < 15 / 31 := by
  sorry

end fraction_ordering_l75_7529


namespace positive_reals_inequality_l75_7569

theorem positive_reals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) : 
  (a + b + c ≥ 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c) ∧ 
  (a^2 + b^2 + c^2 ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end positive_reals_inequality_l75_7569


namespace max_value_theorem_l75_7568

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : 2*x + 2*y + z = 1) : 
  3*x*y + y*z + z*x ≤ 1/5 := by
sorry

end max_value_theorem_l75_7568


namespace grade_assignment_count_l75_7576

theorem grade_assignment_count : (4 : ℕ) ^ 15 = 1073741824 := by
  sorry

end grade_assignment_count_l75_7576


namespace prob_at_least_one_woman_l75_7518

/-- The probability of selecting at least one woman when choosing 4 people at random from a group of 10 men and 5 women is 29/36. -/
theorem prob_at_least_one_woman (total : ℕ) (men : ℕ) (women : ℕ) (selection : ℕ) : 
  total = 15 → men = 10 → women = 5 → selection = 4 → 
  (1 - (men.choose selection / total.choose selection : ℚ)) = 29/36 := by
sorry

end prob_at_least_one_woman_l75_7518


namespace function_constant_l75_7530

/-- A function satisfying the given functional equation is constant -/
theorem function_constant (f : ℝ → ℝ) 
    (h : ∀ (x y : ℝ), x > 0 → y > 0 → f (Real.sqrt (x * y)) = f ((x + y) / 2)) :
  ∀ (a b : ℝ), a > 0 → b > 0 → f a = f b := by sorry

end function_constant_l75_7530


namespace city_population_problem_l75_7558

/-- Given three cities with the following conditions:
    - Richmond has 1000 more people than Victoria
    - Victoria has 4 times as many people as another city
    - Richmond has 3000 people
    Prove that the other city has 500 people. -/
theorem city_population_problem (richmond victoria other : ℕ) : 
  richmond = victoria + 1000 →
  victoria = 4 * other →
  richmond = 3000 →
  other = 500 := by
  sorry

end city_population_problem_l75_7558


namespace solve_equation_l75_7527

theorem solve_equation : ∃ x : ℝ, 15 * x = 5.7 ∧ x = 0.38 := by
  sorry

end solve_equation_l75_7527


namespace hyperbola_to_ellipse_l75_7549

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Theorem statement
theorem hyperbola_to_ellipse :
  ∀ x y : ℝ, hyperbola x y → 
  ∃ a b : ℝ, ellipse a b ∧ 
  (∀ c d : ℝ, hyperbola c 0 → (a = c ∨ a = -c) ∧ (b = 0)) :=
by sorry

end hyperbola_to_ellipse_l75_7549


namespace line_not_in_second_quadrant_l75_7525

/-- Represents a point in 2D space --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space --/
structure Line2D where
  k : ℝ
  b : ℝ

/-- Checks if a point is in the second quadrant --/
def isInSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is on the given line --/
def isOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.y = l.k * p.x + l.b

/-- Theorem: A line with positive slope and negative y-intercept does not pass through the second quadrant --/
theorem line_not_in_second_quadrant (l : Line2D) 
  (h1 : l.k > 0) (h2 : l.b < 0) : 
  ¬ ∃ p : Point2D, isInSecondQuadrant p ∧ isOnLine p l :=
by
  sorry


end line_not_in_second_quadrant_l75_7525


namespace museum_ring_display_height_l75_7579

/-- Calculates the total vertical distance of a sequence of rings -/
def total_vertical_distance (top_diameter : ℕ) (bottom_diameter : ℕ) (thickness : ℕ) : ℕ :=
  let n := (top_diameter - bottom_diameter) / 2 + 1
  let sum_inside_diameters := n * (top_diameter - thickness + bottom_diameter - thickness) / 2
  sum_inside_diameters + 2 * thickness

/-- Theorem stating that the total vertical distance for the given ring sequence is 325 cm -/
theorem museum_ring_display_height : total_vertical_distance 36 4 1 = 325 := by
  sorry

#eval total_vertical_distance 36 4 1

end museum_ring_display_height_l75_7579


namespace extracurricular_activity_selection_l75_7552

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem extracurricular_activity_selection 
  (total_people : ℕ) 
  (boys : ℕ) 
  (girls : ℕ) 
  (leaders : ℕ) 
  (to_select : ℕ) :
  total_people = 13 →
  boys = 8 →
  girls = 5 →
  leaders = 2 →
  to_select = 5 →
  (choose girls 1 * choose boys 4 = 350) ∧
  (choose (total_people - leaders) 3 = 165) ∧
  (choose total_people to_select - choose (total_people - leaders) to_select = 825) :=
by sorry

end extracurricular_activity_selection_l75_7552


namespace initial_cats_in_shelter_l75_7556

theorem initial_cats_in_shelter (initial_cats : ℕ) : 
  (initial_cats / 3 : ℚ) = (initial_cats / 3 : ℕ) →
  (4 * initial_cats / 3 + 8 * initial_cats / 3 : ℚ) = 60 →
  initial_cats = 15 := by
sorry

end initial_cats_in_shelter_l75_7556


namespace subset_sums_determine_set_l75_7561

def three_element_subset_sums (A : Finset ℤ) : Finset ℤ :=
  (A.powerset.filter (λ s => s.card = 3)).image (λ s => s.sum id)

theorem subset_sums_determine_set :
  ∀ A : Finset ℤ,
    A.card = 4 →
    three_element_subset_sums A = {-1, 3, 5, 8} →
    A = {-3, 0, 2, 6} := by
  sorry

end subset_sums_determine_set_l75_7561


namespace triangular_front_view_solids_l75_7545

/-- Enumeration of possible solids --/
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

/-- Definition of a solid with a triangular front view --/
def has_triangular_front_view (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True
  | Solid.Cone => True
  | _ => False

/-- Theorem stating that a solid with a triangular front view must be one of the specified solids --/
theorem triangular_front_view_solids (s : Solid) :
  has_triangular_front_view s →
  (s = Solid.TriangularPyramid ∨ s = Solid.SquarePyramid ∨ s = Solid.TriangularPrism ∨ s = Solid.Cone) :=
by
  sorry

end triangular_front_view_solids_l75_7545


namespace base_10_to_base_8_l75_7511

theorem base_10_to_base_8 : 
  (3 * 8^3 + 1 * 8^2 + 4 * 8^1 + 0 * 8^0 : ℕ) = 1632 := by
  sorry

end base_10_to_base_8_l75_7511


namespace f_properties_l75_7542

open Real

noncomputable def f (x : ℝ) : ℝ := 2 / x + log x

theorem f_properties :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≥ f 2) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ = f x₂ → x₁ + x₂ > 4) :=
by sorry

end f_properties_l75_7542


namespace intersection_A_B_l75_7593

-- Define set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 + 2*x ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 0} := by
  sorry

end intersection_A_B_l75_7593


namespace polynomial_simplification_l75_7510

theorem polynomial_simplification (x : ℝ) :
  (5 * x^12 - 3 * x^9 + 6 * x^8 - 2 * x^7) + 
  (7 * x^12 + 2 * x^11 - x^9 + 4 * x^7 + 2 * x^5 - x + 3) = 
  12 * x^12 + 2 * x^11 - 4 * x^9 + 6 * x^8 + 2 * x^7 + 2 * x^5 - x + 3 :=
by sorry

end polynomial_simplification_l75_7510


namespace computer_multiplications_l75_7587

/-- Represents the number of multiplications a computer can perform per minute -/
def multiplications_per_minute : ℕ := 25000

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Represents the number of hours we're calculating for -/
def hours : ℕ := 3

/-- Theorem stating that the computer will perform 4,500,000 multiplications in three hours -/
theorem computer_multiplications :
  multiplications_per_minute * minutes_per_hour * hours = 4500000 :=
by sorry

end computer_multiplications_l75_7587


namespace geometric_mean_problem_l75_7515

theorem geometric_mean_problem (k : ℝ) :
  (k + 9) * (6 - k) = (2 * k)^2 → k = 3 := by
sorry

end geometric_mean_problem_l75_7515


namespace bill_profit_l75_7500

-- Define the given conditions
def total_milk : ℚ := 16
def butter_ratio : ℚ := 1/4
def sour_cream_ratio : ℚ := 1/4
def milk_to_butter : ℚ := 4
def milk_to_sour_cream : ℚ := 2
def butter_price : ℚ := 5
def sour_cream_price : ℚ := 6
def whole_milk_price : ℚ := 3

-- Define the theorem
theorem bill_profit : 
  let milk_for_butter := total_milk * butter_ratio
  let milk_for_sour_cream := total_milk * sour_cream_ratio
  let butter_gallons := milk_for_butter / milk_to_butter
  let sour_cream_gallons := milk_for_sour_cream / milk_to_sour_cream
  let whole_milk_gallons := total_milk - milk_for_butter - milk_for_sour_cream
  let butter_profit := butter_gallons * butter_price
  let sour_cream_profit := sour_cream_gallons * sour_cream_price
  let whole_milk_profit := whole_milk_gallons * whole_milk_price
  butter_profit + sour_cream_profit + whole_milk_profit = 41 := by
sorry

end bill_profit_l75_7500


namespace perpendicular_lines_b_value_l75_7524

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- First line equation: 3y - 3b = 9x -/
def line1 (x y b : ℝ) : Prop := 3 * y - 3 * b = 9 * x

/-- Second line equation: y - 2 = (b + 9)x -/
def line2 (x y b : ℝ) : Prop := y - 2 = (b + 9) * x

theorem perpendicular_lines_b_value :
  ∀ b : ℝ, (∃ x y : ℝ, line1 x y b ∧ line2 x y b ∧
    perpendicular 3 (b + 9)) → b = -28/3 := by
  sorry

end perpendicular_lines_b_value_l75_7524


namespace geometric_sequence_10th_term_l75_7544

theorem geometric_sequence_10th_term :
  let a₁ : ℚ := 5
  let r : ℚ := 5 / 3
  let n : ℕ := 10
  let aₙ := a₁ * r^(n - 1)
  aₙ = 9765625 / 19683 :=
by sorry

end geometric_sequence_10th_term_l75_7544


namespace binary_11010_equals_octal_32_l75_7583

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The binary representation of the number 11010 -/
def binary_11010 : List Bool := [false, true, false, true, true]

/-- The octal representation of the number 32 -/
def octal_32 : List ℕ := [3, 2]

theorem binary_11010_equals_octal_32 :
  decimal_to_octal (binary_to_decimal binary_11010) = octal_32 := by
  sorry

end binary_11010_equals_octal_32_l75_7583


namespace function_lower_bound_l75_7590

open Real

/-- A function satisfying the given inequality for all real x -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x, Real.sqrt (2 * f x) - Real.sqrt (2 * f x - f (2 * x)) ≥ 2

/-- The main theorem to be proved -/
theorem function_lower_bound
  (f : ℝ → ℝ) (h : SatisfiesInequality f) :
  ∀ x, f x ≥ 7 := by
  sorry

end function_lower_bound_l75_7590


namespace power_fraction_simplification_l75_7523

theorem power_fraction_simplification : (25 ^ 40) / (125 ^ 20) = 5 ^ 20 := by
  sorry

end power_fraction_simplification_l75_7523


namespace subtract_negatives_example_l75_7581

theorem subtract_negatives_example : (-3) - (-5) = 2 := by
  sorry

end subtract_negatives_example_l75_7581


namespace polygon_sides_from_angle_sum_l75_7534

theorem polygon_sides_from_angle_sum (sum_of_angles : ℝ) :
  sum_of_angles = 1080 → ∃ n : ℕ, n = 8 ∧ sum_of_angles = 180 * (n - 2) := by
  sorry

end polygon_sides_from_angle_sum_l75_7534


namespace line_point_k_value_l75_7532

/-- A line contains the points (7, 10), (-3, k), and (-11, 5). The value of k is 65/9. -/
theorem line_point_k_value :
  ∀ (k : ℚ),
  (∃ (m b : ℚ),
    (7 : ℚ) * m + b = 10 ∧
    (-3 : ℚ) * m + b = k ∧
    (-11 : ℚ) * m + b = 5) →
  k = 65 / 9 := by
sorry

end line_point_k_value_l75_7532


namespace frog_jump_probability_l75_7522

/-- Represents a position on the grid -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the grid -/
def Grid := {p : Position // p.x ≤ 5 ∧ p.y ≤ 5}

/-- The blocked cell -/
def blockedCell : Position := ⟨3, 3⟩

/-- Check if a position is on the grid boundary -/
def isOnBoundary (p : Position) : Bool :=
  p.x = 0 ∨ p.x = 5 ∨ p.y = 0 ∨ p.y = 5

/-- Check if a position is on a vertical side of the grid -/
def isOnVerticalSide (p : Position) : Bool :=
  p.x = 0 ∨ p.x = 5

/-- Probability of ending on a vertical side starting from a given position -/
noncomputable def probabilityVerticalSide (p : Position) : Real :=
  sorry

/-- Theorem: The probability of ending on a vertical side starting from (2,2) is 5/8 -/
theorem frog_jump_probability :
  probabilityVerticalSide ⟨2, 2⟩ = 5/8 := by sorry

end frog_jump_probability_l75_7522


namespace final_liquid_X_percentage_l75_7501

/-- Composition of a solution -/
structure Solution :=
  (x : ℝ) -- Percentage of liquid X
  (water : ℝ) -- Percentage of water
  (z : ℝ) -- Percentage of liquid Z

/-- Given conditions -/
def solution_Y : Solution := ⟨20, 55, 25⟩
def initial_weight : ℝ := 12
def evaporated_water : ℝ := 4
def added_Y_weight : ℝ := 3
def solution_B : Solution := ⟨35, 15, 50⟩
def added_B_weight : ℝ := 2
def evaporation_factor : ℝ := 0.75
def solution_D : Solution := ⟨15, 60, 25⟩
def added_D_weight : ℝ := 6

/-- The theorem to prove -/
theorem final_liquid_X_percentage :
  let initial_X := solution_Y.x * initial_weight / 100
  let initial_Z := solution_Y.z * initial_weight / 100
  let remaining_water := solution_Y.water * initial_weight / 100 - evaporated_water
  let added_Y_X := solution_Y.x * added_Y_weight / 100
  let added_Y_water := solution_Y.water * added_Y_weight / 100
  let added_Y_Z := solution_Y.z * added_Y_weight / 100
  let added_B_X := solution_B.x * added_B_weight / 100
  let added_B_water := solution_B.water * added_B_weight / 100
  let added_B_Z := solution_B.z * added_B_weight / 100
  let total_before_evap := initial_X + initial_Z + remaining_water + added_Y_X + added_Y_water + added_Y_Z + added_B_X + added_B_water + added_B_Z
  let total_after_evap := total_before_evap * evaporation_factor
  let evaporated_water_2 := (1 - evaporation_factor) * (remaining_water + added_Y_water + added_B_water)
  let remaining_water_2 := remaining_water + added_Y_water + added_B_water - evaporated_water_2
  let added_D_X := solution_D.x * added_D_weight / 100
  let added_D_water := solution_D.water * added_D_weight / 100
  let added_D_Z := solution_D.z * added_D_weight / 100
  let final_X := initial_X + added_Y_X + added_B_X + added_D_X
  let final_water := remaining_water_2 + added_D_water
  let final_Z := initial_Z + added_Y_Z + added_B_Z + added_D_Z
  let final_total := final_X + final_water + final_Z
  final_X / final_total * 100 = 25.75 := by sorry

end final_liquid_X_percentage_l75_7501


namespace special_function_property_l75_7596

/-- A function satisfying g(xy) = g(x)/y for all positive real numbers x and y -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y

theorem special_function_property (g : ℝ → ℝ) 
    (h1 : SpecialFunction g) 
    (h2 : g 30 = 30) : 
    g 45 = 20 := by
  sorry

end special_function_property_l75_7596


namespace subset_sum_inequality_l75_7539

theorem subset_sum_inequality (n m : ℕ) (A : Finset ℕ) (h_m : m > 0) (h_n : n > 0) 
  (h_subset : A ⊆ Finset.range n)
  (h_closure : ∀ (i j : ℕ), i ∈ A → j ∈ A → i + j ≤ n → i + j ∈ A) :
  (A.sum id) / m ≥ (n + 1) / 2 := by
sorry

end subset_sum_inequality_l75_7539


namespace least_common_multiple_of_primes_l75_7509

theorem least_common_multiple_of_primes : ∃ n : ℕ,
  (n > 0) ∧
  (n % 7 = 0) ∧ (n % 11 = 0) ∧ (n % 13 = 0) ∧
  (∀ m : ℕ, m > 0 ∧ m % 7 = 0 ∧ m % 11 = 0 ∧ m % 13 = 0 → m ≥ n) ∧
  n = 1001 := by
sorry

end least_common_multiple_of_primes_l75_7509


namespace cake_triangles_l75_7536

/-- The number of triangular pieces that can be cut from a rectangular cake -/
theorem cake_triangles (cake_length cake_width triangle_base triangle_height : ℝ) 
  (h1 : cake_length = 24)
  (h2 : cake_width = 20)
  (h3 : triangle_base = 2)
  (h4 : triangle_height = 2) :
  (cake_length * cake_width) / (1/2 * triangle_base * triangle_height) = 240 :=
by sorry

end cake_triangles_l75_7536


namespace camp_attendance_l75_7546

theorem camp_attendance (stay_home : ℕ) (difference : ℕ) (camp : ℕ) : 
  stay_home = 777622 → difference = 574664 → camp + difference = stay_home → camp = 202958 := by
sorry

end camp_attendance_l75_7546


namespace complex_fraction_sum_l75_7588

theorem complex_fraction_sum (z : ℂ) (a b : ℝ) : 
  z = (2 + I) / (1 - 2*I) → 
  z = Complex.mk a b → 
  a + b = 1 :=
by sorry

end complex_fraction_sum_l75_7588


namespace quadratic_equation_solution_l75_7574

/-- The quadratic equation (k-2)x^2 + 3x + k^2 - 4 = 0 has one solution as x = 0 -/
def has_zero_solution (k : ℝ) : Prop :=
  k^2 - 4 = 0

/-- The coefficient of x^2 is not zero -/
def is_quadratic (k : ℝ) : Prop :=
  k - 2 ≠ 0

theorem quadratic_equation_solution :
  ∀ k : ℝ, has_zero_solution k → is_quadratic k → k = -2 :=
by
  sorry

end quadratic_equation_solution_l75_7574


namespace tylers_and_brothers_age_sum_l75_7508

theorem tylers_and_brothers_age_sum : 
  ∀ (tyler_age brother_age : ℕ),
    tyler_age = 7 →
    brother_age = tyler_age + 3 →
    tyler_age + brother_age = 17 := by
  sorry

end tylers_and_brothers_age_sum_l75_7508


namespace strongest_signal_l75_7537

def signal_strength (x : ℤ) : ℝ := |x|

def is_stronger (x y : ℤ) : Prop := signal_strength x < signal_strength y

theorem strongest_signal :
  let signals : List ℤ := [-50, -60, -70, -80]
  ∀ s ∈ signals, s ≠ -50 → is_stronger (-50) s :=
sorry

end strongest_signal_l75_7537


namespace hexagon_angle_measure_l75_7564

theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  -- ABCDEF is a convex hexagon (sum of angles is 720°)
  A + B + C + D + E + F = 720 →
  -- Angles A, B, C, and D are congruent
  A = B ∧ B = C ∧ C = D →
  -- Angles E and F are congruent
  E = F →
  -- Measure of angle A is 30° less than measure of angle E
  A + 30 = E →
  -- Conclusion: Measure of angle E is 140°
  E = 140 := by
sorry

end hexagon_angle_measure_l75_7564


namespace inequality_equivalence_l75_7570

theorem inequality_equivalence (x : ℝ) : (x - 2) / (x - 4) ≥ 3 ↔ x ∈ Set.Ioo 4 5 ∪ {5} := by
  sorry

end inequality_equivalence_l75_7570


namespace elroy_extra_miles_l75_7571

/-- Proves that Elroy walks 5 more miles than last year's winner to collect the same amount -/
theorem elroy_extra_miles
  (last_year_rate : ℝ)
  (this_year_rate : ℝ)
  (last_year_amount : ℝ)
  (h1 : last_year_rate = 4)
  (h2 : this_year_rate = 2.75)
  (h3 : last_year_amount = 44) :
  (last_year_amount / this_year_rate) - (last_year_amount / last_year_rate) = 5 := by
sorry

end elroy_extra_miles_l75_7571


namespace triangle_area_implies_sin_A_l75_7528

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_inequality : a < b + c ∧ b < a + c ∧ c < a + b)

-- Define the area of the triangle
def area (t : Triangle) : ℝ := t.a^2 - (t.b - t.c)^2

-- State the theorem
theorem triangle_area_implies_sin_A (t : Triangle) (h_area : area t = t.a^2 - (t.b - t.c)^2) :
  let A := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))
  Real.sin A = 8 / 17 := by
  sorry

end triangle_area_implies_sin_A_l75_7528


namespace max_value_constraint_l75_7504

theorem max_value_constraint (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 3) :
  4 * a * b * Real.sqrt 3 + 12 * b * c ≤ Real.sqrt 39 :=
by sorry

end max_value_constraint_l75_7504


namespace equal_intercepts_implies_a_value_l75_7547

/-- Given two points A(0, 1) and B(4, a) on a line, if the x-intercept and y-intercept of the line are equal, then a = -3. -/
theorem equal_intercepts_implies_a_value (a : ℝ) : 
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (4, a)
  let m : ℝ := (a - 1) / 4  -- Slope of the line AB
  let x_intercept : ℝ := 4 / (1 - m)  -- x-intercept formula
  let y_intercept : ℝ := a - m * 4  -- y-intercept formula
  x_intercept = y_intercept → a = -3 :=
by sorry

end equal_intercepts_implies_a_value_l75_7547


namespace arithmetic_equality_l75_7557

theorem arithmetic_equality : 54 + 98 / 14 + 23 * 17 - 200 - 312 / 6 = 200 := by
  sorry

end arithmetic_equality_l75_7557


namespace find_a_and_b_l75_7538

-- Define the curve equation
def curve (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 36

-- Define the theorem
theorem find_a_and_b :
  ∀ a b : ℝ, curve 0 (-12) a b → curve 0 0 a b → a = 0 ∧ b = -6 :=
by
  sorry

end find_a_and_b_l75_7538


namespace inscribed_cone_volume_ratio_l75_7505

/-- A right circular cone inscribed in a right prism -/
structure InscribedCone where
  /-- Radius of the cone's base -/
  r : ℝ
  /-- Height of both the cone and the prism -/
  h : ℝ
  /-- The radius and height are positive -/
  r_pos : r > 0
  h_pos : h > 0

/-- Theorem: The ratio of the volume of the inscribed cone to the volume of the prism is π/12 -/
theorem inscribed_cone_volume_ratio (c : InscribedCone) :
  (1 / 3 * π * c.r^2 * c.h) / (4 * c.r^2 * c.h) = π / 12 := by
  sorry

end inscribed_cone_volume_ratio_l75_7505


namespace initial_velocity_is_three_l75_7580

-- Define the displacement function
def displacement (t : ℝ) : ℝ := 3 * t - t^2

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 3 - 2 * t

-- Theorem statement
theorem initial_velocity_is_three :
  velocity 0 = 3 :=
sorry

end initial_velocity_is_three_l75_7580


namespace sphere_surface_inequality_l75_7512

theorem sphere_surface_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  (x - y) * (y - z) * (x - z) ≤ 1 / Real.sqrt 2 := by
  sorry

end sphere_surface_inequality_l75_7512


namespace rational_sum_product_quotient_zero_l75_7563

theorem rational_sum_product_quotient_zero (a b : ℚ) :
  (a + b) / (a * b) = 0 → a ≠ 0 ∧ b ≠ 0 ∧ a = -b := by
  sorry

end rational_sum_product_quotient_zero_l75_7563


namespace twentieth_base5_is_40_l75_7540

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : ℕ :=
  if n < 5 then n
  else 10 * toBase5 (n / 5) + (n % 5)

/-- The 20th number in base 5 sequence -/
def twentieth_base5 : ℕ := toBase5 20

theorem twentieth_base5_is_40 : twentieth_base5 = 40 := by
  sorry

end twentieth_base5_is_40_l75_7540


namespace hyperbola_properties_l75_7502

/-- Given a hyperbola with equation x²/4 - y² = 1, prove its asymptotes and eccentricity -/
theorem hyperbola_properties (x y : ℝ) :
  x^2 / 4 - y^2 = 1 →
  (∃ (k : ℝ), k = 1/2 ∧ (y = k*x ∨ y = -k*x)) ∧
  (∃ (e : ℝ), e = Real.sqrt 5 / 2 ∧ e > 1) :=
by sorry

end hyperbola_properties_l75_7502


namespace arun_weight_theorem_l75_7553

def arun_weight_conditions (w : ℕ) : Prop :=
  64 < w ∧ w < 72 ∧ w % 3 = 0 ∧  -- Arun's condition
  60 < w ∧ w < 70 ∧ w % 2 = 0 ∧  -- Brother's condition
  w ≤ 67 ∧ Nat.Prime w ∧         -- Mother's condition
  63 ≤ w ∧ w ≤ 71 ∧ w % 5 = 0 ∧  -- Sister's condition
  62 < w ∧ w ≤ 73 ∧ w % 4 = 0    -- Father's condition

theorem arun_weight_theorem :
  ∃! w : ℕ, arun_weight_conditions w ∧ w = 66 := by
  sorry

end arun_weight_theorem_l75_7553


namespace quadratic_trinomial_existence_l75_7514

theorem quadratic_trinomial_existence : ∃ f : ℝ → ℝ, 
  (∀ x, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧ 
  f 2014 = 2015 ∧ 
  f 2015 = 0 ∧ 
  f 2016 = 2015 := by
  sorry

end quadratic_trinomial_existence_l75_7514


namespace chord_intercept_l75_7575

/-- The value of 'a' in the equation of a line that intercepts a chord of length √3 on a circle -/
theorem chord_intercept (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 3 ∧ x + y + a = 0) →  -- Line intersects circle
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 + y1^2 = 3 ∧ x2^2 + y2^2 = 3 ∧  -- Two points on circle
    x1 + y1 + a = 0 ∧ x2 + y2 + a = 0 ∧  -- Two points on line
    (x1 - x2)^2 + (y1 - y2)^2 = 3) →  -- Distance between points is √3
  a = 3 * Real.sqrt 2 / 2 ∨ a = -3 * Real.sqrt 2 / 2 :=
sorry

end chord_intercept_l75_7575


namespace percentage_non_swimmers_basketball_l75_7577

/-- Represents the percentage of students who play basketball -/
def basketball_players : ℝ := 0.7

/-- Represents the percentage of students who swim -/
def swimmers : ℝ := 0.5

/-- Represents the percentage of basketball players who also swim -/
def basketball_and_swim : ℝ := 0.3

/-- Theorem: The percentage of non-swimmers who play basketball is 98% -/
theorem percentage_non_swimmers_basketball : 
  (basketball_players - basketball_players * basketball_and_swim) / (1 - swimmers) = 0.98 := by
  sorry

end percentage_non_swimmers_basketball_l75_7577


namespace profit_and_sales_maximization_max_profit_with_constraints_l75_7582

/-- Represents the daily sales quantity as a function of the selling price -/
def sales_quantity (x : ℝ) : ℝ := -10 * x + 400

/-- Represents the daily profit as a function of the selling price -/
def profit (x : ℝ) : ℝ := sales_quantity x * (x - 10)

/-- The cost price of the item -/
def cost_price : ℝ := 10

/-- The domain constraints for the selling price -/
def price_domain (x : ℝ) : Prop := 10 < x ∧ x ≤ 40

theorem profit_and_sales_maximization (x : ℝ) 
  (h : price_domain x) : 
  profit x = 1250 ∧ 
  (∀ y, price_domain y → sales_quantity x ≥ sales_quantity y) → 
  x = 15 :=
sorry

theorem max_profit_with_constraints (x : ℝ) :
  28 ≤ x ∧ x ≤ 35 →
  profit x ≤ 2160 :=
sorry

end profit_and_sales_maximization_max_profit_with_constraints_l75_7582


namespace forest_foxes_l75_7586

theorem forest_foxes (total : ℕ) (deer_fraction : ℚ) (fox_fraction : ℚ) : 
  total = 160 →
  deer_fraction = 7 / 8 →
  fox_fraction = 1 - deer_fraction →
  (fox_fraction * total : ℚ) = 20 := by
  sorry

end forest_foxes_l75_7586


namespace sandras_mother_contribution_sandras_mother_contribution_proof_l75_7559

theorem sandras_mother_contribution : ℝ → Prop :=
  fun m =>
    let savings : ℝ := 10
    let father_contribution : ℝ := 2 * m
    let total_money : ℝ := savings + m + father_contribution
    let candy_cost : ℝ := 0.5
    let jelly_bean_cost : ℝ := 0.2
    let candy_quantity : ℕ := 14
    let jelly_bean_quantity : ℕ := 20
    let total_cost : ℝ := candy_cost * candy_quantity + jelly_bean_cost * jelly_bean_quantity
    let money_left : ℝ := 11
    total_money = total_cost + money_left → m = 4

theorem sandras_mother_contribution_proof : ∃ m, sandras_mother_contribution m :=
  sorry

end sandras_mother_contribution_sandras_mother_contribution_proof_l75_7559


namespace fencing_requirement_l75_7599

/-- A rectangular field with one side of 20 feet and an area of 80 sq. feet requires 28 feet of fencing for the other three sides. -/
theorem fencing_requirement (length width : ℝ) : 
  length = 20 → 
  length * width = 80 → 
  length + 2 * width = 28 := by sorry

end fencing_requirement_l75_7599


namespace complex_magnitude_problem_l75_7567

theorem complex_magnitude_problem : 
  let z : ℂ := (2 - Complex.I)^2 / Complex.I
  Complex.abs z = 5 := by
sorry

end complex_magnitude_problem_l75_7567


namespace largest_five_digit_divisible_by_six_l75_7520

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def divisible_by_six (n : ℕ) : Prop := n % 6 = 0

theorem largest_five_digit_divisible_by_six :
  ∀ n : ℕ, is_five_digit n → divisible_by_six n → n ≤ 99996 :=
by
  sorry

end largest_five_digit_divisible_by_six_l75_7520


namespace ball_box_problem_l75_7573

/-- Given an opaque box with balls of three colors: red, yellow, and blue. -/
structure BallBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ
  total_eq : total = red + yellow + blue
  yellow_eq : yellow = 2 * blue

/-- The probability of drawing a blue ball from the box -/
def blue_probability (box : BallBox) : ℚ :=
  box.blue / box.total

/-- The number of additional blue balls needed to make the probability 1/2 -/
def additional_blue_balls (box : BallBox) : ℕ :=
  let new_total := box.total + 14
  let new_blue := box.blue + 14
  14

/-- Theorem stating the properties of the specific box in the problem -/
theorem ball_box_problem :
  ∃ (box : BallBox),
    box.total = 30 ∧
    box.red = 6 ∧
    blue_probability box = 4 / 15 ∧
    additional_blue_balls box = 14 ∧
    blue_probability ⟨box.total + 14, box.red, box.blue + 14, box.yellow,
      by sorry, by sorry⟩ = 1 / 2 := by
  sorry


end ball_box_problem_l75_7573


namespace sum_of_squares_problem_l75_7506

theorem sum_of_squares_problem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + c^2 = 75) (h5 : a*b + b*c + c*a = 40) (h6 : c = 5) :
  a + b + c = 5 * Real.sqrt 62 := by
  sorry

end sum_of_squares_problem_l75_7506


namespace eight_digit_increasing_remainder_l75_7566

theorem eight_digit_increasing_remainder (n : ℕ) (h : n = 8) :
  (Nat.choose (n + 9 - 1) n) % 1000 = 870 := by
  sorry

end eight_digit_increasing_remainder_l75_7566


namespace curve_fixed_point_l75_7551

/-- The curve C passes through a fixed point for all k ≠ -1 -/
theorem curve_fixed_point (k : ℝ) (hk : k ≠ -1) :
  ∃ (x y : ℝ), ∀ (k : ℝ), k ≠ -1 →
    x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0 ∧ x = 1 ∧ y = -3 := by
  sorry

end curve_fixed_point_l75_7551


namespace twentieth_term_of_sequence_l75_7560

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem twentieth_term_of_sequence :
  arithmetic_sequence 2 3 20 = 59 := by
  sorry

end twentieth_term_of_sequence_l75_7560


namespace factorization_equality_l75_7562

theorem factorization_equality (c : ℝ) : 196 * c^3 + 28 * c^2 = 28 * c^2 * (7 * c + 1) := by
  sorry

end factorization_equality_l75_7562


namespace min_value_of_u_l75_7592

/-- Given that x and y are real numbers satisfying 2x + y ≥ 1, 
    the function u = x² + 4x + y² - 2y has a minimum value of -9/5 -/
theorem min_value_of_u (x y : ℝ) (h : 2 * x + y ≥ 1) :
  ∃ (min_u : ℝ), min_u = -9/5 ∧ ∀ (x' y' : ℝ), 2 * x' + y' ≥ 1 → 
    x'^2 + 4*x' + y'^2 - 2*y' ≥ min_u :=
by sorry

end min_value_of_u_l75_7592


namespace intersection_of_A_and_B_l75_7503

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l75_7503


namespace arcade_ticket_difference_l75_7598

def arcade_tickets : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ := fun initial toys clothes food accessories =>
  let food_discounted := food * 0.85
  let combined := clothes + food_discounted + accessories
  combined - toys

theorem arcade_ticket_difference : arcade_tickets 250 58 85 60 45.5 = 123.5 := by
  sorry

end arcade_ticket_difference_l75_7598


namespace right_triangle_exists_l75_7531

theorem right_triangle_exists (a : ℤ) (h : a ≥ 5) :
  ∃ b c : ℤ, c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 := by
  sorry

end right_triangle_exists_l75_7531


namespace paris_total_study_hours_l75_7578

/-- The number of hours Paris studies during the semester -/
def paris_study_hours : ℕ :=
  let weeks_in_semester : ℕ := 15
  let weekday_study_hours : ℕ := 3
  let saturday_study_hours : ℕ := 4
  let sunday_study_hours : ℕ := 5
  let weekdays_per_week : ℕ := 5
  let weekly_study_hours : ℕ := weekday_study_hours * weekdays_per_week + saturday_study_hours + sunday_study_hours
  weekly_study_hours * weeks_in_semester

theorem paris_total_study_hours : paris_study_hours = 360 := by
  sorry

end paris_total_study_hours_l75_7578


namespace light_glow_theorem_l75_7585

def seconds_since_midnight (hours minutes seconds : ℕ) : ℕ :=
  hours * 3600 + minutes * 60 + seconds

def light_glow_count (start_time end_time glow_interval : ℕ) : ℕ :=
  (end_time - start_time) / glow_interval

theorem light_glow_theorem (start_a start_b start_c end_time : ℕ) 
  (interval_a interval_b interval_c : ℕ) : 
  let count_a := light_glow_count start_a end_time interval_a
  let count_b := light_glow_count start_b end_time interval_b
  let count_c := light_glow_count start_c end_time interval_c
  ∃ (x y z : ℕ), x = count_a ∧ y = count_b ∧ z = count_c := by
  sorry

#eval light_glow_count (seconds_since_midnight 1 57 58) (seconds_since_midnight 3 20 47) 14
#eval light_glow_count (seconds_since_midnight 2 0 25) (seconds_since_midnight 3 20 47) 21
#eval light_glow_count (seconds_since_midnight 2 10 15) (seconds_since_midnight 3 20 47) 10

end light_glow_theorem_l75_7585


namespace select_defective_theorem_l75_7594

/-- The number of ways to select at least 2 defective products -/
def select_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℕ :=
  Nat.choose defective 2 * Nat.choose (total - defective) (selected - 2) +
  Nat.choose defective 3 * Nat.choose (total - defective) (selected - 3)

/-- Theorem stating the number of ways to select at least 2 defective products
    from 5 randomly selected products out of 200 total products with 3 defective products -/
theorem select_defective_theorem :
  select_defective 200 3 5 = Nat.choose 3 2 * Nat.choose 197 3 + Nat.choose 3 3 * Nat.choose 197 2 := by
  sorry

end select_defective_theorem_l75_7594


namespace remaining_students_l75_7521

/-- Given a group of students divided into 3 groups of 8, with 2 students leaving early,
    prove that 22 students remain. -/
theorem remaining_students (initial_groups : Nat) (students_per_group : Nat) (students_left : Nat) :
  initial_groups = 3 →
  students_per_group = 8 →
  students_left = 2 →
  initial_groups * students_per_group - students_left = 22 := by
  sorry

end remaining_students_l75_7521


namespace smallest_factor_for_perfect_square_l75_7519

theorem smallest_factor_for_perfect_square : ∃ (y : ℕ), 
  (y > 0) ∧ 
  (∃ (n : ℕ), 76545 * y = n^2) ∧
  (y % 3 ≠ 0) ∧ 
  (y % 5 ≠ 0) ∧
  (∀ (z : ℕ), z > 0 ∧ z < y → ¬(∃ (m : ℕ), 76545 * z = m^2) ∨ (z % 3 = 0) ∨ (z % 5 = 0)) ∧
  y = 7 := by
sorry

end smallest_factor_for_perfect_square_l75_7519


namespace cit_beaver_difference_l75_7589

/-- A Beaver-number is a positive 5-digit integer whose digit sum is divisible by 17. -/
def is_beaver_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ (n.digits 10).sum % 17 = 0

/-- A Beaver-pair is a pair of consecutive Beaver-numbers. -/
def is_beaver_pair (m n : ℕ) : Prop :=
  is_beaver_number m ∧ is_beaver_number n ∧ n = m + 1

/-- An MIT Beaver is the smaller number in a Beaver-pair. -/
def is_mit_beaver (m : ℕ) : Prop :=
  ∃ n, is_beaver_pair m n

/-- A CIT Beaver is the larger number in a Beaver-pair. -/
def is_cit_beaver (n : ℕ) : Prop :=
  ∃ m, is_beaver_pair m n

/-- The theorem stating the difference between the maximum and minimum CIT Beaver numbers. -/
theorem cit_beaver_difference : 
  ∃ max min : ℕ, 
    is_cit_beaver max ∧ 
    is_cit_beaver min ∧ 
    (∀ n, is_cit_beaver n → n ≤ max) ∧ 
    (∀ n, is_cit_beaver n → min ≤ n) ∧ 
    max - min = 79200 :=
sorry

end cit_beaver_difference_l75_7589


namespace soccer_team_wins_l75_7555

/-- Given a soccer team that played 140 games and won 50 percent of them,
    prove that the number of games won is 70. -/
theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (games_won : ℕ) :
  total_games = 140 →
  win_percentage = 1/2 →
  games_won = (total_games : ℚ) * win_percentage →
  games_won = 70 := by
  sorry

end soccer_team_wins_l75_7555


namespace division_problem_l75_7507

theorem division_problem : ∃ x : ℝ, (3.242 * 15) / x = 0.04863 := by
  sorry

end division_problem_l75_7507


namespace problem_statement_l75_7541

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem problem_statement (f : ℝ → ℝ) (m n : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f m - f n > f (-m) - f (-n)) :
  m - n < 0 :=
by sorry

end problem_statement_l75_7541


namespace rem_negative_five_ninths_seven_thirds_l75_7554

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_negative_five_ninths_seven_thirds :
  rem (-5/9 : ℚ) (7/3 : ℚ) = 16/9 := by
  sorry

end rem_negative_five_ninths_seven_thirds_l75_7554


namespace flower_arrangement_count_l75_7526

/-- The number of roses available for selection. -/
def num_roses : ℕ := 4

/-- The number of tulips available for selection. -/
def num_tulips : ℕ := 3

/-- The number of flower arrangements where exactly one of the roses or tulips is the same. -/
def arrangements_with_one_same : ℕ := 
  (num_roses * (num_tulips * (num_tulips - 1))) + 
  (num_tulips * (num_roses * (num_roses - 1)))

/-- Theorem stating that the number of flower arrangements where exactly one of the roses or tulips is the same is 60. -/
theorem flower_arrangement_count : arrangements_with_one_same = 60 := by
  sorry

end flower_arrangement_count_l75_7526


namespace sequence_general_formula_l75_7591

/-- Given a sequence {a_n} where the sum of its first n terms is S_n = 3 + 2^n,
    this theorem proves the general formula for a_n. -/
theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 3 + 2^n) :
  (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) := by
  sorry

end sequence_general_formula_l75_7591
