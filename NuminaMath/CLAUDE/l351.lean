import Mathlib

namespace ellipse_with_given_properties_l351_35127

/-- Represents an ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity

/-- The equation of an ellipse in standard form -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

theorem ellipse_with_given_properties :
  ∀ (E : Ellipse),
    E.b = 1 →  -- Half of minor axis length is 1
    E.e = Real.sqrt 2 / 2 →  -- Eccentricity is √2/2
    (∀ x y : ℝ, ellipse_equation E x y ↔ x^2 / 2 + y^2 = 1) :=
by
  sorry

end ellipse_with_given_properties_l351_35127


namespace simplify_product_of_radicals_l351_35178

theorem simplify_product_of_radicals (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 84 * x * Real.sqrt (2 * x) :=
by sorry

end simplify_product_of_radicals_l351_35178


namespace intersection_of_M_and_N_l351_35175

-- Define the set M as the domain of y = log(1-x)
def M : Set ℝ := {x : ℝ | x < 1}

-- Define the set N = {y | y = e^x, x ∈ ℝ}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.exp x}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l351_35175


namespace series_sum_l351_35157

theorem series_sum : ∑' n, (n : ℝ) / 5^n = 5 / 16 := by sorry

end series_sum_l351_35157


namespace distance_to_x_axis_l351_35176

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-2, 3)) : 
  abs (P.2) = 3 := by sorry

end distance_to_x_axis_l351_35176


namespace triangle_inequality_l351_35114

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = Real.pi

-- Theorem statement
theorem triangle_inequality (t : Triangle) (x y z : Real) :
  x^2 + y^2 + z^2 ≥ 2*x*y*(Real.cos t.C) + 2*y*z*(Real.cos t.A) + 2*z*x*(Real.cos t.B) := by
  sorry

end triangle_inequality_l351_35114


namespace ratio_equality_l351_35138

theorem ratio_equality (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) : (a / 5) / (b / 4) = 1 := by
  sorry

end ratio_equality_l351_35138


namespace hyperbola_eccentricity_l351_35198

/-- For a hyperbola with equation x²/a² - y²/b² = 1, if the distance between
    its vertices (2a) is one-third of its focal length (2c), then its
    eccentricity (e) is equal to 3. -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) :
  (2 * a = (1/3) * (2 * c)) → (c / a = 3) :=
by sorry

end hyperbola_eccentricity_l351_35198


namespace sum_of_reciprocals_of_roots_l351_35183

theorem sum_of_reciprocals_of_roots (p q : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + p*x₁ + q = 0 → 
  x₂^2 + p*x₂ + q = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  1/x₁ + 1/x₂ = -p/q :=
by sorry

end sum_of_reciprocals_of_roots_l351_35183


namespace max_cards_48_36_16_12_l351_35189

/-- The maximum number of rectangular cards that can be cut from a rectangular cardboard --/
def max_cards (cardboard_length cardboard_width card_length card_width : ℕ) : ℕ :=
  max ((cardboard_length / card_length) * (cardboard_width / card_width))
      ((cardboard_length / card_width) * (cardboard_width / card_length))

/-- Theorem: The maximum number of 16 cm x 12 cm cards that can be cut from a 48 cm x 36 cm cardboard is 9 --/
theorem max_cards_48_36_16_12 :
  max_cards 48 36 16 12 = 9 := by
  sorry

end max_cards_48_36_16_12_l351_35189


namespace shortest_distance_curve_to_line_l351_35135

/-- The shortest distance from a point on the curve y = 2ln x to the line 2x - y + 3 = 0 is √5 -/
theorem shortest_distance_curve_to_line :
  let curve := fun x : ℝ => 2 * Real.log x
  let line := fun x y : ℝ => 2 * x - y + 3 = 0
  ∃ d : ℝ, d = Real.sqrt 5 ∧
    ∀ x y : ℝ, curve x = y →
      d ≤ (|2 * x - y + 3| / Real.sqrt (2^2 + (-1)^2)) ∧
      ∃ x₀ y₀ : ℝ, curve x₀ = y₀ ∧
        d = (|2 * x₀ - y₀ + 3| / Real.sqrt (2^2 + (-1)^2)) :=
by sorry

end shortest_distance_curve_to_line_l351_35135


namespace quadratic_no_solution_l351_35116

theorem quadratic_no_solution : 
  {x : ℝ | x^2 - 2*x + 3 = 0} = ∅ := by sorry

end quadratic_no_solution_l351_35116


namespace seventieth_even_positive_integer_seventieth_even_positive_integer_is_140_l351_35112

theorem seventieth_even_positive_integer : ℕ → ℕ := 
  fun n => 2 * n

#check seventieth_even_positive_integer 70 = 140

theorem seventieth_even_positive_integer_is_140 : 
  seventieth_even_positive_integer 70 = 140 := by
  sorry

end seventieth_even_positive_integer_seventieth_even_positive_integer_is_140_l351_35112


namespace train_length_l351_35192

/-- Proves that a train traveling at 40 km/hr crossing a pole in 9 seconds has a length of 100 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 40 → -- speed in km/hr
  time = 9 → -- time in seconds
  length = speed * (1000 / 3600) * time → -- convert km/hr to m/s and multiply by time
  length = 100 := by
  sorry

end train_length_l351_35192


namespace y_intercept_of_parallel_line_l351_35158

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def yIntercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := -3, point := (0, 6) } →
  b.point = (3, -2) →
  yIntercept b = 7 := by
  sorry

end y_intercept_of_parallel_line_l351_35158


namespace reflect_F_coordinates_l351_35100

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The original point F -/
def F : ℝ × ℝ := (-1, -1)

theorem reflect_F_coordinates :
  (reflect_y_eq_x (reflect_x F)) = (1, -1) := by
sorry

end reflect_F_coordinates_l351_35100


namespace largest_angle_is_112_5_l351_35111

/-- Represents a quadrilateral formed by folding two sides of a square along its diagonal -/
structure FoldedSquare where
  /-- The side length of the original square -/
  side : ℝ
  /-- Assumption that the side length is positive -/
  side_pos : side > 0

/-- The largest angle in the folded square -/
def largest_angle (fs : FoldedSquare) : ℝ := 112.5

/-- Theorem stating that the largest angle in the folded square is 112.5° -/
theorem largest_angle_is_112_5 (fs : FoldedSquare) :
  largest_angle fs = 112.5 := by sorry

end largest_angle_is_112_5_l351_35111


namespace no_function_satisfies_conditions_l351_35140

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧ 
  (∀ x : ℝ, x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by sorry

end no_function_satisfies_conditions_l351_35140


namespace nancys_weight_l351_35171

theorem nancys_weight (water_intake : ℝ) (water_percentage : ℝ) :
  water_intake = 54 →
  water_percentage = 0.60 →
  water_intake = water_percentage * 90 :=
by
  sorry

end nancys_weight_l351_35171


namespace sum_of_multiples_of_6_and_9_is_multiple_of_3_l351_35121

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
  sorry

end sum_of_multiples_of_6_and_9_is_multiple_of_3_l351_35121


namespace distance_between_points_l351_35187

-- Define the equation of the curve
def on_curve (x y : ℝ) : Prop := y^2 + x^3 = 2*x*y + 4

-- Define the theorem
theorem distance_between_points (e a b : ℝ) 
  (h1 : on_curve e a) 
  (h2 : on_curve e b) 
  (h3 : a ≠ b) : 
  |a - b| = 2 * Real.sqrt (e^2 - e^3 + 4) := by
  sorry

end distance_between_points_l351_35187


namespace mom_initial_money_l351_35180

/-- The amount of money Mom spent on bananas -/
def banana_cost : ℕ := 2 * 4

/-- The amount of money Mom spent on pears -/
def pear_cost : ℕ := 2

/-- The amount of money Mom spent on asparagus -/
def asparagus_cost : ℕ := 6

/-- The amount of money Mom spent on chicken -/
def chicken_cost : ℕ := 11

/-- The amount of money Mom has left after shopping -/
def money_left : ℕ := 28

/-- The total amount Mom spent on groceries -/
def total_spent : ℕ := banana_cost + pear_cost + asparagus_cost + chicken_cost

/-- Theorem stating that Mom had €55 when she left for the market -/
theorem mom_initial_money : total_spent + money_left = 55 := by
  sorry

end mom_initial_money_l351_35180


namespace max_f_and_min_side_a_l351_35133

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

theorem max_f_and_min_side_a :
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ 2) ∧
  (∀ (A B C a b c : ℝ),
    0 < A ∧ A < Real.pi →
    0 < B ∧ B < Real.pi →
    0 < C ∧ C < Real.pi →
    A + B + C = Real.pi →
    f (B + C) = 3 / 2 →
    b + c = 2 →
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
    a ≥ 1) :=
by sorry

end max_f_and_min_side_a_l351_35133


namespace imaginary_sum_equals_neg_i_l351_35104

theorem imaginary_sum_equals_neg_i :
  let i : ℂ := Complex.I
  (1 / i) + (1 / i^3) + (1 / i^5) + (1 / i^7) + (1 / i^9) = -i :=
by sorry

end imaginary_sum_equals_neg_i_l351_35104


namespace overall_average_marks_l351_35118

/-- Given three batches of students with their respective sizes and average marks,
    calculate the overall average marks for all students combined. -/
theorem overall_average_marks
  (batch1_size batch2_size batch3_size : ℕ)
  (batch1_avg batch2_avg batch3_avg : ℚ)
  (h1 : batch1_size = 40)
  (h2 : batch2_size = 50)
  (h3 : batch3_size = 60)
  (h4 : batch1_avg = 45)
  (h5 : batch2_avg = 55)
  (h6 : batch3_avg = 65) :
  (batch1_size * batch1_avg + batch2_size * batch2_avg + batch3_size * batch3_avg) /
  (batch1_size + batch2_size + batch3_size) = 8450 / 150 := by
  sorry

#eval (8450 : ℚ) / 150  -- To verify the result

end overall_average_marks_l351_35118


namespace cone_lateral_surface_area_l351_35182

theorem cone_lateral_surface_area 
  (r : ℝ) 
  (h : ℝ) 
  (lateral_area : ℝ) 
  (h_r : r = 3) 
  (h_h : h = 1) :
  lateral_area = 3 * Real.sqrt 10 * Real.pi := by
  sorry

end cone_lateral_surface_area_l351_35182


namespace sylvia_incorrect_fraction_l351_35161

/-- Proves that Sylvia's fraction of incorrect answers is 1/5 given the conditions -/
theorem sylvia_incorrect_fraction (total_questions : ℕ) (sergio_incorrect : ℕ) (difference : ℕ) :
  total_questions = 50 →
  sergio_incorrect = 4 →
  difference = 6 →
  (total_questions - (total_questions - sergio_incorrect - difference)) / total_questions = 1 / 5 := by
  sorry

end sylvia_incorrect_fraction_l351_35161


namespace carton_height_calculation_l351_35113

/-- Calculates the height of a carton given its base dimensions, soap box dimensions, and maximum capacity -/
theorem carton_height_calculation (carton_length carton_width : ℕ) 
  (box_length box_width box_height : ℕ) (max_boxes : ℕ) : 
  carton_length = 25 ∧ carton_width = 42 ∧ 
  box_length = 7 ∧ box_width = 6 ∧ box_height = 5 ∧
  max_boxes = 300 →
  (max_boxes / ((carton_length / box_length) * (carton_width / box_width))) * box_height = 70 :=
by sorry

end carton_height_calculation_l351_35113


namespace vinces_bus_ride_l351_35164

theorem vinces_bus_ride (zachary_ride : ℝ) (vince_difference : ℝ) :
  zachary_ride = 0.5 →
  vince_difference = 0.125 →
  zachary_ride + vince_difference = 0.625 :=
by
  sorry

end vinces_bus_ride_l351_35164


namespace angle_B_value_min_side_b_value_l351_35181

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def TriangleConditions (t : Triangle) : Prop :=
  (Real.cos t.C / Real.cos t.B = (2 * t.a - t.c) / t.b) ∧
  (t.a + t.c = 2)

theorem angle_B_value (t : Triangle) (h : TriangleConditions t) : t.B = π / 3 := by
  sorry

theorem min_side_b_value (t : Triangle) (h : TriangleConditions t) : 
  ∃ (b_min : ℝ), b_min = 1 ∧ ∀ (t' : Triangle), TriangleConditions t' → t'.b ≥ b_min := by
  sorry

end angle_B_value_min_side_b_value_l351_35181


namespace solution_set_part_I_value_of_a_part_II_l351_35137

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 1

-- Part I
theorem solution_set_part_I (a : ℝ) (h : a > 1) :
  let f := f a
  a = 2 →
  {x : ℝ | f x ≥ 4 - |x - 4|} = {x : ℝ | x ≥ 11/2 ∨ x ≤ 1/2} :=
sorry

-- Part II
theorem value_of_a_part_II (a : ℝ) (h : a > 1) :
  let f := f a
  ({x : ℝ | |f (2*x + a) - 2*f x| ≤ 1} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 1}) →
  a = 2 :=
sorry

end solution_set_part_I_value_of_a_part_II_l351_35137


namespace tetrahedron_triangle_existence_l351_35169

/-- Represents a tetrahedron with edge lengths -/
structure Tetrahedron where
  edges : Fin 6 → ℝ
  edge_positive : ∀ i, edges i > 0

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Main theorem: In any tetrahedron, there exists a vertex such that 
    the edges connected to it can form a triangle -/
theorem tetrahedron_triangle_existence (t : Tetrahedron) : 
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    can_form_triangle (t.edges i) (t.edges j) (t.edges k) :=
sorry

end tetrahedron_triangle_existence_l351_35169


namespace find_number_l351_35145

theorem find_number : ∃! x : ℝ, ((x * 2) - 37 + 25) / 8 = 5 := by sorry

end find_number_l351_35145


namespace ball_trajectory_l351_35148

-- Define the quadratic function
def f (t : ℚ) : ℚ := -4.9 * t^2 + 7 * t + 10

-- State the theorem
theorem ball_trajectory :
  f (5/7) = 15 ∧
  f (10/7) = 0 ∧
  ∀ t : ℚ, 5/7 < t → t < 10/7 → f t ≠ 15 ∧ f t ≠ 0 :=
sorry

end ball_trajectory_l351_35148


namespace earth_surface_utilization_l351_35174

theorem earth_surface_utilization (
  exposed_land : ℚ)
  (inhabitable_land : ℚ)
  (utilized_land : ℚ)
  (h1 : exposed_land = 1 / 3)
  (h2 : inhabitable_land = 2 / 5 * exposed_land)
  (h3 : utilized_land = 3 / 4 * inhabitable_land) :
  utilized_land = 1 / 10 := by
  sorry

end earth_surface_utilization_l351_35174


namespace team_selection_count_l351_35173

/-- Represents the number of ways to select a team under given conditions -/
def selectTeam (totalMale totalFemale teamSize : ℕ) 
               (maleCaptains femaleCaptains : ℕ) : ℕ := 
  Nat.choose (totalMale + totalFemale - 1) (teamSize - 1) + 
  Nat.choose (totalMale + totalFemale - maleCaptains - 1) (teamSize - 1) - 
  Nat.choose (totalMale - maleCaptains) (teamSize - 1)

/-- Theorem stating the number of ways to select a team of 5 from 6 male (1 captain) 
    and 4 female (1 captain) athletes, including at least 1 female and a captain -/
theorem team_selection_count : 
  selectTeam 6 4 5 1 1 = 191 := by sorry

end team_selection_count_l351_35173


namespace divisibility_rules_l351_35141

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ := n % 10

-- Define a function to get the last two digits of a natural number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define a function to check if a number is even
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define a function to sum the digits of a natural number
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

theorem divisibility_rules (n : ℕ) :
  (n % 2 = 0 ↔ isEven (lastDigit n)) ∧
  (n % 5 = 0 ↔ lastDigit n = 0 ∨ lastDigit n = 5) ∧
  (n % 3 = 0 ↔ sumOfDigits n % 3 = 0) ∧
  (n % 4 = 0 ↔ lastTwoDigits n % 4 = 0) ∧
  (n % 25 = 0 ↔ lastTwoDigits n % 25 = 0) :=
by sorry


end divisibility_rules_l351_35141


namespace longest_side_is_72_l351_35193

def rectangle_problem (length width : ℝ) : Prop :=
  length > 0 ∧ 
  width > 0 ∧ 
  2 * (length + width) = 240 ∧ 
  length * width = 12 * 240

theorem longest_side_is_72 : 
  ∃ (length width : ℝ), 
    rectangle_problem length width ∧ 
    (length ≥ width → length = 72) ∧
    (width > length → width = 72) :=
sorry

end longest_side_is_72_l351_35193


namespace cycle_gain_percent_l351_35152

/-- Calculates the gain percent given the cost price and selling price -/
def gainPercent (costPrice sellingPrice : ℚ) : ℚ :=
  ((sellingPrice - costPrice) / costPrice) * 100

/-- Proves that the gain percent on a cycle bought for Rs. 930 and sold for Rs. 1210 is approximately 30.11% -/
theorem cycle_gain_percent :
  let costPrice : ℚ := 930
  let sellingPrice : ℚ := 1210
  abs (gainPercent costPrice sellingPrice - 30.11) < 0.01 := by
  sorry

end cycle_gain_percent_l351_35152


namespace equilateral_triangle_formation_l351_35155

/-- Function to calculate the sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate to check if a number is divisible by 3 -/
def divisible_by_three (n : ℕ) : Prop := n % 3 = 0

/-- Predicate to check if it's possible to form an equilateral triangle from n sticks -/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  divisible_by_three (sum_to_n n) ∧ 
  ∃ (partition : ℕ → ℕ → ℕ), 
    (∀ i j, i < j → j ≤ n → partition i j ≤ sum_to_n n / 3) ∧
    (∀ i, i ≤ n → ∃ j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ j ≤ n ∧ k ≤ n ∧
      partition i j + partition j k + partition k i = sum_to_n n / 3)

theorem equilateral_triangle_formation :
  ¬can_form_equilateral_triangle 100 ∧ can_form_equilateral_triangle 99 := by
  sorry

end equilateral_triangle_formation_l351_35155


namespace exists_crocodile_coloring_l351_35159

/-- A coloring function for the infinite chess grid -/
def GridColoring := ℤ → ℤ → Fin 2

/-- The crocodile move property for a given coloring -/
def IsCrocodileColoring (f : GridColoring) (m n : ℕ+) : Prop :=
  ∀ x y : ℤ, f x y ≠ f (x + m) (y + n) ∧ f x y ≠ f (x + n) (y + m)

/-- Theorem: For any positive integers m and n, there exists a valid crocodile coloring -/
theorem exists_crocodile_coloring (m n : ℕ+) :
  ∃ f : GridColoring, IsCrocodileColoring f m n := by
  sorry

end exists_crocodile_coloring_l351_35159


namespace largest_prime_factor_of_expression_l351_35131

theorem largest_prime_factor_of_expression : 
  (Nat.factors (12^3 + 8^4 - 4^5)).maximum = some 3 := by
  sorry

end largest_prime_factor_of_expression_l351_35131


namespace sunflower_height_comparison_l351_35149

/-- Given that sunflowers from Packet A are 20% taller than those from Packet B,
    and sunflowers from Packet A are 192 inches tall,
    prove that sunflowers from Packet B are 160 inches tall. -/
theorem sunflower_height_comparison (height_A height_B : ℝ) : 
  height_A = height_B * 1.2 → height_A = 192 → height_B = 160 := by
  sorry

end sunflower_height_comparison_l351_35149


namespace intersection_of_A_and_B_l351_35120

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {x | ∃ m ∈ A, x = 3 * m - 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 4} := by
  sorry

end intersection_of_A_and_B_l351_35120


namespace hilary_corn_shucking_l351_35126

/-- The number of ears of corn per stalk -/
def ears_per_stalk : ℕ := 4

/-- The number of stalks Hilary has -/
def total_stalks : ℕ := 108

/-- The number of kernels on half of the ears -/
def kernels_first_half : ℕ := 500

/-- The additional number of kernels on the other half of the ears -/
def additional_kernels : ℕ := 100

/-- The total number of kernels Hilary has to shuck -/
def total_kernels : ℕ := 
  let total_ears := ears_per_stalk * total_stalks
  let ears_per_half := total_ears / 2
  let kernels_second_half := kernels_first_half + additional_kernels
  ears_per_half * kernels_first_half + ears_per_half * kernels_second_half

theorem hilary_corn_shucking :
  total_kernels = 237600 := by
  sorry

end hilary_corn_shucking_l351_35126


namespace davids_physics_marks_l351_35123

def english_marks : ℕ := 45
def math_marks : ℕ := 35
def chemistry_marks : ℕ := 47
def biology_marks : ℕ := 55
def average_marks : ℚ := 46.8
def num_subjects : ℕ := 5

theorem davids_physics_marks :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    physics_marks = 52 := by
  sorry

end davids_physics_marks_l351_35123


namespace necessary_sufficient_condition_l351_35186

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) ↔
  (a ≤ -2 ∨ a = 1) :=
sorry

end necessary_sufficient_condition_l351_35186


namespace sharp_constant_is_20_l351_35105

/-- The function # defined for any real number -/
def sharp (C : ℝ) (p : ℝ) : ℝ := 2 * p - C

/-- Theorem stating that the constant in the sharp function is 20 -/
theorem sharp_constant_is_20 : ∃ C : ℝ, 
  (sharp C (sharp C (sharp C 18.25)) = 6) ∧ C = 20 := by
  sorry

end sharp_constant_is_20_l351_35105


namespace garage_wheels_count_l351_35129

def total_wheels (cars bicycles : Nat) (lawnmower tricycle unicycle skateboard wheelbarrow wagon : Nat) : Nat :=
  cars * 4 + bicycles * 2 + lawnmower * 4 + tricycle * 3 + unicycle + skateboard * 4 + wheelbarrow + wagon * 4

theorem garage_wheels_count :
  total_wheels 2 3 1 1 1 1 1 1 = 31 := by
  sorry

end garage_wheels_count_l351_35129


namespace expression_evaluation_l351_35107

theorem expression_evaluation : 6 * 199 + 4 * 199 + 3 * 199 + 199 + 100 = 2886 := by
  sorry

end expression_evaluation_l351_35107


namespace diophantine_equation_solutions_l351_35163

def diophantine_equation (x y : ℤ) : Prop :=
  2 * x^4 - 4 * y^4 - 7 * x^2 * y^2 - 27 * x^2 + 63 * y^2 + 85 = 0

def solution_set : Set (ℤ × ℤ) :=
  {(3, 1), (3, -1), (-3, 1), (-3, -1), (2, 3), (2, -3), (-2, 3), (-2, -3)}

theorem diophantine_equation_solutions :
  ∀ (x y : ℤ), diophantine_equation x y ↔ (x, y) ∈ solution_set :=
by sorry

end diophantine_equation_solutions_l351_35163


namespace jeans_prices_l351_35142

/-- Represents the shopping scenario with Mary and her children --/
structure ShoppingScenario where
  coat_original_price : ℝ
  coat_discount_rate : ℝ
  backpack_cost : ℝ
  shoes_cost : ℝ
  subtotal : ℝ
  jeans_price_difference : ℝ
  sales_tax_rate : ℝ

/-- Theorem stating the prices of Jamie's jeans --/
theorem jeans_prices (scenario : ShoppingScenario)
  (h_coat : scenario.coat_original_price = 50)
  (h_discount : scenario.coat_discount_rate = 0.1)
  (h_backpack : scenario.backpack_cost = 25)
  (h_shoes : scenario.shoes_cost = 30)
  (h_subtotal : scenario.subtotal = 139)
  (h_difference : scenario.jeans_price_difference = 15)
  (h_tax : scenario.sales_tax_rate = 0.07) :
  ∃ (cheap_jeans expensive_jeans : ℝ),
    cheap_jeans = 12 ∧
    expensive_jeans = 27 ∧
    cheap_jeans + expensive_jeans = scenario.subtotal -
      (scenario.coat_original_price * (1 - scenario.coat_discount_rate) +
       scenario.backpack_cost + scenario.shoes_cost) ∧
    expensive_jeans - cheap_jeans = scenario.jeans_price_difference :=
by sorry

end jeans_prices_l351_35142


namespace child_ticket_cost_is_4_l351_35170

/-- The cost of a child's ticket at a ball game -/
def child_ticket_cost (num_adults num_children : ℕ) (adult_ticket_cost total_bill : ℚ) : ℚ :=
  (total_bill - num_adults * adult_ticket_cost) / num_children

theorem child_ticket_cost_is_4 :
  child_ticket_cost 10 11 8 124 = 4 := by
  sorry

end child_ticket_cost_is_4_l351_35170


namespace paul_homework_hours_l351_35172

/-- Calculates the total hours of homework on weeknights for Paul --/
def weeknight_homework (total_weeknights : ℕ) (practice_nights : ℕ) (average_hours : ℕ) : ℕ :=
  (total_weeknights - practice_nights) * average_hours

/-- Proves that Paul has 9 hours of homework on weeknights --/
theorem paul_homework_hours :
  let total_weeknights := 5
  let practice_nights := 2
  let average_hours := 3
  weeknight_homework total_weeknights practice_nights average_hours = 9 := by
  sorry

#eval weeknight_homework 5 2 3

end paul_homework_hours_l351_35172


namespace student_committee_candidates_l351_35115

theorem student_committee_candidates :
  ∃ n : ℕ, 
    n > 0 ∧ 
    n * (n - 1) = 132 ∧ 
    (∀ m : ℕ, m > 0 ∧ m * (m - 1) = 132 → m = n) ∧
    n = 12 := by
  sorry

end student_committee_candidates_l351_35115


namespace sphere_surface_area_rectangular_solid_l351_35146

/-- The surface area of a sphere circumscribing a rectangular solid -/
theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (S : ℝ) :
  a = 3 →
  b = 4 →
  c = 5 →
  S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
  S = 50 * Real.pi :=
by sorry

end sphere_surface_area_rectangular_solid_l351_35146


namespace union_equals_A_l351_35185

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {y | m*y + 2 = 0}

theorem union_equals_A : {m : ℝ | A ∪ B m = A} = {0, -1, -2/3} := by sorry

end union_equals_A_l351_35185


namespace smallest_positive_integer_linear_combination_l351_35153

theorem smallest_positive_integer_linear_combination : 
  ∃ (k : ℕ), k > 0 ∧ (∀ (m n p : ℤ), ∃ (x : ℤ), 1234 * m + 56789 * n + 345 * p = x * k) ∧ 
  (∀ (l : ℕ), l > 0 → (∀ (m n p : ℤ), ∃ (x : ℤ), 1234 * m + 56789 * n + 345 * p = x * l) → l ≥ k) :=
by sorry

end smallest_positive_integer_linear_combination_l351_35153


namespace least_product_of_primes_above_50_l351_35191

theorem least_product_of_primes_above_50 (p q : ℕ) : 
  p.Prime → q.Prime → p > 50 → q > 50 → p ≠ q → 
  ∀ r s : ℕ, r.Prime → s.Prime → r > 50 → s > 50 → r ≠ s → 
  p * q ≤ r * s :=
sorry

end least_product_of_primes_above_50_l351_35191


namespace biology_magnet_combinations_l351_35190

def word : String := "BIOLOGY"

def num_vowels : Nat := 3
def num_consonants : Nat := 2
def num_Os : Nat := 2

def vowels : Finset Char := {'I', 'O'}
def consonants : Finset Char := {'B', 'L', 'G', 'Y'}

theorem biology_magnet_combinations : 
  (Finset.card (Finset.powerset vowels) * Finset.card (Finset.powerset consonants)) +
  (Finset.card (Finset.powerset {0, 1}) * Finset.card (Finset.powerset consonants)) = 42 := by
  sorry

end biology_magnet_combinations_l351_35190


namespace g_inverse_composition_l351_35124

def g : Fin 5 → Fin 5
| 0 => 3  -- Representing g(1) = 4
| 1 => 2  -- Representing g(2) = 3
| 2 => 0  -- Representing g(3) = 1
| 3 => 4  -- Representing g(4) = 5
| 4 => 1  -- Representing g(5) = 2

theorem g_inverse_composition (h : Function.Bijective g) :
  (Function.invFun g) ((Function.invFun g) ((Function.invFun g) 2)) = 3 := by
  sorry

end g_inverse_composition_l351_35124


namespace quadratic_expression_value_l351_35188

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 4 * x + y = 12) (h2 : x + 4 * y = 16) : 
  17 * x^2 + 18 * x * y + 17 * y^2 = 400 := by
  sorry

end quadratic_expression_value_l351_35188


namespace complex_multiplication_simplification_l351_35139

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem: For any real number t, (2+t i)(2-t i) = 4 + t^2 -/
theorem complex_multiplication_simplification (t : ℝ) : 
  (2 + t * i) * (2 - t * i) = (4 : ℂ) + t^2 := by
  sorry

end complex_multiplication_simplification_l351_35139


namespace conference_handshakes_l351_35128

/-- The number of handshakes in a conference of n people where each person
    shakes hands exactly once with every other person. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a conference of 12 people where each person shakes hands
    exactly once with every other person, there are 66 handshakes. -/
theorem conference_handshakes :
  handshakes 12 = 66 := by
  sorry

#eval handshakes 12

end conference_handshakes_l351_35128


namespace quadratic_form_ratio_l351_35132

theorem quadratic_form_ratio (x : ℝ) : 
  ∃ (d e : ℝ), x^2 + 800*x + 500 = (x + d)^2 + e ∧ e / d = -398.75 := by
  sorry

end quadratic_form_ratio_l351_35132


namespace cookie_distribution_l351_35184

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) :
  total_cookies = 24 →
  num_people = 6 →
  total_cookies = num_people * cookies_per_person →
  cookies_per_person = 4 := by
  sorry

end cookie_distribution_l351_35184


namespace santiago_garrett_rose_difference_l351_35125

/-- Mrs. Santiago has 58 red roses and Mrs. Garrett has 24 red roses. 
    The theorem proves that Mrs. Santiago has 34 more red roses than Mrs. Garrett. -/
theorem santiago_garrett_rose_difference :
  ∀ (santiago_roses garrett_roses : ℕ),
    santiago_roses = 58 →
    garrett_roses = 24 →
    santiago_roses - garrett_roses = 34 :=
by
  sorry

end santiago_garrett_rose_difference_l351_35125


namespace larger_number_proof_l351_35106

theorem larger_number_proof (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  Nat.gcd a b = 60 → Nat.lcm a b = 9900 → max a b = 900 := by
sorry

end larger_number_proof_l351_35106


namespace remainder_property_l351_35197

theorem remainder_property (n : ℕ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end remainder_property_l351_35197


namespace min_exponent_sum_520_l351_35109

/-- Given a natural number n, returns the minimum sum of exponents when expressing n as a sum of at least two distinct powers of 2 -/
def min_exponent_sum (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the minimum sum of exponents when expressing 520 as a sum of at least two distinct powers of 2 is 12 -/
theorem min_exponent_sum_520 : min_exponent_sum 520 = 12 := by
  sorry

end min_exponent_sum_520_l351_35109


namespace division_problem_l351_35199

theorem division_problem (A : ℕ) (h1 : 59 / A = 6) (h2 : 59 % A = 5) : A = 9 := by
  sorry

end division_problem_l351_35199


namespace x_coordinate_difference_l351_35147

theorem x_coordinate_difference (m n k : ℝ) : 
  (m = 2*n + 5) → 
  (m + k = 2*(n + 2) + 5) → 
  k = 4 := by
sorry

end x_coordinate_difference_l351_35147


namespace lcm_gcd_problem_l351_35156

theorem lcm_gcd_problem : (Nat.lcm 12 9 * Nat.gcd 12 9) - Nat.gcd 15 9 = 105 := by
  sorry

end lcm_gcd_problem_l351_35156


namespace ethanol_percentage_in_fuel_A_l351_35194

/-- Proves that the percentage of ethanol in fuel A is 12% -/
theorem ethanol_percentage_in_fuel_A :
  let tank_capacity : ℝ := 208
  let fuel_A_volume : ℝ := 82
  let fuel_B_ethanol_percentage : ℝ := 0.16
  let total_ethanol : ℝ := 30
  let fuel_B_volume : ℝ := tank_capacity - fuel_A_volume
  let fuel_A_ethanol_percentage : ℝ := (total_ethanol - fuel_B_ethanol_percentage * fuel_B_volume) / fuel_A_volume
  fuel_A_ethanol_percentage = 0.12 := by sorry

end ethanol_percentage_in_fuel_A_l351_35194


namespace perpendicular_line_implies_parallel_planes_perpendicular_lines_to_plane_implies_parallel_lines_l351_35103

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the perpendicular relation between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Theorem 1: If a line is perpendicular to two planes, then those two planes are parallel
theorem perpendicular_line_implies_parallel_planes
  (m : Line) (α β : Plane) :
  perpendicular_line_plane m α →
  perpendicular_line_plane m β →
  parallel_plane_plane α β :=
sorry

-- Theorem 2: If two lines are both perpendicular to the same plane, then those two lines are parallel
theorem perpendicular_lines_to_plane_implies_parallel_lines
  (m n : Line) (α : Plane) :
  perpendicular_line_plane m α →
  perpendicular_line_plane n α →
  parallel_line_line m n :=
sorry

end perpendicular_line_implies_parallel_planes_perpendicular_lines_to_plane_implies_parallel_lines_l351_35103


namespace largest_angle_not_less_than_60_degrees_l351_35130

open Real

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : Point) (b : Point)

/-- Calculates the angle between two lines -/
noncomputable def angle (l1 l2 : Line) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (a b c : Point) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (m : Point) (a b : Point) : Prop := sorry

/-- Main theorem -/
theorem largest_angle_not_less_than_60_degrees 
  (a b c : Point) 
  (h_equilateral : isEquilateral a b c)
  (c₁ : Point) (h_c₁_midpoint : isMidpoint c₁ a b)
  (a₁ : Point) (h_a₁_midpoint : isMidpoint a₁ b c)
  (b₁ : Point) (h_b₁_midpoint : isMidpoint b₁ c a)
  (p : Point) :
  let angle1 := angle (Line.mk a b) (Line.mk p c₁)
  let angle2 := angle (Line.mk b c) (Line.mk p a₁)
  let angle3 := angle (Line.mk c a) (Line.mk p b₁)
  max angle1 (max angle2 angle3) ≥ π/3 := by sorry

end largest_angle_not_less_than_60_degrees_l351_35130


namespace perpendicular_line_equation_l351_35160

/-- A line passing through (-1, 2) and perpendicular to y = 2/3x has the equation 3x + 2y - 1 = 0 -/
theorem perpendicular_line_equation :
  let l : Set (ℝ × ℝ) := {(x, y) | 3 * x + 2 * y - 1 = 0}
  let point : ℝ × ℝ := (-1, 2)
  let perpendicular_slope : ℝ := 2 / 3
  (point ∈ l) ∧
  (∀ (x y : ℝ), (x, y) ∈ l → (3 : ℝ) * perpendicular_slope = -1) :=
by sorry

end perpendicular_line_equation_l351_35160


namespace scientific_notation_of_12417_l351_35134

theorem scientific_notation_of_12417 : ∃ (a : ℝ) (n : ℤ), 
  12417 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.2417 ∧ n = 4 := by
  sorry

end scientific_notation_of_12417_l351_35134


namespace right_triangle_sides_l351_35168

theorem right_triangle_sides : ∃ (a b c : ℝ), 
  a = Real.sqrt 3 ∧ 
  b = Real.sqrt 13 ∧ 
  c = 4 ∧ 
  a^2 + b^2 = c^2 := by
  sorry

end right_triangle_sides_l351_35168


namespace problem_solution_l351_35179

theorem problem_solution (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : 
  |x - y| = 10 := by sorry

end problem_solution_l351_35179


namespace largest_c_for_negative_five_in_range_l351_35102

/-- The function f(x) defined as x^2 + 5x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + c

/-- Theorem stating that 5/4 is the largest value of c such that -5 is in the range of f(x) -/
theorem largest_c_for_negative_five_in_range :
  ∀ c : ℝ, (∃ x : ℝ, f c x = -5) ↔ c ≤ 5/4 := by sorry

end largest_c_for_negative_five_in_range_l351_35102


namespace parallel_lines_a_value_l351_35162

/-- Two lines are parallel if and only if their slopes are equal but they are not identical --/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ / n₁ = m₂ / n₂ ∧ m₁ / c₁ ≠ m₂ / c₂

/-- The problem statement --/
theorem parallel_lines_a_value :
  ∀ a : ℝ,
  are_parallel 1 a 6 (a - 2) 3 (2 * a) →
  a = -1 :=
sorry

end parallel_lines_a_value_l351_35162


namespace age_difference_of_children_l351_35196

/-- Proves that the age difference between children is 4 years given the conditions -/
theorem age_difference_of_children (n : ℕ) (sum_ages : ℕ) (eldest_age : ℕ) (d : ℕ) :
  n = 4 ∧ 
  sum_ages = 48 ∧ 
  eldest_age = 18 ∧ 
  sum_ages = n * eldest_age - (d * (n * (n - 1)) / 2) →
  d = 4 :=
by sorry


end age_difference_of_children_l351_35196


namespace oil_distribution_l351_35165

/-- Represents the problem of minimizing the number of small barrels --/
def MinimizeSmallBarrels (total_oil : ℕ) (large_barrel_capacity : ℕ) (small_barrel_capacity : ℕ) : Prop :=
  ∃ (large_barrels small_barrels : ℕ),
    large_barrel_capacity * large_barrels + small_barrel_capacity * small_barrels = total_oil ∧
    small_barrels = 1 ∧
    ∀ (l s : ℕ), large_barrel_capacity * l + small_barrel_capacity * s = total_oil →
      s ≥ small_barrels

theorem oil_distribution :
  MinimizeSmallBarrels 745 11 7 :=
sorry

end oil_distribution_l351_35165


namespace smallest_a_for_two_roots_less_than_one_l351_35117

theorem smallest_a_for_two_roots_less_than_one : 
  ∃ (a b c : ℤ), 
    (a > 0) ∧ 
    (∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 0 < r₁ ∧ r₁ < 1 ∧ 0 < r₂ ∧ r₂ < 1 ∧ 
      (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = r₁ ∨ x = r₂)) ∧
    (∀ a' : ℤ, 0 < a' ∧ a' < a → 
      ¬∃ (b' c' : ℤ), ∃ (s₁ s₂ : ℝ), s₁ ≠ s₂ ∧ 0 < s₁ ∧ s₁ < 1 ∧ 0 < s₂ ∧ s₂ < 1 ∧ 
        (∀ x : ℝ, a' * x^2 + b' * x + c' = 0 ↔ x = s₁ ∨ x = s₂)) ∧
    a = 4 :=
by sorry

end smallest_a_for_two_roots_less_than_one_l351_35117


namespace equation_solution_l351_35177

theorem equation_solution (y : ℝ) (h : y ≠ 0) :
  (3 / y - (4 / y) * (2 / y) = 1.5) → y = 1 + Real.sqrt (19 / 3) :=
by sorry

end equation_solution_l351_35177


namespace coinciding_rest_days_main_theorem_l351_35150

/-- Chris's schedule cycle length -/
def chris_cycle : ℕ := 7

/-- Dana's schedule cycle length -/
def dana_cycle : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 500

/-- Number of rest days for Chris in one cycle -/
def chris_rest_days : ℕ := 2

/-- Number of rest days for Dana in one cycle -/
def dana_rest_days : ℕ := 1

/-- The number of days both Chris and Dana have rest-days on the same day
    within the first 500 days of their schedules -/
theorem coinciding_rest_days : ℕ := by
  sorry

/-- The main theorem stating that the number of coinciding rest days is 28 -/
theorem main_theorem : coinciding_rest_days = 28 := by
  sorry

end coinciding_rest_days_main_theorem_l351_35150


namespace tom_gave_two_seashells_l351_35136

/-- The number of seashells Tom gave to Jessica -/
def seashells_given (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Theorem stating that Tom gave 2 seashells to Jessica -/
theorem tom_gave_two_seashells :
  seashells_given 5 3 = 2 := by
  sorry

end tom_gave_two_seashells_l351_35136


namespace quarters_fraction_l351_35110

/-- The number of state quarters in Stephanie's collection -/
def total_quarters : ℕ := 25

/-- The number of states that joined the union from 1800 to 1809 -/
def states_1800_1809 : ℕ := 8

/-- The fraction of quarters representing states that joined from 1800 to 1809 -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_quarters

theorem quarters_fraction :
  fraction_1800_1809 = 8 / 25 := by sorry

end quarters_fraction_l351_35110


namespace BA_equals_AB_l351_35101

def matrix_2x2 (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![a, b; c, d]

theorem BA_equals_AB (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B)
  (h2 : A * B = matrix_2x2 5 2 (-2) 4) :
  B * A = matrix_2x2 5 2 (-2) 4 := by
  sorry

end BA_equals_AB_l351_35101


namespace quadratic_has_two_real_roots_roots_equal_absolute_value_l351_35108

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (m + 3) * x + m + 2

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Theorem 2: The absolute values of the roots are equal iff m = -1 or m = -3
theorem roots_equal_absolute_value (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ |x₁| = |x₂|) ↔
  (m = -1 ∨ m = -3) :=
sorry

end quadratic_has_two_real_roots_roots_equal_absolute_value_l351_35108


namespace brian_watching_time_l351_35154

def cat_video_length : ℕ := 4

def dog_video_length (cat_length : ℕ) : ℕ := 2 * cat_length

def gorilla_video_length (cat_length dog_length : ℕ) : ℕ := 2 * (cat_length + dog_length)

def total_watching_time (cat_length dog_length gorilla_length : ℕ) : ℕ :=
  cat_length + dog_length + gorilla_length

theorem brian_watching_time :
  total_watching_time cat_video_length 
    (dog_video_length cat_video_length) 
    (gorilla_video_length cat_video_length (dog_video_length cat_video_length)) = 36 := by
  sorry

end brian_watching_time_l351_35154


namespace candy_game_theorem_l351_35143

/-- The maximum number of candies that can be eaten in the candy-eating game. -/
def max_candies (n : ℕ) : ℕ :=
  n.choose 2

/-- The candy-eating game theorem. -/
theorem candy_game_theorem :
  max_candies 27 = 351 :=
by sorry

end candy_game_theorem_l351_35143


namespace odd_function_sum_l351_35195

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_even : is_even (λ x => f (x + 2))) 
  (h_f1 : f 1 = 1) : 
  f 8 + f 9 = 1 := by
  sorry

end odd_function_sum_l351_35195


namespace shaded_area_calculation_l351_35166

/-- Calculates the total shaded area of a right triangle and half of an adjacent rectangle -/
theorem shaded_area_calculation (triangle_base : ℝ) (triangle_height : ℝ) (rectangle_width : ℝ) :
  triangle_base = 6 →
  triangle_height = 8 →
  rectangle_width = 5 →
  (1 / 2 * triangle_base * triangle_height) + (1 / 2 * rectangle_width * triangle_height) = 44 := by
  sorry

end shaded_area_calculation_l351_35166


namespace map_scale_conversion_l351_35167

/-- Given a map scale where 10 cm represents 50 km, prove that 15 cm represents 75 km -/
theorem map_scale_conversion (scale : ℝ) (h1 : scale = 50 / 10) : 15 * scale = 75 := by
  sorry

end map_scale_conversion_l351_35167


namespace javier_to_anna_fraction_l351_35151

/-- Represents the number of stickers each person has -/
structure StickerCount where
  lee : ℕ
  anna : ℕ
  javier : ℕ

/-- Calculates the fraction of stickers Javier should give to Anna -/
def fraction_to_anna (initial : StickerCount) (final : StickerCount) : ℚ :=
  (final.anna - initial.anna : ℚ) / initial.javier

/-- Theorem stating that Javier should give 0 fraction of his stickers to Anna -/
theorem javier_to_anna_fraction (l : ℕ) : 
  let initial := StickerCount.mk l (3 * l) (12 * l)
  let final := StickerCount.mk (2 * l) (3 * l) (6 * l)
  fraction_to_anna initial final = 0 := by
  sorry

#check javier_to_anna_fraction

end javier_to_anna_fraction_l351_35151


namespace ninas_inheritance_l351_35122

theorem ninas_inheritance (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧                    -- Both investments are positive
  0.06 * x + 0.08 * y = 860 ∧        -- Total yearly interest
  (x = 5000 ∨ y = 5000) →            -- $5000 invested at one rate
  x + y = 12000 :=                   -- Total inheritance
by sorry

end ninas_inheritance_l351_35122


namespace solutions_to_equation_l351_35144

def solution_set : Set ℂ := {
  (3 * Real.sqrt 2) / 2 + (3 * Real.sqrt 2) / 2 * Complex.I,
  -(3 * Real.sqrt 2) / 2 - (3 * Real.sqrt 2) / 2 * Complex.I,
  (3 * Real.sqrt 2) / 2 * Complex.I - (3 * Real.sqrt 2) / 2,
  -(3 * Real.sqrt 2) / 2 * Complex.I + (3 * Real.sqrt 2) / 2
}

theorem solutions_to_equation : 
  ∀ x : ℂ, x^4 + 81 = 0 ↔ x ∈ solution_set := by sorry

end solutions_to_equation_l351_35144


namespace exists_multiple_without_zero_l351_35119

def has_no_zero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

theorem exists_multiple_without_zero (k : ℕ) : 
  k > 0 → ∃ n : ℕ, 5^k ∣ n ∧ has_no_zero n :=
sorry

end exists_multiple_without_zero_l351_35119
