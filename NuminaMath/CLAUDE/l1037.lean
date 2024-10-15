import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_for_B_exceeds_A_l1037_103749

def A (n : ℕ) : ℚ := 490 * n - 10 * n^2

def B (n : ℕ) : ℚ := 500 * n + 400 - 500 / 2^(n-1)

theorem smallest_n_for_B_exceeds_A :
  ∀ k : ℕ, k < 4 → B k ≤ A k ∧ B 4 > A 4 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_B_exceeds_A_l1037_103749


namespace NUMINAMATH_CALUDE_sugar_calculation_l1037_103795

/-- The amount of sugar needed for one chocolate bar in grams -/
def sugar_per_bar : ℝ := 1.5

/-- The number of chocolate bars produced per minute -/
def bars_per_minute : ℕ := 36

/-- The number of minutes of production -/
def production_time : ℕ := 2

/-- Calculates the total amount of sugar used in grams -/
def total_sugar_used : ℝ := sugar_per_bar * bars_per_minute * production_time

theorem sugar_calculation :
  total_sugar_used = 108 := by sorry

end NUMINAMATH_CALUDE_sugar_calculation_l1037_103795


namespace NUMINAMATH_CALUDE_l_shaped_area_l1037_103741

/-- The area of an L-shaped region formed by subtracting three squares from a larger square -/
theorem l_shaped_area (outer_side : ℝ) (inner_side1 inner_side2 inner_side3 : ℝ) :
  outer_side = 6 ∧ 
  inner_side1 = 1 ∧ 
  inner_side2 = 2 ∧ 
  inner_side3 = 3 →
  outer_side ^ 2 - (inner_side1 ^ 2 + inner_side2 ^ 2 + inner_side3 ^ 2) = 22 :=
by sorry

end NUMINAMATH_CALUDE_l_shaped_area_l1037_103741


namespace NUMINAMATH_CALUDE_intersection_theorem_l1037_103768

-- Define the hyperbola and line equations
def hyperbola (x y : ℝ) : Prop := y = 9 / (x^2 + 1)
def line (x y : ℝ) : Prop := x + y = 4

-- Define the intersection points
def intersection_points : Set ℝ := {1, (3 + Real.sqrt 29) / 2, (3 - Real.sqrt 29) / 2}

-- Theorem statement
theorem intersection_theorem :
  ∀ x ∈ intersection_points, ∃ y, hyperbola x y ∧ line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1037_103768


namespace NUMINAMATH_CALUDE_cos_cube_decomposition_sum_of_squares_l1037_103727

open Real

theorem cos_cube_decomposition_sum_of_squares :
  (∃ b₁ b₂ b₃ : ℝ, ∀ θ : ℝ, cos θ ^ 3 = b₁ * cos θ + b₂ * cos (2 * θ) + b₃ * cos (3 * θ)) →
  (∃ b₁ b₂ b₃ : ℝ, 
    (∀ θ : ℝ, cos θ ^ 3 = b₁ * cos θ + b₂ * cos (2 * θ) + b₃ * cos (3 * θ)) ∧
    b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 = 5 / 8) :=
by sorry

end NUMINAMATH_CALUDE_cos_cube_decomposition_sum_of_squares_l1037_103727


namespace NUMINAMATH_CALUDE_circle_rectangles_l1037_103773

/-- The number of points on the circle's circumference -/
def n : ℕ := 12

/-- The number of diameters in the circle -/
def num_diameters : ℕ := n / 2

/-- The number of rectangles that can be formed -/
def num_rectangles : ℕ := Nat.choose num_diameters 2

theorem circle_rectangles :
  num_rectangles = 15 :=
sorry

end NUMINAMATH_CALUDE_circle_rectangles_l1037_103773


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1037_103764

/-- Given two unit vectors a and b in a real inner product space,
    prove that the angle between them is 2π/3 if |a-2b| = √7 -/
theorem angle_between_vectors 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (h : ‖a - 2 • b‖ = Real.sqrt 7) :
  Real.arccos (inner a b) = 2 * Real.pi / 3 := by
  sorry

#check angle_between_vectors

end NUMINAMATH_CALUDE_angle_between_vectors_l1037_103764


namespace NUMINAMATH_CALUDE_shooting_competition_probabilities_l1037_103712

theorem shooting_competition_probabilities 
  (p_A_not_losing : ℝ) 
  (p_B_losing : ℝ) 
  (h1 : p_A_not_losing = 0.59) 
  (h2 : p_B_losing = 0.44) : 
  ∃ (p_A_not_winning p_A_B_drawing : ℝ),
    p_A_not_winning = 0.56 ∧ 
    p_A_B_drawing = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_probabilities_l1037_103712


namespace NUMINAMATH_CALUDE_remainder_of_196c_pow_2008_mod_97_l1037_103728

theorem remainder_of_196c_pow_2008_mod_97 (c : ℤ) : (196 * c)^2008 % 97 = 44 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_196c_pow_2008_mod_97_l1037_103728


namespace NUMINAMATH_CALUDE_toothpicks_in_45x25_grid_with_gaps_l1037_103729

/-- Calculates the number of effective lines in a grid with gaps every fifth line -/
def effectiveLines (total : ℕ) : ℕ :=
  total + 1 - (total + 1) / 5

/-- Calculates the total number of toothpicks in a rectangular grid with gaps -/
def toothpicksInGrid (length width : ℕ) : ℕ :=
  let verticalLines := effectiveLines length
  let horizontalLines := effectiveLines width
  verticalLines * width + horizontalLines * length

/-- Theorem: A 45x25 grid with every fifth row and column missing uses 1722 toothpicks -/
theorem toothpicks_in_45x25_grid_with_gaps :
  toothpicksInGrid 45 25 = 1722 := by
  sorry

#eval toothpicksInGrid 45 25

end NUMINAMATH_CALUDE_toothpicks_in_45x25_grid_with_gaps_l1037_103729


namespace NUMINAMATH_CALUDE_tricycle_count_l1037_103797

/-- The number of tricycles in a group of children -/
def num_tricycles (total_children : ℕ) (total_wheels : ℕ) : ℕ :=
  total_children - (total_wheels - 3 * total_children) / 1

/-- Theorem stating that given 10 children and 26 wheels, there are 6 tricycles -/
theorem tricycle_count : num_tricycles 10 26 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_count_l1037_103797


namespace NUMINAMATH_CALUDE_courtyard_area_difference_l1037_103787

/-- The difference in area between a circular courtyard and a rectangular courtyard -/
theorem courtyard_area_difference :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter : ℝ := 2 * (rect_length + rect_width)
  let rect_area : ℝ := rect_length * rect_width
  let circle_radius : ℝ := rect_perimeter / (2 * Real.pi)
  let circle_area : ℝ := Real.pi * circle_radius ^ 2
  circle_area - rect_area = (6400 - 1200 * Real.pi) / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_courtyard_area_difference_l1037_103787


namespace NUMINAMATH_CALUDE_equilateral_hyperbola_equation_l1037_103725

/-- An equilateral hyperbola centered at the origin and passing through (0, 3) -/
structure EquilateralHyperbola where
  /-- The equation of the hyperbola in the form y² - x² = a -/
  equation : ℝ → ℝ → ℝ
  /-- The hyperbola passes through the point (0, 3) -/
  passes_through_point : equation 0 3 = equation 0 3
  /-- The hyperbola is centered at the origin -/
  centered_at_origin : ∀ x y, equation x y = equation (-x) (-y)
  /-- The hyperbola is equilateral -/
  equilateral : ∀ x y, equation x y = equation y x

/-- The equation of the equilateral hyperbola is y² - x² = 9 -/
theorem equilateral_hyperbola_equation (h : EquilateralHyperbola) :
  ∀ x y, h.equation x y = y^2 - x^2 - 9 := by sorry

end NUMINAMATH_CALUDE_equilateral_hyperbola_equation_l1037_103725


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l1037_103711

def num_fruits : ℕ := 4
def num_meals : ℕ := 3

theorem joe_fruit_probability :
  let p_same := (1 / num_fruits : ℚ) ^ num_meals * num_fruits
  1 - p_same = 15 / 16 := by sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l1037_103711


namespace NUMINAMATH_CALUDE_sum_gcd_lcm_18_30_45_l1037_103717

def A : ℕ := Nat.gcd 18 (Nat.gcd 30 45)
def B : ℕ := Nat.lcm 18 (Nat.lcm 30 45)

theorem sum_gcd_lcm_18_30_45 : A + B = 93 := by
  sorry

end NUMINAMATH_CALUDE_sum_gcd_lcm_18_30_45_l1037_103717


namespace NUMINAMATH_CALUDE_high_school_population_l1037_103736

/-- Represents a high school with three grades and a stratified sampling method. -/
structure HighSchool where
  grade10_students : ℕ
  total_sample : ℕ
  grade11_sample : ℕ
  grade12_sample : ℕ

/-- Calculates the total number of students in the high school based on stratified sampling. -/
def total_students (hs : HighSchool) : ℕ :=
  let grade10_sample := hs.total_sample - hs.grade11_sample - hs.grade12_sample
  (hs.grade10_students * hs.total_sample) / grade10_sample

/-- Theorem stating that given the specific conditions, the total number of students is 1800. -/
theorem high_school_population (hs : HighSchool)
  (h1 : hs.grade10_students = 600)
  (h2 : hs.total_sample = 45)
  (h3 : hs.grade11_sample = 20)
  (h4 : hs.grade12_sample = 10) :
  total_students hs = 1800 := by
  sorry

#eval total_students { grade10_students := 600, total_sample := 45, grade11_sample := 20, grade12_sample := 10 }

end NUMINAMATH_CALUDE_high_school_population_l1037_103736


namespace NUMINAMATH_CALUDE_quadratic_positivity_l1037_103774

theorem quadratic_positivity (a : ℝ) : 
  (∀ x, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_positivity_l1037_103774


namespace NUMINAMATH_CALUDE_logo_shaded_area_l1037_103763

/-- Calculates the shaded area of a logo design with a rectangle and four tangent circles -/
theorem logo_shaded_area (length width : ℝ) (h1 : length = 30) (h2 : width = 15) : 
  let rectangle_area := length * width
  let circle_radius := width / 4
  let circle_area := π * circle_radius^2
  let total_circle_area := 4 * circle_area
  rectangle_area - total_circle_area = 450 - 56.25 * π := by
  sorry

end NUMINAMATH_CALUDE_logo_shaded_area_l1037_103763


namespace NUMINAMATH_CALUDE_probability_heart_then_club_is_13_204_l1037_103740

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Suits in a deck -/
inductive Suit
| Hearts
| Clubs
| Diamonds
| Spades

/-- A card in the deck -/
structure Card :=
  (number : Fin 13)
  (suit : Suit)

/-- The probability of drawing a heart first and a club second from a standard deck -/
def probability_heart_then_club (d : Deck) : ℚ :=
  13 / 204

/-- Theorem: The probability of drawing a heart first and a club second from a standard deck is 13/204 -/
theorem probability_heart_then_club_is_13_204 (d : Deck) :
  probability_heart_then_club d = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_then_club_is_13_204_l1037_103740


namespace NUMINAMATH_CALUDE_solve_bus_problem_l1037_103739

def bus_problem (initial : ℕ) (stop_a_off stop_a_on : ℕ) (stop_b_off stop_b_on : ℕ) 
                 (stop_c_off stop_c_on : ℕ) (stop_d_off : ℕ) (final : ℕ) : Prop :=
  let after_a := initial - stop_a_off + stop_a_on
  let after_b := after_a - stop_b_off + stop_b_on
  let after_c := after_b - stop_c_off + stop_c_on
  let after_d := after_c - stop_d_off
  ∃ (stop_d_on : ℕ), after_d + stop_d_on = final ∧ stop_d_on = 10

theorem solve_bus_problem : 
  bus_problem 64 8 12 4 6 14 22 10 78 := by
  sorry

end NUMINAMATH_CALUDE_solve_bus_problem_l1037_103739


namespace NUMINAMATH_CALUDE_class_composition_l1037_103775

theorem class_composition (total : ℕ) (boys girls : ℕ) : 
  total = 20 →
  boys + girls = total →
  (boys : ℚ) / total = 3/4 * ((girls : ℚ) / total) →
  boys = 12 ∧ girls = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_class_composition_l1037_103775


namespace NUMINAMATH_CALUDE_tom_read_18_books_l1037_103760

/-- The number of books Tom read in May -/
def may_books : ℕ := 2

/-- The number of books Tom read in June -/
def june_books : ℕ := 6

/-- The number of books Tom read in July -/
def july_books : ℕ := 10

/-- The total number of books Tom read -/
def total_books : ℕ := may_books + june_books + july_books

theorem tom_read_18_books : total_books = 18 := by
  sorry

end NUMINAMATH_CALUDE_tom_read_18_books_l1037_103760


namespace NUMINAMATH_CALUDE_star_example_l1037_103791

-- Define the ⋆ operation
def star (a b c d : ℚ) : ℚ := a * c * (d / (2 * b))

-- Theorem statement
theorem star_example : star 5 6 9 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l1037_103791


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1037_103702

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b * c * Real.cos A = 4 →
  a * c * Real.sin B = 8 * Real.sin A →
  A = π / 3 ∧ 0 < Real.sin A * Real.sin B * Real.sin C ∧ 
  Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1037_103702


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1037_103770

theorem max_value_sqrt_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 →
    Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) + Real.sqrt (2 * c + 1) ≤
    Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1)) →
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1037_103770


namespace NUMINAMATH_CALUDE_rectangle_with_hole_to_square_l1037_103709

/-- Represents a rectangle with a hole -/
structure RectangleWithHole where
  width : ℝ
  height : ℝ
  hole_width : ℝ
  hole_height : ℝ

/-- Calculates the usable area of a rectangle with a hole -/
def usable_area (r : RectangleWithHole) : ℝ :=
  r.width * r.height - r.hole_width * r.hole_height

/-- Theorem: A 9x12 rectangle with a 1x8 hole can be cut into two equal parts that form a 10x10 square -/
theorem rectangle_with_hole_to_square :
  ∃ (r : RectangleWithHole),
    r.width = 9 ∧
    r.height = 12 ∧
    r.hole_width = 1 ∧
    r.hole_height = 8 ∧
    usable_area r = 100 ∧
    ∃ (side_length : ℝ),
      side_length * side_length = usable_area r ∧
      side_length = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_to_square_l1037_103709


namespace NUMINAMATH_CALUDE_sundae_cost_calculation_l1037_103744

-- Define constants for prices and discount thresholds
def scoop_price : ℚ := 2
def topping_a_price : ℚ := 0.5
def topping_b_price : ℚ := 0.75
def topping_c_price : ℚ := 0.6
def topping_d_price : ℚ := 0.8
def topping_e_price : ℚ := 0.9

def topping_a_discount_threshold : ℕ := 3
def topping_b_discount_threshold : ℕ := 2
def topping_c_discount_threshold : ℕ := 4

def topping_a_discount : ℚ := 0.3
def topping_b_discount : ℚ := 0.4
def topping_c_discount : ℚ := 0.5

-- Define the function to calculate the total cost
def calculate_sundae_cost (scoops topping_a topping_b topping_c topping_d topping_e : ℕ) : ℚ :=
  let ice_cream_cost := scoops * scoop_price
  let topping_a_cost := topping_a * topping_a_price - (topping_a / topping_a_discount_threshold) * topping_a_discount
  let topping_b_cost := topping_b * topping_b_price - (topping_b / topping_b_discount_threshold) * topping_b_discount
  let topping_c_cost := topping_c * topping_c_price - (topping_c / topping_c_discount_threshold) * topping_c_discount
  let topping_d_cost := topping_d * topping_d_price
  let topping_e_cost := topping_e * topping_e_price
  ice_cream_cost + topping_a_cost + topping_b_cost + topping_c_cost + topping_d_cost + topping_e_cost

-- Theorem statement
theorem sundae_cost_calculation :
  calculate_sundae_cost 3 5 3 7 2 1 = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_sundae_cost_calculation_l1037_103744


namespace NUMINAMATH_CALUDE_sum_of_median_scores_l1037_103778

-- Define the type for basketball scores
def Score := ℕ

-- Define a function to calculate the median of a list of scores
noncomputable def median (scores : List Score) : ℝ := sorry

-- Define the scores for player A
def scoresA : List Score := sorry

-- Define the scores for player B
def scoresB : List Score := sorry

-- Theorem to prove
theorem sum_of_median_scores : 
  median scoresA + median scoresB = 64 := by sorry

end NUMINAMATH_CALUDE_sum_of_median_scores_l1037_103778


namespace NUMINAMATH_CALUDE_swap_digits_l1037_103765

theorem swap_digits (x : ℕ) (h : 9 < x ∧ x < 100) : 
  (x % 10) * 10 + (x / 10) = 10 * (x % 10) + (x / 10) := by sorry

end NUMINAMATH_CALUDE_swap_digits_l1037_103765


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1037_103708

/-- Calculate the interest rate given simple interest, principal, and time -/
theorem interest_rate_calculation
  (simple_interest : ℝ)
  (principal : ℝ)
  (time : ℝ)
  (h1 : simple_interest = 4016.25)
  (h2 : principal = 10040.625)
  (h3 : time = 5)
  (h4 : simple_interest = principal * (rate / 100) * time) :
  rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1037_103708


namespace NUMINAMATH_CALUDE_trapezoidal_sequence_624_l1037_103738

/-- The trapezoidal sequence -/
def trapezoidal_sequence : ℕ → ℕ
| 0 => 5
| n + 1 => trapezoidal_sequence n + (n + 4)

/-- The 624th term of the trapezoidal sequence is 196250 -/
theorem trapezoidal_sequence_624 : trapezoidal_sequence 623 = 196250 := by
  sorry

end NUMINAMATH_CALUDE_trapezoidal_sequence_624_l1037_103738


namespace NUMINAMATH_CALUDE_sine_cosine_extreme_value_l1037_103703

open Real

theorem sine_cosine_extreme_value (a b : ℝ) (h : a < b) :
  ∃ f g : ℝ → ℝ,
    (∀ x ∈ Set.Icc a b, f x = sin x ∧ g x = cos x) ∧
    g a * g b < 0 ∧
    ¬(∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, g x ≤ g y ∨ g x ≥ g y) :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_extreme_value_l1037_103703


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1037_103707

theorem inequality_solution_set (y : ℝ) :
  (2 / (y - 2) + 5 / (y + 3) ≤ 2) ↔ (y ∈ Set.Ioc (-3) (-1) ∪ Set.Ioo 2 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1037_103707


namespace NUMINAMATH_CALUDE_max_marks_calculation_l1037_103745

theorem max_marks_calculation (passing_threshold : ℚ) (scored_marks : ℕ) (short_marks : ℕ) :
  passing_threshold = 30 / 100 →
  scored_marks = 212 →
  short_marks = 13 →
  ∃ max_marks : ℕ,
    max_marks = 750 ∧
    (scored_marks + short_marks : ℚ) / max_marks = passing_threshold :=
by sorry

end NUMINAMATH_CALUDE_max_marks_calculation_l1037_103745


namespace NUMINAMATH_CALUDE_rational_representation_condition_l1037_103713

theorem rational_representation_condition (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (∀ (q : ℚ), q > 0 → ∃ (r : ℚ), r > 0 ∧ q = (r * x) / (r * y)) ↔ x * y < 0 :=
sorry

end NUMINAMATH_CALUDE_rational_representation_condition_l1037_103713


namespace NUMINAMATH_CALUDE_inequality_solution_l1037_103734

theorem inequality_solution (m n : ℤ) : 
  (∀ x : ℝ, x > 0 → (m * x + 5) * (x^2 - n) ≤ 0) →
  (m + n ∈ ({-4, 24} : Set ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1037_103734


namespace NUMINAMATH_CALUDE_roots_greater_than_five_k_range_l1037_103714

theorem roots_greater_than_five_k_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11*x + (30 + k) = 0 → x > 5) → 
  0 < k ∧ k ≤ 1/4 := by
sorry

end NUMINAMATH_CALUDE_roots_greater_than_five_k_range_l1037_103714


namespace NUMINAMATH_CALUDE_candies_left_l1037_103789

def initial_candies : ℕ := 88
def candies_taken : ℕ := 6

theorem candies_left : initial_candies - candies_taken = 82 := by
  sorry

end NUMINAMATH_CALUDE_candies_left_l1037_103789


namespace NUMINAMATH_CALUDE_triangle_angle_cosine_l1037_103788

theorem triangle_angle_cosine (A B C : Real) : 
  A + B + C = Real.pi →  -- Sum of angles in a triangle is π radians
  A + C = 2 * B →
  1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B →
  Real.cos ((A - C) / 2) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_cosine_l1037_103788


namespace NUMINAMATH_CALUDE_ellen_dough_balls_l1037_103735

/-- Represents the time it takes for a ball of dough to rise -/
def rise_time : ℕ := 3

/-- Represents the time it takes to bake a ball of dough -/
def bake_time : ℕ := 2

/-- Represents the total time for the entire baking process -/
def total_time : ℕ := 20

/-- Calculates the total time taken for a given number of dough balls -/
def time_for_n_balls (n : ℕ) : ℕ :=
  rise_time + bake_time + (n - 1) * rise_time

/-- The theorem stating the number of dough balls Ellen makes -/
theorem ellen_dough_balls :
  ∃ n : ℕ, n > 0 ∧ time_for_n_balls n = total_time ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellen_dough_balls_l1037_103735


namespace NUMINAMATH_CALUDE_division_remainder_l1037_103706

theorem division_remainder : ∃ A : ℕ, 28 = 3 * 9 + A ∧ A < 3 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1037_103706


namespace NUMINAMATH_CALUDE_intersection_point_count_l1037_103700

theorem intersection_point_count :
  ∃! p : ℝ × ℝ, 
    (p.1 + p.2 - 5) * (2 * p.1 - 3 * p.2 + 5) = 0 ∧ 
    (p.1 - p.2 + 1) * (3 * p.1 + 2 * p.2 - 12) = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_count_l1037_103700


namespace NUMINAMATH_CALUDE_quartic_trinomial_m_value_l1037_103731

theorem quartic_trinomial_m_value (m : ℤ) : 
  (abs (m - 3) = 4) → (m - 7 ≠ 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quartic_trinomial_m_value_l1037_103731


namespace NUMINAMATH_CALUDE_inequality_solution_l1037_103704

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 5 / (x + 4) ≥ 1) ↔ (x < -4 ∨ x ≥ 5) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1037_103704


namespace NUMINAMATH_CALUDE_second_square_weight_l1037_103769

/-- Represents a square piece of metal -/
structure MetalSquare where
  side_length : ℝ
  weight : ℝ

/-- The density of the metal in ounces per square inch -/
def metal_density : ℝ := 0.5

theorem second_square_weight
  (first_square : MetalSquare)
  (h1 : first_square.side_length = 4)
  (h2 : first_square.weight = 8)
  (second_square : MetalSquare)
  (h3 : second_square.side_length = 7) :
  second_square.weight = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_second_square_weight_l1037_103769


namespace NUMINAMATH_CALUDE_equation_solution_l1037_103750

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (7 * x / (x - 2) - 5 / (x - 2) = 2 / (x - 2)) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1037_103750


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1037_103733

theorem arithmetic_expression_equality : (11 * 24 - 23 * 9) / 3 + 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1037_103733


namespace NUMINAMATH_CALUDE_jiangxia_is_first_largest_bidirectional_l1037_103782

structure TidalPowerPlant where
  location : String
  year_built : Nat
  is_bidirectional : Bool
  is_largest : Bool

def china_tidal_plants : Nat := 9

def jiangxia_plant : TidalPowerPlant := {
  location := "Jiangxia",
  year_built := 1980,
  is_bidirectional := true,
  is_largest := true
}

theorem jiangxia_is_first_largest_bidirectional :
  ∃ (plant : TidalPowerPlant),
    plant.year_built = 1980 ∧
    plant.is_bidirectional = true ∧
    plant.is_largest = true ∧
    plant.location = "Jiangxia" :=
by
  sorry

#check jiangxia_is_first_largest_bidirectional

end NUMINAMATH_CALUDE_jiangxia_is_first_largest_bidirectional_l1037_103782


namespace NUMINAMATH_CALUDE_equation_solution_l1037_103759

theorem equation_solution :
  let f (n : ℚ) := (2 - n) / (n + 1) + (2 * n - 4) / (2 - n)
  ∃ (n : ℚ), f n = 1 ∧ n = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1037_103759


namespace NUMINAMATH_CALUDE_slope_135_implies_y_negative_four_l1037_103761

/-- Given two points A and B, if the slope of the line passing through them is 135°, then the y-coordinate of A is -4. -/
theorem slope_135_implies_y_negative_four (x_a y_a x_b y_b : ℝ) :
  x_a = 3 →
  x_b = 2 →
  y_b = -3 →
  (y_a - y_b) / (x_a - x_b) = Real.tan (135 * π / 180) →
  y_a = -4 := by
  sorry

#check slope_135_implies_y_negative_four

end NUMINAMATH_CALUDE_slope_135_implies_y_negative_four_l1037_103761


namespace NUMINAMATH_CALUDE_village_population_l1037_103792

theorem village_population (p : ℝ) : p = 939 ↔ 0.92 * p = 1.15 * p + 216 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1037_103792


namespace NUMINAMATH_CALUDE_investment_sum_l1037_103718

theorem investment_sum (P : ℝ) : 
  P * (18 / 100) * 2 - P * (12 / 100) * 2 = 240 → P = 2000 := by sorry

end NUMINAMATH_CALUDE_investment_sum_l1037_103718


namespace NUMINAMATH_CALUDE_circle_symmetry_ab_range_l1037_103786

/-- Given a circle x^2 + y^2 - 4x + 2y + 1 = 0 symmetric about the line ax - 2by - 1 = 0 (a, b ∈ ℝ),
    the range of ab is (-∞, 1/16]. -/
theorem circle_symmetry_ab_range (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 → 
    (∃ x' y' : ℝ, x'^2 + y'^2 - 4*x' + 2*y' + 1 = 0 ∧ 
      a*x - 2*b*y - 1 = a*x' - 2*b*y' - 1 ∧ 
      (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2)) →
  a * b ≤ 1/16 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_ab_range_l1037_103786


namespace NUMINAMATH_CALUDE_geometric_sequence_log_sum_l1037_103796

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_log_sum 
  (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a) 
  (h_prod : a 2 * a 5 = 10) : 
  Real.log (a 3) + Real.log (a 4) = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_log_sum_l1037_103796


namespace NUMINAMATH_CALUDE_second_graders_borrowed_books_l1037_103780

theorem second_graders_borrowed_books (initial_books : ℕ) (remaining_books : ℕ) 
  (h1 : initial_books = 75) 
  (h2 : remaining_books = 57) : 
  initial_books - remaining_books = 18 := by
  sorry

end NUMINAMATH_CALUDE_second_graders_borrowed_books_l1037_103780


namespace NUMINAMATH_CALUDE_flat_cost_calculation_l1037_103781

theorem flat_cost_calculation (x : ℝ) : 
  x > 0 →  -- Assuming the cost is positive
  0.11 * x - (-0.11 * x) = 1.21 →
  x = 5.50 := by
  sorry

end NUMINAMATH_CALUDE_flat_cost_calculation_l1037_103781


namespace NUMINAMATH_CALUDE_tv_production_average_l1037_103723

/-- Proves that given the average production of 60 TVs/day for the first 25 days 
    of a 30-day month, and an overall monthly average of 58 TVs/day, 
    the average production for the last 5 days of the month is 48 TVs/day. -/
theorem tv_production_average (first_25_avg : ℕ) (total_days : ℕ) (monthly_avg : ℕ) :
  first_25_avg = 60 →
  total_days = 30 →
  monthly_avg = 58 →
  (monthly_avg * total_days - first_25_avg * 25) / 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_tv_production_average_l1037_103723


namespace NUMINAMATH_CALUDE_pages_per_day_l1037_103742

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 96) (h2 : days = 12) :
  total_pages / days = 8 := by
sorry

end NUMINAMATH_CALUDE_pages_per_day_l1037_103742


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l1037_103793

theorem quadratic_is_square_of_binomial (d : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 80*x + d = (x + a)^2 + b^2) → d = 1600 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l1037_103793


namespace NUMINAMATH_CALUDE_work_completion_time_l1037_103777

/-- The efficiency ratio between p and q -/
def efficiency_ratio : ℝ := 1.6

/-- The time taken by p and q working together -/
def combined_time : ℝ := 16

/-- The time taken by p working alone -/
def p_time : ℝ := 26

theorem work_completion_time :
  (efficiency_ratio * combined_time) / (efficiency_ratio + 1) = p_time := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1037_103777


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l1037_103783

theorem sum_of_four_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) = 34 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l1037_103783


namespace NUMINAMATH_CALUDE_f_properties_l1037_103794

def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

def M : Set ℝ := {x : ℝ | f x ≥ 3}

theorem f_properties :
  (M = {x : ℝ | x ≤ -1/2 ∨ x ≥ 2}) ∧
  (∀ a ∈ M, ∀ x : ℝ, |x + a| + |x - 1/a| ≥ 5/2) := by sorry

end NUMINAMATH_CALUDE_f_properties_l1037_103794


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1037_103722

-- Define the diamond operation
def diamond (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

-- State the theorem
theorem diamond_equation_solution :
  ∃! A : ℝ, diamond A 10 = 57 ∧ A = 20 := by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l1037_103722


namespace NUMINAMATH_CALUDE_volume_ratio_cone_cylinder_l1037_103776

/-- Given a cylinder and a cone with the same radius, where the cone's height is 1/3 of the cylinder's height,
    the ratio of the volume of the cone to the volume of the cylinder is 1/9. -/
theorem volume_ratio_cone_cylinder (r h : ℝ) (h_pos : 0 < r) (h_height : 0 < h) :
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_volume_ratio_cone_cylinder_l1037_103776


namespace NUMINAMATH_CALUDE_division_problem_l1037_103757

theorem division_problem (total : ℕ) (p q r : ℕ) : 
  total = 1210 →
  p * 4 = q * 5 →
  q * 10 = r * 9 →
  p + q + r = total →
  r = 400 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1037_103757


namespace NUMINAMATH_CALUDE_min_diagonals_regular_2017_gon_l1037_103715

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2017

/-- The number of different diagonal lengths in a regular n-gon -/
def num_different_lengths (n : ℕ) : ℕ := (n - 3) / 2

/-- The minimum number of diagonals to select to guarantee two of the same length -/
def min_diagonals_same_length (n : ℕ) : ℕ := num_different_lengths n + 1

theorem min_diagonals_regular_2017_gon :
  min_diagonals_same_length n = 1008 :=
sorry

end NUMINAMATH_CALUDE_min_diagonals_regular_2017_gon_l1037_103715


namespace NUMINAMATH_CALUDE_total_moving_time_l1037_103753

/-- The time (in minutes) spent filling the car for each trip. -/
def fill_time : ℕ := 15

/-- The time (in minutes) spent driving one-way for each trip. -/
def drive_time : ℕ := 30

/-- The total number of trips made. -/
def num_trips : ℕ := 6

/-- The total time spent moving, in hours. -/
def total_time : ℚ := (fill_time + drive_time) * num_trips / 60

/-- Proves that the total time spent moving is 4.5 hours. -/
theorem total_moving_time : total_time = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_total_moving_time_l1037_103753


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1037_103710

theorem inequality_solution_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ m ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1037_103710


namespace NUMINAMATH_CALUDE_fourth_number_proof_l1037_103756

theorem fourth_number_proof (x : ℝ) : 
  3 + 33 + 333 + x = 399.6 → x = 30.6 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l1037_103756


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l1037_103726

theorem unique_solution_quadratic_equation :
  ∃! (x y : ℝ), (4 * x^2 + 6 * x + 4) * (4 * y^2 - 12 * y + 25) = 28 ∧
                x = -3/4 ∧ y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l1037_103726


namespace NUMINAMATH_CALUDE_total_cars_l1037_103705

/-- The number of cars owned by each person --/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  carol : ℕ
  susan : ℕ
  erica : ℕ
  jack : ℕ
  kevin : ℕ

/-- The conditions of car ownership --/
def carOwnershipConditions (c : CarOwnership) : Prop :=
  c.cathy = 5 ∧
  c.lindsey = c.cathy + 4 ∧
  c.susan = c.carol - 2 ∧
  c.carol = 2 * c.cathy ∧
  c.erica = c.lindsey + (c.lindsey / 4) ∧
  c.jack = (c.susan + c.carol) / 2 ∧
  c.kevin = ((c.lindsey + c.cathy) * 9) / 10

/-- The theorem stating the total number of cars --/
theorem total_cars (c : CarOwnership) (h : carOwnershipConditions c) : 
  c.cathy + c.lindsey + c.carol + c.susan + c.erica + c.jack + c.kevin = 65 := by
  sorry


end NUMINAMATH_CALUDE_total_cars_l1037_103705


namespace NUMINAMATH_CALUDE_T_perimeter_is_20_l1037_103798

/-- The perimeter of a T shape formed by two 2-inch × 4-inch rectangles -/
def T_perimeter : ℝ :=
  let rectangle_width : ℝ := 2
  let rectangle_length : ℝ := 4
  let rectangle_perimeter : ℝ := 2 * (rectangle_width + rectangle_length)
  let overlap : ℝ := 2 * rectangle_width
  2 * rectangle_perimeter - overlap

/-- Theorem stating that the perimeter of the T shape is 20 inches -/
theorem T_perimeter_is_20 : T_perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_T_perimeter_is_20_l1037_103798


namespace NUMINAMATH_CALUDE_total_interest_is_1800_l1037_103719

/-- Calculates the total interest over 10 years when the principal is trebled after 5 years -/
def totalInterest (P R : ℚ) : ℚ :=
  let firstHalfInterest := (P * R * 5) / 100
  let secondHalfInterest := (3 * P * R * 5) / 100
  firstHalfInterest + secondHalfInterest

/-- Theorem stating that the total interest is 1800 given the problem conditions -/
theorem total_interest_is_1800 (P R : ℚ) 
    (h : (P * R * 10) / 100 = 900) : totalInterest P R = 1800 := by
  sorry

#eval totalInterest 1000 9  -- This should evaluate to 1800

end NUMINAMATH_CALUDE_total_interest_is_1800_l1037_103719


namespace NUMINAMATH_CALUDE_incorrect_page_number_l1037_103724

theorem incorrect_page_number (n : ℕ) (x : ℕ) : 
  (n ≥ 1) →
  (x ≤ n) →
  (n * (n + 1) / 2 + x = 2076) →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_incorrect_page_number_l1037_103724


namespace NUMINAMATH_CALUDE_average_salary_non_officers_l1037_103790

/-- Prove that the average salary of non-officers is 110 Rs/month -/
theorem average_salary_non_officers (
  total_avg : ℝ) (officer_avg : ℝ) (num_officers : ℕ) (num_non_officers : ℕ)
  (h1 : total_avg = 120)
  (h2 : officer_avg = 420)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 450)
  : (((total_avg * (num_officers + num_non_officers : ℝ)) - 
     (officer_avg * num_officers)) / num_non_officers) = 110 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_non_officers_l1037_103790


namespace NUMINAMATH_CALUDE_ramsey_theorem_for_interns_l1037_103799

/-- Represents the relationship between two interns -/
inductive Relationship
  | Knows
  | DoesNotKnow

/-- Defines a group of interns and their relationships -/
structure InternGroup :=
  (size : Nat)
  (relationships : Fin size → Fin size → Relationship)

/-- The main theorem -/
theorem ramsey_theorem_for_interns (group : InternGroup) (h : group.size = 6) :
  ∃ (a b c : Fin group.size),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    ((group.relationships a b = Relationship.Knows ∧
      group.relationships b c = Relationship.Knows ∧
      group.relationships a c = Relationship.Knows) ∨
     (group.relationships a b = Relationship.DoesNotKnow ∧
      group.relationships b c = Relationship.DoesNotKnow ∧
      group.relationships a c = Relationship.DoesNotKnow)) :=
sorry

end NUMINAMATH_CALUDE_ramsey_theorem_for_interns_l1037_103799


namespace NUMINAMATH_CALUDE_calculation_proof_l1037_103762

theorem calculation_proof :
  (∃ x, x = Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 ∧ x = 4 + Real.sqrt 6) ∧
  (∃ y, y = (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 - Real.sqrt 27 * Real.sqrt 3 + (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) ∧ y = -4) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1037_103762


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1037_103752

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 500)
  (h3 : crossing_time = 8) :
  (train_length + bridge_length) / crossing_time = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1037_103752


namespace NUMINAMATH_CALUDE_midpoint_rectangle_area_l1037_103779

/-- Given a rectangle with area 48 and length-to-width ratio 3:2, 
    the area of the rectangle formed by connecting its side midpoints is 12. -/
theorem midpoint_rectangle_area (length width : ℝ) : 
  length * width = 48 →
  length / width = 3 / 2 →
  (length / 2) * (width / 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_rectangle_area_l1037_103779


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1037_103754

theorem sufficient_condition_for_inequality (a : ℝ) :
  a ≥ 5 → ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1037_103754


namespace NUMINAMATH_CALUDE_man_speed_man_speed_specific_l1037_103755

/-- Calculates the speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : Real) (train_speed_kmh : Real) (pass_time : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / pass_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- The speed of the man given specific values -/
theorem man_speed_specific : 
  man_speed 150 83.99280057595394 6 = 6.007199827245052 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_specific_l1037_103755


namespace NUMINAMATH_CALUDE_sine_ratio_equals_one_l1037_103701

theorem sine_ratio_equals_one (c : ℝ) (h : c = 2 * π / 13) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c) * Real.sin (16 * c) * Real.sin (20 * c)) /
  (Real.sin (2 * c) * Real.sin (4 * c) * Real.sin (6 * c) * Real.sin (8 * c) * Real.sin (10 * c)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_sine_ratio_equals_one_l1037_103701


namespace NUMINAMATH_CALUDE_local_min_implies_a_eq_4_l1037_103771

/-- The function f(x) = x^3 - ax^2 + 4x - 8 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4*x - 8

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + 4

/-- Theorem: If f(x) has a local minimum at x = 2, then a = 4 -/
theorem local_min_implies_a_eq_4 (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 2| < δ → f a x ≥ f a 2) →
  f_deriv a 2 = 0 →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_local_min_implies_a_eq_4_l1037_103771


namespace NUMINAMATH_CALUDE_fibonacci_lucas_power_relation_l1037_103766

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Main theorem -/
theorem fibonacci_lucas_power_relation (n p : ℕ) :
  (((lucas n : ℝ) + Real.sqrt 5 * (fib n : ℝ)) / 2) ^ p =
  ((lucas (n * p) : ℝ) + Real.sqrt 5 * (fib (n * p) : ℝ)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_lucas_power_relation_l1037_103766


namespace NUMINAMATH_CALUDE_people_per_column_l1037_103716

theorem people_per_column (total_people : ℕ) (people_per_column : ℕ) : 
  (total_people = 16 * people_per_column) ∧ 
  (total_people = 15 * 32) → 
  people_per_column = 30 := by
  sorry

end NUMINAMATH_CALUDE_people_per_column_l1037_103716


namespace NUMINAMATH_CALUDE_asymptote_sum_l1037_103748

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x^3 + A*x^2 + B*x + C = (x + 1)*(x - 3)*(x - 4)) →
  A + B + C = 11 := by
sorry

end NUMINAMATH_CALUDE_asymptote_sum_l1037_103748


namespace NUMINAMATH_CALUDE_school_boys_count_l1037_103747

theorem school_boys_count (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) (other_count : ℕ) :
  muslim_percent = 44/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  other_count = 117 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℚ) / total = 1) ∧
    total = 650 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l1037_103747


namespace NUMINAMATH_CALUDE_four_touching_circles_l1037_103758

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Returns true if the circle touches the line -/
def touches (c : Circle) (l : Line) : Prop :=
  sorry

/-- The main theorem stating that there are exactly 4 circles of a given radius
    touching two given lines -/
theorem four_touching_circles 
  (r : ℝ) 
  (l₁ l₂ : Line) 
  (h_r : r > 0) 
  (h_distinct : l₁ ≠ l₂) : 
  ∃! (s : Finset Circle), 
    s.card = 4 ∧ 
    ∀ c ∈ s, c.radius = r ∧ touches c l₁ ∧ touches c l₂ :=
sorry

end NUMINAMATH_CALUDE_four_touching_circles_l1037_103758


namespace NUMINAMATH_CALUDE_ant_position_after_2020_moves_l1037_103767

/-- Represents the direction the ant is facing -/
inductive Direction
| East
| North
| West
| South

/-- Represents the position and state of the ant -/
structure AntState :=
  (x : Int) (y : Int) (direction : Direction) (moveCount : Nat)

/-- Function to update the ant's state after one move -/
def move (state : AntState) : AntState :=
  match state.direction with
  | Direction.East => { x := state.x + state.moveCount + 1, y := state.y, direction := Direction.North, moveCount := state.moveCount + 1 }
  | Direction.North => { x := state.x, y := state.y + state.moveCount + 1, direction := Direction.West, moveCount := state.moveCount + 1 }
  | Direction.West => { x := state.x - state.moveCount - 1, y := state.y, direction := Direction.South, moveCount := state.moveCount + 1 }
  | Direction.South => { x := state.x, y := state.y - state.moveCount - 1, direction := Direction.East, moveCount := state.moveCount + 1 }

/-- Function to update the ant's state after n moves -/
def moveN (state : AntState) (n : Nat) : AntState :=
  match n with
  | 0 => state
  | Nat.succ m => move (moveN state m)

/-- The main theorem to prove -/
theorem ant_position_after_2020_moves :
  let initialState : AntState := { x := -20, y := 20, direction := Direction.East, moveCount := 0 }
  let finalState := moveN initialState 2020
  finalState.x = -1030 ∧ finalState.y = -990 := by sorry

end NUMINAMATH_CALUDE_ant_position_after_2020_moves_l1037_103767


namespace NUMINAMATH_CALUDE_train_clicks_theorem_l1037_103743

/-- Represents the number of clicks heard in 30 seconds for a train accelerating from 30 to 60 mph over 5 miles --/
def train_clicks : ℕ := 40

/-- Rail length in feet --/
def rail_length : ℝ := 50

/-- Initial speed in miles per hour --/
def initial_speed : ℝ := 30

/-- Final speed in miles per hour --/
def final_speed : ℝ := 60

/-- Acceleration distance in miles --/
def acceleration_distance : ℝ := 5

/-- Time period in seconds --/
def time_period : ℝ := 30

/-- Theorem stating that the number of clicks heard in 30 seconds is approximately 40 --/
theorem train_clicks_theorem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |train_clicks - (((initial_speed + final_speed) / 2 * 5280 / 60) / rail_length * (time_period / 60))| < ε :=
sorry

end NUMINAMATH_CALUDE_train_clicks_theorem_l1037_103743


namespace NUMINAMATH_CALUDE_circle_angle_change_l1037_103772

theorem circle_angle_change (R L α r l β : ℝ) : 
  r = R / 2 → 
  l = 3 * L / 2 → 
  L = R * α → 
  l = r * β → 
  β / α = 3 := by sorry

end NUMINAMATH_CALUDE_circle_angle_change_l1037_103772


namespace NUMINAMATH_CALUDE_lawn_width_proof_l1037_103721

theorem lawn_width_proof (length width : ℝ) (road_width cost_per_sqm total_cost : ℝ) : 
  length = 80 →
  road_width = 15 →
  cost_per_sqm = 3 →
  total_cost = 5625 →
  (road_width * width + road_width * length - road_width * road_width) * cost_per_sqm = total_cost →
  width = 60 := by
sorry

end NUMINAMATH_CALUDE_lawn_width_proof_l1037_103721


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1037_103751

/-- A quadratic function with positive leading coefficient -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The roots of f(x) - x = 0 for a quadratic function f -/
structure QuadraticRoots (f : QuadraticFunction) where
  x₁ : ℝ
  x₂ : ℝ
  root_order : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / f.a

theorem quadratic_function_properties (f : QuadraticFunction) (roots : QuadraticRoots f) :
  (∀ x, 0 < x ∧ x < roots.x₁ → x < f.a * x^2 + f.b * x + f.c ∧ f.a * x^2 + f.b * x + f.c < roots.x₁) ∧
  roots.x₁ < roots.x₂ / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1037_103751


namespace NUMINAMATH_CALUDE_theater_ticket_price_l1037_103730

theorem theater_ticket_price (adult_price : ℕ) 
  (total_attendance : ℕ) (total_revenue : ℕ) (child_attendance : ℕ) :
  total_attendance = 280 →
  total_revenue = 14000 →
  child_attendance = 80 →
  (total_attendance - child_attendance) * adult_price + child_attendance * 25 = total_revenue →
  adult_price = 60 := by
sorry

end NUMINAMATH_CALUDE_theater_ticket_price_l1037_103730


namespace NUMINAMATH_CALUDE_soccer_match_draw_probability_l1037_103732

theorem soccer_match_draw_probability 
  (p_win : ℝ) 
  (p_not_lose : ℝ) 
  (h1 : p_win = 0.3) 
  (h2 : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_soccer_match_draw_probability_l1037_103732


namespace NUMINAMATH_CALUDE_set_operation_proof_l1037_103784

theorem set_operation_proof (A B C : Set ℕ) : 
  A = {1, 2} → B = {1, 2, 3} → C = {2, 3, 4} → 
  (A ∩ B) ∪ C = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_proof_l1037_103784


namespace NUMINAMATH_CALUDE_distance_between_trees_441_22_l1037_103785

/-- The distance between consecutive trees in a yard -/
def distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : ℚ :=
  (yard_length : ℚ) / (num_trees - 1 : ℚ)

/-- Theorem: The distance between consecutive trees in a 441-metre yard with 22 trees is 21 metres -/
theorem distance_between_trees_441_22 :
  distance_between_trees 441 22 = 21 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_441_22_l1037_103785


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1037_103737

universe u

def U : Set (Fin 5) := {0, 1, 2, 3, 4}
def M : Set (Fin 5) := {0, 2, 3}
def N : Set (Fin 5) := {1, 3, 4}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1037_103737


namespace NUMINAMATH_CALUDE_quadratic_trinomial_condition_l1037_103746

/-- Given a constant m, if x^|m| + (m-2)x - 10 is a quadratic trinomial, then m = -2 -/
theorem quadratic_trinomial_condition (m : ℝ) : 
  (∃ (a b c : ℝ), ∀ x, x^(|m|) + (m-2)*x - 10 = a*x^2 + b*x + c) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_condition_l1037_103746


namespace NUMINAMATH_CALUDE_marcia_blouses_l1037_103720

/-- Calculates the number of blouses Marcia can add to her wardrobe given the following conditions:
  * Marcia needs 3 skirts, 2 pairs of pants, and some blouses
  * Skirts cost $20.00 each
  * Blouses cost $15.00 each
  * Pants cost $30.00 each
  * There's a sale on pants: buy 1 pair get 1 pair 1/2 off
  * Total budget is $180.00
-/
def calculate_blouses (skirt_count : Nat) (skirt_price : Nat) (pant_count : Nat) (pant_price : Nat) (blouse_price : Nat) (total_budget : Nat) : Nat :=
  let skirt_total := skirt_count * skirt_price
  let pant_total := pant_price + (pant_price / 2)
  let remaining_budget := total_budget - skirt_total - pant_total
  remaining_budget / blouse_price

theorem marcia_blouses :
  calculate_blouses 3 20 2 30 15 180 = 5 := by
  sorry

end NUMINAMATH_CALUDE_marcia_blouses_l1037_103720
