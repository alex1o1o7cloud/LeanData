import Mathlib

namespace NUMINAMATH_CALUDE_fraction_product_value_l2786_278615

/-- The product of fractions from 8/4 to 2008/2004 following the pattern (4n+4)/(4n) -/
def fraction_product : ℚ :=
  (2008 : ℚ) / 4

theorem fraction_product_value : fraction_product = 502 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_value_l2786_278615


namespace NUMINAMATH_CALUDE_sqrt_five_squared_l2786_278667

theorem sqrt_five_squared : (Real.sqrt 5) ^ 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_l2786_278667


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2786_278678

-- Define the inequality
def inequality (x : ℝ) : Prop := (1 - x) * (x - 3) < 0

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | x < 1 ∨ x > 3}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2786_278678


namespace NUMINAMATH_CALUDE_vertical_line_condition_l2786_278605

/-- Given two points A and B, if the line AB has an angle of inclination of 90°, then a = 0 -/
theorem vertical_line_condition (a : ℝ) : 
  let A : ℝ × ℝ := (1 + a, 2 * a)
  let B : ℝ × ℝ := (1 - a, 3)
  (A.1 = B.1) →  -- This condition represents a vertical line (90° inclination)
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_vertical_line_condition_l2786_278605


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l2786_278646

theorem root_difference_implies_k_value :
  ∀ (k : ℝ) (r s : ℝ),
  (r^2 + k*r + 12 = 0) ∧ (s^2 + k*s + 12 = 0) →
  ((r+3)^2 - k*(r+3) + 12 = 0) ∧ ((s+3)^2 - k*(s+3) + 12 = 0) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l2786_278646


namespace NUMINAMATH_CALUDE_ashton_pencils_left_l2786_278684

/-- The number of pencils Ashton has left after giving some away -/
def pencils_left (initial_boxes : ℕ) (pencils_per_box : ℕ) (given_to_brother : ℕ) (given_to_friends : ℕ) : ℕ :=
  initial_boxes * pencils_per_box - given_to_brother - given_to_friends

/-- Theorem stating that Ashton has 24 pencils left -/
theorem ashton_pencils_left : pencils_left 3 14 6 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ashton_pencils_left_l2786_278684


namespace NUMINAMATH_CALUDE_no_real_some_complex_solutions_l2786_278657

-- Define the system of equations
def equation1 (x y : ℂ) : Prop := y = (x + 1)^2
def equation2 (x y : ℂ) : Prop := x * y^2 + y = 1

-- Theorem statement
theorem no_real_some_complex_solutions :
  (∀ x y : ℝ, ¬(equation1 x y ∧ equation2 x y)) ∧
  (∃ x y : ℂ, equation1 x y ∧ equation2 x y) :=
sorry

end NUMINAMATH_CALUDE_no_real_some_complex_solutions_l2786_278657


namespace NUMINAMATH_CALUDE_square_area_ratio_l2786_278625

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 3 * (4 * b)) : a^2 = 9 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2786_278625


namespace NUMINAMATH_CALUDE_divisor_problem_l2786_278607

theorem divisor_problem (x d : ℝ) (h1 : x = 33) (h2 : x / d + 9 = 15) : d = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2786_278607


namespace NUMINAMATH_CALUDE_students_with_all_pets_l2786_278699

theorem students_with_all_pets (total_students : ℕ) 
  (dog_ratio : ℚ) (cat_ratio : ℚ) (other_pets : ℕ) (no_pets : ℕ)
  (only_dogs : ℕ) (dogs_and_other : ℕ) (only_cats : ℕ) :
  total_students = 40 →
  dog_ratio = 1/2 →
  cat_ratio = 2/5 →
  other_pets = 8 →
  no_pets = 7 →
  only_dogs = 12 →
  dogs_and_other = 3 →
  only_cats = 11 →
  ∃ (all_pets : ℕ),
    all_pets = 5 ∧
    total_students * dog_ratio = only_dogs + dogs_and_other + all_pets ∧
    total_students * cat_ratio = only_cats + all_pets ∧
    other_pets = dogs_and_other + all_pets ∧
    total_students - no_pets = only_dogs + dogs_and_other + only_cats + all_pets :=
by sorry

end NUMINAMATH_CALUDE_students_with_all_pets_l2786_278699


namespace NUMINAMATH_CALUDE_calculate_expression_l2786_278631

theorem calculate_expression : 3000 * (3000^2999 - 3000^2998) = 3000^2999 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2786_278631


namespace NUMINAMATH_CALUDE_triangle_angle_60_degrees_l2786_278647

theorem triangle_angle_60_degrees (A B C : Real) (hABC : A + B + C = Real.pi)
  (h_eq : Real.sin A ^ 2 - Real.sin C ^ 2 + Real.sin B ^ 2 = Real.sin A * Real.sin B) :
  C = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_60_degrees_l2786_278647


namespace NUMINAMATH_CALUDE_remainder_987654_div_8_l2786_278641

theorem remainder_987654_div_8 : 987654 % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_987654_div_8_l2786_278641


namespace NUMINAMATH_CALUDE_largest_consecutive_composite_l2786_278692

theorem largest_consecutive_composite : ∃ (n : ℕ), 
  (n < 50) ∧ 
  (n ≥ 10) ∧ 
  (∀ i ∈ Finset.range 10, ¬(Nat.Prime (n - i))) ∧
  (∀ m : ℕ, m > n → ¬(∀ i ∈ Finset.range 10, ¬(Nat.Prime (m - i)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_composite_l2786_278692


namespace NUMINAMATH_CALUDE_book_pages_proof_l2786_278653

/-- Calculates the number of digits used to number pages from 1 to n -/
def digits_used (n : ℕ) : ℕ := sorry

/-- The number of pages in the book -/
def num_pages : ℕ := 155

/-- The total number of digits used to number all pages -/
def total_digits : ℕ := 357

theorem book_pages_proof : digits_used num_pages = total_digits := by sorry

end NUMINAMATH_CALUDE_book_pages_proof_l2786_278653


namespace NUMINAMATH_CALUDE_unique_solution_l2786_278645

/-- Represents the ages of two people satisfying the given conditions -/
structure AgesPair where
  first : ℕ
  second : ℕ
  sum_is_35 : first + second = 35
  age_relation : 2 * first - second = second - first

/-- The unique solution to the age problem -/
theorem unique_solution : ∃! (ages : AgesPair), ages.first = 20 ∧ ages.second = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2786_278645


namespace NUMINAMATH_CALUDE_equivalent_discount_l2786_278613

theorem equivalent_discount (original_price : ℝ) (first_discount second_discount : ℝ) :
  original_price = 50 →
  first_discount = 0.3 →
  second_discount = 0.4 →
  let discounted_price := original_price * (1 - first_discount)
  let final_price := discounted_price * (1 - second_discount)
  let equivalent_discount := (original_price - final_price) / original_price
  equivalent_discount = 0.58 := by
sorry

end NUMINAMATH_CALUDE_equivalent_discount_l2786_278613


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequalities_l2786_278612

theorem greatest_whole_number_satisfying_inequalities :
  ∃ (n : ℕ), n = 1 ∧
  (∀ (x : ℝ), (x > n → ¬(3 * x - 5 < 1 - x ∧ 2 * x + 4 ≤ 8))) ∧
  (3 * n - 5 < 1 - n ∧ 2 * n + 4 ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequalities_l2786_278612


namespace NUMINAMATH_CALUDE_f_2004_equals_2003_l2786_278659

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem f_2004_equals_2003 
  (f g : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_relation : ∀ x, g x = f (x - 1))
  (h_g1 : g 1 = 2003) :
  f 2004 = 2003 := by
  sorry

end NUMINAMATH_CALUDE_f_2004_equals_2003_l2786_278659


namespace NUMINAMATH_CALUDE_turtle_distribution_theorem_l2786_278644

/-- The ratio of turtles received by Marion, Martha, and Martin -/
def turtle_ratio : Fin 3 → ℕ
| 0 => 3  -- Marion
| 1 => 2  -- Martha
| 2 => 1  -- Martin

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The total number of turtles received by all three -/
def total_turtles : ℕ := martha_turtles * (turtle_ratio 0 + turtle_ratio 1 + turtle_ratio 2) / turtle_ratio 1

theorem turtle_distribution_theorem : total_turtles = 120 := by
  sorry

end NUMINAMATH_CALUDE_turtle_distribution_theorem_l2786_278644


namespace NUMINAMATH_CALUDE_minoxidil_mixture_l2786_278685

theorem minoxidil_mixture (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_volume : ℝ) (added_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 70 ∧ 
  initial_concentration = 0.02 ∧ 
  added_volume = 35 ∧ 
  added_concentration = 0.05 ∧ 
  final_concentration = 0.03 →
  (initial_volume * initial_concentration + added_volume * added_concentration) / 
    (initial_volume + added_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_minoxidil_mixture_l2786_278685


namespace NUMINAMATH_CALUDE_regression_line_not_most_points_l2786_278628

/-- A type representing a scatter plot of data points. -/
structure ScatterPlot where
  points : Set (ℝ × ℝ)

/-- A type representing a line in 2D space. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The regression line for a given scatter plot. -/
noncomputable def regressionLine (plot : ScatterPlot) : Line :=
  sorry

/-- The number of points a line passes through in a scatter plot. -/
def pointsPassed (line : Line) (plot : ScatterPlot) : ℕ :=
  sorry

/-- The statement that the regression line passes through the most points. -/
def regressionLinePassesMostPoints (plot : ScatterPlot) : Prop :=
  ∀ l : Line, pointsPassed (regressionLine plot) plot ≥ pointsPassed l plot

/-- Theorem stating that the regression line does not necessarily pass through the most points. -/
theorem regression_line_not_most_points :
  ∃ plot : ScatterPlot, ¬(regressionLinePassesMostPoints plot) :=
sorry

end NUMINAMATH_CALUDE_regression_line_not_most_points_l2786_278628


namespace NUMINAMATH_CALUDE_integer_root_values_l2786_278626

theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, x^3 + 3*x^2 + a*x + 9 = 0) ↔ 
  a ∈ ({-109, -21, -13, 3, 11, 53} : Set ℤ) :=
sorry

end NUMINAMATH_CALUDE_integer_root_values_l2786_278626


namespace NUMINAMATH_CALUDE_circle_common_chord_l2786_278673

theorem circle_common_chord (a : ℝ) (h : a > 0) : 
  ∃ (x y : ℝ), 
    (x^2 + y^2 = 4 ∧ 
     x^2 + y^2 + 2*x + 2*a*y - 6 = 0 ∧ 
     ∃ (x₁ y₁ x₂ y₂ : ℝ), 
       (x₁^2 + y₁^2 = 4 ∧ 
        x₁^2 + y₁^2 + 2*x₁ + 2*a*y₁ - 6 = 0 ∧
        x₂^2 + y₂^2 = 4 ∧ 
        x₂^2 + y₂^2 + 2*x₂ + 2*a*y₂ - 6 = 0 ∧
        (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12)) →
    a = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_common_chord_l2786_278673


namespace NUMINAMATH_CALUDE_min_value_2n_plus_k_l2786_278686

theorem min_value_2n_plus_k (n k : ℕ) : 
  (144 + n) * 2 = n * k → -- total coins after sharing
  n > 0 → -- at least one person joins
  k > 0 → -- each person carries at least one coin
  2 * n + k ≥ 50 ∧ ∃ (n' k' : ℕ), 2 * n' + k' = 50 ∧ (144 + n') * 2 = n' * k' ∧ n' > 0 ∧ k' > 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_2n_plus_k_l2786_278686


namespace NUMINAMATH_CALUDE_campers_fed_specific_l2786_278630

/-- The number of campers that can be fed given the caught fish --/
def campers_fed (trout_weight : ℕ) (bass_count bass_weight : ℕ) (salmon_count salmon_weight : ℕ) (consumption_per_camper : ℕ) : ℕ :=
  (trout_weight + bass_count * bass_weight + salmon_count * salmon_weight) / consumption_per_camper

/-- Theorem stating the number of campers that can be fed given the specific fishing scenario --/
theorem campers_fed_specific : campers_fed 8 6 2 2 12 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_campers_fed_specific_l2786_278630


namespace NUMINAMATH_CALUDE_smallest_integer_absolute_value_l2786_278608

theorem smallest_integer_absolute_value (x : ℤ) :
  (∀ y : ℤ, |3 * y - 4| ≤ 22 → x ≤ y) ↔ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_absolute_value_l2786_278608


namespace NUMINAMATH_CALUDE_line_direction_vector_l2786_278606

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := (2, 0) + t • d
  let d : ℝ × ℝ := direction_vector line
  let y (x : ℝ) : ℝ := (5 * x - 7) / 6
  ∀ x ≥ 2, (x - 2) ^ 2 + (y x) ^ 2 = t ^ 2 →
  d = (6 / Real.sqrt 61, 5 / Real.sqrt 61) :=
by sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2786_278606


namespace NUMINAMATH_CALUDE_cone_base_radius_l2786_278696

/-- Given a sector with radius 5 and central angle 144°, prove that when wrapped into a cone, 
    the radius of the base of the cone is 2. -/
theorem cone_base_radius (r : ℝ) (θ : ℝ) : 
  r = 5 → θ = 144 → (θ / 360) * (2 * π * r) = 2 * π * 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2786_278696


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2786_278634

theorem complex_modulus_problem (z : ℂ) :
  (2017 * z - 25) / (z - 2017) = (3 : ℂ) + 4 * I →
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2786_278634


namespace NUMINAMATH_CALUDE_average_b_c_l2786_278674

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : a - c = 80) : 
  (b + c) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_b_c_l2786_278674


namespace NUMINAMATH_CALUDE_sequence_property_initial_condition_main_theorem_l2786_278600

def sequence_a (n : ℕ) : ℝ :=
  sorry

theorem sequence_property (n : ℕ) :
  (2 * n + 3 : ℝ) * sequence_a (n + 1) - (2 * n + 5 : ℝ) * sequence_a n =
  (2 * n + 3 : ℝ) * (2 * n + 5 : ℝ) * Real.log (1 + 1 / (n : ℝ)) :=
  sorry

theorem initial_condition : sequence_a 1 = 5 :=
  sorry

theorem main_theorem (n : ℕ) (hn : n > 0) :
  sequence_a n / (2 * n + 3 : ℝ) = 1 + Real.log n :=
  sorry

end NUMINAMATH_CALUDE_sequence_property_initial_condition_main_theorem_l2786_278600


namespace NUMINAMATH_CALUDE_equation_solution_l2786_278629

theorem equation_solution (x : ℝ) (h : 9 - 16/x + 9/x^2 = 0) : 3/x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2786_278629


namespace NUMINAMATH_CALUDE_divisible_by_three_l2786_278639

theorem divisible_by_three (A B : ℤ) (h : A > B) :
  ∃ x : ℤ, (x = A ∨ x = B ∨ x = A + B ∨ x = A - B) ∧ x % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l2786_278639


namespace NUMINAMATH_CALUDE_viewers_scientific_notation_l2786_278652

/-- Represents 1 billion -/
def billion : ℝ := 1000000000

/-- The number of viewers who watched the Spring Festival Gala live broadcast -/
def viewers : ℝ := 1.173 * billion

/-- Theorem stating that the number of viewers in billions is equal to its scientific notation -/
theorem viewers_scientific_notation : viewers = 1.173 * (10 : ℝ)^9 := by
  sorry

end NUMINAMATH_CALUDE_viewers_scientific_notation_l2786_278652


namespace NUMINAMATH_CALUDE_rain_probability_l2786_278618

/-- The probability of rain on Friday -/
def prob_friday : ℝ := 0.7

/-- The probability of rain on Saturday -/
def prob_saturday : ℝ := 0.5

/-- The probability of rain on Sunday -/
def prob_sunday : ℝ := 0.3

/-- The events are independent -/
axiom independence : True

/-- The probability of rain on all three days -/
def prob_all_days : ℝ := prob_friday * prob_saturday * prob_sunday

/-- Theorem: The probability of rain on all three days is 10.5% -/
theorem rain_probability : prob_all_days = 0.105 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l2786_278618


namespace NUMINAMATH_CALUDE_parabola_vertex_l2786_278654

/-- The parabola defined by y = -x^2 + cx + d -/
def parabola (c d : ℝ) (x : ℝ) : ℝ := -x^2 + c*x + d

/-- The solution set of the inequality -x^2 + cx + d ≤ 0 -/
def solution_set (c d : ℝ) : Set ℝ := {x | x ∈ Set.Icc (-6) (-1) ∨ x ∈ Set.Ici 4}

theorem parabola_vertex (c d : ℝ) :
  (solution_set c d = {x | x ∈ Set.Icc (-6) (-1) ∨ x ∈ Set.Ici 4}) →
  (∃ (x y : ℝ), x = 7/2 ∧ y = -171/4 ∧
    ∀ (t : ℝ), parabola c d t ≤ parabola c d x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2786_278654


namespace NUMINAMATH_CALUDE_investment_duration_theorem_l2786_278642

def initial_investment : ℝ := 2000
def interest_rate_1 : ℝ := 0.08
def interest_rate_2 : ℝ := 0.12
def final_value : ℝ := 6620
def years_at_rate_1 : ℕ := 2

def investment_equation (t : ℝ) : Prop :=
  initial_investment * (1 + interest_rate_1) ^ years_at_rate_1 * (1 + interest_rate_2) ^ (t - years_at_rate_1) = final_value

theorem investment_duration_theorem :
  ∃ t : ℕ, (∀ s : ℝ, investment_equation s → t ≥ ⌈s⌉) ∧ investment_equation (t : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_investment_duration_theorem_l2786_278642


namespace NUMINAMATH_CALUDE_money_distribution_problem_l2786_278638

/-- The number of people in the money distribution problem -/
def num_people : ℕ := 195

/-- The amount of coins the first person receives -/
def first_person_coins : ℕ := 3

/-- The amount of coins each person receives after redistribution -/
def redistribution_coins : ℕ := 100

theorem money_distribution_problem :
  ∃ (n : ℕ), n = num_people ∧
  first_person_coins * n + (n * (n - 1)) / 2 = redistribution_coins * n :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_problem_l2786_278638


namespace NUMINAMATH_CALUDE_cherry_tree_leaves_l2786_278623

/-- The number of cherry trees originally planned to be planted -/
def original_plan : ℕ := 7

/-- The actual number of cherry trees planted -/
def actual_trees : ℕ := 2 * original_plan

/-- The number of leaves each tree drops -/
def leaves_per_tree : ℕ := 100

/-- The total number of leaves falling from all cherry trees -/
def total_leaves : ℕ := actual_trees * leaves_per_tree

theorem cherry_tree_leaves : total_leaves = 1400 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tree_leaves_l2786_278623


namespace NUMINAMATH_CALUDE_terminal_side_of_half_angle_l2786_278668

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  (∃ n : ℤ, n * 360 + 90 < α ∧ α < n * 360 + 135) ∨
  (∃ n : ℤ, n * 360 + 270 < α ∧ α < n * 360 + 315)

theorem terminal_side_of_half_angle (α : Real) :
  is_in_third_quadrant α → is_in_second_or_fourth_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_terminal_side_of_half_angle_l2786_278668


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2786_278698

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the line x = a²/c intersects its asymptotes at points A and B,
    and triangle ABF is a right-angled triangle (where F is the right focus),
    then the eccentricity of the hyperbola is √2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (A B F : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}) →
  A.1 = a^2 / c →
  B.1 = a^2 / c →
  F.1 = c →
  F.2 = 0 →
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 →
  c / a = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2786_278698


namespace NUMINAMATH_CALUDE_min_value_h_positive_m_l2786_278616

/-- The minimum value of ax - ln x for x > 0 and a ≥ 1 is 1 + ln a -/
theorem min_value_h (a : ℝ) (ha : a ≥ 1) :
  ∀ x > 0, a * x - Real.log x ≥ 1 + Real.log a := by sorry

/-- For all x > 0 and a ≥ 1, ax - ln(x + 1) > 0 -/
theorem positive_m (a : ℝ) (ha : a ≥ 1) :
  ∀ x > 0, a * x - Real.log (x + 1) > 0 := by sorry

end NUMINAMATH_CALUDE_min_value_h_positive_m_l2786_278616


namespace NUMINAMATH_CALUDE_rectangle_width_equals_eight_l2786_278603

theorem rectangle_width_equals_eight (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ)
  (h1 : square_side = 4)
  (h2 : rect_length = 2)
  (h3 : square_side * square_side = rect_length * rect_width) :
  rect_width = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_equals_eight_l2786_278603


namespace NUMINAMATH_CALUDE_representative_selection_counts_l2786_278611

def num_boys : Nat := 5
def num_girls : Nat := 3
def num_representatives : Nat := 5
def num_subjects : Nat := 5

theorem representative_selection_counts :
  let scenario1 := (num_girls.choose 1) * (num_boys.choose 4) * (num_representatives.factorial) +
                   (num_girls.choose 2) * (num_boys.choose 3) * (num_representatives.factorial)
  let scenario2 := ((num_boys + num_girls - 1).choose (num_representatives - 1)) * ((num_representatives - 1).factorial)
  let scenario3 := ((num_boys + num_girls - 1).choose (num_representatives - 1)) * ((num_representatives - 1).factorial) * (num_subjects - 1)
  let scenario4 := ((num_boys + num_girls - 2).choose (num_representatives - 2)) * ((num_representatives - 2).factorial) * (num_subjects - 1)
  (∃ (count1 count2 count3 count4 : Nat),
    count1 = scenario1 ∧
    count2 = scenario2 ∧
    count3 = scenario3 ∧
    count4 = scenario4) := by sorry

end NUMINAMATH_CALUDE_representative_selection_counts_l2786_278611


namespace NUMINAMATH_CALUDE_box_dimensions_sum_l2786_278660

-- Define the dimensions of the box
variable (P Q R : ℝ)

-- Define the conditions
def condition1 : Prop := P * Q = 30
def condition2 : Prop := P * R = 50
def condition3 : Prop := Q * R = 90

-- Theorem statement
theorem box_dimensions_sum 
  (h1 : condition1 P Q)
  (h2 : condition2 P R)
  (h3 : condition3 Q R) :
  P + Q + R = 18 * Real.sqrt 1.5 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_sum_l2786_278660


namespace NUMINAMATH_CALUDE_range_of_a_range_of_b_l2786_278656

-- Define propositions
def p (a : ℝ) : Prop := ∀ x, 2^x + 1 ≥ a

def q (a : ℝ) : Prop := ∀ x, a * x^2 - x + a > 0

def m (a b : ℝ) : Prop := ∃ x, x^2 + b*x + a = 0

-- Theorem for part (1)
theorem range_of_a : 
  (∃ a, p a ∧ q a) → (∀ a, p a ∧ q a → a > 1/2 ∧ a ≤ 1) :=
sorry

-- Theorem for part (2)
theorem range_of_b :
  (∀ a b, (¬p a → ¬m a b) ∧ ¬(m a b → ¬p a)) →
  (∀ b, (∃ a, ¬p a ∧ m a b) → b > -2 ∧ b < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_b_l2786_278656


namespace NUMINAMATH_CALUDE_gcd_of_840_and_1764_l2786_278665

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_840_and_1764_l2786_278665


namespace NUMINAMATH_CALUDE_hash_twelve_six_l2786_278693

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- State the theorem
theorem hash_twelve_six :
  (∀ r s : ℝ, hash r 0 = r) →
  (∀ r s : ℝ, hash r s = hash s r) →
  (∀ r s : ℝ, hash (r + 2) s = hash r s + 2 * s + 2) →
  hash 12 6 = 168 :=
by
  sorry

end NUMINAMATH_CALUDE_hash_twelve_six_l2786_278693


namespace NUMINAMATH_CALUDE_employees_using_public_transportation_l2786_278662

theorem employees_using_public_transportation 
  (total_employees : ℕ) 
  (drive_percentage : ℚ) 
  (public_transport_fraction : ℚ) :
  total_employees = 100 →
  drive_percentage = 60 / 100 →
  public_transport_fraction = 1 / 2 →
  (total_employees : ℚ) * (1 - drive_percentage) * public_transport_fraction = 20 := by
  sorry

end NUMINAMATH_CALUDE_employees_using_public_transportation_l2786_278662


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2786_278666

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (40 - a) + b / (75 - b) + c / (85 - c) = 8) :
  8 / (40 - a) + 15 / (75 - b) + 17 / (85 - c) = 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2786_278666


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2786_278651

def M : ℕ := 36 * 36 * 98 * 210

-- Sum of odd divisors
def sum_odd_divisors (n : ℕ) : ℕ := sorry

-- Sum of even divisors
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 60 = sum_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2786_278651


namespace NUMINAMATH_CALUDE_probability_one_match_l2786_278680

/-- Represents the two topics that can be chosen. -/
inductive Topic : Type
  | A : Topic
  | B : Topic

/-- Represents a selection of topics by the three teachers. -/
def Selection := Topic × Topic × Topic

/-- The set of all possible selections. -/
def allSelections : Finset Selection := sorry

/-- Predicate for selections where exactly one male and the female choose the same topic. -/
def exactlyOneMatch (s : Selection) : Prop := sorry

/-- The set of selections where exactly one male and the female choose the same topic. -/
def matchingSelections : Finset Selection := sorry

/-- Theorem stating that the probability of exactly one male and the female choosing the same topic is 1/2. -/
theorem probability_one_match :
  (matchingSelections.card : ℚ) / allSelections.card = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_one_match_l2786_278680


namespace NUMINAMATH_CALUDE_min_product_of_three_distinct_l2786_278636

def S : Finset Int := {-10, -5, -3, 0, 4, 6, 9}

theorem min_product_of_three_distinct (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
  a * b * c ≤ x * y * z :=
by sorry

end NUMINAMATH_CALUDE_min_product_of_three_distinct_l2786_278636


namespace NUMINAMATH_CALUDE_min_swaps_to_reverse_l2786_278697

/-- Represents a strip of cells containing tokens -/
def Strip := Fin 100 → ℕ

/-- Reverses the order of tokens in the strip -/
def reverse (s : Strip) : Strip :=
  fun i => s (99 - i)

/-- Represents a swap operation -/
inductive Swap
  | adjacent : Fin 100 → Swap
  | free : Fin 96 → Swap

/-- Applies a swap operation to a strip -/
def applySwap (s : Strip) (swap : Swap) : Strip :=
  match swap with
  | Swap.adjacent i => 
      if i < 99 then
        fun j => if j = i then s (i+1) 
                 else if j = i+1 then s i
                 else s j
      else s
  | Swap.free i => 
      fun j => if j = i then s (i+4)
               else if j = i+4 then s i
               else s j

/-- A sequence of swap operations -/
def SwapSequence := List Swap

/-- Applies a sequence of swaps to a strip -/
def applySwaps (s : Strip) : SwapSequence → Strip
  | [] => s
  | (swap :: rest) => applySwaps (applySwap s swap) rest

/-- Counts the number of adjacent swaps in a sequence -/
def countAdjacentSwaps : SwapSequence → ℕ
  | [] => 0
  | (Swap.adjacent _ :: rest) => 1 + countAdjacentSwaps rest
  | (_ :: rest) => countAdjacentSwaps rest

/-- The main theorem: proving that 50 adjacent swaps are required to reverse the strip -/
theorem min_swaps_to_reverse (s : Strip) : 
  (∃ swaps : SwapSequence, applySwaps s swaps = reverse s) → 
  (∃ minSwaps : SwapSequence, 
    applySwaps s minSwaps = reverse s ∧ 
    countAdjacentSwaps minSwaps = 50 ∧
    ∀ swaps : SwapSequence, applySwaps s swaps = reverse s → 
      countAdjacentSwaps minSwaps ≤ countAdjacentSwaps swaps) :=
by sorry

end NUMINAMATH_CALUDE_min_swaps_to_reverse_l2786_278697


namespace NUMINAMATH_CALUDE_exists_containing_quadrilateral_l2786_278627

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool

/-- A point in 2D space -/
def Point := Real × Real

/-- Check if a point is inside a convex polygon -/
def is_inside (p : Point) (poly : ConvexPolygon) : Bool := sorry

/-- Check if four points form a quadrilateral -/
def is_quadrilateral (a b c d : Point) : Bool := sorry

/-- Check if a quadrilateral contains a point -/
def quadrilateral_contains (a b c d : Point) (p : Point) : Bool := sorry

theorem exists_containing_quadrilateral 
  (poly : ConvexPolygon) (p1 p2 : Point) 
  (h1 : is_inside p1 poly) (h2 : is_inside p2 poly) :
  ∃ (a b c d : Point), 
    a ∈ poly.vertices ∧ 
    b ∈ poly.vertices ∧ 
    c ∈ poly.vertices ∧ 
    d ∈ poly.vertices ∧
    is_quadrilateral a b c d ∧
    quadrilateral_contains a b c d p1 ∧
    quadrilateral_contains a b c d p2 := by
  sorry

end NUMINAMATH_CALUDE_exists_containing_quadrilateral_l2786_278627


namespace NUMINAMATH_CALUDE_stadium_length_conversion_l2786_278679

/-- Conversion factor from feet to yards -/
def feet_per_yard : ℚ := 3

/-- Length of the stadium in feet -/
def stadium_length_feet : ℚ := 183

/-- Length of the stadium in yards -/
def stadium_length_yards : ℚ := stadium_length_feet / feet_per_yard

theorem stadium_length_conversion :
  stadium_length_yards = 61 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_conversion_l2786_278679


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l2786_278677

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.9 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l2786_278677


namespace NUMINAMATH_CALUDE_inequality_properties_l2786_278601

theorem inequality_properties (a b c d : ℝ) :
  (a > b ∧ c > d → a + c > b + d) ∧
  (a > b ∧ b > 0 ∧ c < 0 → c / a > c / b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_properties_l2786_278601


namespace NUMINAMATH_CALUDE_parabola_chord_midpoint_to_directrix_l2786_278650

/-- Given a parabola y² = 4x and a chord AB of length 7 intersecting the parabola at points A(x₁, y₁) and B(x₂, y₂),
    the distance from the midpoint M of the chord to the parabola's directrix is 7/2. -/
theorem parabola_chord_midpoint_to_directrix
  (x₁ y₁ x₂ y₂ : ℝ) 
  (on_parabola_A : y₁^2 = 4*x₁)
  (on_parabola_B : y₂^2 = 4*x₂)
  (chord_length : Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 7) :
  let midpoint_x := (x₁ + x₂) / 2
  (midpoint_x + 1) = 7/2 := by sorry

end NUMINAMATH_CALUDE_parabola_chord_midpoint_to_directrix_l2786_278650


namespace NUMINAMATH_CALUDE_portraits_not_taken_l2786_278676

theorem portraits_not_taken (total_students : ℕ) (before_lunch : ℕ) (after_lunch : ℕ) : 
  total_students = 24 → 
  before_lunch = total_students / 3 →
  after_lunch = 10 →
  total_students - (before_lunch + after_lunch) = 6 := by
sorry

end NUMINAMATH_CALUDE_portraits_not_taken_l2786_278676


namespace NUMINAMATH_CALUDE_f_inequality_range_l2786_278661

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem f_inequality_range (x : ℝ) : f (2*x) > f (x - 3) ↔ x < -3 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_range_l2786_278661


namespace NUMINAMATH_CALUDE_lucy_calculation_l2786_278643

theorem lucy_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 13) 
  (h2 : x - y - z = -1) : 
  x - y = 6 := by sorry

end NUMINAMATH_CALUDE_lucy_calculation_l2786_278643


namespace NUMINAMATH_CALUDE_even_sum_probability_l2786_278687

def wheel1_sections : ℕ := 6
def wheel1_even_sections : ℕ := 2
def wheel1_odd_sections : ℕ := 4

def wheel2_sections : ℕ := 4
def wheel2_even_sections : ℕ := 1
def wheel2_odd_sections : ℕ := 3

theorem even_sum_probability :
  let p_even_sum := (wheel1_even_sections / wheel1_sections) * (wheel2_even_sections / wheel2_sections) +
                    (wheel1_odd_sections / wheel1_sections) * (wheel2_odd_sections / wheel2_sections)
  p_even_sum = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_probability_l2786_278687


namespace NUMINAMATH_CALUDE_sin_negative_945_degrees_l2786_278694

theorem sin_negative_945_degrees : Real.sin ((-945 : ℝ) * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_945_degrees_l2786_278694


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_3_l2786_278669

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The first line equation: ax + 2y + 3a = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0

/-- The second line equation: 3x + (a - 1)y = a - 7 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := 3 * x + (a - 1) * y = a - 7

/-- The theorem stating that if the two lines are parallel, then a = 3 -/
theorem parallel_lines_imply_a_eq_3 :
  ∀ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_3_l2786_278669


namespace NUMINAMATH_CALUDE_male_population_in_village_l2786_278682

theorem male_population_in_village (total_population : ℕ) 
  (h1 : total_population = 800) 
  (num_groups : ℕ) 
  (h2 : num_groups = 4) 
  (h3 : total_population % num_groups = 0) 
  (h4 : ∃ (male_group : ℕ), male_group ≤ num_groups ∧ 
    male_group * (total_population / num_groups) = total_population / num_groups) :
  total_population / num_groups = 200 :=
by sorry

end NUMINAMATH_CALUDE_male_population_in_village_l2786_278682


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_three_l2786_278637

theorem sum_of_x_and_y_is_three (x y : ℝ) 
  (hx : (x - 1)^2003 + 2002*(x - 1) = -1)
  (hy : (y - 2)^2003 + 2002*(y - 2) = 1) : 
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_three_l2786_278637


namespace NUMINAMATH_CALUDE_candy_distribution_l2786_278691

theorem candy_distribution (total_candies : ℕ) 
  (lollipops_per_boy : ℕ) (candy_canes_per_girl : ℕ) : 
  total_candies = 90 →
  lollipops_per_boy = 3 →
  candy_canes_per_girl = 2 →
  (total_candies / 3 : ℕ) % lollipops_per_boy = 0 →
  ((2 * total_candies / 3) : ℕ) % candy_canes_per_girl = 0 →
  (total_candies / 3 / lollipops_per_boy : ℕ) + 
  ((2 * total_candies / 3) / candy_canes_per_girl : ℕ) = 40 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2786_278691


namespace NUMINAMATH_CALUDE_tanC_over_tanA_max_tanB_l2786_278610

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + 2*t.b^2 = t.c^2

-- Theorem 1: If the condition is satisfied, then tan C / tan A = -3
theorem tanC_over_tanA (t : Triangle) (h : satisfiesCondition t) :
  Real.tan t.C / Real.tan t.A = -3 :=
sorry

-- Theorem 2: If the condition is satisfied, then the maximum value of tan B is √3/3
theorem max_tanB (t : Triangle) (h : satisfiesCondition t) :
  ∃ (max : ℝ), max = Real.sqrt 3 / 3 ∧ Real.tan t.B ≤ max :=
sorry

end NUMINAMATH_CALUDE_tanC_over_tanA_max_tanB_l2786_278610


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2786_278635

def polynomial (x : ℤ) : ℤ := x^3 - 2*x^2 + 3*x - 17

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = {-17, -1, 1, 17} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2786_278635


namespace NUMINAMATH_CALUDE_max_b_value_l2786_278675

theorem max_b_value (b : ℕ+) (x : ℤ) (h : x^2 + b*x = -21) : b ≤ 22 := by
  sorry

end NUMINAMATH_CALUDE_max_b_value_l2786_278675


namespace NUMINAMATH_CALUDE_min_c_value_l2786_278619

/-- Given natural numbers a, b, c where a < b < c, and a system of equations with exactly one solution,
    prove that the minimum possible value of c is 1018. -/
theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
    (h3 : ∃! (x y : ℝ), 2 * x + y = 2035 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1018 ∧ ∃ (a' b' : ℕ), a' < b' ∧ b' < 1018 ∧
    ∃! (x y : ℝ), 2 * x + y = 2035 ∧ y = |x - a'| + |x - b'| + |x - 1018| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l2786_278619


namespace NUMINAMATH_CALUDE_greatest_difference_multiple_of_five_l2786_278690

theorem greatest_difference_multiple_of_five : ∀ a b : ℕ,
  (a < 10) →
  (b < 10) →
  (700 + 10 * a + b) % 5 = 0 →
  ((a + b) % 5 = 0) →
  ∃ c d : ℕ,
    (c < 10) ∧
    (d < 10) ∧
    (700 + 10 * c + d) % 5 = 0 ∧
    ((c + d) % 5 = 0) ∧
    (∀ e f : ℕ,
      (e < 10) →
      (f < 10) →
      (700 + 10 * e + f) % 5 = 0 →
      ((e + f) % 5 = 0) →
      (a + b) - (c + d) ≤ (e + f) - (c + d)) ∧
    (a + b) - (c + d) = 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_difference_multiple_of_five_l2786_278690


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2786_278648

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_uniqueness
  (f : ℝ → ℝ)
  (h_quad : is_quadratic f)
  (h_solution_set : ∀ x, f x < 0 ↔ 0 < x ∧ x < 5)
  (h_max_value : ∀ x ∈ Set.Icc (-1) 4, f x ≤ 12)
  (h_attains_max : ∃ x ∈ Set.Icc (-1) 4, f x = 12) :
  ∀ x, f x = 2 * x^2 - 10 * x :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2786_278648


namespace NUMINAMATH_CALUDE_marco_card_trade_ratio_l2786_278602

theorem marco_card_trade_ratio : 
  ∀ (total_cards duplicates_traded new_cards : ℕ),
    total_cards = 500 →
    duplicates_traded = new_cards →
    new_cards = 25 →
    (duplicates_traded : ℚ) / (total_cards / 4 : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_marco_card_trade_ratio_l2786_278602


namespace NUMINAMATH_CALUDE_train_speed_l2786_278632

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 2500) (h2 : time = 100) :
  length / time = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2786_278632


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2786_278689

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2786_278689


namespace NUMINAMATH_CALUDE_incorrect_operation_l2786_278658

theorem incorrect_operation : (4 + 5)^2 ≠ 4^2 + 5^2 := by sorry

end NUMINAMATH_CALUDE_incorrect_operation_l2786_278658


namespace NUMINAMATH_CALUDE_man_work_days_l2786_278640

theorem man_work_days (man_son_days : ℝ) (son_days : ℝ) (man_days : ℝ) : 
  man_son_days = 4 → son_days = 20 → man_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_man_work_days_l2786_278640


namespace NUMINAMATH_CALUDE_total_odd_green_and_red_marbles_l2786_278664

/-- Represents a person's marble collection --/
structure MarbleCollection where
  green : Nat
  red : Nat
  blue : Nat

/-- Counts odd numbers of green and red marbles --/
def countOddGreenAndRed (mc : MarbleCollection) : Nat :=
  (if mc.green % 2 = 1 then mc.green else 0) +
  (if mc.red % 2 = 1 then mc.red else 0)

theorem total_odd_green_and_red_marbles :
  let sara := MarbleCollection.mk 3 5 6
  let tom := MarbleCollection.mk 4 7 2
  let lisa := MarbleCollection.mk 5 3 7
  countOddGreenAndRed sara + countOddGreenAndRed tom + countOddGreenAndRed lisa = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_odd_green_and_red_marbles_l2786_278664


namespace NUMINAMATH_CALUDE_comprehensive_survey_suitable_for_grade_8_1_l2786_278671

/-- Represents a type of survey -/
inductive SurveyType
| Sampling
| Comprehensive

/-- Represents a population to be surveyed -/
structure Population where
  size : ℕ
  accessibility : Bool
  variability : Bool

/-- Determines if a survey type is suitable for a given population -/
def is_suitable (st : SurveyType) (p : Population) : Prop :=
  match st with
  | SurveyType.Sampling => p.size > 1000 ∨ p.accessibility = false ∨ p.variability = true
  | SurveyType.Comprehensive => p.size ≤ 1000 ∧ p.accessibility = true ∧ p.variability = false

/-- Represents the population of Grade 8 (1) students in a certain school -/
def grade_8_1_population : Population :=
  { size := 50,  -- Assuming a typical class size
    accessibility := true,
    variability := false }

/-- Theorem stating that a comprehensive survey is suitable for the Grade 8 (1) population -/
theorem comprehensive_survey_suitable_for_grade_8_1 :
  is_suitable SurveyType.Comprehensive grade_8_1_population :=
by
  sorry


end NUMINAMATH_CALUDE_comprehensive_survey_suitable_for_grade_8_1_l2786_278671


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l2786_278681

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 := by sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l2786_278681


namespace NUMINAMATH_CALUDE_triangle_pairs_theorem_l2786_278617

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle_pair (t1 t2 : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := t1
  let (d, e, f) := t2
  is_triangle a b c ∧ is_triangle d e f ∧ a + b + c + d + e + f = 16

theorem triangle_pairs_theorem :
  ∀ t1 t2 : ℕ × ℕ × ℕ,
  valid_triangle_pair t1 t2 →
  ((t1 = (4, 4, 3) ∧ t2 = (1, 2, 2)) ∨
   (t1 = (4, 4, 2) ∧ t2 = (2, 2, 2)) ∨
   (t1 = (4, 4, 1) ∧ t2 = (3, 2, 2)) ∨
   (t1 = (4, 4, 1) ∧ t2 = (3, 3, 1)) ∨
   (t2 = (4, 4, 3) ∧ t1 = (1, 2, 2)) ∨
   (t2 = (4, 4, 2) ∧ t1 = (2, 2, 2)) ∨
   (t2 = (4, 4, 1) ∧ t1 = (3, 2, 2)) ∨
   (t2 = (4, 4, 1) ∧ t1 = (3, 3, 1))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_pairs_theorem_l2786_278617


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l2786_278633

theorem arccos_equation_solution (x : ℝ) : 
  Real.arccos (3 * x) - Real.arccos (2 * x) = π / 6 →
  x = 1 / (2 * Real.sqrt (12 - 6 * Real.sqrt 3)) ∨
  x = -1 / (2 * Real.sqrt (12 - 6 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l2786_278633


namespace NUMINAMATH_CALUDE_egg_processing_plant_l2786_278614

theorem egg_processing_plant (E : ℕ) : 
  (96 : ℚ) / 100 * E + (4 : ℚ) / 100 * E = E → -- Original ratio
  ((96 : ℚ) / 100 * E + 12) / E = (99 : ℚ) / 100 → -- New ratio with 12 additional accepted eggs
  E = 400 := by
sorry

end NUMINAMATH_CALUDE_egg_processing_plant_l2786_278614


namespace NUMINAMATH_CALUDE_triangle_special_angle_l2786_278655

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a^2 + b^2 = c^2 - √2ab, then angle C = 3π/4 -/
theorem triangle_special_angle (a b c : ℝ) (h : a^2 + b^2 = c^2 - Real.sqrt 2 * a * b) :
  let angle_C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  angle_C = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l2786_278655


namespace NUMINAMATH_CALUDE_cafe_sign_white_area_l2786_278688

/-- Represents a rectangular sign with painted letters -/
structure Sign :=
  (width : ℕ)
  (height : ℕ)
  (c_area : ℕ)
  (a_area : ℕ)
  (f_area : ℕ)
  (e_area : ℕ)

/-- Calculates the white area of the sign -/
def white_area (s : Sign) : ℕ :=
  s.width * s.height - (s.c_area + s.a_area + s.f_area + s.e_area)

/-- Theorem stating that the white area of the given sign is 66 square units -/
theorem cafe_sign_white_area :
  ∃ (s : Sign),
    s.width = 6 ∧
    s.height = 18 ∧
    s.c_area = 11 ∧
    s.a_area = 10 ∧
    s.f_area = 12 ∧
    s.e_area = 9 ∧
    white_area s = 66 :=
sorry

end NUMINAMATH_CALUDE_cafe_sign_white_area_l2786_278688


namespace NUMINAMATH_CALUDE_g_of_fifty_l2786_278622

/-- A function g satisfying g(xy) = xg(y) for all real x and y, and g(1) = 30 -/
def g : ℝ → ℝ :=
  fun x => x * 30

/-- Theorem stating that g(50) = 1500 -/
theorem g_of_fifty : g 50 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_g_of_fifty_l2786_278622


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2786_278649

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^6 + b^6) / (a + b)^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2786_278649


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2786_278683

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2786_278683


namespace NUMINAMATH_CALUDE_marc_tv_watching_l2786_278604

/-- Given the number of episodes Marc watches per day and the total number of episodes,
    prove the relationship between x, y, and z. -/
theorem marc_tv_watching
  (friends_total : ℕ)
  (seinfeld_total : ℕ)
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h1 : friends_total = 50)
  (h2 : seinfeld_total = 75)
  (h3 : x * z = friends_total)
  (h4 : y * z = seinfeld_total) :
  y = (3 / 2) * x ∧ z = 50 / x :=
by sorry

end NUMINAMATH_CALUDE_marc_tv_watching_l2786_278604


namespace NUMINAMATH_CALUDE_sum_of_squared_digits_l2786_278621

/-- The number of digits in 222222222 -/
def n : ℕ := 9

/-- The number whose square we're considering -/
def num : ℕ := 222222222

/-- Function to calculate the sum of digits of a number -/
def sum_of_digits (m : ℕ) : ℕ := sorry

theorem sum_of_squared_digits : sum_of_digits (num ^ 2) = 162 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_digits_l2786_278621


namespace NUMINAMATH_CALUDE_derivative_inequality_implies_function_inequality_l2786_278620

theorem derivative_inequality_implies_function_inequality 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x > 0, deriv f x - f x / x > 0) → 3 * f 4 > 4 * f 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_inequality_implies_function_inequality_l2786_278620


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_i_l2786_278670

theorem complex_sum_of_powers_i (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_i_l2786_278670


namespace NUMINAMATH_CALUDE_point_on_line_l2786_278672

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let p1 : Point := ⟨3, 0⟩
  let p2 : Point := ⟨11, 4⟩
  let p3 : Point := ⟨19, 8⟩
  collinear p1 p2 p3 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l2786_278672


namespace NUMINAMATH_CALUDE_small_circle_radius_l2786_278695

theorem small_circle_radius (R : ℝ) (h : R = 5) :
  let d := Real.sqrt (2 * R^2)
  let r := (d - 2*R) / 2
  r = (Real.sqrt 200 - 10) / 2 := by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l2786_278695


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2786_278663

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}
def B : Set ℝ := {y | y > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2786_278663


namespace NUMINAMATH_CALUDE_odd_cube_plus_linear_plus_constant_l2786_278609

theorem odd_cube_plus_linear_plus_constant (o n m : ℤ) 
  (ho : ∃ k : ℤ, o = 2*k + 1) : 
  Odd (o^3 + n*o + m) ↔ Even m := by
  sorry

end NUMINAMATH_CALUDE_odd_cube_plus_linear_plus_constant_l2786_278609


namespace NUMINAMATH_CALUDE_min_dot_product_l2786_278624

open Real

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the fixed point M
def M : ℝ × ℝ := (1, 0)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the vector from M to a point P
def vector_MP (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 - M.1, P.2 - M.2)

theorem min_dot_product :
  ∃ (min : ℝ),
    (∀ (A B : ℝ × ℝ),
      ellipse A.1 A.2 →
      ellipse B.1 B.2 →
      dot_product (vector_MP A) (vector_MP B) = 0 →
      dot_product (vector_MP A) (A.1 - B.1, A.2 - B.2) ≥ min) ∧
    (∃ (A B : ℝ × ℝ),
      ellipse A.1 A.2 ∧
      ellipse B.1 B.2 ∧
      dot_product (vector_MP A) (vector_MP B) = 0 ∧
      dot_product (vector_MP A) (A.1 - B.1, A.2 - B.2) = min) ∧
    min = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_l2786_278624
