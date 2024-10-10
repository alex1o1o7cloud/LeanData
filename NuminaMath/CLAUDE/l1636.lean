import Mathlib

namespace dispersion_measures_l1636_163612

-- Define a sample as a list of real numbers
def Sample := List ℝ

-- Define the statistics
def standardDeviation (s : Sample) : ℝ := sorry
def range (s : Sample) : ℝ := sorry
def mean (s : Sample) : ℝ := sorry
def median (s : Sample) : ℝ := sorry

-- Define a predicate for whether a statistic measures dispersion
def measuresDispersion (f : Sample → ℝ) : Prop := sorry

-- Theorem stating which statistics measure dispersion
theorem dispersion_measures (s : Sample) :
  measuresDispersion (standardDeviation) ∧
  measuresDispersion (range) ∧
  ¬measuresDispersion (mean) ∧
  ¬measuresDispersion (median) :=
sorry

end dispersion_measures_l1636_163612


namespace sequence_sum_l1636_163608

theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  a 1 = 1 → 
  (∀ n : ℕ, 2 * S n = a (n + 1) - 1) → 
  a 3 + a 4 + a 5 = 117 := by
sorry

end sequence_sum_l1636_163608


namespace novelists_to_poets_ratio_l1636_163638

def total_people : ℕ := 24
def novelists : ℕ := 15

def poets : ℕ := total_people - novelists

theorem novelists_to_poets_ratio :
  (novelists : ℚ) / (poets : ℚ) = 5 / 3 := by sorry

end novelists_to_poets_ratio_l1636_163638


namespace rachel_homework_difference_l1636_163681

/-- Rachel's homework problem -/
theorem rachel_homework_difference (math_pages reading_pages : ℕ) : 
  math_pages = 8 →
  reading_pages = 14 →
  reading_pages > math_pages →
  reading_pages - math_pages = 6 := by
  sorry

end rachel_homework_difference_l1636_163681


namespace division_problem_l1636_163657

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 136 → 
  quotient = 9 → 
  remainder = 1 → 
  dividend = divisor * quotient + remainder → 
  divisor = 15 := by
sorry

end division_problem_l1636_163657


namespace rhombuses_in_five_by_five_grid_l1636_163689

/-- Represents a grid of equilateral triangles -/
structure TriangleGrid where
  rows : Nat
  cols : Nat

/-- Calculates the number of rhombuses in a triangle grid -/
def count_rhombuses (grid : TriangleGrid) : Nat :=
  sorry

/-- Theorem stating that a 5x5 grid of equilateral triangles contains 30 rhombuses -/
theorem rhombuses_in_five_by_five_grid :
  let grid : TriangleGrid := { rows := 5, cols := 5 }
  count_rhombuses grid = 30 := by
  sorry

end rhombuses_in_five_by_five_grid_l1636_163689


namespace world_cup_viewers_scientific_notation_l1636_163609

/-- Expresses a number in millions as scientific notation -/
def scientific_notation_millions (x : ℝ) : ℝ × ℤ :=
  (x, 7)

theorem world_cup_viewers_scientific_notation :
  scientific_notation_millions 70.62 = (7.062, 7) := by
  sorry

end world_cup_viewers_scientific_notation_l1636_163609


namespace lifting_capacity_proof_l1636_163675

/-- Calculates the total weight a person can lift with both hands after training and specializing,
    given their initial lifting capacity per hand. -/
def totalLiftingCapacity (initialCapacity : ℝ) : ℝ :=
  let doubledCapacity := initialCapacity * 2
  let specializedCapacity := doubledCapacity * 1.1
  specializedCapacity * 2

/-- Proves that given an initial lifting capacity of 80 kg per hand,
    the total weight that can be lifted with both hands after training and specializing is 352 kg. -/
theorem lifting_capacity_proof :
  totalLiftingCapacity 80 = 352 := by
  sorry

#eval totalLiftingCapacity 80

end lifting_capacity_proof_l1636_163675


namespace fruit_seller_apples_l1636_163659

/-- Proves that if a fruit seller sells 50% of his apples and is left with 5000 apples, 
    then he originally had 10000 apples. -/
theorem fruit_seller_apples (original : ℕ) (sold_percentage : ℚ) (remaining : ℕ) 
    (h1 : sold_percentage = 1/2)
    (h2 : remaining = 5000)
    (h3 : (1 - sold_percentage) * original = remaining) : 
  original = 10000 := by
  sorry

end fruit_seller_apples_l1636_163659


namespace women_reseating_l1636_163635

/-- The number of ways to reseat n women in a circle, where each woman can sit in her original seat,
    an adjacent seat, or two seats away. -/
def C : ℕ → ℕ
  | 0 => 0  -- Added for completeness
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | n + 4 => 2 * C (n + 3) + 2 * C (n + 2) + C (n + 1)

/-- The number of ways to reseat 9 women in a circle, where each woman can sit in her original seat,
    an adjacent seat, or two seats away, is equal to 3086. -/
theorem women_reseating : C 9 = 3086 := by
  sorry

end women_reseating_l1636_163635


namespace trapezoid_bc_length_l1636_163643

/-- Trapezoid properties -/
structure Trapezoid :=
  (area : ℝ)
  (altitude : ℝ)
  (ab : ℝ)
  (cd : ℝ)

/-- Theorem: For a trapezoid with given properties, BC = 10 cm -/
theorem trapezoid_bc_length (t : Trapezoid) 
  (h1 : t.area = 200)
  (h2 : t.altitude = 10)
  (h3 : t.ab = 12)
  (h4 : t.cd = 22) :
  ∃ bc : ℝ, bc = 10 := by
  sorry


end trapezoid_bc_length_l1636_163643


namespace parents_age_difference_l1636_163656

/-- The difference between Sobha's parents' ages -/
def age_difference (s : ℕ) : ℕ :=
  let f := s + 38  -- father's age
  let b := s - 4   -- brother's age
  let m := b + 36  -- mother's age
  f - m

/-- Theorem stating that the age difference between Sobha's parents is 6 years -/
theorem parents_age_difference (s : ℕ) (h : s ≥ 4) : age_difference s = 6 := by
  sorry

end parents_age_difference_l1636_163656


namespace negation_of_statement_l1636_163642

theorem negation_of_statement :
  (¬ (∀ x : ℝ, (x = 0 ∨ x = 1) → x^2 - x = 0)) ↔
  (∀ x : ℝ, (x ≠ 0 ∧ x ≠ 1) → x^2 - x ≠ 0) :=
by sorry

end negation_of_statement_l1636_163642


namespace f_is_decreasing_l1636_163645

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0
axiom additivity : ∀ x y, f (x + y) = f x + f y
axiom negative_for_positive : ∀ x, x > 0 → f x < 0

-- State the theorem
theorem f_is_decreasing : 
  (∀ x y, x < y → f x > f y) :=
sorry

end f_is_decreasing_l1636_163645


namespace cubic_monotonic_and_odd_l1636_163634

def f (x : ℝ) : ℝ := x^3

theorem cubic_monotonic_and_odd :
  (∀ x y, x < y → f x < f y) ∧ 
  (∀ x, f (-x) = -f x) :=
sorry

end cubic_monotonic_and_odd_l1636_163634


namespace cylinder_base_radius_l1636_163633

/-- Given a cylinder with generatrix length 3 cm and lateral area 12π cm², 
    prove that the radius of the base is 2 cm. -/
theorem cylinder_base_radius 
  (generatrix : ℝ) 
  (lateral_area : ℝ) 
  (h1 : generatrix = 3) 
  (h2 : lateral_area = 12 * Real.pi) : 
  lateral_area / (2 * Real.pi * generatrix) = 2 := by
  sorry

end cylinder_base_radius_l1636_163633


namespace point_on_extension_line_l1636_163600

theorem point_on_extension_line (θ : ℝ) (M : ℝ × ℝ) :
  (∃ k : ℝ, k > 1 ∧ M = (k * Real.cos θ, k * Real.sin θ)) →
  (M.1^2 + M.2^2 = 4) →
  M = (-2 * Real.cos θ, -2 * Real.sin θ) := by
  sorry

end point_on_extension_line_l1636_163600


namespace baking_time_per_batch_l1636_163624

/-- Proves that the time to bake one batch of cupcakes is 20 minutes -/
theorem baking_time_per_batch (
  num_batches : ℕ)
  (icing_time_per_batch : ℕ)
  (total_time : ℕ)
  (h1 : num_batches = 4)
  (h2 : icing_time_per_batch = 30)
  (h3 : total_time = 200)
  : ∃ (baking_time_per_batch : ℕ),
    baking_time_per_batch * num_batches + icing_time_per_batch * num_batches = total_time ∧
    baking_time_per_batch = 20 :=
by
  sorry

end baking_time_per_batch_l1636_163624


namespace sum_of_x_and_y_l1636_163629

theorem sum_of_x_and_y (x y : ℝ) (h : (2 : ℝ)^x = (18 : ℝ)^y ∧ (2 : ℝ)^x = (6 : ℝ)^(x*y)) :
  x + y = 0 ∨ x + y = 2 := by
  sorry

end sum_of_x_and_y_l1636_163629


namespace distribute_10_8_l1636_163664

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins,
    with each bin receiving at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem stating that distributing 10 objects into 8 bins, with each bin receiving at least one,
    results in 36 possible distributions. -/
theorem distribute_10_8 : distribute 10 8 = 36 := by
  sorry

end distribute_10_8_l1636_163664


namespace negation_of_existence_negation_of_quadratic_inequality_l1636_163630

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ ∀ x, ¬ P x :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 4 < 0) ↔ (∀ x : ℝ, x^2 - x + 4 ≥ 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_inequality_l1636_163630


namespace word_permutations_l1636_163685

-- Define the number of distinct letters in the word
def num_distinct_letters : ℕ := 6

-- Define the function to calculate factorial
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem word_permutations :
  factorial num_distinct_letters = 720 := by
  sorry

end word_permutations_l1636_163685


namespace isosceles_triangle_perimeter_l1636_163691

/-- The perimeter of an isosceles triangle given specific conditions -/
theorem isosceles_triangle_perimeter : 
  ∀ (equilateral_perimeter isosceles_base : ℝ),
  equilateral_perimeter = 60 →
  isosceles_base = 15 →
  ∃ (isosceles_perimeter : ℝ),
  isosceles_perimeter = equilateral_perimeter / 3 + equilateral_perimeter / 3 + isosceles_base ∧
  isosceles_perimeter = 55 := by
sorry

end isosceles_triangle_perimeter_l1636_163691


namespace reflection_of_circle_center_l1636_163618

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (-3, 7)
  let reflected_center : ℝ × ℝ := (-7, 3)
  reflect_about_y_eq_neg_x original_center = reflected_center := by
sorry

end reflection_of_circle_center_l1636_163618


namespace cube_root_of_x_plus_3y_is_3_l1636_163663

theorem cube_root_of_x_plus_3y_is_3 (x y : ℝ) 
  (h : y = Real.sqrt (x - 3) + Real.sqrt (3 - x) + 8) : 
  (x + 3 * y) ^ (1/3 : ℝ) = 3 := by sorry

end cube_root_of_x_plus_3y_is_3_l1636_163663


namespace set_intersection_theorem_l1636_163603

def S : Set ℝ := {x | 2 * x + 1 > 0}
def T : Set ℝ := {x | 3 * x - 5 < 0}

theorem set_intersection_theorem :
  S ∩ T = {x : ℝ | -1/2 < x ∧ x < 5/3} :=
by sorry

end set_intersection_theorem_l1636_163603


namespace expression_value_l1636_163677

theorem expression_value : (49 + 5)^2 - (5^2 + 49^2) = 490 := by
  sorry

end expression_value_l1636_163677


namespace sum_of_roots_quadratic_l1636_163668

theorem sum_of_roots_quadratic : ∃ (x₁ x₂ : ℤ),
  (x₁^2 = x₁ + 272) ∧ 
  (x₂^2 = x₂ + 272) ∧ 
  (∀ x : ℤ, x^2 = x + 272 → x = x₁ ∨ x = x₂) ∧
  (x₁ + x₂ = 1) := by
  sorry

end sum_of_roots_quadratic_l1636_163668


namespace smallest_c_value_l1636_163684

theorem smallest_c_value (c : ℝ) (h : (3 * c + 4) * (c - 2) = 7 * c + 6) :
  c ≥ (9 - Real.sqrt 249) / 6 ∧ ∃ (c₀ : ℝ), (3 * c₀ + 4) * (c₀ - 2) = 7 * c₀ + 6 ∧ c₀ = (9 - Real.sqrt 249) / 6 := by
  sorry

end smallest_c_value_l1636_163684


namespace pet_store_gerbils_l1636_163651

theorem pet_store_gerbils : 
  ∀ (initial_gerbils sold_gerbils remaining_gerbils : ℕ),
  sold_gerbils = 69 →
  remaining_gerbils = 16 →
  initial_gerbils = sold_gerbils + remaining_gerbils →
  initial_gerbils = 85 := by
sorry

end pet_store_gerbils_l1636_163651


namespace larger_integer_problem_l1636_163607

theorem larger_integer_problem (x y : ℤ) : 
  y - x = 8 → x * y = 272 → max x y = 17 := by sorry

end larger_integer_problem_l1636_163607


namespace sum_of_fractions_l1636_163671

theorem sum_of_fractions : (3 : ℚ) / 4 + (6 : ℚ) / 9 = (17 : ℚ) / 12 := by
  sorry

end sum_of_fractions_l1636_163671


namespace solve_equation_l1636_163687

theorem solve_equation (x : ℝ) (h : (40 / x) - 1 = 19) : x = 2 := by
  sorry

end solve_equation_l1636_163687


namespace not_property_P_if_cong_4_mod_9_l1636_163623

/-- Property P: An integer n has property P if there exist integers x, y, z 
    such that n = x³ + y³ + z³ - 3xyz -/
def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3*x*y*z

/-- Theorem: If an integer n is congruent to 4 modulo 9, then it does not have property P -/
theorem not_property_P_if_cong_4_mod_9 (n : ℤ) (h : n % 9 = 4) : 
  ¬(has_property_P n) := by
  sorry

#check not_property_P_if_cong_4_mod_9

end not_property_P_if_cong_4_mod_9_l1636_163623


namespace extended_line_segment_vector_representation_l1636_163601

/-- Given a line segment AB extended to P such that AP:PB = 10:3,
    prove that the position vector of P can be expressed as P = -3/7*A + 10/7*B,
    where A and B are the position vectors of points A and B respectively. -/
theorem extended_line_segment_vector_representation 
  (A B P : ℝ × ℝ) -- A, B, and P are points in 2D space
  (h : (dist A P) / (dist P B) = 10 / 3) -- AP:PB = 10:3
  : ∃ (t u : ℝ), P = t • A + u • B ∧ t = -3/7 ∧ u = 10/7 := by
  sorry

end extended_line_segment_vector_representation_l1636_163601


namespace sum_of_repeating_decimals_l1636_163669

def repeating_decimal_6 : ℚ := 2/3
def repeating_decimal_3 : ℚ := 1/3

theorem sum_of_repeating_decimals : 
  repeating_decimal_6 + repeating_decimal_3 = 1 := by
  sorry

end sum_of_repeating_decimals_l1636_163669


namespace binomial_12_3_l1636_163619

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_l1636_163619


namespace bus_problem_l1636_163683

/-- Calculates the number of students remaining on a bus after a given number of stops,
    where one-third of the students get off at each stop. -/
def studentsRemaining (initialStudents : ℚ) (stops : ℕ) : ℚ :=
  initialStudents * (2/3)^stops

/-- Proves that if a bus starts with 60 students and loses one-third of its passengers
    at each of four stops, the number of students remaining after the fourth stop is 320/27. -/
theorem bus_problem : studentsRemaining 60 4 = 320/27 := by
  sorry

#eval studentsRemaining 60 4

end bus_problem_l1636_163683


namespace triangle_area_l1636_163696

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A + B + C = π →
  a = 2 * Real.sqrt 3 →
  b + c = 4 →
  Real.cos B * Real.cos C - Real.sin B * Real.sin C = 1/2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 := by
  sorry

end triangle_area_l1636_163696


namespace shaded_area_circles_l1636_163617

theorem shaded_area_circles (R : ℝ) (r : ℝ) : 
  R^2 * π = 100 * π →
  r = R / 2 →
  (2 / 3) * (π * R^2) + (1 / 3) * (π * r^2) = 75 * π := by
  sorry

end shaded_area_circles_l1636_163617


namespace fifth_match_goals_l1636_163699

/-- Represents the goal-scoring statistics of a football player over 5 matches -/
structure FootballStats where
  total_goals : ℕ
  avg_increase : ℚ

/-- Theorem stating that under given conditions, the player scored 3 goals in the fifth match -/
theorem fifth_match_goals (stats : FootballStats) 
  (h1 : stats.total_goals = 11)
  (h2 : stats.avg_increase = 1/5) : 
  (stats.total_goals : ℚ) - 4 * ((stats.total_goals : ℚ) / 5 - stats.avg_increase) = 3 := by
  sorry

#check fifth_match_goals

end fifth_match_goals_l1636_163699


namespace cube_volume_increase_cube_volume_not_8_times_l1636_163637

theorem cube_volume_increase (edge : ℝ) (edge_positive : 0 < edge) : 
  (2 * edge)^3 = 27 * edge^3 := by sorry

theorem cube_volume_not_8_times (edge : ℝ) (edge_positive : 0 < edge) : 
  (2 * edge)^3 ≠ 8 * edge^3 := by sorry

end cube_volume_increase_cube_volume_not_8_times_l1636_163637


namespace rectangle_perimeter_from_triangle_l1636_163627

/-- Given a triangle with sides 5, 12, and 13 units, and a rectangle with width 5 units
    and area equal to the triangle's area, the perimeter of the rectangle is 22 units. -/
theorem rectangle_perimeter_from_triangle : 
  ∀ (triangle_side1 triangle_side2 triangle_side3 rectangle_width : ℝ),
  triangle_side1 = 5 →
  triangle_side2 = 12 →
  triangle_side3 = 13 →
  rectangle_width = 5 →
  (1/2) * triangle_side1 * triangle_side2 = rectangle_width * (1/2 * triangle_side1 * triangle_side2 / rectangle_width) →
  2 * (rectangle_width + (1/2 * triangle_side1 * triangle_side2 / rectangle_width)) = 22 :=
by
  sorry


end rectangle_perimeter_from_triangle_l1636_163627


namespace sqrt_2_irrational_sqrt_2_only_irrational_in_set_l1636_163614

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_2_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

-- Define the set of numbers from the original problem
def problem_numbers : Set ℝ := {0, -1, Real.sqrt 2, 3.14}

-- State that √2 is the only irrational number in the set
theorem sqrt_2_only_irrational_in_set : 
  ∀ x ∈ problem_numbers, IsIrrational x ↔ x = Real.sqrt 2 := by
  sorry

end sqrt_2_irrational_sqrt_2_only_irrational_in_set_l1636_163614


namespace crop_allocation_theorem_l1636_163648

/-- Represents the yield function for crop A -/
def yield_A (x : ℝ) : ℝ := (2 + x) * (1.2 - 0.1 * x)

/-- Represents the maximum yield for crop A -/
def max_yield_A : ℝ := 4.9

/-- Represents the yield for crop B -/
def yield_B : ℝ := 10 * 0.5

/-- The total land area in square meters -/
def total_area : ℝ := 100

/-- The minimum required total yield in kg -/
def min_total_yield : ℝ := 496

theorem crop_allocation_theorem :
  ∃ (a : ℝ), a ≤ 40 ∧ a ≥ 0 ∧
  ∀ (x : ℝ), x ≤ 40 ∧ x ≥ 0 →
    max_yield_A * a + yield_B * (total_area - a) ≥ min_total_yield ∧
    (x > a → max_yield_A * x + yield_B * (total_area - x) < min_total_yield) :=
by sorry

end crop_allocation_theorem_l1636_163648


namespace equation_solution_l1636_163667

theorem equation_solution : ∃! x : ℝ, (1 / (x - 1) = 3 / (2 * x - 3)) := by
  sorry

end equation_solution_l1636_163667


namespace incorrect_locus_proof_l1636_163673

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 25

/-- The set of points satisfying the circle equation -/
def circle_set : Set (ℝ × ℝ) := {p | circle_equation p.1 p.2}

/-- The statement to be proven false -/
def incorrect_statement : Prop :=
  (∀ p : ℝ × ℝ, ¬(circle_equation p.1 p.2) → p ∉ circle_set) →
  (circle_set = {p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - 0)^2 = 5^2})

theorem incorrect_locus_proof : ¬incorrect_statement := by
  sorry

end incorrect_locus_proof_l1636_163673


namespace intersection_point_l1636_163688

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 3 = 0

-- Define a point on the y-axis
def on_y_axis (x y : ℝ) : Prop := x = 0

-- Theorem statement
theorem intersection_point :
  ∃ (y : ℝ), line_equation 0 y ∧ on_y_axis 0 y ∧ y = 3 :=
sorry

end intersection_point_l1636_163688


namespace functional_equation_solutions_l1636_163674

-- Define the function types
variable (f g : ℝ → ℝ)

-- Define the functional equation condition
def functional_equation : Prop :=
  ∀ x y : ℝ, f (x - f y) = x * f y - y * f x + g x

-- State the theorem
theorem functional_equation_solutions :
  functional_equation f g →
  ((∀ x, f x = 0 ∧ g x = 0) ∨ (∀ x, f x = x ∧ g x = 0)) :=
sorry

end functional_equation_solutions_l1636_163674


namespace inequality_solution_l1636_163666

theorem inequality_solution : 
  ∀ x y : ℤ, 
    (x - 3*y + 2 ≥ 1) → 
    (-x + 2*y + 1 ≥ 1) → 
    (x^2 / Real.sqrt (x - 3*y + 2 : ℝ) + y^2 / Real.sqrt (-x + 2*y + 1 : ℝ) ≥ y^2 + 2*x^2 - 2*x - 1) →
    ((x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 1)) := by
  sorry

#check inequality_solution

end inequality_solution_l1636_163666


namespace smallest_A_for_divisibility_l1636_163610

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def six_digit_number (A : ℕ) : ℕ := 4 * 100000 + A * 10000 + 88851

theorem smallest_A_for_divisibility :
  ∀ A : ℕ, A ≥ 1 →
    (is_divisible_by_3 (six_digit_number A) → A ≥ 1) ∧
    is_divisible_by_3 (six_digit_number 1) :=
by sorry

end smallest_A_for_divisibility_l1636_163610


namespace green_apples_count_l1636_163644

theorem green_apples_count (total : ℕ) (red : ℕ) (yellow : ℕ) (h1 : total = 19) (h2 : red = 3) (h3 : yellow = 14) :
  total - (red + yellow) = 2 := by
  sorry

end green_apples_count_l1636_163644


namespace absolute_value_equation_solutions_l1636_163661

theorem absolute_value_equation_solutions :
  ∀ x : ℝ, (3 * x + 9 = |(-20 + 4 * x)|) ↔ (x = 29 ∨ x = 11/7) :=
by sorry

end absolute_value_equation_solutions_l1636_163661


namespace ella_age_l1636_163695

/-- Given the ages of Sam, Tim, and Ella, prove that Ella is 15 years old. -/
theorem ella_age (s t e : ℕ) : 
  (s + t + e) / 3 = 12 →  -- The average of their ages is 12
  e - 5 = s →             -- Five years ago, Ella was the same age as Sam is now
  t + 4 = (3 * (s + 4)) / 4 →  -- In 4 years, Tim's age will be 3/4 of Sam's age at that time
  e = 15 := by
sorry


end ella_age_l1636_163695


namespace function_bounds_l1636_163613

/-- A strictly increasing function from ℕ to ℕ -/
def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ m n, m < n → f m < f n

theorem function_bounds
  (k : ℕ)
  (f : ℕ → ℕ)
  (h_strict : StrictlyIncreasing f)
  (h_comp : ∀ n, f (f n) = k * n) :
  ∀ n, (2 * k : ℚ) / (k + 1) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1 : ℚ) / 2 * n :=
by sorry

end function_bounds_l1636_163613


namespace negation_of_universal_nonnegative_naturals_l1636_163654

theorem negation_of_universal_nonnegative_naturals :
  (¬ ∀ (x : ℕ), x ≥ 0) ↔ (∃ (x : ℕ), x < 0) := by
  sorry

end negation_of_universal_nonnegative_naturals_l1636_163654


namespace reflection_line_sum_l1636_163655

/-- Given a line y = mx + b, if the reflection of point (1,2) across this line is (7,6), then m + b = 8.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 7 ∧ y = 6 ∧ 
    (x - 1)^2 + (y - 2)^2 = (7 - x)^2 + (6 - y)^2 ∧
    (y - 2) = m * (x - 1) ∧
    y = m * x + b) →
  m + b = 8.5 := by sorry

end reflection_line_sum_l1636_163655


namespace max_abs_Z_on_circle_l1636_163647

open Complex

theorem max_abs_Z_on_circle (Z : ℂ) (h : abs (Z - (3 + 4*I)) = 1) :
  ∃ (M : ℝ), M = 6 ∧ ∀ (W : ℂ), abs (W - (3 + 4*I)) = 1 → abs W ≤ M :=
sorry

end max_abs_Z_on_circle_l1636_163647


namespace sqrt_square_eq_abs_l1636_163694

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

end sqrt_square_eq_abs_l1636_163694


namespace sodas_bought_example_l1636_163686

/-- Given a total cost, sandwich price, number of sandwiches, and soda price,
    calculate the number of sodas bought. -/
def sodas_bought (total_cost sandwich_price num_sandwiches soda_price : ℚ) : ℚ :=
  (total_cost - num_sandwiches * sandwich_price) / soda_price

theorem sodas_bought_example : 
  sodas_bought 8.38 2.45 2 0.87 = 4 := by sorry

end sodas_bought_example_l1636_163686


namespace min_sum_absolute_values_l1636_163640

theorem min_sum_absolute_values (x : ℝ) :
  ∃ (m : ℝ), m = -2 ∧ (∀ x, |x + 3| + |x + 5| + |x + 6| ≥ m) ∧ (∃ x, |x + 3| + |x + 5| + |x + 6| = m) := by
  sorry

end min_sum_absolute_values_l1636_163640


namespace factorization_equality_l1636_163611

theorem factorization_equality (x y : ℝ) : x^2 - 1 + 2*x*y + y^2 = (x+y+1)*(x+y-1) := by
  sorry

end factorization_equality_l1636_163611


namespace first_candidate_percentage_l1636_163662

theorem first_candidate_percentage (total_votes : ℕ) (invalid_percent : ℚ) (second_candidate_votes : ℕ) :
  total_votes = 5500 →
  invalid_percent = 20/100 →
  second_candidate_votes = 1980 →
  let valid_votes := total_votes * (1 - invalid_percent)
  let first_candidate_votes := valid_votes - second_candidate_votes
  (first_candidate_votes : ℚ) / valid_votes * 100 = 55 := by
  sorry

end first_candidate_percentage_l1636_163662


namespace quadratic_function_max_min_l1636_163628

theorem quadratic_function_max_min (a b : ℝ) (h1 : a ≠ 0) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 2 * a * x + b
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, f y ≤ f x) ∧
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, f y ≥ f x) ∧
  (∃ x ∈ Set.Icc 1 2, f x = 0) ∧
  (∃ x ∈ Set.Icc 1 2, f x = -1) →
  ((a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = -1)) :=
by sorry

end quadratic_function_max_min_l1636_163628


namespace opposite_vectors_properties_l1636_163606

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def are_opposite (a b : V) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ b = -a

theorem opposite_vectors_properties {a b : V} (h : are_opposite a b) :
  (∃ (k : ℝ), b = k • a) ∧  -- a is parallel to b
  a ≠ b ∧                   -- a ≠ b
  ‖a‖ = ‖b‖ ∧              -- |a| = |b|
  b = -a :=                 -- b = -a
by sorry

end opposite_vectors_properties_l1636_163606


namespace multi_digit_perfect_square_distinct_digits_l1636_163658

theorem multi_digit_perfect_square_distinct_digits :
  ∀ n : ℕ, n > 9 → (∃ m : ℕ, n = m^2) →
    ∃ d₁ d₂ : ℕ, d₁ ≠ d₂ ∧ d₁ < 10 ∧ d₂ < 10 ∧
    ∃ k : ℕ, n = d₁ + 10 * k ∧ ∃ l : ℕ, k = d₂ + 10 * l :=
by sorry

end multi_digit_perfect_square_distinct_digits_l1636_163658


namespace enrique_commission_l1636_163652

/-- Calculates the commission for a given item --/
def calculate_commission (price : ℝ) (quantity : ℕ) (commission_rate : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate) * quantity * commission_rate

/-- Calculates the total commission for all items sold --/
def total_commission (suit_price suit_quantity : ℕ) (shirt_price shirt_quantity : ℕ) 
                     (loafer_price loafer_quantity : ℕ) (tie_price tie_quantity : ℕ) 
                     (sock_price sock_quantity : ℕ) : ℝ :=
  let suit_commission := calculate_commission (suit_price : ℝ) suit_quantity 0.15 0.1 0
  let shirt_commission := calculate_commission (shirt_price : ℝ) shirt_quantity 0.15 0 0.05
  let loafer_commission := calculate_commission (loafer_price : ℝ) loafer_quantity 0.1 0 0.05
  let tie_commission := calculate_commission (tie_price : ℝ) tie_quantity 0.1 0 0.05
  let sock_commission := calculate_commission (sock_price : ℝ) sock_quantity 0.1 0 0.05
  suit_commission + shirt_commission + loafer_commission + tie_commission + sock_commission

theorem enrique_commission : 
  total_commission 700 2 50 6 150 2 30 4 10 5 = 285.60 := by
  sorry

end enrique_commission_l1636_163652


namespace fib_equation_solutions_l1636_163626

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The set of solutions to the Fibonacci equation -/
def fibSolutions : Set (ℕ × ℕ) :=
  {p : ℕ × ℕ | 5 * fib p.1 - 3 * fib p.2 = 1}

theorem fib_equation_solutions :
  fibSolutions = {(2, 3), (5, 8), (8, 13)} := by sorry

end fib_equation_solutions_l1636_163626


namespace no_arithmetic_progression_l1636_163649

theorem no_arithmetic_progression (m : ℕ+) :
  ∃ σ : Fin (2^m.val) ↪ Fin (2^m.val),
    ∀ (i j k : Fin (2^m.val)), i < j → j < k →
      σ j - σ i ≠ σ k - σ j :=
by sorry

end no_arithmetic_progression_l1636_163649


namespace sqrt_equation_solution_l1636_163620

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end sqrt_equation_solution_l1636_163620


namespace tangent_perpendicular_implies_negative_a_l1636_163622

theorem tangent_perpendicular_implies_negative_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * a * x^2 + 1 / x = 0)) → a < 0 := by
  sorry

end tangent_perpendicular_implies_negative_a_l1636_163622


namespace constant_sum_area_one_iff_identical_l1636_163690

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 1000
  col : Fin 1000
  value : ℝ

/-- Represents the entire 1000 × 1000 grid -/
def Grid := Cell → ℝ

/-- A rectangle within the grid -/
structure Rectangle where
  top_left : Cell
  bottom_right : Cell

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  (r.bottom_right.row - r.top_left.row + 1) * (r.bottom_right.col - r.top_left.col + 1)

/-- The sum of values in a rectangle -/
def sum_rectangle (g : Grid) (r : Rectangle) : ℝ := sorry

/-- Predicate: all rectangles of area s have the same sum -/
def constant_sum_for_area (g : Grid) (s : ℕ) : Prop :=
  ∀ r₁ r₂ : Rectangle, area r₁ = s → area r₂ = s → sum_rectangle g r₁ = sum_rectangle g r₂

/-- Predicate: all cells in the grid have the same value -/
def all_cells_identical (g : Grid) : Prop :=
  ∀ c₁ c₂ : Cell, g c₁ = g c₂

/-- Main theorem: constant sum for area 1 implies all cells are identical -/
theorem constant_sum_area_one_iff_identical :
  ∀ g : Grid, constant_sum_for_area g 1 ↔ all_cells_identical g :=
sorry

end constant_sum_area_one_iff_identical_l1636_163690


namespace unique_solution_system_l1636_163636

theorem unique_solution_system (x y : ℝ) :
  (x + y = (5 - x) + (5 - y)) ∧ (x - y = (x - 1) + (y - 1)) →
  x = 4 ∧ y = 1 := by
  sorry

end unique_solution_system_l1636_163636


namespace intersection_distance_and_difference_l1636_163615

theorem intersection_distance_and_difference : ∃ (x₁ x₂ : ℝ),
  (4 * x₁^2 + x₁ - 1 = 5) ∧
  (4 * x₂^2 + x₂ - 1 = 5) ∧
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = Real.sqrt 97 / 4) ∧
  (97 - 4 = 93) := by
  sorry

end intersection_distance_and_difference_l1636_163615


namespace abs_le_2_set_equality_l1636_163616

def set_of_integers_abs_le_2 : Set ℤ := {x | |x| ≤ 2}

theorem abs_le_2_set_equality : set_of_integers_abs_le_2 = {-2, -1, 0, 1, 2} := by
  sorry

end abs_le_2_set_equality_l1636_163616


namespace polar_to_cartesian_circle_l1636_163679

/-- Given a curve C in polar coordinates with equation ρ = 6 * cos(θ),
    prove that its equivalent Cartesian equation is x² + y² = 6x -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 6 * Real.cos θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  x^2 + y^2 = 6*x := by
  sorry

end polar_to_cartesian_circle_l1636_163679


namespace composite_rectangle_area_l1636_163670

theorem composite_rectangle_area : 
  let rect1_area := 6 * 9
  let rect2_area := 4 * 6
  let rect3_area := 5 * 2
  rect1_area + rect2_area + rect3_area = 88 := by
  sorry

end composite_rectangle_area_l1636_163670


namespace largest_number_in_set_l1636_163631

theorem largest_number_in_set (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- a, b, c are in ascending order
  (a + b + c) / 3 = 6 ∧  -- mean is 6
  b = 6 ∧  -- median is 6
  a = 2  -- smallest number is 2
  → c = 10 := by sorry

end largest_number_in_set_l1636_163631


namespace num_squares_6x6_l1636_163602

/-- A square on a grid --/
structure GridSquare where
  size : ℕ
  rotation : Bool  -- False for regular, True for diagonal

/-- The set of all possible non-congruent squares on a 6x6 grid --/
def squares_6x6 : Finset GridSquare := sorry

/-- The number of non-congruent squares on a 6x6 grid --/
theorem num_squares_6x6 : Finset.card squares_6x6 = 75 := by sorry

end num_squares_6x6_l1636_163602


namespace ellipse_focus_distance_l1636_163625

/-- An ellipse with equation x²/25 + y²/9 = 1 -/
structure Ellipse :=
  (x y : ℝ)
  (eq : x^2/25 + y^2/9 = 1)

/-- The distance from a point to a focus of the ellipse -/
def distance_to_focus (P : Ellipse) (focus : ℝ × ℝ) : ℝ :=
  sorry

theorem ellipse_focus_distance (P : Ellipse) (F1 F2 : ℝ × ℝ) :
  distance_to_focus P F1 = 3 →
  distance_to_focus P F2 = 7 :=
sorry

end ellipse_focus_distance_l1636_163625


namespace hyperbola_tangent_property_l1636_163653

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- Define the line AB
def line_AB (x y : ℝ) : Prop := 2*x - 3*y = 2

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the hyperbola
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the angle between two vectors
def angle (v₁ v₂ : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_tangent_property :
  ∀ (A B : ℝ × ℝ),
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  A.1 < B.1 →
  line_AB A.1 A.2 →
  line_AB B.1 B.2 →
  line_AB P.1 P.2 →
  (∀ (x y : ℝ), hyperbola x y → line_AB x y → (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) →
  (line_AB A.1 A.2 ∧ line_AB B.1 B.2) ∧
  angle (A.1 - F₁.1, A.2 - F₁.2) (P.1 - F₁.1, P.2 - F₁.2) =
  angle (B.1 - F₂.1, B.2 - F₂.2) (P.1 - F₂.1, P.2 - F₂.2) :=
by sorry

end hyperbola_tangent_property_l1636_163653


namespace f_satisfies_condition_l1636_163697

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(1 / (log x))

-- State the theorem
theorem f_satisfies_condition (a : ℝ) (h_a : a > 1) :
  ∀ (x y u v : ℝ), x > 1 → y > 1 → u > 0 → v > 0 →
    f a (x^u * y^v) ≤ (f a x)^(1/(1*u)) * (f a y)^(1/10) :=
by sorry

end f_satisfies_condition_l1636_163697


namespace proof_arrangements_l1636_163621

/-- The number of letters in the word PROOF -/
def word_length : ℕ := 5

/-- The number of times the letter 'O' appears in PROOF -/
def o_count : ℕ := 2

/-- Formula for calculating the number of arrangements -/
def arrangements (n : ℕ) (k : ℕ) : ℕ := n.factorial / k.factorial

/-- Theorem stating that the number of unique arrangements of PROOF is 60 -/
theorem proof_arrangements : arrangements word_length o_count = 60 := by
  sorry

end proof_arrangements_l1636_163621


namespace systematic_sampling_interval_and_exclusion_l1636_163680

theorem systematic_sampling_interval_and_exclusion 
  (total_stores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_stores = 92) 
  (h2 : sample_size = 30) :
  ∃ (interval : ℕ) (excluded : ℕ),
    interval * sample_size + excluded = total_stores ∧ 
    interval = 3 ∧ 
    excluded = 2 := by
sorry

end systematic_sampling_interval_and_exclusion_l1636_163680


namespace complex_angle_90_degrees_l1636_163646

theorem complex_angle_90_degrees (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) 
  (h : Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂)) : 
  Real.cos (Complex.arg z₁ - Complex.arg z₂) = 0 :=
sorry

end complex_angle_90_degrees_l1636_163646


namespace train_length_calculation_l1636_163693

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 30 → time_s = 6 → 
  ∃ (length_m : ℝ), (abs (length_m - 50) < 1 ∧ length_m = speed_kmh * (1000 / 3600) * time_s) :=
by sorry

end train_length_calculation_l1636_163693


namespace halloween_jelly_beans_l1636_163692

theorem halloween_jelly_beans 
  (initial_jelly_beans : ℕ)
  (total_children : ℕ)
  (jelly_beans_per_child : ℕ)
  (remaining_jelly_beans : ℕ)
  (h1 : initial_jelly_beans = 100)
  (h2 : total_children = 40)
  (h3 : jelly_beans_per_child = 2)
  (h4 : remaining_jelly_beans = 36)
  : (((initial_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child) / total_children) * 100 = 80 := by
  sorry

end halloween_jelly_beans_l1636_163692


namespace exactly_three_two_digit_multiples_l1636_163676

theorem exactly_three_two_digit_multiples :
  ∃! (s : Finset ℕ), 
    (∀ x ∈ s, x > 0 ∧ (∃! (m : Finset ℕ), 
      (∀ y ∈ m, y ≥ 10 ∧ y ≤ 99 ∧ ∃ k : ℕ, y = k * x) ∧ 
      m.card = 3)) ∧ 
    s.card = 9 :=
sorry

end exactly_three_two_digit_multiples_l1636_163676


namespace pet_store_birds_l1636_163678

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 9

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds :
  total_birds = 36 := by sorry

end pet_store_birds_l1636_163678


namespace absolute_value_symmetry_axis_of_symmetry_is_three_l1636_163605

/-- The axis of symmetry for the absolute value function y = |x-a| --/
def axisOfSymmetry (a : ℝ) : ℝ := a

/-- A function is symmetric about a vertical line if it remains unchanged when reflected about that line --/
def isSymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem absolute_value_symmetry (a : ℝ) :
  isSymmetricAbout (fun x ↦ |x - a|) (axisOfSymmetry a) := by sorry

theorem axis_of_symmetry_is_three (a : ℝ) :
  axisOfSymmetry a = 3 → a = 3 := by sorry

end absolute_value_symmetry_axis_of_symmetry_is_three_l1636_163605


namespace trig_identity_proof_l1636_163604

theorem trig_identity_proof : 
  2 * Real.cos (π / 6) - Real.tan (π / 3) + Real.sin (π / 4) * Real.cos (π / 4) = 1 / 2 := by
  sorry

end trig_identity_proof_l1636_163604


namespace scaling_2_3_to_3_2_l1636_163698

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  x_scale : ℝ
  y_scale : ℝ

/-- Apply a scaling transformation to a point -/
def apply_scaling (t : ScalingTransformation) (p : ℝ × ℝ) : ℝ × ℝ :=
  (t.x_scale * p.1, t.y_scale * p.2)

/-- The scaling transformation that changes (2, 3) to (3, 2) -/
theorem scaling_2_3_to_3_2 : 
  ∃ (t : ScalingTransformation), apply_scaling t (2, 3) = (3, 2) ∧ 
    t.x_scale = 3/2 ∧ t.y_scale = 2/3 := by
  sorry

end scaling_2_3_to_3_2_l1636_163698


namespace josh_extracurricular_hours_l1636_163660

/-- Represents the number of days Josh has soccer practice in a week -/
def soccer_days : ℕ := 3

/-- Represents the number of hours Josh spends on soccer practice each day -/
def soccer_hours_per_day : ℝ := 2

/-- Represents the number of days Josh has band practice in a week -/
def band_days : ℕ := 2

/-- Represents the number of hours Josh spends on band practice each day -/
def band_hours_per_day : ℝ := 1.5

/-- Calculates the total hours Josh spends on extracurricular activities in a week -/
def total_extracurricular_hours : ℝ :=
  (soccer_days : ℝ) * soccer_hours_per_day + (band_days : ℝ) * band_hours_per_day

/-- Theorem stating that Josh spends 9 hours on extracurricular activities in a week -/
theorem josh_extracurricular_hours :
  total_extracurricular_hours = 9 := by
  sorry

end josh_extracurricular_hours_l1636_163660


namespace book_sales_ratio_l1636_163672

theorem book_sales_ratio : 
  ∀ (T : ℕ), -- Number of books sold on Thursday
  15 + T + T / 5 = 69 → -- Total sales equation
  T / 15 = 3 -- Ratio of Thursday to Wednesday sales
  := by sorry

end book_sales_ratio_l1636_163672


namespace notebook_distribution_l1636_163665

theorem notebook_distribution (C : ℕ) (H : ℕ) : 
  (C * (C / 8) = 512) →
  (H / 8 = 16) →
  (H : ℚ) / C = 2 :=
by
  sorry

end notebook_distribution_l1636_163665


namespace linear_system_solution_l1636_163682

theorem linear_system_solution (x y m : ℝ) : 
  (2 * x + y = 7) → 
  (x + 2 * y = m - 3) → 
  (x - y = 2) → 
  m = 8 := by
sorry

end linear_system_solution_l1636_163682


namespace janinas_pancakes_sales_l1636_163641

/-- The number of pancakes Janina must sell daily to cover her expenses -/
def pancakes_to_sell (daily_rent : ℕ) (daily_supplies : ℕ) (price_per_pancake : ℕ) : ℕ :=
  (daily_rent + daily_supplies) / price_per_pancake

/-- Theorem stating that Janina must sell 21 pancakes daily to cover her expenses -/
theorem janinas_pancakes_sales : pancakes_to_sell 30 12 2 = 21 := by
  sorry

end janinas_pancakes_sales_l1636_163641


namespace integer_pairs_sum_product_l1636_163632

theorem integer_pairs_sum_product (m n : ℤ) : m + n + m * n = 6 ↔ (m = 0 ∧ n = 6) ∨ (m = 6 ∧ n = 0) := by
  sorry

end integer_pairs_sum_product_l1636_163632


namespace diamond_fifteen_two_l1636_163650

-- Define the diamond operation
def diamond (a b : ℤ) : ℚ := a + a / (b + 1)

-- State the theorem
theorem diamond_fifteen_two : diamond 15 2 = 20 := by sorry

end diamond_fifteen_two_l1636_163650


namespace reading_time_difference_l1636_163639

/-- Prove that the difference in reading time between two people is 360 minutes -/
theorem reading_time_difference 
  (xanthia_speed molly_speed : ℕ) 
  (book_length : ℕ) 
  (h1 : xanthia_speed = 120)
  (h2 : molly_speed = 40)
  (h3 : book_length = 360) :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 360 := by
  sorry

#check reading_time_difference

end reading_time_difference_l1636_163639
