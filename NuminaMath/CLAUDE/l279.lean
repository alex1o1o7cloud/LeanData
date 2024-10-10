import Mathlib

namespace smallest_b_for_quadratic_inequality_seven_satisfies_inequality_seven_is_smallest_l279_27940

theorem smallest_b_for_quadratic_inequality :
  ∀ b : ℝ, b^2 - 16*b + 63 ≤ 0 → b ≥ 7 :=
by
  sorry

theorem seven_satisfies_inequality : 
  7^2 - 16*7 + 63 ≤ 0 :=
by
  sorry

theorem seven_is_smallest :
  ∀ b : ℝ, b^2 - 16*b + 63 ≤ 0 → b ≥ 7 ∧ 
  (∃ ε > 0, (7 - ε)^2 - 16*(7 - ε) + 63 > 0) :=
by
  sorry

end smallest_b_for_quadratic_inequality_seven_satisfies_inequality_seven_is_smallest_l279_27940


namespace houses_before_boom_count_l279_27905

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before_boom : ℕ := 2000 - 574

/-- The current number of houses in Lawrence County -/
def current_houses : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 574

theorem houses_before_boom_count : houses_before_boom = 1426 := by
  sorry

end houses_before_boom_count_l279_27905


namespace percentage_of_amount_twenty_five_percent_of_300_l279_27924

theorem percentage_of_amount (amount : ℝ) (percentage : ℝ) :
  (percentage / 100) * amount = (percentage * amount) / 100 := by sorry

theorem twenty_five_percent_of_300 :
  (25 : ℝ) / 100 * 300 = 75 := by sorry

end percentage_of_amount_twenty_five_percent_of_300_l279_27924


namespace parallelogram_base_length_l279_27969

theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 108) 
  (h2 : height = 9) :
  area / height = 12 := by
sorry

end parallelogram_base_length_l279_27969


namespace basket_balls_count_l279_27909

/-- Given a basket of balls where the ratio of white to red balls is 5:3 and there are 15 white balls, prove that there are 9 red balls. -/
theorem basket_balls_count (white_balls : ℕ) (red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 5 / 3 → white_balls = 15 → red_balls = 9 := by
  sorry

end basket_balls_count_l279_27909


namespace equation_solution_l279_27902

theorem equation_solution :
  ∃ x : ℚ, (x + 3*x = 300 - (4*x + 5*x)) ∧ (x = 300/13) := by sorry

end equation_solution_l279_27902


namespace iteration_convergence_l279_27959

theorem iteration_convergence (a b : ℝ) (h : a > b) :
  ∃ k : ℕ, (2 : ℝ)^(-k : ℤ) * (a - b) < 1 / 2002 := by
  sorry

end iteration_convergence_l279_27959


namespace tan_105_degrees_l279_27989

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l279_27989


namespace car_cost_l279_27926

/-- The cost of a car given an initial payment and monthly installments -/
theorem car_cost (initial_payment : ℕ) (num_installments : ℕ) (installment_amount : ℕ) : 
  initial_payment + num_installments * installment_amount = 18000 :=
by
  sorry

#check car_cost 3000 6 2500

end car_cost_l279_27926


namespace triangle_angle_arithmetic_sequence_l279_27971

theorem triangle_angle_arithmetic_sequence (A B C : ℝ) (AC BC : ℝ) : 
  -- Angles form an arithmetic sequence
  2 * B = A + C →
  -- Sum of angles in a triangle is π
  A + B + C = Real.pi →
  -- Given side lengths
  AC = Real.sqrt 6 →
  BC = 2 →
  -- A is positive and less than π/3
  0 < A →
  A < Real.pi / 3 →
  -- Conclusion: A equals π/4 (45°)
  A = Real.pi / 4 := by
sorry

end triangle_angle_arithmetic_sequence_l279_27971


namespace intersection_distance_l279_27942

/-- The distance between the intersection points of a line and a circle --/
theorem intersection_distance (x y : ℝ) : 
  (x - y + 1 = 0) → -- Line equation
  (x^2 + (y-2)^2 = 4) → -- Circle equation
  ∃ A B : ℝ × ℝ, -- Two intersection points
    A ≠ B ∧
    (A.1 - A.2 + 1 = 0) ∧ (A.1^2 + (A.2-2)^2 = 4) ∧ -- A satisfies both equations
    (B.1 - B.2 + 1 = 0) ∧ (B.1^2 + (B.2-2)^2 = 4) ∧ -- B satisfies both equations
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 14 -- Distance between A and B is √14
  := by sorry

end intersection_distance_l279_27942


namespace sphere_radius_from_hole_l279_27943

theorem sphere_radius_from_hole (hole_width : ℝ) (hole_depth : ℝ) (sphere_radius : ℝ) : 
  hole_width = 30 ∧ hole_depth = 10 → 
  sphere_radius^2 = (hole_width/2)^2 + (sphere_radius - hole_depth)^2 →
  sphere_radius = 16.25 := by
sorry

end sphere_radius_from_hole_l279_27943


namespace quadratic_function_positive_range_l279_27933

theorem quadratic_function_positive_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x → x < 3 → a * x^2 - 2 * a * x + 3 > 0) ↔ 
  (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3) :=
by sorry

end quadratic_function_positive_range_l279_27933


namespace circle_area_with_special_condition_l279_27937

theorem circle_area_with_special_condition (r : ℝ) (h : r > 0) :
  (5 : ℝ) * (1 / (2 * Real.pi * r)) = r / 2 → π * r^2 = 5 := by
  sorry

end circle_area_with_special_condition_l279_27937


namespace min_value_of_reciprocal_sum_l279_27984

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ a*x - b*y + 2 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁^2 + y₁^2 + 2*x₁ - 4*y₁ + 1 = 0 ∧ 
                         x₂^2 + y₂^2 + 2*x₂ - 4*y₂ + 1 = 0 ∧
                         a*x₁ - b*y₁ + 2 = 0 ∧ a*x₂ - b*y₂ + 2 = 0 ∧
                         (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16) →
  (∀ c d : ℝ, c > 0 → d > 0 → 
    (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ c*x - d*y + 2 = 0) →
    1/a + 1/b ≤ 1/c + 1/d) →
  1/a + 1/b = 3/2 + Real.sqrt 2 :=
by sorry

end min_value_of_reciprocal_sum_l279_27984


namespace complex_number_simplification_l279_27967

theorem complex_number_simplification :
  (6 - 3*Complex.I) + 3*(2 - 7*Complex.I) = 12 - 24*Complex.I := by
  sorry

end complex_number_simplification_l279_27967


namespace solve_equation_l279_27922

theorem solve_equation : ∃ x : ℝ, (x - 6) ^ 4 = (1 / 16)⁻¹ ∧ x = 8 := by
  sorry

end solve_equation_l279_27922


namespace wendys_bouquets_l279_27950

/-- Calculates the number of bouquets that can be made given the initial number of flowers,
    number of wilted flowers, and number of flowers per bouquet. -/
def calculateBouquets (initialFlowers : ℕ) (wiltedFlowers : ℕ) (flowersPerBouquet : ℕ) : ℕ :=
  (initialFlowers - wiltedFlowers) / flowersPerBouquet

/-- Proves that Wendy can make 2 bouquets with the given conditions. -/
theorem wendys_bouquets :
  calculateBouquets 45 35 5 = 2 := by
  sorry

end wendys_bouquets_l279_27950


namespace function_minimum_at_one_l279_27912

/-- The function f(x) = (x^2 + 1) / (x + a) has a minimum at x = 1 -/
theorem function_minimum_at_one (a : ℝ) :
  let f : ℝ → ℝ := λ x => (x^2 + 1) / (x + a)
  ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), x ≠ -a → f x ≥ f 1 :=
by
  sorry

end function_minimum_at_one_l279_27912


namespace farm_animals_l279_27963

theorem farm_animals (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 4 * initial_cows →
  (initial_horses - 15) / (initial_cows + 15) = 13 / 7 →
  (initial_horses - 15) - (initial_cows + 15) = 30 := by
sorry

end farm_animals_l279_27963


namespace product_of_integers_l279_27932

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 22)
  (diff_squares_eq : x^2 - y^2 = 44) :
  x * y = 120 := by
  sorry

end product_of_integers_l279_27932


namespace geometric_series_ratio_l279_27986

theorem geometric_series_ratio (a r : ℝ) (hr : 0 < r) (hr1 : r < 1) :
  (a * r^4 / (1 - r)) = (a / (1 - r)) / 81 → r = 1/3 := by
  sorry

end geometric_series_ratio_l279_27986


namespace trapezoid_larger_base_length_l279_27910

/-- A trapezoid with a midline of length 10 and a diagonal that divides the midline
    into two parts with a difference of 3 has a larger base of length 13. -/
theorem trapezoid_larger_base_length (x y : ℝ) 
  (h1 : (x + y) / 2 = 10)  -- midline length is 10
  (h2 : x - y = 6)         -- difference between parts of divided midline is 3 * 2
  : x = 13 := by  -- x represents the larger base
  sorry

end trapezoid_larger_base_length_l279_27910


namespace girls_in_school_l279_27968

/-- Proves that in a school with 1600 students, if a stratified sample of 200 students
    contains 10 fewer girls than boys, then the total number of girls in the school is 760. -/
theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girls_in_sample : ℕ) :
  total_students = 1600 →
  sample_size = 200 →
  girls_in_sample = sample_size / 2 - 5 →
  (girls_in_sample : ℚ) / (total_students : ℚ) = (sample_size : ℚ) / (total_students : ℚ) →
  girls_in_sample * (total_students / sample_size) = 760 :=
by sorry

end girls_in_school_l279_27968


namespace principal_calculation_l279_27964

/-- Proves that given specific conditions, the principal amount is 1600 --/
theorem principal_calculation (rate : ℚ) (time : ℚ) (amount : ℚ) :
  rate = 5 / 100 →
  time = 12 / 5 →
  amount = 1792 →
  amount = (1600 : ℚ) * (1 + rate * time) :=
by
  sorry

end principal_calculation_l279_27964


namespace exists_positive_value_for_expression_l279_27934

theorem exists_positive_value_for_expression : ∃ n : ℕ+, n.val^2 - 8*n.val + 7 > 0 := by
  sorry

end exists_positive_value_for_expression_l279_27934


namespace soda_survey_l279_27983

/-- Given a survey of 600 people and a central angle of 108° for the "Soda" sector,
    prove that the number of people who chose "Soda" is 180. -/
theorem soda_survey (total_people : ℕ) (soda_angle : ℕ) :
  total_people = 600 →
  soda_angle = 108 →
  (total_people * soda_angle) / 360 = 180 := by
  sorry

end soda_survey_l279_27983


namespace initial_average_weight_l279_27916

theorem initial_average_weight 
  (initial_count : ℕ) 
  (new_student_weight : ℝ) 
  (new_average : ℝ) : 
  initial_count = 29 →
  new_student_weight = 10 →
  new_average = 27.4 →
  ∃ (initial_average : ℝ),
    initial_average * initial_count + new_student_weight = 
    new_average * (initial_count + 1) ∧
    initial_average = 28 := by
  sorry

end initial_average_weight_l279_27916


namespace count_random_events_l279_27995

-- Define the set of events
inductive Event
| DiceRoll
| Rain
| Lottery
| SumGreaterThanTwo
| WaterBoiling

-- Define a function to check if an event is random
def isRandomEvent : Event → Bool
| Event.DiceRoll => true
| Event.Rain => true
| Event.Lottery => true
| Event.SumGreaterThanTwo => false
| Event.WaterBoiling => false

-- Theorem: The number of random events is 3
theorem count_random_events :
  (List.filter isRandomEvent [Event.DiceRoll, Event.Rain, Event.Lottery, Event.SumGreaterThanTwo, Event.WaterBoiling]).length = 3 :=
by sorry

end count_random_events_l279_27995


namespace square_tiles_count_l279_27919

/-- Represents the number of edges for each tile type -/
def edges_per_tile : Fin 3 → ℕ
| 0 => 3  -- Triangle
| 1 => 4  -- Square
| 2 => 5  -- Rectangle

/-- Proves that given the conditions, the number of square tiles is 10 -/
theorem square_tiles_count 
  (total_tiles : ℕ) 
  (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 32)
  (h_total_edges : total_edges = 114) :
  ∃ (triangles squares rectangles : ℕ),
    triangles + squares + rectangles = total_tiles ∧
    3 * triangles + 4 * squares + 5 * rectangles = total_edges ∧
    squares = 10 :=
by sorry

end square_tiles_count_l279_27919


namespace marble_bag_count_l279_27908

/-- Given a bag of marbles with red, blue, and green marbles in the ratio 2:4:6,
    and 36 green marbles, prove that the total number of marbles is 72. -/
theorem marble_bag_count (red blue green total : ℕ) : 
  red + blue + green = total →
  2 * blue = 4 * red →
  3 * blue = 2 * green →
  green = 36 →
  total = 72 := by
sorry

end marble_bag_count_l279_27908


namespace symmetric_points_difference_l279_27957

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Given that point A(-2, b) is symmetric to point B(a, 3) with respect to the origin, prove that a - b = 5 -/
theorem symmetric_points_difference (a b : ℝ) 
  (h : symmetric_wrt_origin (-2, b) (a, 3)) : a - b = 5 := by
  sorry

end symmetric_points_difference_l279_27957


namespace henry_correct_answers_l279_27923

/-- Represents a mathematics contest with given scoring rules and a participant's performance. -/
structure MathContest where
  total_problems : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the number of correct answers given a MathContest instance. -/
def correct_answers (contest : MathContest) : ℕ :=
  sorry

/-- Theorem stating that for the given contest conditions, Henry had 10 correct answers. -/
theorem henry_correct_answers : 
  let contest : MathContest := {
    total_problems := 15,
    correct_points := 6,
    incorrect_points := -3,
    total_score := 45
  }
  correct_answers contest = 10 := by
  sorry

end henry_correct_answers_l279_27923


namespace hyperbola_eccentricity_l279_27958

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    left focus F₁, right focus F₂, and a point P on the hyperbola,
    if PF₂ is perpendicular to the x-axis, |F₁F₂| = 12, and |PF₂| = 5,
    then the eccentricity of the hyperbola is 3/2. -/
theorem hyperbola_eccentricity 
  (a b : ℝ) (F₁ F₂ P : ℝ × ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (h_foci : F₁.1 < F₂.1 ∧ F₁.2 = 0 ∧ F₂.2 = 0)
  (h_on_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (h_perpendicular : P.1 = F₂.1)
  (h_distance_foci : Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) = 12)
  (h_distance_PF₂ : Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 5) :
  (Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)) / (2 * a) = 3/2 := by
  sorry

end hyperbola_eccentricity_l279_27958


namespace umars_age_l279_27973

/-- Given the ages of Ali, Yusaf, and Umar, prove Umar's age -/
theorem umars_age (ali_age yusaf_age umar_age : ℕ) : 
  ali_age = 8 →
  ali_age = yusaf_age + 3 →
  umar_age = 2 * yusaf_age →
  umar_age = 10 := by
sorry

end umars_age_l279_27973


namespace fraction_simplification_l279_27975

theorem fraction_simplification 
  (a b x y : ℝ) : 
  (3*b*x*(a^3*x^3 + 3*a^2*y^2 + 2*b^2*y^2) + 2*a*y*(2*a^2*x^2 + 3*b^2*x^2 + b^3*y^3)) / (3*b*x + 2*a*y) 
  = a^3*x^3 + 3*a^2*x*y + 2*b^2*y^2 := by
sorry

end fraction_simplification_l279_27975


namespace star_equation_solution_l279_27918

def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

theorem star_equation_solution :
  ∀ x : ℝ, star 3 x = 15 → x = 7/2 := by
  sorry

end star_equation_solution_l279_27918


namespace inequality_proof_l279_27961

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (b / a + c / b + a / c) ≥ (1 / 3) * (a + b + c) * (1 / a + 1 / b + 1 / c) := by
  sorry

end inequality_proof_l279_27961


namespace triangle_vector_properties_l279_27929

/-- Given a triangle ABC with internal angles A, B, C, this theorem proves
    properties related to vectors m and n, and the side lengths of the triangle. -/
theorem triangle_vector_properties (A B C : Real) (m n : Real × Real) :
  let m : Real × Real := (2 * Real.sqrt 3, 1)
  let n : Real × Real := (Real.cos (A / 2) ^ 2, Real.sin A)
  C = 2 * Real.pi / 3 →
  ‖(1, 0) - (Real.cos A, Real.sin A)‖ = 3 →
  (A = Real.pi / 2 → ‖n‖ = Real.sqrt 5 / 2) ∧
  (∀ θ, m.1 * (Real.cos (θ / 2) ^ 2) + m.2 * Real.sin θ ≤ m.1 * (Real.cos (Real.pi / 12) ^ 2) + m.2 * Real.sin (Real.pi / 6)) ∧
  (‖(Real.cos (Real.pi / 6), Real.sin (Real.pi / 6)) - (Real.cos (5 * Real.pi / 6), Real.sin (5 * Real.pi / 6))‖ = Real.sqrt 3) :=
by sorry

end triangle_vector_properties_l279_27929


namespace polynomial_solution_l279_27979

theorem polynomial_solution (a b c : ℤ) : 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  (∀ X : ℤ, a^3 + a*a*X + b*X + c = a^3) ∧
  (∀ X : ℤ, b^3 + a*b*X + b*X + c = b^3) →
  a = 1 ∧ b = -1 ∧ c = -2 := by
sorry

end polynomial_solution_l279_27979


namespace probability_exactly_once_l279_27960

theorem probability_exactly_once (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (1 - (1 - p)^3 = 26/27) →
  3 * p * (1 - p)^2 = 2/9 :=
by sorry

end probability_exactly_once_l279_27960


namespace elizabeths_husband_weight_l279_27987

/-- Represents a married couple -/
structure Couple where
  husband_weight : ℝ
  wife_weight : ℝ

/-- The problem setup -/
def cannibal_problem (couples : Fin 3 → Couple) : Prop :=
  let wives_weights := (couples 0).wife_weight + (couples 1).wife_weight + (couples 2).wife_weight
  let total_weight := (couples 0).husband_weight + (couples 0).wife_weight +
                      (couples 1).husband_weight + (couples 1).wife_weight +
                      (couples 2).husband_weight + (couples 2).wife_weight
  ∃ (leon victor maurice : Fin 3),
    leon ≠ victor ∧ leon ≠ maurice ∧ victor ≠ maurice ∧
    wives_weights = 171 ∧
    ¬ ∃ n : ℤ, total_weight = n ∧
    (couples leon).husband_weight = (couples leon).wife_weight ∧
    (couples victor).husband_weight = 1.5 * (couples victor).wife_weight ∧
    (couples maurice).husband_weight = 2 * (couples maurice).wife_weight ∧
    (couples 0).wife_weight = (couples 1).wife_weight + 10 ∧
    (couples 1).wife_weight = (couples 2).wife_weight - 5 ∧
    (couples victor).husband_weight = 85.5

/-- The main theorem to prove -/
theorem elizabeths_husband_weight (couples : Fin 3 → Couple) :
  cannibal_problem couples → ∃ i : Fin 3, (couples i).husband_weight = 85.5 :=
sorry

end elizabeths_husband_weight_l279_27987


namespace rectangular_to_polar_conversion_l279_27945

theorem rectangular_to_polar_conversion :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 3 * Real.sqrt 2 ∧ θ = π / 4 ∧
  r * Real.cos θ = 3 ∧ r * Real.sin θ = 3 := by
sorry

end rectangular_to_polar_conversion_l279_27945


namespace square_land_side_length_l279_27978

/-- Given a square-shaped land plot with an area of 625 square units,
    prove that the length of one side is 25 units. -/
theorem square_land_side_length :
  ∀ (side : ℝ), side > 0 → side * side = 625 → side = 25 := by
  sorry

end square_land_side_length_l279_27978


namespace smallest_four_digit_divisible_by_53_l279_27930

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 := by
  sorry

end smallest_four_digit_divisible_by_53_l279_27930


namespace condition_necessary_not_sufficient_l279_27906

theorem condition_necessary_not_sufficient : 
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x ≠ 1) ∧ 
  (∀ x : ℝ, x = 1 → (x - 1) * (x + 2) = 0) := by
  sorry

end condition_necessary_not_sufficient_l279_27906


namespace product_evaluation_l279_27972

theorem product_evaluation (a : ℕ) (h : a = 7) : 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 5040 := by
  sorry

end product_evaluation_l279_27972


namespace min_value_sum_reciprocal_squares_l279_27917

/-- Given two circles with equations (x^2 + y^2 + 2ax + a^2 - 4 = 0) and (x^2 + y^2 - 4by - 1 + 4b^2 = 0),
    where a ∈ ℝ, ab ≠ 0, and the circles have exactly three common tangents,
    prove that the minimum value of (1/a^2 + 1/b^2) is 1. -/
theorem min_value_sum_reciprocal_squares (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∨ x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0) →
  (∃! (t1 t2 t3 : ℝ × ℝ → ℝ), t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    (∀ x y : ℝ, (t1 (x, y) = 0 ∨ t2 (x, y) = 0 ∨ t3 (x, y) = 0) ↔ 
      ((x^2 + y^2 + 2*a*x + a^2 - 4 = 0) ∨ (x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0)))) →
  a ≠ 0 →
  b ≠ 0 →
  ∃ (m : ℝ), m = 1 ∧ ∀ (k : ℝ), k ≥ 0 → (1 / a^2 + 1 / b^2) ≥ m + k :=
by sorry

end min_value_sum_reciprocal_squares_l279_27917


namespace buses_in_parking_lot_l279_27914

theorem buses_in_parking_lot (initial_buses additional_buses : ℕ) : 
  initial_buses = 7 → additional_buses = 6 → initial_buses + additional_buses = 13 :=
by sorry

end buses_in_parking_lot_l279_27914


namespace total_heads_is_48_l279_27962

/-- Represents the number of feet an animal has -/
def feet_count (animal : String) : ℕ :=
  if animal = "hen" then 2 else 4

/-- The total number of animals -/
def total_animals (hens cows : ℕ) : ℕ := hens + cows

/-- The total number of feet -/
def total_feet (hens cows : ℕ) : ℕ := feet_count "hen" * hens + feet_count "cow" * cows

/-- Theorem stating that the total number of heads is 48 -/
theorem total_heads_is_48 (hens cows : ℕ) 
  (h1 : total_feet hens cows = 140) 
  (h2 : hens = 26) : 
  total_animals hens cows = 48 := by sorry

end total_heads_is_48_l279_27962


namespace x0_value_l279_27904

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem x0_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 3) → x₀ = Real.exp 2 := by
  sorry

end x0_value_l279_27904


namespace smallest_prime_angle_in_inscribed_triangle_l279_27947

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The theorem statement -/
theorem smallest_prime_angle_in_inscribed_triangle :
  ∀ q : ℕ,
  q > 0 →
  isPrime q →
  isPrime (2 * q) →
  isPrime (180 - 3 * q) →
  (∀ p : ℕ, p < q → p > 0 → ¬(isPrime p ∧ isPrime (2 * p) ∧ isPrime (180 - 3 * p))) →
  q = 7 := by
  sorry

#check smallest_prime_angle_in_inscribed_triangle

end smallest_prime_angle_in_inscribed_triangle_l279_27947


namespace houses_with_neither_feature_l279_27954

theorem houses_with_neither_feature (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) :
  total = 90 →
  garage = 50 →
  pool = 40 →
  both = 35 →
  total - (garage + pool - both) = 35 := by
sorry

end houses_with_neither_feature_l279_27954


namespace arwen_tulips_l279_27948

/-- Proves that Arwen picked 20 tulips, given the conditions of the problem -/
theorem arwen_tulips : 
  ∀ (a e : ℕ), 
    e = 2 * a →  -- Elrond picked twice as many tulips as Arwen
    a + e = 60 →  -- They picked 60 tulips in total
    a = 20  -- Arwen picked 20 tulips
    := by sorry

end arwen_tulips_l279_27948


namespace probability_adjacent_ascending_five_cds_l279_27925

/-- The probability of two specific CDs being adjacent in ascending order when n CDs are randomly arranged -/
def probability_adjacent_ascending (n : ℕ) : ℚ :=
  if n ≥ 2 then (4 * (n - 2).factorial) / n.factorial else 0

/-- Theorem: The probability of CDs 1 and 2 being next to each other in ascending order 
    when 5 CDs are randomly placed in a cassette holder is 1/5 -/
theorem probability_adjacent_ascending_five_cds : 
  probability_adjacent_ascending 5 = 1 / 5 := by
  sorry

end probability_adjacent_ascending_five_cds_l279_27925


namespace min_colors_for_distribution_centers_l279_27927

def combinations (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem min_colors_for_distribution_centers : 
  (∃ (n : ℕ), n ≥ 6 ∧ combinations n 3 ≥ 20) ∧
  (∀ (m : ℕ), m < 6 → combinations m 3 < 20) := by
  sorry

end min_colors_for_distribution_centers_l279_27927


namespace smallest_w_l279_27920

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^5) (936 * w) → 
  is_factor (3^3) (936 * w) → 
  is_factor (14^2) (936 * w) → 
  w ≥ 1764 :=
sorry

end smallest_w_l279_27920


namespace wait_hare_is_random_l279_27981

-- Define the type for events
inductive Event
| StrongYouth
| ScoopMoon
| WaitHare
| GreenMountains

-- Define what it means for an event to be random
def isRandom (e : Event) : Prop :=
  match e with
  | Event.WaitHare => True
  | _ => False

-- Theorem statement
theorem wait_hare_is_random :
  isRandom Event.WaitHare :=
sorry

end wait_hare_is_random_l279_27981


namespace min_value_xyz_one_min_value_achievable_l279_27965

theorem min_value_xyz_one (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  3 * x^2 + 12 * x * y + 9 * y^2 + 15 * y * z + 3 * z^2 ≥ 243 / Real.rpow 4 (1/9) :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 1 ∧
  3 * x^2 + 12 * x * y + 9 * y^2 + 15 * y * z + 3 * z^2 = 243 / Real.rpow 4 (1/9) :=
by sorry

end min_value_xyz_one_min_value_achievable_l279_27965


namespace min_sum_reciprocals_l279_27976

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 ∧ a + b = 49 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → c + d ≥ 49 :=
by sorry

end min_sum_reciprocals_l279_27976


namespace raccoons_pepper_sprayed_l279_27988

theorem raccoons_pepper_sprayed (num_raccoons : ℕ) (num_squirrels : ℕ) : 
  num_squirrels = 6 * num_raccoons →
  num_raccoons + num_squirrels = 84 →
  num_raccoons = 12 := by
sorry

end raccoons_pepper_sprayed_l279_27988


namespace particle_position_at_2004_l279_27996

/-- Represents the position of a particle -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Defines the movement pattern of the particle -/
def next_position (p : Position) : Position :=
  if p.x = p.y then Position.mk (p.x + 1) p.y
  else if p.x > p.y then Position.mk p.x (p.y + 1)
  else Position.mk (p.x + 1) p.y

/-- Calculates the position of the particle after n seconds -/
def position_at_time (n : ℕ) : Position :=
  match n with
  | 0 => Position.mk 0 0
  | n + 1 => next_position (position_at_time n)

/-- The main theorem stating the position of the particle after 2004 seconds -/
theorem particle_position_at_2004 :
  position_at_time 2004 = Position.mk 20 44 := by
  sorry


end particle_position_at_2004_l279_27996


namespace dilation_rotation_theorem_l279_27903

/-- The matrix representing a dilation by scale factor 4 centered at the origin -/
def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![4, 0; 0, 4]

/-- The matrix representing a 90-degree counterclockwise rotation -/
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- The combined transformation matrix -/
def combined_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -4; 4, 0]

theorem dilation_rotation_theorem :
  rotation_matrix * dilation_matrix = combined_matrix :=
sorry

end dilation_rotation_theorem_l279_27903


namespace jerry_current_average_l279_27980

/-- Jerry's current average score on the first 3 tests -/
def current_average : ℝ := sorry

/-- Jerry's score on the fourth test -/
def fourth_test_score : ℝ := 93

/-- The increase in average score after the fourth test -/
def average_increase : ℝ := 2

theorem jerry_current_average : 
  (current_average * 3 + fourth_test_score) / 4 = current_average + average_increase ∧ 
  current_average = 85 := by sorry

end jerry_current_average_l279_27980


namespace sin_15_cos_15_eq_one_fourth_l279_27999

theorem sin_15_cos_15_eq_one_fourth : 4 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 := by
  sorry

end sin_15_cos_15_eq_one_fourth_l279_27999


namespace abs_neg_five_eq_five_l279_27955

theorem abs_neg_five_eq_five : |(-5 : ℤ)| = 5 := by
  sorry

end abs_neg_five_eq_five_l279_27955


namespace mrs_petersons_tumblers_l279_27966

/-- The number of tumblers bought given the price per tumbler, 
    the amount paid, and the change received. -/
def number_of_tumblers (price_per_tumbler : ℕ) (amount_paid : ℕ) (change : ℕ) : ℕ :=
  (amount_paid - change) / price_per_tumbler

/-- Theorem stating that Mrs. Petersons bought 10 tumblers -/
theorem mrs_petersons_tumblers : 
  number_of_tumblers 45 500 50 = 10 := by
  sorry

end mrs_petersons_tumblers_l279_27966


namespace clock_solution_l279_27953

/-- The time in minutes after 9:00 when the minute hand will be exactly opposite
    the place where the hour hand was two minutes ago, five minutes from now. -/
def clock_problem : ℝ → Prop := λ t =>
  0 < t ∧ t < 60 ∧  -- Time is between 9:00 and 10:00
  abs (6 * (t + 5) - (270 + 0.5 * (t - 2))) = 180  -- Opposite hands condition

theorem clock_solution : ∃ t, clock_problem t ∧ t = 10.75 := by
  sorry

end clock_solution_l279_27953


namespace total_lives_l279_27938

theorem total_lives (num_friends : ℕ) (lives_per_friend : ℕ) 
  (h1 : num_friends = 8) (h2 : lives_per_friend = 8) : 
  num_friends * lives_per_friend = 64 := by
  sorry

end total_lives_l279_27938


namespace temperature_conversion_l279_27941

theorem temperature_conversion (t k some_number : ℝ) :
  t = 5 / 9 * (k - some_number) →
  t = 105 →
  k = 221 →
  some_number = 32 := by
sorry

end temperature_conversion_l279_27941


namespace sqrt_equation_solution_l279_27994

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (2 - 5 * x) = 5 → x = -4.6 := by
  sorry

end sqrt_equation_solution_l279_27994


namespace intersection_line_circle_l279_27946

/-- Given a line y = kx + 3 intersecting the circle (x-1)^2 + (y-2)^2 = 4 at points M and N,
    if |MN| ≥ 2√3, then k ≤ 0. -/
theorem intersection_line_circle (k : ℝ) (M N : ℝ × ℝ) : 
  (∀ x y, y = k * x + 3 → (x - 1)^2 + (y - 2)^2 = 4) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12 →
  k ≤ 0 := by
  sorry

end intersection_line_circle_l279_27946


namespace triangles_in_regular_decagon_l279_27928

def regular_decagon_vertices : ℕ := 10

theorem triangles_in_regular_decagon :
  (regular_decagon_vertices.choose 3) = 120 :=
by sorry

end triangles_in_regular_decagon_l279_27928


namespace trapezium_area_and_shorter_side_l279_27915

theorem trapezium_area_and_shorter_side (a b h : ℝ) :
  a = 24 ∧ b = 18 ∧ h = 15 →
  (1/2 : ℝ) * (a + b) * h = 315 ∧ min a b = 18 :=
by sorry

end trapezium_area_and_shorter_side_l279_27915


namespace derivative_y_wrt_x_l279_27936

noncomputable section

variable (t : ℝ)

def x : ℝ := Real.arcsin (Real.sin t)
def y : ℝ := Real.arccos (Real.cos t)

theorem derivative_y_wrt_x : 
  deriv (fun x => y x) (x t) = 1 :=
sorry

end derivative_y_wrt_x_l279_27936


namespace complex_cube_real_iff_l279_27970

def is_real (z : ℂ) : Prop := z.im = 0

theorem complex_cube_real_iff (z : ℂ) : 
  is_real (z^3) ↔ z.im = 0 ∨ z.im = Real.sqrt 3 * z.re ∨ z.im = -Real.sqrt 3 * z.re :=
sorry

end complex_cube_real_iff_l279_27970


namespace circle_area_from_square_perimeter_l279_27991

/-- The area of a circle that shares a center with a square of perimeter 40 feet -/
theorem circle_area_from_square_perimeter : ∃ (circle_area : ℝ), 
  circle_area = 50 * Real.pi ∧ 
  ∃ (square_side : ℝ), 
    4 * square_side = 40 ∧
    circle_area = Real.pi * (square_side * Real.sqrt 2 / 2)^2 := by
  sorry

end circle_area_from_square_perimeter_l279_27991


namespace min_groups_for_photography_class_l279_27998

theorem min_groups_for_photography_class (total_students : ℕ) (max_group_size : ℕ) 
  (h1 : total_students = 30) (h2 : max_group_size = 6) : 
  Nat.ceil (total_students / max_group_size) = 5 :=
by sorry

end min_groups_for_photography_class_l279_27998


namespace star_equation_equiv_two_distinct_real_roots_l279_27993

/-- The star operation defined as m ☆ n = mn² - mn - 1 -/
def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

/-- The equation 1 ☆ x = 0 is equivalent to x² - x - 1 = 0 -/
theorem star_equation_equiv (x : ℝ) : star 1 x = 0 ↔ x^2 - x - 1 = 0 := by sorry

/-- The equation x² - x - 1 = 0 has two distinct real roots -/
theorem two_distinct_real_roots :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ r₁^2 - r₁ - 1 = 0 ∧ r₂^2 - r₂ - 1 = 0 := by sorry

end star_equation_equiv_two_distinct_real_roots_l279_27993


namespace paths_from_A_to_E_l279_27944

/-- The number of paths between two consecutive points -/
def paths_between_consecutive : ℕ := 2

/-- The number of direct paths from A to E -/
def direct_paths : ℕ := 1

/-- The number of intermediate points between A and E -/
def intermediate_points : ℕ := 4

/-- The total number of paths from A to E -/
def total_paths : ℕ := paths_between_consecutive ^ intermediate_points + direct_paths

theorem paths_from_A_to_E : total_paths = 17 := by sorry

end paths_from_A_to_E_l279_27944


namespace no_simultaneous_greater_value_l279_27951

theorem no_simultaneous_greater_value : ¬∃ x : ℝ, (x + 3) / 5 > 2 * x + 3 ∧ (x + 3) / 5 > 1 - x := by
  sorry

end no_simultaneous_greater_value_l279_27951


namespace three_digit_power_ending_theorem_l279_27900

/-- A three-digit number is between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A number N satisfies the property if for all k ≥ 1, N^k ≡ N (mod 1000) -/
def SatisfiesProperty (N : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 1 → N^k ≡ N [MOD 1000]

theorem three_digit_power_ending_theorem :
  ∀ N : ℕ, ThreeDigitNumber N → SatisfiesProperty N ↔ (N = 625 ∨ N = 376) :=
sorry

end three_digit_power_ending_theorem_l279_27900


namespace parabola_intersection_l279_27997

theorem parabola_intersection (a b : ℝ) (h1 : a ≠ 0) : 
  (∀ x, a * (x - b) * (x - 1) = 0 → x = 3 ∨ x = 1) ∧
  a * (3 - b) * (3 - 1) = 0 →
  ∃ x, x ≠ 3 ∧ a * (x - b) * (x - 1) = 0 ∧ x = 1 :=
by sorry

end parabola_intersection_l279_27997


namespace ferry_passengers_with_hats_l279_27977

theorem ferry_passengers_with_hats (total_passengers : ℕ) 
  (percent_men : ℚ) (percent_women_with_hats : ℚ) (percent_men_with_hats : ℚ) :
  total_passengers = 1500 →
  percent_men = 2/5 →
  percent_women_with_hats = 3/20 →
  percent_men_with_hats = 3/25 →
  ∃ (total_with_hats : ℕ), total_with_hats = 207 :=
by
  sorry

end ferry_passengers_with_hats_l279_27977


namespace sum_not_prime_l279_27985

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b + c + d = x * y :=
sorry

end sum_not_prime_l279_27985


namespace parallelogram_not_symmetrical_l279_27907

-- Define a type for shapes
inductive Shape
  | Circle
  | Rectangle
  | Parallelogram
  | IsoscelesTrapezoid

-- Define a property for symmetry
def isSymmetrical (s : Shape) : Prop :=
  match s with
  | Shape.Circle => True
  | Shape.Rectangle => True
  | Shape.Parallelogram => False
  | Shape.IsoscelesTrapezoid => True

-- Theorem statement
theorem parallelogram_not_symmetrical :
  ¬(isSymmetrical Shape.Parallelogram) :=
by
  sorry

#check parallelogram_not_symmetrical

end parallelogram_not_symmetrical_l279_27907


namespace tree_planting_temperature_reduction_l279_27990

theorem tree_planting_temperature_reduction 
  (initial_temp : ℝ) 
  (cost_per_tree : ℝ) 
  (temp_reduction_per_tree : ℝ) 
  (total_cost : ℝ) 
  (h1 : initial_temp = 80)
  (h2 : cost_per_tree = 6)
  (h3 : temp_reduction_per_tree = 0.1)
  (h4 : total_cost = 108) :
  initial_temp - (total_cost / cost_per_tree * temp_reduction_per_tree) = 78.2 := by
  sorry

end tree_planting_temperature_reduction_l279_27990


namespace characterization_of_k_l279_27956

theorem characterization_of_k (k m n : ℕ+) (h : m * (m + k) = n * (n + 1)) :
  k = 1 ∨ k ≥ 4 := by
  sorry

end characterization_of_k_l279_27956


namespace geometric_sequence_formula_l279_27901

-- Define the geometric sequence and its sum
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 2

-- Define the general formula for the sequence
def general_formula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 * (3 ^ (n - 1))

-- Theorem statement
theorem geometric_sequence_formula 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : geometric_sequence a S) : 
  general_formula a :=
sorry

end geometric_sequence_formula_l279_27901


namespace average_service_hours_l279_27974

theorem average_service_hours (n : ℕ) (h1 h2 h3 : ℕ) (s1 s2 s3 : ℕ) :
  n = 10 →
  h1 = 15 →
  h2 = 16 →
  h3 = 20 →
  s1 = 2 →
  s2 = 5 →
  s3 = 3 →
  s1 + s2 + s3 = n →
  (h1 * s1 + h2 * s2 + h3 * s3) / n = 17 :=
by
  sorry

end average_service_hours_l279_27974


namespace sqrt_of_point_zero_nine_equals_point_three_l279_27939

theorem sqrt_of_point_zero_nine_equals_point_three : 
  Real.sqrt 0.09 = 0.3 := by
  sorry

end sqrt_of_point_zero_nine_equals_point_three_l279_27939


namespace nine_team_league_games_l279_27952

/-- The total number of games played in a baseball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a 9-team league where each team plays 3 games with every other team, 
    the total number of games played is 108 -/
theorem nine_team_league_games :
  total_games 9 3 = 108 := by
  sorry

end nine_team_league_games_l279_27952


namespace intersection_distance_to_pole_l279_27949

-- Define the polar coordinate system
def PolarCoordinate := ℝ × ℝ

-- Define the distance function in polar coordinates
def distance_to_pole (p : PolarCoordinate) : ℝ := p.1

-- Define the curves
def curve1 (ρ θ : ℝ) : Prop := ρ = 2 * θ + 1
def curve2 (ρ θ : ℝ) : Prop := ρ * θ = 1

theorem intersection_distance_to_pole :
  ∀ (p : PolarCoordinate),
    p.1 > 0 →
    curve1 p.1 p.2 →
    curve2 p.1 p.2 →
    distance_to_pole p = 2 := by
  sorry

end intersection_distance_to_pole_l279_27949


namespace pure_imaginary_complex_number_l279_27911

theorem pure_imaginary_complex_number (m : ℝ) : 
  (((m^2 - m - 2) : ℂ) + (m + 1) * Complex.I).re = 0 ∧ 
  (((m^2 - m - 2) : ℂ) + (m + 1) * Complex.I).im ≠ 0 → 
  m = 2 := by sorry

end pure_imaginary_complex_number_l279_27911


namespace annuity_duration_exists_l279_27982

/-- The duration of the original annuity in years -/
def original_duration : ℝ := 20

/-- The interest rate as a decimal -/
def interest_rate : ℝ := 0.04

/-- The equation that the new duration must satisfy -/
def annuity_equation (x : ℝ) : Prop :=
  Real.exp x = 2 * Real.exp original_duration / (Real.exp original_duration + 1)

/-- Theorem stating the existence of a solution to the annuity equation -/
theorem annuity_duration_exists :
  ∃ x : ℝ, annuity_equation x :=
sorry

end annuity_duration_exists_l279_27982


namespace matts_baseball_cards_value_l279_27913

theorem matts_baseball_cards_value (n : ℕ) (x : ℚ) : 
  n = 8 →  -- Matt has 8 baseball cards
  2 * x + 3 = (3 * 2 + 9) →  -- He trades 2 cards for 3 $2 cards and 1 $9 card, making a $3 profit
  x = 6 :=  -- Each of Matt's baseball cards is worth $6
by
  sorry

end matts_baseball_cards_value_l279_27913


namespace fifth_term_of_geometric_sequence_l279_27992

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n - 1)

theorem fifth_term_of_geometric_sequence
  (a₁ a₂ : ℚ)
  (h₁ : a₁ = 2)
  (h₂ : a₂ = 1/4)
  (h₃ : a₂ = a₁ * (a₂ / a₁)) :
  geometric_sequence a₁ (a₂ / a₁) 5 = 1/2048 := by
sorry

end fifth_term_of_geometric_sequence_l279_27992


namespace arabella_first_step_time_l279_27935

/-- Represents the time spent learning dance steps -/
structure DanceSteps where
  first : ℝ
  second : ℝ
  third : ℝ

/-- The conditions for Arabella's dance step learning -/
def arabella_dance_conditions (steps : DanceSteps) : Prop :=
  steps.second = steps.first / 2 ∧
  steps.third = steps.first + steps.second ∧
  steps.first + steps.second + steps.third = 90

/-- Theorem stating that under the given conditions, the time spent on the first step is 30 minutes -/
theorem arabella_first_step_time (steps : DanceSteps) 
  (h : arabella_dance_conditions steps) : steps.first = 30 := by
  sorry

end arabella_first_step_time_l279_27935


namespace stratified_sampling_l279_27921

theorem stratified_sampling (total_male : ℕ) (total_female : ℕ) (selected_male : ℕ) :
  total_male = 56 →
  total_female = 42 →
  selected_male = 8 →
  (total_male : ℚ) / total_female = 4 / 3 →
  ∃ selected_female : ℕ, 
    (selected_female : ℚ) / selected_male = total_female / total_male ∧
    selected_female = 6 :=
by sorry

end stratified_sampling_l279_27921


namespace arithmetic_calculation_l279_27931

theorem arithmetic_calculation : 5 * 7 + 10 * 4 - 35 / 5 + 18 / 3 = 74 := by
  sorry

end arithmetic_calculation_l279_27931
