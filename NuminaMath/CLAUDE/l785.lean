import Mathlib

namespace NUMINAMATH_CALUDE_complement_of_equal_sets_l785_78552

def U : Set Nat := {1, 3}
def A : Set Nat := {1, 3}

theorem complement_of_equal_sets :
  (U \ A : Set Nat) = ∅ :=
sorry

end NUMINAMATH_CALUDE_complement_of_equal_sets_l785_78552


namespace NUMINAMATH_CALUDE_sea_island_arithmetic_author_l785_78587

-- Define the type for mathematicians
inductive Mathematician
| YangHui
| ZuChongzhi
| LiuHui
| QinJiushao

-- Define the properties of "The Sea Island Arithmetic"
structure SeaIslandArithmetic where
  is_significant : Prop
  advanced_surveying : Prop
  years_ahead : ℕ

-- Define the authorship relation
def is_author_of (m : Mathematician) (s : SeaIslandArithmetic) : Prop := sorry

-- State the theorem
theorem sea_island_arithmetic_author :
  ∃ (s : SeaIslandArithmetic),
    s.is_significant ∧
    s.advanced_surveying ∧
    s.years_ahead ≥ 1300 ∧
    s.years_ahead ≤ 1500 ∧
    is_author_of Mathematician.LiuHui s :=
  sorry

end NUMINAMATH_CALUDE_sea_island_arithmetic_author_l785_78587


namespace NUMINAMATH_CALUDE_max_tickets_purchasable_l785_78571

theorem max_tickets_purchasable (ticket_price : ℚ) (budget : ℚ) : 
  ticket_price = 15.75 → budget = 200 → 
  ∃ n : ℕ, n * ticket_price ≤ budget ∧ 
           ∀ m : ℕ, m * ticket_price ≤ budget → m ≤ n ∧
           n = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_tickets_purchasable_l785_78571


namespace NUMINAMATH_CALUDE_sasha_sticker_collection_l785_78537

theorem sasha_sticker_collection (m n : ℕ) (t : ℝ) : 
  m < n →
  m > 0 →
  t > 1 →
  m * t + n = 100 →
  m + n * t = 101 →
  (n = 34 ∨ n = 66) ∧ ∀ k : ℕ, (k ≠ 34 ∧ k ≠ 66) → 
    ¬(∃ m' : ℕ, ∃ t' : ℝ, 
      m' < k ∧ 
      m' > 0 ∧ 
      t' > 1 ∧ 
      m' * t' + k = 100 ∧ 
      m' + k * t' = 101) :=
by sorry

end NUMINAMATH_CALUDE_sasha_sticker_collection_l785_78537


namespace NUMINAMATH_CALUDE_estimate_student_population_l785_78507

theorem estimate_student_population (first_survey : ℕ) (second_survey : ℕ) (overlap : ℕ) 
  (h1 : first_survey = 80)
  (h2 : second_survey = 100)
  (h3 : overlap = 20) :
  (first_survey * second_survey) / overlap = 400 := by
  sorry

end NUMINAMATH_CALUDE_estimate_student_population_l785_78507


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_is_sqrt_5_l785_78525

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Predicate indicating that the focus of a hyperbola is symmetric with respect to its asymptote -/
def focus_symmetric_to_asymptote (h : Hyperbola a b) : Prop := sorry

/-- Predicate indicating that the focus of a hyperbola lies on the hyperbola -/
def focus_on_hyperbola (h : Hyperbola a b) : Prop := sorry

/-- Theorem stating that if the focus of a hyperbola is symmetric with respect to its asymptote
    and lies on the hyperbola, then its eccentricity is √5 -/
theorem hyperbola_eccentricity_is_sqrt_5 {a b : ℝ} (h : Hyperbola a b)
  (h_sym : focus_symmetric_to_asymptote h) (h_on : focus_on_hyperbola h) :
  eccentricity h = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_is_sqrt_5_l785_78525


namespace NUMINAMATH_CALUDE_negative_three_times_b_minus_a_l785_78509

theorem negative_three_times_b_minus_a (a b : ℚ) (h : a - b = 1/2) : -3 * (b - a) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_b_minus_a_l785_78509


namespace NUMINAMATH_CALUDE_percentage_increase_calculation_l785_78528

def original_earnings : ℝ := 60
def new_earnings : ℝ := 68

theorem percentage_increase_calculation :
  (new_earnings - original_earnings) / original_earnings * 100 = 13.33333333333333 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_calculation_l785_78528


namespace NUMINAMATH_CALUDE_coin_drop_probability_l785_78533

theorem coin_drop_probability : 
  let square_side : ℝ := 10
  let black_square_side : ℝ := 1
  let coin_diameter : ℝ := 2
  let coin_radius : ℝ := coin_diameter / 2
  let drop_area_side : ℝ := square_side - coin_diameter
  let drop_area : ℝ := drop_area_side ^ 2
  let extended_black_square_side : ℝ := black_square_side + coin_diameter
  let extended_black_area : ℝ := 4 * (extended_black_square_side ^ 2)
  extended_black_area / drop_area = 9 / 16 := by sorry

end NUMINAMATH_CALUDE_coin_drop_probability_l785_78533


namespace NUMINAMATH_CALUDE_eggs_to_examine_l785_78560

def number_of_trays : ℕ := 7
def eggs_per_tray : ℕ := 10
def percentage_to_examine : ℚ := 70 / 100

theorem eggs_to_examine :
  (number_of_trays * (eggs_per_tray * percentage_to_examine).floor) = 49 := by
  sorry

end NUMINAMATH_CALUDE_eggs_to_examine_l785_78560


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l785_78568

theorem complex_purely_imaginary (z : ℂ) : 
  (∃ y : ℝ, z = y * I) →  -- z is purely imaginary
  (∃ w : ℝ, (z + 2)^2 - 8*I = w * I) →  -- (z + 2)² - 8i is purely imaginary
  z = -2 * I :=  -- z = -2i
by sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l785_78568


namespace NUMINAMATH_CALUDE_average_cost_is_two_l785_78558

/-- The average cost of fruit given the prices and quantities of apples, bananas, and oranges. -/
def average_cost (apple_price banana_price orange_price : ℚ) 
                 (apple_qty banana_qty orange_qty : ℕ) : ℚ :=
  let total_cost := apple_price * apple_qty + banana_price * banana_qty + orange_price * orange_qty
  let total_qty := apple_qty + banana_qty + orange_qty
  total_cost / total_qty

/-- Theorem stating that the average cost of fruit is $2 given the specified prices and quantities. -/
theorem average_cost_is_two :
  average_cost 2 1 3 12 4 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_cost_is_two_l785_78558


namespace NUMINAMATH_CALUDE_power_division_multiplication_l785_78586

theorem power_division_multiplication (m : ℝ) :
  ((-m^4)^5) / (m^5) * m = -m^14 := by
  sorry

end NUMINAMATH_CALUDE_power_division_multiplication_l785_78586


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l785_78554

theorem complex_number_in_second_quadrant :
  let z : ℂ := -1 + Complex.I
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l785_78554


namespace NUMINAMATH_CALUDE_analogical_reasoning_is_specific_to_specific_l785_78550

-- Define the types of reasoning
inductive ReasoningType
  | Reasonable
  | Inductive
  | Deductive
  | Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
  | GeneralToSpecific
  | SpecificToGeneral
  | SpecificToSpecific

-- Define the property of a reasoning type
def reasoning_direction (r : ReasoningType) : ReasoningDirection :=
  match r with
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific
  | _ => ReasoningDirection.GeneralToSpecific -- Default for other types, not relevant for this problem

-- Theorem statement
theorem analogical_reasoning_is_specific_to_specific :
  reasoning_direction ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific :=
by sorry

end NUMINAMATH_CALUDE_analogical_reasoning_is_specific_to_specific_l785_78550


namespace NUMINAMATH_CALUDE_crayfish_yield_theorem_l785_78565

theorem crayfish_yield_theorem (last_year_total : ℝ) (this_year_total : ℝ) 
  (yield_difference : ℝ) (h1 : last_year_total = 4800) 
  (h2 : this_year_total = 6000) (h3 : yield_difference = 60) : 
  ∃ (x : ℝ), x = 300 ∧ this_year_total / x = last_year_total / (x - yield_difference) :=
by sorry

end NUMINAMATH_CALUDE_crayfish_yield_theorem_l785_78565


namespace NUMINAMATH_CALUDE_tea_boxes_problem_l785_78514

/-- Proves that if there are four boxes of tea, and after removing 9 kg from each box,
    the total remaining quantity equals the original quantity in one box,
    then each box initially contained 12 kg of tea. -/
theorem tea_boxes_problem (x : ℝ) : 
  (4 * (x - 9) = x) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_tea_boxes_problem_l785_78514


namespace NUMINAMATH_CALUDE_infinitely_many_squares_in_ap_l785_78570

/-- An arithmetic progression of positive integers. -/
def ArithmeticProgression (a d : ℕ) : ℕ → ℕ
  | n => a + n * d

/-- Predicate to check if a number is a perfect square. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- The main theorem to be proved. -/
theorem infinitely_many_squares_in_ap (a d : ℕ) (h : d > 0) :
  (∃ n : ℕ, IsPerfectSquare (ArithmeticProgression a d n)) →
  (∀ m : ℕ, ∃ n : ℕ, n > m ∧ IsPerfectSquare (ArithmeticProgression a d n)) :=
by sorry


end NUMINAMATH_CALUDE_infinitely_many_squares_in_ap_l785_78570


namespace NUMINAMATH_CALUDE_isosceles_right_triangles_in_quadrilateral_l785_78555

-- Define the points
variable (A B C D O₁ O₂ O₃ O₄ : Point)

-- Define the quadrilateral
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define isosceles right triangle
def is_isosceles_right_triangle (X Y Z : Point) : Prop := sorry

-- State the theorem
theorem isosceles_right_triangles_in_quadrilateral 
  (h_quad : is_convex_quadrilateral A B C D)
  (h_ABO₁ : is_isosceles_right_triangle A B O₁)
  (h_BCO₂ : is_isosceles_right_triangle B C O₂)
  (h_CDO₃ : is_isosceles_right_triangle C D O₃)
  (h_DAO₄ : is_isosceles_right_triangle D A O₄)
  (h_O₁_O₃ : O₁ = O₃) :
  O₂ = O₄ := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangles_in_quadrilateral_l785_78555


namespace NUMINAMATH_CALUDE_power_inequality_l785_78538

theorem power_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a*b*c)^(a/5) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l785_78538


namespace NUMINAMATH_CALUDE_meal_cost_45_dollars_l785_78598

/-- The cost of a meal consisting of one pizza and three burgers -/
def meal_cost (burger_price : ℝ) : ℝ :=
  let pizza_price := 2 * burger_price
  pizza_price + 3 * burger_price

/-- Theorem: The cost of one pizza and three burgers is $45 when a burger costs $9 -/
theorem meal_cost_45_dollars :
  meal_cost 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_45_dollars_l785_78598


namespace NUMINAMATH_CALUDE_inverse_f_at_486_l785_78591

/-- Given a function f with the properties f(5) = 2 and f(3x) = 3f(x) for all x,
    prove that the inverse function f⁻¹ evaluated at 486 is equal to 1215. -/
theorem inverse_f_at_486 (f : ℝ → ℝ) (h1 : f 5 = 2) (h2 : ∀ x, f (3 * x) = 3 * f x) :
  Function.invFun f 486 = 1215 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_486_l785_78591


namespace NUMINAMATH_CALUDE_union_of_sets_l785_78547

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4}
  let B : Set ℕ := {1, 3, 5, 7}
  A ∪ B = {1, 2, 3, 4, 5, 7} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l785_78547


namespace NUMINAMATH_CALUDE_probability_in_tournament_of_26_l785_78567

/-- The probability of two specific participants playing against each other in a tournament. -/
def probability_of_match (n : ℕ) : ℚ :=
  (n - 1) / (n * (n - 1) / 2)

/-- Theorem: In a tournament with 26 participants, the probability of two specific participants
    playing against each other is 1/13. -/
theorem probability_in_tournament_of_26 :
  probability_of_match 26 = 1 / 13 := by
  sorry

#eval probability_of_match 26  -- To check the result

end NUMINAMATH_CALUDE_probability_in_tournament_of_26_l785_78567


namespace NUMINAMATH_CALUDE_perpendicular_condition_l785_78597

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m : ℝ) : Prop :=
  (-m / (2*m - 1)) * (-3 / m) = -1

/-- The first line equation -/
def line1 (m : ℝ) (x y : ℝ) : Prop :=
  m*x + (2*m - 1)*y + 1 = 0

/-- The second line equation -/
def line2 (m : ℝ) (x y : ℝ) : Prop :=
  3*x + m*y + 2 = 0

/-- m = -1 is sufficient but not necessary for the lines to be perpendicular -/
theorem perpendicular_condition (m : ℝ) :
  (m = -1 → are_perpendicular m) ∧
  ¬(are_perpendicular m → m = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l785_78597


namespace NUMINAMATH_CALUDE_walnut_trees_count_l785_78544

/-- The number of walnut trees in the park after planting and removing trees -/
def final_tree_count (initial_trees : ℕ) (planted_group1 : ℕ) (planted_group2 : ℕ) (planted_group3 : ℕ) (removed_trees : ℕ) : ℕ :=
  initial_trees + planted_group1 + planted_group2 + planted_group3 - removed_trees

/-- Theorem stating that the final number of walnut trees in the park is 55 -/
theorem walnut_trees_count : final_tree_count 22 12 15 10 4 = 55 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_count_l785_78544


namespace NUMINAMATH_CALUDE_triangle_problem_l785_78518

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  a = 6 →
  1/2 * a * c * Real.sin B = 6 * Real.sqrt 3 →
  ((B = π/3 ∨ B = 2*π/3) ∧ (b = 2 * Real.sqrt 7 ∨ b = Real.sqrt 76)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l785_78518


namespace NUMINAMATH_CALUDE_sequence_inequality_l785_78549

theorem sequence_inequality (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a 1 + a (n + 1)) / a n > 1 + 1 / n :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l785_78549


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l785_78561

def euler_family_ages : List ℝ := [5, 5, 10, 15, 8, 12, 16]

theorem euler_family_mean_age :
  let ages := euler_family_ages
  let sum_ages := ages.sum
  let num_children := ages.length
  sum_ages / num_children = 10.14 := by
sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l785_78561


namespace NUMINAMATH_CALUDE_sara_apples_l785_78577

theorem sara_apples (total : ℕ) (ali_factor : ℕ) (sara_apples : ℕ) 
  (h1 : total = 80)
  (h2 : ali_factor = 4)
  (h3 : total = sara_apples + ali_factor * sara_apples) :
  sara_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_sara_apples_l785_78577


namespace NUMINAMATH_CALUDE_sin_absolute_value_equality_l785_78539

theorem sin_absolute_value_equality (α : ℝ) : 
  |Real.sin α| = -Real.sin α ↔ ∃ k : ℤ, α ∈ Set.Icc ((2 * k - 1) * Real.pi) (2 * k * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_sin_absolute_value_equality_l785_78539


namespace NUMINAMATH_CALUDE_sum_difference_1500_l785_78508

/-- The sum of the first n odd counting numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The sum of the first n even counting numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even counting numbers
    and the sum of the first n odd counting numbers -/
def sumDifference (n : ℕ) : ℕ := sumEvenNumbers n - sumOddNumbers n

theorem sum_difference_1500 :
  sumDifference 1500 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_1500_l785_78508


namespace NUMINAMATH_CALUDE_sin_arithmetic_sequence_l785_78546

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sin_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.sin (a 2 + a 8) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_arithmetic_sequence_l785_78546


namespace NUMINAMATH_CALUDE_rock_collection_total_l785_78595

theorem rock_collection_total (igneous sedimentary : ℕ) : 
  igneous = sedimentary / 2 →
  (2 : ℕ) * igneous / 3 = 40 →
  igneous + sedimentary = 180 :=
by
  sorry

#check rock_collection_total

end NUMINAMATH_CALUDE_rock_collection_total_l785_78595


namespace NUMINAMATH_CALUDE_divisible_by_six_l785_78574

theorem divisible_by_six (a : ℤ) : 6 ∣ a * (a + 1) * (2 * a + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l785_78574


namespace NUMINAMATH_CALUDE_largest_A_for_divisibility_by_3_l785_78572

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_A_for_divisibility_by_3 :
  ∀ A : ℕ, A ≤ 9 →
    is_divisible_by_3 (3 * 100000 + A * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + 2) →
    A ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_largest_A_for_divisibility_by_3_l785_78572


namespace NUMINAMATH_CALUDE_pencil_distribution_l785_78515

theorem pencil_distribution (total_pencils : ℕ) (pencils_per_row : ℕ) (rows : ℕ) : 
  total_pencils = 12 → 
  pencils_per_row = 4 → 
  total_pencils = rows * pencils_per_row → 
  rows = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l785_78515


namespace NUMINAMATH_CALUDE_problem_solution_l785_78589

theorem problem_solution (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l785_78589


namespace NUMINAMATH_CALUDE_initial_num_pipes_is_three_l785_78530

-- Define the fill time for the initial number of pipes
def initial_fill_time : ℝ := 8

-- Define the fill time for two pipes
def two_pipes_fill_time : ℝ := 12

-- Define the number of pipes we want to prove
def target_num_pipes : ℕ := 3

-- Theorem statement
theorem initial_num_pipes_is_three :
  ∃ (n : ℕ), n > 0 ∧
  (1 : ℝ) / initial_fill_time = (n : ℝ) * ((1 : ℝ) / two_pipes_fill_time / 2) ∧
  n = target_num_pipes :=
sorry

end NUMINAMATH_CALUDE_initial_num_pipes_is_three_l785_78530


namespace NUMINAMATH_CALUDE_vaccine_effective_l785_78532

/-- Represents the contingency table for vaccine effectiveness study -/
structure VaccineStudy where
  total_mice : ℕ
  infected_mice : ℕ
  not_infected_mice : ℕ
  prob_infected_not_vaccinated : ℚ

/-- Calculates the chi-square statistic for the vaccine study -/
def chi_square (study : VaccineStudy) : ℚ :=
  let a := study.not_infected_mice - (study.total_mice / 2 - study.infected_mice * study.prob_infected_not_vaccinated)
  let b := study.not_infected_mice - a
  let c := study.total_mice / 2 - study.infected_mice * study.prob_infected_not_vaccinated
  let d := study.infected_mice - c
  let n := study.total_mice
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 95% confidence in the chi-square test -/
def chi_square_critical : ℚ := 3841 / 1000

/-- Theorem stating that the vaccine is effective with 95% confidence -/
theorem vaccine_effective (study : VaccineStudy) 
  (h1 : study.total_mice = 200)
  (h2 : study.infected_mice = 100)
  (h3 : study.not_infected_mice = 100)
  (h4 : study.prob_infected_not_vaccinated = 3/5) :
  chi_square study > chi_square_critical := by
  sorry

end NUMINAMATH_CALUDE_vaccine_effective_l785_78532


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l785_78505

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 12 = 0 → y = x) ↔ 
  (m = -10 ∨ m = 14) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l785_78505


namespace NUMINAMATH_CALUDE_platform_length_l785_78559

/-- Given a train of length 300 m that crosses a platform in 39 seconds
    and a signal pole in 12 seconds, the length of the platform is 675 m. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 39)
  (h3 : pole_crossing_time = 12) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 675 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l785_78559


namespace NUMINAMATH_CALUDE_half_girls_probability_l785_78527

def n : ℕ := 7
def p : ℚ := 1/2

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem half_girls_probability :
  binomial_probability n 3 p + binomial_probability n 4 p = 35/64 := by
  sorry

end NUMINAMATH_CALUDE_half_girls_probability_l785_78527


namespace NUMINAMATH_CALUDE_remainder_x_plus_2_pow_2022_l785_78580

theorem remainder_x_plus_2_pow_2022 (x : ℤ) :
  (x^3 % (x^2 + x + 1) = 1) →
  ((x + 2)^2022 % (x^2 + x + 1) = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_x_plus_2_pow_2022_l785_78580


namespace NUMINAMATH_CALUDE_intersection_P_complement_M_l785_78516

def U : Set Int := Set.univ

def M : Set Int := {1, 2}

def P : Set Int := {-2, -1, 0, 1, 2}

theorem intersection_P_complement_M :
  P ∩ (U \ M) = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_P_complement_M_l785_78516


namespace NUMINAMATH_CALUDE_expression_evaluation_l785_78531

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/5
  (2*x - 3)^2 - (x + 2*y)*(x - 2*y) - 3*y^2 + 3 = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l785_78531


namespace NUMINAMATH_CALUDE_min_balls_to_draw_l785_78557

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- Represents the maximum number of balls that can be drawn for each color -/
structure MaxDrawnBalls where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee the desired outcome -/
def minBallsToGuarantee : Nat := 57

/-- The threshold for a single color to be guaranteed -/
def singleColorThreshold : Nat := 12

/-- Theorem stating the minimum number of balls to be drawn -/
theorem min_balls_to_draw (initial : BallCounts) (max_drawn : MaxDrawnBalls) : 
  initial.red = 30 ∧ 
  initial.green = 25 ∧ 
  initial.yellow = 20 ∧ 
  initial.blue = 15 ∧ 
  initial.white = 10 ∧ 
  initial.black = 5 ∧
  max_drawn.red < singleColorThreshold ∧
  max_drawn.green < singleColorThreshold ∧ 
  max_drawn.yellow < singleColorThreshold ∧
  max_drawn.blue < singleColorThreshold ∧
  max_drawn.white < singleColorThreshold ∧
  max_drawn.black < singleColorThreshold ∧
  max_drawn.green % 2 = 0 ∧
  max_drawn.white % 2 = 0 ∧
  max_drawn.green ≤ initial.green ∧
  max_drawn.white ≤ initial.white →
  minBallsToGuarantee = 
    max_drawn.red + max_drawn.green + max_drawn.yellow + 
    max_drawn.blue + max_drawn.white + max_drawn.black + 1 :=
by sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_l785_78557


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l785_78517

/-- Given a point P(-2, 1), prove that its symmetric point Q with respect to the x-axis has coordinates (-2, -1) -/
theorem symmetric_point_wrt_x_axis :
  let P : ℝ × ℝ := (-2, 1)
  let Q : ℝ × ℝ := (-2, -1)
  (Q.1 = P.1) ∧ (Q.2 = -P.2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l785_78517


namespace NUMINAMATH_CALUDE_opposite_numbers_expression_l785_78553

theorem opposite_numbers_expression (a b c d : ℤ) : 
  (a + b = 0) →  -- a and b are opposite numbers
  (c = -1) →     -- c is the largest negative integer
  (d = 1) →      -- d is the smallest positive integer
  (a + b) * d + d - c = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_expression_l785_78553


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l785_78599

def is_fibonacci_like (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) = b (n + 1) + b n

theorem fibonacci_like_sequence (b : ℕ → ℕ) :
  is_fibonacci_like b →
  (∀ n m : ℕ, n < m → b n < b m) →
  b 6 = 96 →
  b 7 = 184 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l785_78599


namespace NUMINAMATH_CALUDE_product_as_difference_of_squares_l785_78583

theorem product_as_difference_of_squares (a b : ℝ) : 
  a * b = ((a + b) / 2)^2 - ((a - b) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_as_difference_of_squares_l785_78583


namespace NUMINAMATH_CALUDE_square_side_length_difference_l785_78542

/-- Given four squares with known side length differences, prove that the total difference
    in side length from the largest to the smallest square is the sum of these differences. -/
theorem square_side_length_difference (AB CD FE : ℝ) (hAB : AB = 11) (hCD : CD = 5) (hFE : FE = 13) :
  ∃ (GH : ℝ), GH = AB + CD + FE :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_difference_l785_78542


namespace NUMINAMATH_CALUDE_solve_equation_l785_78500

theorem solve_equation (n : ℚ) : (1 / (2 * n)) + (1 / (4 * n)) = 3 / 12 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l785_78500


namespace NUMINAMATH_CALUDE_password_guess_probability_l785_78512

/-- The number of digits in the password -/
def password_length : ℕ := 6

/-- The number of possible digits for each position -/
def digit_options : ℕ := 10

/-- The number of attempts allowed -/
def max_attempts : ℕ := 2

/-- The probability of guessing the correct last digit in no more than 2 attempts -/
theorem password_guess_probability :
  (1 : ℚ) / digit_options + (digit_options - 1 : ℚ) / digit_options * (1 : ℚ) / (digit_options - 1) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_password_guess_probability_l785_78512


namespace NUMINAMATH_CALUDE_slope_of_line_intersecting_hyperbola_l785_78501

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 3)

-- Define the distance from a point to the focus
def dist_to_focus (x y : ℝ) : ℝ := 2*x - 1

theorem slope_of_line_intersecting_hyperbola (k : ℝ) :
  (∃ A B : ℝ × ℝ,
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    line k A.1 A.2 ∧
    line k B.1 B.2 ∧
    A.1 > 1 ∧
    B.1 > 1 ∧
    dist_to_focus A.1 A.2 + dist_to_focus B.1 B.2 = 16) →
  k = 3 ∨ k = -3 :=
sorry

end NUMINAMATH_CALUDE_slope_of_line_intersecting_hyperbola_l785_78501


namespace NUMINAMATH_CALUDE_sum_divisible_by_three_probability_l785_78506

/-- Given a sequence of positive integers, the probability that the sum of three
    independently and randomly selected elements is divisible by 3 is at least 1/4. -/
theorem sum_divisible_by_three_probability (n : ℕ) (seq : Fin n → ℕ+) :
  ∃ (p q r : ℝ), p + q + r = 1 ∧ p ≥ 0 ∧ q ≥ 0 ∧ r ≥ 0 ∧
  p^3 + q^3 + r^3 + 6*p*q*r ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_three_probability_l785_78506


namespace NUMINAMATH_CALUDE_firewood_collection_l785_78510

theorem firewood_collection (total kimberley ela : ℕ) (h1 : total = 35) (h2 : kimberley = 10) (h3 : ela = 13) :
  total - kimberley - ela = 12 := by
  sorry

end NUMINAMATH_CALUDE_firewood_collection_l785_78510


namespace NUMINAMATH_CALUDE_sum_of_a_equals_2673_l785_78564

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 18 = 0 then 15
  else if n % 18 = 0 ∧ n % 12 = 0 then 16
  else if n % 12 = 0 ∧ n % 15 = 0 then 17
  else 0

theorem sum_of_a_equals_2673 :
  (Finset.range 3000).sum (fun n => a (n + 1)) = 2673 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_equals_2673_l785_78564


namespace NUMINAMATH_CALUDE_smallest_base_for_xyxy_cube_l785_78513

/-- Represents a number in the form xyxy in base b -/
def xyxy_form (x y b : ℕ) : ℕ := x * b^3 + y * b^2 + x * b + y

/-- Checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

/-- The statement to be proved -/
theorem smallest_base_for_xyxy_cube : 
  (∀ x y : ℕ, ¬ is_perfect_cube (xyxy_form x y 10)) →
  (∃ x y : ℕ, is_perfect_cube (xyxy_form x y 7)) ∧
  (∀ b : ℕ, 1 < b → b < 7 → ∀ x y : ℕ, ¬ is_perfect_cube (xyxy_form x y b)) :=
sorry

end NUMINAMATH_CALUDE_smallest_base_for_xyxy_cube_l785_78513


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l785_78545

/-- Given a square with perimeter 160 units divided into 4 rectangles, where each rectangle
    has one side equal to half of the square's side length and the other side equal to the
    full side length of the square, the perimeter of one of these rectangles is 120 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (rectangle_count : ℕ) 
  (h1 : square_perimeter = 160)
  (h2 : rectangle_count = 4)
  (h3 : ∀ r : ℝ, r > 0 → ∃ (s w : ℝ), s = r ∧ w = r / 2 ∧ 
       4 * r = square_perimeter ∧
       rectangle_count * (s * w) = r * r) :
  ∃ (rectangle_perimeter : ℝ), rectangle_perimeter = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l785_78545


namespace NUMINAMATH_CALUDE_custom_mul_result_l785_78569

/-- Custom multiplication operation -/
def custom_mul (a b c : ℚ) (x y : ℚ) : ℚ := a * x + b * y + c

theorem custom_mul_result (a b c : ℚ) :
  custom_mul a b c 1 2 = 9 →
  custom_mul a b c (-3) 3 = 6 →
  custom_mul a b c 0 1 = 2 →
  custom_mul a b c (-2) 5 = 18 := by
sorry

end NUMINAMATH_CALUDE_custom_mul_result_l785_78569


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l785_78573

theorem trigonometric_equation_solution 
  (a b c : ℝ) 
  (h : a ≠ 0 ∨ b ≠ 0) :
  ∃ (n : ℤ), 
    ∀ (x : ℝ), 
      a * Real.sin x + b * Real.cos x = c → 
        x = Real.arctan (b / a) + n * Real.pi :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l785_78573


namespace NUMINAMATH_CALUDE_complex_equation_solution_l785_78584

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (1 + 2 * Complex.I) / (a + b * Complex.I) = 1 + Complex.I →
  a = 3/2 ∧ b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l785_78584


namespace NUMINAMATH_CALUDE_coeff_x_squared_in_expansion_l785_78576

/-- The coefficient of x^2 in the expansion of (1+2x)^6 is 60 -/
theorem coeff_x_squared_in_expansion : 
  (Finset.range 7).sum (fun k => (Nat.choose 6 k) * (1^(6-k)) * ((2:ℕ)^k)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_squared_in_expansion_l785_78576


namespace NUMINAMATH_CALUDE_midpoint_triangle_area_l785_78548

/-- The area of a triangle formed by midpoints in a square --/
theorem midpoint_triangle_area (s : ℝ) (h : s = 12) :
  let square_area := s^2
  let midpoint_triangle_area := s^2 / 8
  midpoint_triangle_area = 18 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_triangle_area_l785_78548


namespace NUMINAMATH_CALUDE_friday_production_to_meet_target_l785_78556

/-- The number of toys that need to be produced on Friday to meet the weekly target -/
def friday_production (weekly_target : ℕ) (mon_to_wed_daily : ℕ) (thursday : ℕ) : ℕ :=
  weekly_target - (3 * mon_to_wed_daily + thursday)

/-- Theorem stating the required Friday production to meet the weekly target -/
theorem friday_production_to_meet_target :
  friday_production 6500 1200 800 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_friday_production_to_meet_target_l785_78556


namespace NUMINAMATH_CALUDE_cubic_divisibility_l785_78594

theorem cubic_divisibility (t : ℤ) : (((125 * t - 12) ^ 3 + 2 * (125 * t - 12) + 2) % 125 = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_divisibility_l785_78594


namespace NUMINAMATH_CALUDE_arithmetic_sequence_of_primes_l785_78535

theorem arithmetic_sequence_of_primes : ∃ (p q r : ℕ), 
  (Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r) ∧ 
  (p = 127 ∧ q = 3697 ∧ r = 5527) ∧
  (∃ (d : ℕ), 
    q * (q + 1) - p * (p + 1) = d ∧
    r * (r + 1) - q * (q + 1) = d ∧
    p * (p + 1) < q * (q + 1) ∧ q * (q + 1) < r * (r + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_of_primes_l785_78535


namespace NUMINAMATH_CALUDE_new_rectangle_area_greater_than_square_l785_78541

theorem new_rectangle_area_greater_than_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let new_base := 2 * (a + b)
  let new_height := (2 * b + a) / 3
  let square_side := a + b
  new_base * new_height > square_side * square_side :=
by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_greater_than_square_l785_78541


namespace NUMINAMATH_CALUDE_negative_five_greater_than_negative_seventeen_l785_78582

theorem negative_five_greater_than_negative_seventeen : -5 > -17 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_greater_than_negative_seventeen_l785_78582


namespace NUMINAMATH_CALUDE_lawn_care_supplies_l785_78579

theorem lawn_care_supplies (blade_cost : ℕ) (string_cost : ℕ) (total_cost : ℕ) (num_blades : ℕ) :
  blade_cost = 8 →
  string_cost = 7 →
  total_cost = 39 →
  blade_cost * num_blades + string_cost = total_cost →
  num_blades = 4 := by
sorry

end NUMINAMATH_CALUDE_lawn_care_supplies_l785_78579


namespace NUMINAMATH_CALUDE_complex_multiplication_l785_78520

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (1 - i)^2 * i = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l785_78520


namespace NUMINAMATH_CALUDE_class_size_l785_78526

theorem class_size (n : ℕ) 
  (h1 : n ≤ 30) 
  (h2 : ∃ d l : ℕ, d + l + 2 = n ∧ 4 * d = n - d - 1 ∧ l = (n - l - 1) / 3) :
  n = 21 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l785_78526


namespace NUMINAMATH_CALUDE_carlas_marbles_l785_78511

/-- Given that Carla had some marbles and bought more, prove how many she has now. -/
theorem carlas_marbles (initial : ℝ) (bought : ℝ) (total : ℝ) 
  (h1 : initial = 187.0) 
  (h2 : bought = 134.0) 
  (h3 : total = initial + bought) : 
  total = 321.0 := by
  sorry


end NUMINAMATH_CALUDE_carlas_marbles_l785_78511


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_batch_l785_78519

/-- Represents the number of units drawn from each batch in a stratified sampling -/
structure BatchSampling where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given BatchSampling forms an arithmetic sequence -/
def is_arithmetic_sequence (s : BatchSampling) : Prop :=
  s.c - s.b = s.b - s.a

/-- The theorem stating that in a stratified sampling of 60 units from three batches
    forming an arithmetic sequence, the number of units drawn from the middle batch is 20 -/
theorem stratified_sampling_middle_batch :
  ∀ s : BatchSampling,
    is_arithmetic_sequence s →
    s.a + s.b + s.c = 60 →
    s.b = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_batch_l785_78519


namespace NUMINAMATH_CALUDE_sqrt_sqrt_two_power_ten_l785_78590

theorem sqrt_sqrt_two_power_ten : (Real.sqrt ((Real.sqrt 2) ^ 4)) ^ 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sqrt_two_power_ten_l785_78590


namespace NUMINAMATH_CALUDE_grid_paths_7x6_l785_78551

/-- The number of paths in a grid from (0,0) to (m,n) where each step is either right or up -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The dimensions of our grid -/
def gridWidth : ℕ := 7
def gridHeight : ℕ := 6

/-- The total number of steps in our grid -/
def totalSteps : ℕ := gridWidth + gridHeight

theorem grid_paths_7x6 : gridPaths gridWidth gridHeight = 1716 := by sorry

end NUMINAMATH_CALUDE_grid_paths_7x6_l785_78551


namespace NUMINAMATH_CALUDE_squirrel_acorns_l785_78578

theorem squirrel_acorns (num_squirrels : ℕ) (total_acorns : ℕ) (additional_acorns : ℕ) :
  num_squirrels = 5 →
  total_acorns = 575 →
  additional_acorns = 15 →
  ∃ (acorns_per_squirrel : ℕ),
    acorns_per_squirrel = 130 ∧
    num_squirrels * (acorns_per_squirrel - additional_acorns) = total_acorns :=
by sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l785_78578


namespace NUMINAMATH_CALUDE_triangle_properties_l785_78529

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  b = a * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin A →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  ((1/4) * a^2 + (1/4) * b^2 - (1/4) * c^2) * 2 / a = 2 →
  -- Conclusions
  A = π/3 ∧ b = Real.sqrt 2 ∧ c = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l785_78529


namespace NUMINAMATH_CALUDE_product_closest_to_2400_l785_78593

def options : List ℝ := [210, 240, 2100, 2400, 24000]

theorem product_closest_to_2400 : 
  let product := 0.000315 * 7928564
  ∀ x ∈ options, x ≠ 2400 → |product - 2400| < |product - x| := by
  sorry

end NUMINAMATH_CALUDE_product_closest_to_2400_l785_78593


namespace NUMINAMATH_CALUDE_abs_neg_2023_l785_78543

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l785_78543


namespace NUMINAMATH_CALUDE_max_product_under_constraints_l785_78592

theorem max_product_under_constraints (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : 10 * x + 15 * y = 150) (h2 : x^2 + y^2 ≤ 100) :
  x * y ≤ 37.5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
  10 * x₀ + 15 * y₀ = 150 ∧ x₀^2 + y₀^2 ≤ 100 ∧ x₀ * y₀ = 37.5 :=
sorry

end NUMINAMATH_CALUDE_max_product_under_constraints_l785_78592


namespace NUMINAMATH_CALUDE_max_x_value_l785_78502

/-- Represents the linear relationship between x and y --/
def linear_relation (x y : ℝ) : Prop := y = x - 5

/-- The maximum forecast value for y --/
def max_y : ℝ := 10

/-- Theorem stating the maximum value of x given the conditions --/
theorem max_x_value (h : linear_relation max_y max_x) : max_x = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l785_78502


namespace NUMINAMATH_CALUDE_decimal_to_percentage_example_l785_78575

/-- Converts a decimal fraction to a percentage -/
def decimal_to_percentage (d : ℝ) : ℝ := d * 100

/-- The decimal fraction we're working with -/
def given_decimal : ℝ := 0.01

/-- Theorem stating that converting 0.01 to a percentage results in 1 -/
theorem decimal_to_percentage_example :
  decimal_to_percentage given_decimal = 1 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_example_l785_78575


namespace NUMINAMATH_CALUDE_pigeonhole_principle_buttons_l785_78522

theorem pigeonhole_principle_buttons : ∀ (r w b : ℕ),
  r ≥ 3 ∧ w ≥ 3 ∧ b ≥ 3 →
  ∀ n : ℕ, n ≥ 7 →
  ∀ f : Fin n → Fin 3,
  ∃ c : Fin 3, ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
  f i = c ∧ f j = c ∧ f k = c :=
by
  sorry

#check pigeonhole_principle_buttons

end NUMINAMATH_CALUDE_pigeonhole_principle_buttons_l785_78522


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l785_78503

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8| + 3

theorem sum_of_max_min_g : 
  ∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → g x ≤ max) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ g x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → min ≤ g x) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ g x = min) ∧
    max + min = 18 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l785_78503


namespace NUMINAMATH_CALUDE_fitness_center_member_ratio_l785_78521

theorem fitness_center_member_ratio 
  (f : ℕ) (m : ℕ) -- f: number of female members, m: number of male members
  (h1 : (35 * f + 30 * m) / (f + m) = 32) : -- average age of all members is 32
  f / m = 2 / 3 := by
sorry


end NUMINAMATH_CALUDE_fitness_center_member_ratio_l785_78521


namespace NUMINAMATH_CALUDE_cab_driver_income_l785_78524

theorem cab_driver_income (income : List ℝ) (average : ℝ) : 
  income.length = 5 →
  income[0]! = 600 →
  income[1]! = 250 →
  income[2]! = 450 →
  income[4]! = 800 →
  average = (income.sum / income.length) →
  average = 500 →
  income[3]! = 400 := by
sorry

end NUMINAMATH_CALUDE_cab_driver_income_l785_78524


namespace NUMINAMATH_CALUDE_circle_equation_theorem_l785_78581

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the coefficients of a circle equation -/
structure CircleCoefficients where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Checks if a point lies on a circle with given coefficients -/
def pointOnCircle (p : Point) (c : CircleCoefficients) : Prop :=
  p.x^2 + p.y^2 + c.D * p.x + c.E * p.y + c.F = 0

/-- The theorem stating that the given equation represents the circle passing through the specified points -/
theorem circle_equation_theorem (p1 p2 p3 : Point) : 
  p1 = ⟨0, 0⟩ → 
  p2 = ⟨4, 0⟩ → 
  p3 = ⟨-1, 1⟩ → 
  ∃ (c : CircleCoefficients), 
    c.D = -4 ∧ c.E = -6 ∧ c.F = 0 ∧ 
    pointOnCircle p1 c ∧ 
    pointOnCircle p2 c ∧ 
    pointOnCircle p3 c :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_theorem_l785_78581


namespace NUMINAMATH_CALUDE_subtraction_inequality_l785_78504

theorem subtraction_inequality (a b c : ℝ) (h : a > b) : c - a < c - b := by
  sorry

end NUMINAMATH_CALUDE_subtraction_inequality_l785_78504


namespace NUMINAMATH_CALUDE_h_composition_equals_902_l785_78585

/-- The function h as defined in the problem -/
def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

/-- Theorem stating that h(h(2)) = 902 -/
theorem h_composition_equals_902 : h (h 2) = 902 := by
  sorry

end NUMINAMATH_CALUDE_h_composition_equals_902_l785_78585


namespace NUMINAMATH_CALUDE_complex_trig_identity_l785_78563

theorem complex_trig_identity (θ : Real) (h : π < θ ∧ θ < (3 * π) / 2) :
  Real.sqrt ((1 / 2) + (1 / 2) * Real.sqrt ((1 / 2) + (1 / 2) * Real.cos (2 * θ))) - 
  Real.sqrt (1 - Real.sin θ) = Real.cos (θ / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_trig_identity_l785_78563


namespace NUMINAMATH_CALUDE_infinite_solutions_cube_equation_l785_78566

theorem infinite_solutions_cube_equation :
  ∀ n : ℕ, ∃ x y z : ℤ, 
    x^2 + y^2 + z^2 = x^3 + y^3 + z^3 ∧
    (∀ m : ℕ, m < n → 
      ∃ x' y' z' : ℤ, 
        x'^2 + y'^2 + z'^2 = x'^3 + y'^3 + z'^3 ∧
        (x', y', z') ≠ (x, y, z)) :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_cube_equation_l785_78566


namespace NUMINAMATH_CALUDE_no_distinct_naturals_satisfying_equation_l785_78540

theorem no_distinct_naturals_satisfying_equation :
  ¬ ∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (2 * a + Nat.lcm b c = 2 * b + Nat.lcm a c) ∧
    (2 * b + Nat.lcm a c = 2 * c + Nat.lcm a b) :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_naturals_satisfying_equation_l785_78540


namespace NUMINAMATH_CALUDE_point_on_line_l785_78536

/-- Given a line passing through points (0, 3) and (-8, 0),
    if the point (t, 7) lies on this line, then t = 32/3 -/
theorem point_on_line (t : ℚ) :
  (∀ (x y : ℚ), (y - 3) / (x - 0) = (0 - 3) / (-8 - 0) →
    (7 - 3) / (t - 0) = (0 - 3) / (-8 - 0)) →
  t = 32 / 3 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l785_78536


namespace NUMINAMATH_CALUDE_bounded_sequence_from_constrained_function_l785_78596

def is_bounded_sequence (a : ℕ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ n : ℕ, |a n| ≤ M

theorem bounded_sequence_from_constrained_function
  (f : ℝ → ℝ)
  (hf_diff : Differentiable ℝ f)
  (hf_cont : Continuous (deriv f))
  (hf_bound : ∀ x : ℝ, 0 ≤ |deriv f x| ∧ |deriv f x| ≤ (1 : ℝ) / 2)
  (a : ℕ → ℝ)
  (ha_init : a 1 = 1)
  (ha_rec : ∀ n : ℕ, a (n + 1) = f (a n)) :
  is_bounded_sequence a :=
by
  sorry

end NUMINAMATH_CALUDE_bounded_sequence_from_constrained_function_l785_78596


namespace NUMINAMATH_CALUDE_yellow_flags_count_l785_78562

/-- Represents the number of yellow flags in a cycle -/
def yellow_per_cycle : ℕ := 2

/-- Represents the length of the repeating cycle -/
def cycle_length : ℕ := 5

/-- Represents the total number of flags we're considering -/
def total_flags : ℕ := 200

/-- Theorem: The number of yellow flags in the first 200 flags is 80 -/
theorem yellow_flags_count : 
  (total_flags / cycle_length) * yellow_per_cycle = 80 := by
sorry

end NUMINAMATH_CALUDE_yellow_flags_count_l785_78562


namespace NUMINAMATH_CALUDE_simplify_expression_l785_78534

theorem simplify_expression (k : ℝ) (h : k ≠ 0) :
  ∃ (a b c : ℤ), (8 * k + 3 + 6 * k^2) + (5 * k^2 + 4 * k + 7) = a * k^2 + b * k + c ∧ a + b + c = 33 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l785_78534


namespace NUMINAMATH_CALUDE_angle_multiplication_l785_78523

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define multiplication of an angle by a natural number
def multiplyAngle (a : Angle) (n : ℕ) : Angle :=
  let totalMinutes := a.degrees * 60 + a.minutes
  let newTotalMinutes := totalMinutes * n
  ⟨newTotalMinutes / 60, newTotalMinutes % 60⟩

-- Define equality for angles
def angleEq (a b : Angle) : Prop :=
  a.degrees * 60 + a.minutes = b.degrees * 60 + b.minutes

-- Theorem statement
theorem angle_multiplication :
  angleEq (multiplyAngle ⟨21, 17⟩ 5) ⟨106, 25⟩ := by
  sorry

end NUMINAMATH_CALUDE_angle_multiplication_l785_78523


namespace NUMINAMATH_CALUDE_chord_length_theorem_l785_78588

/-- In a right triangle ABC with inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of leg AB -/
  a : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- a and r are positive -/
  a_pos : 0 < a
  r_pos : 0 < r

/-- The chord length theorem -/
theorem chord_length_theorem (t : RightTriangleWithInscribedCircle) :
  ∃ (chord_length : ℝ),
    chord_length = (2 * t.a * t.r) / Real.sqrt (t.a^2 + t.r^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l785_78588
