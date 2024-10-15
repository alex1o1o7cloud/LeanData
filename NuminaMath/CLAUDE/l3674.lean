import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_max_value_l3674_367477

theorem quadratic_inequality_max_value (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) →
  (∃ M, M = -4 ∧ ∀ a' b' c', (∀ x, a'*x^2 + b'*x + c' > 0 ↔ -1 < x ∧ x < 2) → b' - c' + 4/a' ≤ M) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_max_value_l3674_367477


namespace NUMINAMATH_CALUDE_emily_spent_12_dollars_l3674_367469

def flower_cost : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

theorem emily_spent_12_dollars :
  flower_cost * (roses_bought + daisies_bought) = 12 := by
  sorry

end NUMINAMATH_CALUDE_emily_spent_12_dollars_l3674_367469


namespace NUMINAMATH_CALUDE_ten_row_triangle_count_l3674_367457

/-- Calculates the sum of the first n natural numbers. -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of rods in a triangle with n rows. -/
def rods_count (n : ℕ) : ℕ := 3 * triangular_number n

/-- Calculates the number of connectors in a triangle with n rows of rods. -/
def connectors_count (n : ℕ) : ℕ := triangular_number (n + 1)

/-- The total number of rods and connectors in a triangle with n rows of rods. -/
def total_count (n : ℕ) : ℕ := rods_count n + connectors_count n

theorem ten_row_triangle_count :
  total_count 10 = 231 := by sorry

end NUMINAMATH_CALUDE_ten_row_triangle_count_l3674_367457


namespace NUMINAMATH_CALUDE_distance_between_centers_l3674_367452

/-- Given a triangle with sides 6, 8, and 10, the distance between the centers
    of its inscribed and circumscribed circles is √13. -/
theorem distance_between_centers (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inradius := area / s
  let circumradius := (a * b * c) / (4 * area)
  Real.sqrt (circumradius^2 + inradius^2 - 2 * circumradius * inradius * Real.cos (π / 2)) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_centers_l3674_367452


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l3674_367439

theorem quadratic_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l3674_367439


namespace NUMINAMATH_CALUDE_tangent_line_condition_l3674_367488

/-- Given a function f(x) = x³ + ax², prove that if the tangent line
    at point (x₀, f(x₀)) has equation x + y = 0, then x₀ = ±1 and f(x₀) = -x₀ -/
theorem tangent_line_condition (a : ℝ) :
  ∃ x₀ : ℝ, (x₀ = 1 ∨ x₀ = -1) ∧
  let f := λ x : ℝ => x^3 + a*x^2
  let f' := λ x : ℝ => 3*x^2 + 2*a*x
  f' x₀ = -1 ∧ x₀ + f x₀ = 0 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_condition_l3674_367488


namespace NUMINAMATH_CALUDE_subset_coloring_existence_l3674_367429

/-- The coloring function type -/
def ColoringFunction (α : Type*) := Set α → Bool

/-- Theorem statement -/
theorem subset_coloring_existence
  (S : Type*)
  [Fintype S]
  (h_card : Fintype.card S = 2002)
  (N : ℕ)
  (h_N : N ≤ 2^2002) :
  ∃ (f : ColoringFunction S),
    (∀ A B : Set S, f A ∧ f B → f (A ∪ B)) ∧
    (∀ A B : Set S, ¬f A ∧ ¬f B → ¬f (A ∪ B)) ∧
    (Fintype.card {A : Set S | f A} = N) :=
by sorry

end NUMINAMATH_CALUDE_subset_coloring_existence_l3674_367429


namespace NUMINAMATH_CALUDE_bee_count_l3674_367483

theorem bee_count (initial_bees additional_bees : ℕ) : 
  initial_bees = 16 → additional_bees = 10 → initial_bees + additional_bees = 26 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l3674_367483


namespace NUMINAMATH_CALUDE_unique_pairs_from_ten_l3674_367454

theorem unique_pairs_from_ten (n : ℕ) (h : n = 10) : n * (n - 1) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_unique_pairs_from_ten_l3674_367454


namespace NUMINAMATH_CALUDE_grandfather_grandson_ages_l3674_367463

def isComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem grandfather_grandson_ages :
  ∀ (grandfather grandson : ℕ),
    isComposite grandfather →
    isComposite grandson →
    (grandfather + 1) * (grandson + 1) = 1610 →
    grandfather = 69 ∧ grandson = 22 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_grandson_ages_l3674_367463


namespace NUMINAMATH_CALUDE_greatest_gcd_6Tn_n_minus_1_l3674_367400

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem greatest_gcd_6Tn_n_minus_1 :
  (∀ n : ℕ+, Nat.gcd (6 * triangular_number n) (n - 1) ≤ 3) ∧
  (∃ n : ℕ+, Nat.gcd (6 * triangular_number n) (n - 1) = 3) :=
by sorry

end NUMINAMATH_CALUDE_greatest_gcd_6Tn_n_minus_1_l3674_367400


namespace NUMINAMATH_CALUDE_closest_to_M_div_N_l3674_367433

-- Define the state space complexity of Go
def M : ℝ := 3^361

-- Define the number of atoms in the observable universe
def N : ℝ := 10^80

-- Define the options
def options : List ℝ := [10^33, 10^53, 10^73, 10^93]

-- Theorem statement
theorem closest_to_M_div_N :
  let ratio := M / N
  (∀ x ∈ options, |ratio - 10^93| ≤ |ratio - x|) ∧ (10^93 ∈ options) :=
by sorry

end NUMINAMATH_CALUDE_closest_to_M_div_N_l3674_367433


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l3674_367417

theorem shortest_side_of_right_triangle (a b c : ℝ) : 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → min a (min b c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l3674_367417


namespace NUMINAMATH_CALUDE_probability_of_winning_l3674_367443

def total_balls : ℕ := 10
def red_balls : ℕ := 5
def white_balls : ℕ := 5
def drawn_balls : ℕ := 5

def winning_outcomes : ℕ := Nat.choose red_balls 4 * Nat.choose white_balls 1 + Nat.choose red_balls 5

def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

theorem probability_of_winning :
  (winning_outcomes : ℚ) / total_outcomes = 26 / 252 :=
sorry

end NUMINAMATH_CALUDE_probability_of_winning_l3674_367443


namespace NUMINAMATH_CALUDE_line_y_coordinate_l3674_367403

/-- 
Given a line that:
- passes through a point (3, y)
- has a slope of 2
- has an x-intercept of 1

Prove that the y-coordinate of the point (3, y) is 4.
-/
theorem line_y_coordinate (y : ℝ) : 
  (∃ (m b : ℝ), m = 2 ∧ b = -2 ∧ 
    (∀ x : ℝ, y = m * (3 - x) + (m * x + b)) ∧
    (0 = m * 1 + b)) → 
  y = 4 :=
by sorry

end NUMINAMATH_CALUDE_line_y_coordinate_l3674_367403


namespace NUMINAMATH_CALUDE_john_climbed_45_feet_l3674_367445

/-- Calculates the total distance climbed given the number of steps in three staircases and the height of each step -/
def total_distance_climbed (first_staircase : ℕ) (step_height : ℝ) : ℝ :=
  let second_staircase := 2 * first_staircase
  let third_staircase := second_staircase - 10
  let total_steps := first_staircase + second_staircase + third_staircase
  total_steps * step_height

/-- Theorem stating that John climbed 45 feet given the problem conditions -/
theorem john_climbed_45_feet :
  total_distance_climbed 20 0.5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_john_climbed_45_feet_l3674_367445


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3674_367449

-- Define the original number
def original_number : ℕ := 150000000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.5 * (10 ^ 11)

-- Theorem to prove the equality
theorem original_equals_scientific : (original_number : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3674_367449


namespace NUMINAMATH_CALUDE_circle_and_five_lines_max_regions_circle_divides_plane_two_parts_circle_and_one_line_max_four_parts_circle_and_two_lines_max_eight_parts_l3674_367471

/-- The maximum number of regions into which n lines can divide a plane -/
def max_regions_lines (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The maximum number of additional regions created when k lines intersect a circle -/
def max_additional_regions (k : ℕ) : ℕ := k * 2

/-- The maximum number of regions into which a plane can be divided by 1 circle and n lines -/
def max_regions_circle_and_lines (n : ℕ) : ℕ :=
  max_regions_lines n + max_additional_regions n

theorem circle_and_five_lines_max_regions :
  max_regions_circle_and_lines 5 = 26 :=
by sorry

theorem circle_divides_plane_two_parts :
  max_regions_circle_and_lines 0 = 2 :=
by sorry

theorem circle_and_one_line_max_four_parts :
  max_regions_circle_and_lines 1 = 4 :=
by sorry

theorem circle_and_two_lines_max_eight_parts :
  max_regions_circle_and_lines 2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_and_five_lines_max_regions_circle_divides_plane_two_parts_circle_and_one_line_max_four_parts_circle_and_two_lines_max_eight_parts_l3674_367471


namespace NUMINAMATH_CALUDE_fruit_shop_problem_l3674_367478

/-- The price per kilogram of apples in yuan -/
def apple_price : ℝ := 8

/-- The price per kilogram of pears in yuan -/
def pear_price : ℝ := 6

/-- The maximum number of kilograms of apples that can be purchased -/
def max_apples : ℝ := 5

theorem fruit_shop_problem :
  (1 * apple_price + 3 * pear_price = 26) ∧
  (2 * apple_price + 1 * pear_price = 22) ∧
  (∀ x y : ℝ, x + y = 15 → x * apple_price + y * pear_price ≤ 100 → x ≤ max_apples) :=
by sorry

end NUMINAMATH_CALUDE_fruit_shop_problem_l3674_367478


namespace NUMINAMATH_CALUDE_system_solution_l3674_367447

theorem system_solution : ∃ (s t : ℝ), 
  (7 * s + 3 * t = 102) ∧ 
  (s = (t - 3)^2) ∧ 
  (abs (t - 6.44) < 0.01) ∧ 
  (abs (s - 11.83) < 0.01) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3674_367447


namespace NUMINAMATH_CALUDE_complement_of_supplement_30_l3674_367409

/-- The supplement of an angle in degrees -/
def supplement (angle : ℝ) : ℝ := 180 - angle

/-- The complement of an angle in degrees -/
def complement (angle : ℝ) : ℝ := 90 - angle

/-- The degree measure of the complement of the supplement of a 30-degree angle is 60° -/
theorem complement_of_supplement_30 : complement (supplement 30) = 60 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_supplement_30_l3674_367409


namespace NUMINAMATH_CALUDE_parabola_properties_l3674_367412

/-- A parabola with vertex at the origin, focus on the x-axis, and passing through (2, 2) -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 2*x

theorem parabola_properties :
  (parabola_equation 0 0) ∧ 
  (∃ p : ℝ, p > 0 ∧ parabola_equation p 0) ∧
  (parabola_equation 2 2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3674_367412


namespace NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l3674_367494

/-- Given a point M with coordinates (3,-4), its symmetric point M' about the x-axis has coordinates (3,4). -/
theorem symmetric_point_about_x_axis :
  let M : ℝ × ℝ := (3, -4)
  let M' : ℝ × ℝ := (M.1, -M.2)
  M' = (3, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l3674_367494


namespace NUMINAMATH_CALUDE_meaningful_sqrt_range_l3674_367405

theorem meaningful_sqrt_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x - 1) ↔ x ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_range_l3674_367405


namespace NUMINAMATH_CALUDE_genetic_material_not_equal_l3674_367467

/-- Represents a cell involved in fertilization -/
structure Cell where
  nucleus : Bool
  cytoplasm : Nat

/-- Represents the process of fertilization -/
def fertilization (sperm : Cell) (egg : Cell) : Prop :=
  sperm.nucleus ∧ egg.nucleus ∧ sperm.cytoplasm < egg.cytoplasm

/-- Represents the zygote formed after fertilization -/
def zygote (sperm : Cell) (egg : Cell) : Prop :=
  fertilization sperm egg

/-- Theorem stating that genetic material in the zygote does not come equally from both parents -/
theorem genetic_material_not_equal (sperm egg : Cell) 
  (h_sperm : sperm.nucleus ∧ sperm.cytoplasm = 0)
  (h_egg : egg.nucleus ∧ egg.cytoplasm > 0)
  (h_zygote : zygote sperm egg) :
  ¬(∃ (x : Nat), x > 0 ∧ x = sperm.cytoplasm ∧ x = egg.cytoplasm) := by
  sorry


end NUMINAMATH_CALUDE_genetic_material_not_equal_l3674_367467


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l3674_367458

/-- Given a restaurant that made hamburgers and served some, 
    calculate the number of hamburgers left over. -/
theorem hamburgers_left_over 
  (total_made : ℕ) 
  (served : ℕ) 
  (h1 : total_made = 25) 
  (h2 : served = 11) : 
  total_made - served = 14 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_l3674_367458


namespace NUMINAMATH_CALUDE_eggs_in_box_l3674_367414

/-- The number of eggs in the box after adding more eggs -/
def total_eggs (initial : Float) (added : Float) : Float :=
  initial + added

/-- Theorem stating that adding 5.0 eggs to 47.0 eggs results in 52.0 eggs -/
theorem eggs_in_box : total_eggs 47.0 5.0 = 52.0 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_box_l3674_367414


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l3674_367479

theorem trigonometric_expression_equals_one : 
  (Real.sin (15 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (5 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l3674_367479


namespace NUMINAMATH_CALUDE_proportion_solution_l3674_367436

theorem proportion_solution (x : ℝ) : (0.60 / x = 6 / 4) → x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3674_367436


namespace NUMINAMATH_CALUDE_five_cubed_sum_equals_five_to_fourth_l3674_367464

theorem five_cubed_sum_equals_five_to_fourth : 5^3 + 5^3 + 5^3 + 5^3 + 5^3 = 5^4 := by
  sorry

end NUMINAMATH_CALUDE_five_cubed_sum_equals_five_to_fourth_l3674_367464


namespace NUMINAMATH_CALUDE_impossibleToKnowDreamIfDiedAsleep_l3674_367493

/-- Represents a person's state --/
inductive PersonState
  | Awake
  | Asleep
  | Dead

/-- Represents a dream --/
structure Dream where
  content : String

/-- Represents a person --/
structure Person where
  state : PersonState
  currentDream : Option Dream

/-- Represents the ability to share dream content --/
def canShareDream (p : Person) : Prop :=
  p.state = PersonState.Awake ∧ p.currentDream.isSome

/-- Represents the event of a person dying while asleep --/
def diedWhileAsleep (p : Person) : Prop :=
  p.state = PersonState.Dead ∧ p.currentDream.isSome

/-- Theorem: If a person died while asleep, it's impossible for others to know their exact dream --/
theorem impossibleToKnowDreamIfDiedAsleep (p : Person) :
  diedWhileAsleep p → ¬(canShareDream p) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibleToKnowDreamIfDiedAsleep_l3674_367493


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3674_367466

theorem complex_fraction_simplification :
  (2 + 4 * Complex.I) / ((1 + Complex.I)^2) = 2 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3674_367466


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3674_367489

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_one : 
  (deriv f) 1 = 24 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3674_367489


namespace NUMINAMATH_CALUDE_f_even_and_periodic_l3674_367428

-- Define a non-constant function f on ℝ
variable (f : ℝ → ℝ)

-- Condition: f is non-constant
axiom f_non_constant : ∃ x y, f x ≠ f y

-- Condition: f(10 + x) is an even function
axiom f_10_even : ∀ x, f (10 + x) = f (10 - x)

-- Condition: f(5 - x) = f(5 + x)
axiom f_5_symmetric : ∀ x, f (5 - x) = f (5 + x)

-- Theorem to prove
theorem f_even_and_periodic :
  (∀ x, f x = f (-x)) ∧ (∃ T > 0, ∀ x, f (x + T) = f x) :=
sorry

end NUMINAMATH_CALUDE_f_even_and_periodic_l3674_367428


namespace NUMINAMATH_CALUDE_diamond_commutative_eq_four_lines_l3674_367472

/-- Diamond operation -/
def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^3 * b - a * b^3

/-- The set of points (x, y) where x ◇ y = y ◇ x -/
def diamond_commutative_set : Set (ℝ × ℝ) :=
  {p | diamond p.1 p.2 = diamond p.2 p.1}

/-- The union of four lines: x = 0, y = 0, y = x, and y = -x -/
def four_lines : Set (ℝ × ℝ) :=
  {p | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

theorem diamond_commutative_eq_four_lines :
  diamond_commutative_set = four_lines := by sorry

end NUMINAMATH_CALUDE_diamond_commutative_eq_four_lines_l3674_367472


namespace NUMINAMATH_CALUDE_square_side_length_l3674_367408

theorem square_side_length (d : ℝ) (h : d = 24) :
  ∃ s : ℝ, s > 0 ∧ s * s + s * s = d * d ∧ s = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3674_367408


namespace NUMINAMATH_CALUDE_base_for_five_digit_100_l3674_367498

theorem base_for_five_digit_100 :
  ∃! (b : ℕ), b > 1 ∧ b^4 ≤ 100 ∧ 100 < b^5 :=
by sorry

end NUMINAMATH_CALUDE_base_for_five_digit_100_l3674_367498


namespace NUMINAMATH_CALUDE_circle_radius_l3674_367474

/-- The radius of a circle described by the equation x² + y² + 12 = 10x - 6y is √22. -/
theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 + 12 = 10*x - 6*y) → ∃ (center_x center_y : ℝ), 
    ∀ (point_x point_y : ℝ), 
      (point_x - center_x)^2 + (point_y - center_y)^2 = 22 := by
sorry


end NUMINAMATH_CALUDE_circle_radius_l3674_367474


namespace NUMINAMATH_CALUDE_johns_remaining_money_l3674_367473

/-- Calculates the remaining money after John's expenses -/
def remaining_money (initial : ℚ) (sweets : ℚ) (friend_gift : ℚ) (num_friends : ℕ) : ℚ :=
  initial - sweets - (friend_gift * num_friends)

/-- Theorem stating that John will be left with $2.45 -/
theorem johns_remaining_money :
  remaining_money 10.10 3.25 2.20 2 = 2.45 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l3674_367473


namespace NUMINAMATH_CALUDE_journey_time_ratio_l3674_367456

/-- Proves that the ratio of return journey time to initial journey time is 3:2 
    given specific speed conditions -/
theorem journey_time_ratio 
  (initial_speed : ℝ) 
  (average_speed : ℝ) 
  (h1 : initial_speed = 51)
  (h2 : average_speed = 34) :
  (1 / average_speed) / (1 / initial_speed) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_ratio_l3674_367456


namespace NUMINAMATH_CALUDE_total_charge_2_hours_l3674_367496

/-- Represents the pricing structure and total charge calculation for a psychologist's therapy sessions. -/
structure TherapyPricing where
  /-- The charge for the first hour of therapy -/
  first_hour_charge : ℕ
  /-- The charge for each additional hour of therapy -/
  additional_hour_charge : ℕ
  /-- The difference between the first hour charge and additional hour charge -/
  charge_difference : first_hour_charge = additional_hour_charge + 25
  /-- The total charge for 5 hours of therapy -/
  total_charge_5_hours : first_hour_charge + 4 * additional_hour_charge = 250

/-- Theorem stating that the total charge for 2 hours of therapy is $115 -/
theorem total_charge_2_hours (p : TherapyPricing) : 
  p.first_hour_charge + p.additional_hour_charge = 115 := by
  sorry


end NUMINAMATH_CALUDE_total_charge_2_hours_l3674_367496


namespace NUMINAMATH_CALUDE_velocity_maximum_at_lowest_point_l3674_367450

/-- Represents a point on the roller coaster track -/
structure TrackPoint where
  height : ℝ
  velocity : ℝ

/-- Represents the roller coaster system -/
structure RollerCoaster where
  points : List TrackPoint
  initial_velocity : ℝ
  g : ℝ  -- Acceleration due to gravity

/-- The total mechanical energy of the system -/
def total_energy (rc : RollerCoaster) (p : TrackPoint) : ℝ :=
  0.5 * p.velocity^2 + rc.g * p.height

/-- The point with minimum height has maximum velocity -/
theorem velocity_maximum_at_lowest_point (rc : RollerCoaster) :
  ∀ p q : TrackPoint,
    p ∈ rc.points →
    q ∈ rc.points →
    p.height < q.height →
    total_energy rc p = total_energy rc q →
    p.velocity > q.velocity :=
sorry

end NUMINAMATH_CALUDE_velocity_maximum_at_lowest_point_l3674_367450


namespace NUMINAMATH_CALUDE_edward_money_left_l3674_367441

def initial_money : ℕ := 41
def books_cost : ℕ := 6
def pens_cost : ℕ := 16

theorem edward_money_left : initial_money - (books_cost + pens_cost) = 19 := by
  sorry

end NUMINAMATH_CALUDE_edward_money_left_l3674_367441


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l3674_367424

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Returns true if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Returns true if a point (x, y) is on the given line -/
def on_line (l : Line) (x y : ℝ) : Prop := y = l.slope * x + l.y_intercept

theorem y_intercept_of_parallel_line 
  (line1 : Line) 
  (hline1 : line1.slope = -3 ∧ line1.y_intercept = 6) 
  (line2 : Line)
  (hparallel : parallel line1 line2)
  (hon_line : on_line line2 3 1) : 
  line2.y_intercept = 10 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l3674_367424


namespace NUMINAMATH_CALUDE_maggie_earnings_proof_l3674_367427

/-- Calculates Maggie's earnings from magazine subscriptions -/
def maggieEarnings (pricePerSubscription : ℕ) 
                   (parentsSubscriptions : ℕ)
                   (grandfatherSubscriptions : ℕ)
                   (nextDoorNeighborSubscriptions : ℕ) : ℕ :=
  let otherNeighborSubscriptions := 2 * nextDoorNeighborSubscriptions
  let totalSubscriptions := parentsSubscriptions + grandfatherSubscriptions + 
                            nextDoorNeighborSubscriptions + otherNeighborSubscriptions
  pricePerSubscription * totalSubscriptions

/-- Proves that Maggie's earnings are $55.00 -/
theorem maggie_earnings_proof : 
  maggieEarnings 5 4 1 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_maggie_earnings_proof_l3674_367427


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3674_367462

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : 
  z = -1 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3674_367462


namespace NUMINAMATH_CALUDE_good_carrots_count_l3674_367411

/-- The number of good carrots given the number of carrots picked by Faye and her mom, and the number of bad carrots. -/
def goodCarrots (fayeCarrots momCarrots badCarrots : ℕ) : ℕ :=
  fayeCarrots + momCarrots - badCarrots

/-- Theorem stating that the number of good carrots is 12 given the problem conditions. -/
theorem good_carrots_count : goodCarrots 23 5 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_good_carrots_count_l3674_367411


namespace NUMINAMATH_CALUDE_problem_statement_l3674_367468

theorem problem_statement (x y : ℝ) (h : 2 * x - y = 8) : 6 - 2 * x + y = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3674_367468


namespace NUMINAMATH_CALUDE_max_value_quadratic_swap_l3674_367491

/-- Given real numbers a, b, and c where |ax^2 + bx + c| has a maximum value of 1 
    on the interval x ∈ [-1,1], the maximum possible value of |cx^2 + bx + a| 
    on the interval x ∈ [-1,1] is 2. -/
theorem max_value_quadratic_swap (a b c : ℝ) 
  (h : ∀ x ∈ Set.Icc (-1) 1, |a * x^2 + b * x + c| ≤ 1) :
  (⨆ x ∈ Set.Icc (-1) 1, |c * x^2 + b * x + a|) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_swap_l3674_367491


namespace NUMINAMATH_CALUDE_equation_solution_l3674_367495

theorem equation_solution : 
  ∃ x : ℝ, (4 : ℝ) ^ x = 2 ^ (x + 1) - 1 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3674_367495


namespace NUMINAMATH_CALUDE_bridge_length_l3674_367481

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 265 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3674_367481


namespace NUMINAMATH_CALUDE_standard_deviation_from_age_range_job_applicants_standard_deviation_l3674_367416

/-- Given an average age and a number of distinct integer ages within one standard deviation,
    calculate the standard deviation. -/
theorem standard_deviation_from_age_range (average_age : ℕ) (distinct_ages : ℕ) : ℕ :=
  let standard_deviation := (distinct_ages - 1) / 2
  standard_deviation

/-- Prove that for an average age of 20 and 17 distinct integer ages within one standard deviation,
    the standard deviation is 8. -/
theorem job_applicants_standard_deviation : 
  standard_deviation_from_age_range 20 17 = 8 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_from_age_range_job_applicants_standard_deviation_l3674_367416


namespace NUMINAMATH_CALUDE_square_side_length_l3674_367437

theorem square_side_length (perimeter : ℝ) (h1 : perimeter = 16) : 
  perimeter / 4 = 4 := by
  sorry

#check square_side_length

end NUMINAMATH_CALUDE_square_side_length_l3674_367437


namespace NUMINAMATH_CALUDE_no_valid_numbers_l3674_367486

theorem no_valid_numbers :
  ¬∃ (a b c : ℕ), 
    (100 ≤ 100 * a + 10 * b + c) ∧ 
    (100 * a + 10 * b + c < 1000) ∧ 
    (100 * a + 10 * b + c) % 15 = 0 ∧ 
    (10 * b + c) % 4 = 0 ∧ 
    a > b ∧ b > c :=
by sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l3674_367486


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l3674_367461

theorem arctan_sum_equals_pi_over_four :
  ∃ (n : ℕ), n > 0 ∧
  Real.arctan (1 / 6) + Real.arctan (1 / 7) + Real.arctan (1 / 5) + Real.arctan (1 / n) = π / 4 ∧
  n = 311 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l3674_367461


namespace NUMINAMATH_CALUDE_eighth_odd_multiple_of_5_l3674_367421

/-- The nth positive integer that is both odd and a multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ :=
  10 * n - 5

theorem eighth_odd_multiple_of_5 : nthOddMultipleOf5 8 = 75 := by
  sorry

end NUMINAMATH_CALUDE_eighth_odd_multiple_of_5_l3674_367421


namespace NUMINAMATH_CALUDE_x_squared_mod_20_l3674_367426

theorem x_squared_mod_20 (x : ℕ) (h1 : 5 * x ≡ 10 [ZMOD 20]) (h2 : 2 * x ≡ 14 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_20_l3674_367426


namespace NUMINAMATH_CALUDE_meeting_participants_l3674_367440

theorem meeting_participants :
  ∀ (F M : ℕ),
  F > 0 ∧ M > 0 →
  F / 2 + M / 4 = (F + M) / 3 →
  F / 2 = 110 →
  F + M = 330 :=
by
  sorry

end NUMINAMATH_CALUDE_meeting_participants_l3674_367440


namespace NUMINAMATH_CALUDE_coin_flip_experiment_l3674_367485

theorem coin_flip_experiment (total_flips : ℕ) (heads_count : ℕ) (is_fair : Bool) :
  total_flips = 800 →
  heads_count = 440 →
  is_fair = true →
  (heads_count : ℚ) / (total_flips : ℚ) = 11/20 ∧ 
  (1 : ℚ) / 2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_experiment_l3674_367485


namespace NUMINAMATH_CALUDE_alpha_range_l3674_367455

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sin x

theorem alpha_range (α : ℝ) :
  α > 0 ∧
  (∀ x, x ∈ Set.Icc 0 α → f x ∈ Set.Icc 1 (3/2)) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc 0 α ∧ x₂ ∈ Set.Icc 0 α ∧ f x₁ = 1 ∧ f x₂ = 3/2) →
  α ∈ Set.Icc (π/6) π :=
sorry

end NUMINAMATH_CALUDE_alpha_range_l3674_367455


namespace NUMINAMATH_CALUDE_distance_between_last_two_points_l3674_367448

def cube_vertices : List (Fin 3 → ℝ) := [
  ![0, 0, 0], ![0, 0, 6], ![0, 6, 0], ![0, 6, 6],
  ![6, 0, 0], ![6, 0, 6], ![6, 6, 0], ![6, 6, 6]
]

def plane_intersections : List (Fin 3 → ℝ) := [
  ![0, 3, 0], ![2, 0, 0], ![2, 6, 6], ![4, 0, 6], ![0, 6, 6]
]

theorem distance_between_last_two_points :
  let S := plane_intersections[3]
  let T := plane_intersections[4]
  Real.sqrt ((S 0 - T 0)^2 + (S 1 - T 1)^2 + (S 2 - T 2)^2) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_last_two_points_l3674_367448


namespace NUMINAMATH_CALUDE_painting_time_equation_l3674_367480

theorem painting_time_equation (t : ℝ) : t > 0 → t = 5/2 := by
  intro h
  have alice_rate : ℝ := 1/4
  have bob_rate : ℝ := 1/6
  have charlie_rate : ℝ := 1/12
  have combined_rate : ℝ := alice_rate + bob_rate + charlie_rate
  have break_time : ℝ := 1/2
  have painting_equation : (combined_rate * (t - break_time) = 1) := by sorry
  sorry

end NUMINAMATH_CALUDE_painting_time_equation_l3674_367480


namespace NUMINAMATH_CALUDE_batsman_average_l3674_367413

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman after their last innings -/
def calculateAverage (b : Batsman) : Nat :=
  (b.totalRuns + b.lastInningsScore) / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after the 12th innings is 82 runs -/
theorem batsman_average (b : Batsman)
  (h1 : b.innings = 12)
  (h2 : b.lastInningsScore = 115)
  (h3 : b.averageIncrease = 3)
  (h4 : calculateAverage b = calculateAverage { b with innings := b.innings - 1 } + b.averageIncrease) :
  calculateAverage b = 82 := by
  sorry

#check batsman_average

end NUMINAMATH_CALUDE_batsman_average_l3674_367413


namespace NUMINAMATH_CALUDE_zoo_recovery_time_l3674_367490

/-- The total time spent recovering escaped animals from a zoo. -/
def total_recovery_time (lions rhinos recovery_time_per_animal : ℕ) : ℕ :=
  (lions + rhinos) * recovery_time_per_animal

/-- Theorem stating that given 3 lions, 2 rhinos, and 2 hours recovery time per animal,
    the total recovery time is 10 hours. -/
theorem zoo_recovery_time :
  total_recovery_time 3 2 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_zoo_recovery_time_l3674_367490


namespace NUMINAMATH_CALUDE_daria_savings_weeks_l3674_367419

/-- The number of weeks required for Daria to save enough money for a vacuum cleaner. -/
def weeks_to_save (initial_savings : ℕ) (weekly_contribution : ℕ) (vacuum_cost : ℕ) : ℕ :=
  ((vacuum_cost - initial_savings) + weekly_contribution - 1) / weekly_contribution

/-- Theorem: Daria needs 10 weeks to save for the vacuum cleaner. -/
theorem daria_savings_weeks : weeks_to_save 20 10 120 = 10 := by
  sorry

end NUMINAMATH_CALUDE_daria_savings_weeks_l3674_367419


namespace NUMINAMATH_CALUDE_initial_kittens_l3674_367444

theorem initial_kittens (kittens_to_jessica kittens_to_sara kittens_left : ℕ) :
  kittens_to_jessica = 3 →
  kittens_to_sara = 6 →
  kittens_left = 9 →
  kittens_to_jessica + kittens_to_sara + kittens_left = 18 :=
by sorry

end NUMINAMATH_CALUDE_initial_kittens_l3674_367444


namespace NUMINAMATH_CALUDE_parallelogram_probability_l3674_367410

-- Define the vertices of the parallelogram
def P : ℝ × ℝ := (4, 4)
def Q : ℝ × ℝ := (-2, -2)
def R : ℝ × ℝ := (-8, -2)
def S : ℝ × ℝ := (-2, 4)

-- Define the line y = -1
def line (x : ℝ) : ℝ := -1

-- Define the area of a parallelogram given base and height
def parallelogram_area (base height : ℝ) : ℝ := base * height

-- Theorem statement
theorem parallelogram_probability : 
  let total_area := parallelogram_area (P.1 - S.1) (P.2 - Q.2)
  let below_line_area := parallelogram_area (P.1 - S.1) 1
  below_line_area / total_area = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_parallelogram_probability_l3674_367410


namespace NUMINAMATH_CALUDE_median_bisects_perimeter_implies_isosceles_l3674_367459

/-- A triangle is represented by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- A median of a triangle -/
structure Median (t : Triangle) where
  base : ℝ
  is_median : base = t.a ∨ base = t.b ∨ base = t.c

/-- A triangle is isosceles if at least two of its sides are equal -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The theorem statement -/
theorem median_bisects_perimeter_implies_isosceles (t : Triangle) (m : Median t) :
  (m.base / 2 + (t.perimeter - m.base) / 2 = t.perimeter / 2) → t.isIsosceles :=
by
  sorry

end NUMINAMATH_CALUDE_median_bisects_perimeter_implies_isosceles_l3674_367459


namespace NUMINAMATH_CALUDE_n_gon_regions_l3674_367453

/-- The number of regions into which the diagonals of an n-gon divide it -/
def R (n : ℕ) : ℕ := (n*(n-1)*(n-2)*(n-3))/24 + (n*(n-3))/2 + 1

/-- Theorem stating the number of regions in an n-gon divided by its diagonals -/
theorem n_gon_regions (n : ℕ) (h : n ≥ 3) :
  R n = (n*(n-1)*(n-2)*(n-3))/24 + (n*(n-3))/2 + 1 :=
by sorry


end NUMINAMATH_CALUDE_n_gon_regions_l3674_367453


namespace NUMINAMATH_CALUDE_prime_composite_inequality_l3674_367476

theorem prime_composite_inequality (n : ℕ) : 
  (Prime (2 * n - 1) → 
    ∀ (a : Fin n → ℕ+), (∀ i j, i ≠ j → a i ≠ a j) → 
      ∃ i j, (a i + a j : ℝ) / (Nat.gcd (a i) (a j)) ≥ 2 * n - 1) ∧
  (¬Prime (2 * n - 1) → 
    ∃ (a : Fin n → ℕ+), (∀ i j, i ≠ j → a i ≠ a j) ∧ 
      ∀ i j, (a i + a j : ℝ) / (Nat.gcd (a i) (a j)) < 2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_composite_inequality_l3674_367476


namespace NUMINAMATH_CALUDE_total_skips_eq_2450_l3674_367438

/-- Represents the number of skips completed by a person given their skipping rate and duration. -/
def skips_completed (rate : ℚ) (duration : ℚ) : ℚ := rate * duration

/-- Calculates the total number of skips completed by Roberto, Valerie, and Lucas. -/
def total_skips : ℚ :=
  let roberto_rate : ℚ := 4200 / 60  -- skips per minute
  let valerie_rate : ℚ := 80         -- skips per minute
  let lucas_rate : ℚ := 150 / 5      -- skips per minute
  let roberto_duration : ℚ := 15     -- minutes
  let valerie_duration : ℚ := 10     -- minutes
  let lucas_duration : ℚ := 20       -- minutes
  skips_completed roberto_rate roberto_duration +
  skips_completed valerie_rate valerie_duration +
  skips_completed lucas_rate lucas_duration

theorem total_skips_eq_2450 : total_skips = 2450 := by
  sorry

end NUMINAMATH_CALUDE_total_skips_eq_2450_l3674_367438


namespace NUMINAMATH_CALUDE_hyuksu_meat_consumption_l3674_367484

/-- The amount of meat Hyuksu ate yesterday in kilograms -/
def meat_yesterday : ℝ := 2.6

/-- The amount of meat Hyuksu ate today in kilograms -/
def meat_today : ℝ := 5.98

/-- The total amount of meat Hyuksu ate in two days in kilograms -/
def total_meat : ℝ := meat_yesterday + meat_today

theorem hyuksu_meat_consumption : total_meat = 8.58 := by
  sorry

end NUMINAMATH_CALUDE_hyuksu_meat_consumption_l3674_367484


namespace NUMINAMATH_CALUDE_largest_valid_number_l3674_367497

def is_valid (n : ℕ) : Prop :=
  n < 10000 ∧
  (∃ a : ℕ, 4^a ≤ n ∧ n < 4^(a+1) ∧ 4^a ≤ 3*n ∧ 3*n < 4^(a+1)) ∧
  (∃ b : ℕ, 8^b ≤ n ∧ n < 8^(b+1) ∧ 8^b ≤ 7*n ∧ 7*n < 8^(b+1)) ∧
  (∃ c : ℕ, 16^c ≤ n ∧ n < 16^(c+1) ∧ 16^c ≤ 15*n ∧ 15*n < 16^(c+1))

theorem largest_valid_number : 
  is_valid 4369 ∧ ∀ m : ℕ, m > 4369 → ¬(is_valid m) :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3674_367497


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l3674_367407

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l3674_367407


namespace NUMINAMATH_CALUDE_probability_two_blue_jellybeans_l3674_367432

-- Define the total number of jellybeans and the number of each color
def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 3
def blue_jellybeans : ℕ := 4
def white_jellybeans : ℕ := 5

-- Define the number of jellybeans to be picked
def picked_jellybeans : ℕ := 3

-- Define the probability of picking exactly two blue jellybeans
def prob_two_blue : ℚ := 12 / 55

-- Theorem statement
theorem probability_two_blue_jellybeans : 
  prob_two_blue = (Nat.choose blue_jellybeans 2 * Nat.choose (total_jellybeans - blue_jellybeans) 1) / 
                  Nat.choose total_jellybeans picked_jellybeans :=
by sorry

end NUMINAMATH_CALUDE_probability_two_blue_jellybeans_l3674_367432


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3674_367482

theorem min_value_quadratic_sum (x y z : ℝ) (h : x - 2*y + 2*z = 5) :
  ∃ (m : ℝ), m = 36 ∧ ∀ (a b c : ℝ), a - 2*b + 2*c = 5 → (a + 5)^2 + (b - 1)^2 + (c + 3)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3674_367482


namespace NUMINAMATH_CALUDE_sum_mod_eight_l3674_367401

theorem sum_mod_eight :
  (7145 + 7146 + 7147 + 7148 + 7149) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_eight_l3674_367401


namespace NUMINAMATH_CALUDE_tan_alpha_values_l3674_367420

theorem tan_alpha_values (α : Real) (h : 2 * Real.sin (2 * α) = 1 - Real.cos (2 * α)) :
  Real.tan α = 2 ∨ Real.tan α = 0 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l3674_367420


namespace NUMINAMATH_CALUDE_max_value_f_l3674_367487

/-- The function f(x) = x(1 - x^2) -/
def f (x : ℝ) : ℝ := x * (1 - x^2)

/-- The maximum value of f(x) on [0, 1] is 2√3/9 -/
theorem max_value_f : ∃ (c : ℝ), c = (2 * Real.sqrt 3) / 9 ∧ 
  (∀ x ∈ Set.Icc 0 1, f x ≤ c) ∧ 
  (∃ x ∈ Set.Icc 0 1, f x = c) := by
  sorry

end NUMINAMATH_CALUDE_max_value_f_l3674_367487


namespace NUMINAMATH_CALUDE_speed_conversion_l3674_367492

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The speed in meters per second -/
def speed_mps : ℝ := 50

/-- Theorem: Converting 50 mps to kmph equals 180 kmph -/
theorem speed_conversion : speed_mps * mps_to_kmph = 180 := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l3674_367492


namespace NUMINAMATH_CALUDE_tuesday_temperature_l3674_367425

-- Define temperatures for each day
def tuesday_temp : ℝ := sorry
def wednesday_temp : ℝ := sorry
def thursday_temp : ℝ := sorry
def friday_temp : ℝ := 53

-- Define the conditions
axiom avg_tue_wed_thu : (tuesday_temp + wednesday_temp + thursday_temp) / 3 = 52
axiom avg_wed_thu_fri : (wednesday_temp + thursday_temp + friday_temp) / 3 = 54

-- Theorem to prove
theorem tuesday_temperature : tuesday_temp = 47 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_temperature_l3674_367425


namespace NUMINAMATH_CALUDE_lcm_gcd_product_36_60_l3674_367470

theorem lcm_gcd_product_36_60 : Nat.lcm 36 60 * Nat.gcd 36 60 = 36 * 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_36_60_l3674_367470


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3674_367423

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -1
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 3*x + y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3674_367423


namespace NUMINAMATH_CALUDE_first_part_multiplier_l3674_367404

theorem first_part_multiplier (x : ℝ) : x + 7 * x = 55 → x = 5 → 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_part_multiplier_l3674_367404


namespace NUMINAMATH_CALUDE_loop_termination_min_n_value_l3674_367451

def s (n : ℕ) : ℕ := 2010 / 2^n + 3 * (2^n - 1) / 2^(n-1)

theorem loop_termination : 
  ∀ k : ℕ, k < 5 → s k ≥ 120 ∧ s 5 < 120 :=
sorry

theorem min_n_value : (∃ n : ℕ, s n < 120) ∧ (∀ k : ℕ, s k < 120 → k ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_loop_termination_min_n_value_l3674_367451


namespace NUMINAMATH_CALUDE_max_point_difference_is_n_l3674_367430

/-- Represents a hockey tournament with n teams -/
structure HockeyTournament where
  n : ℕ  -- number of teams
  n_pos : 0 < n  -- n is positive

/-- The maximum point difference between consecutively ranked teams in a hockey tournament -/
def maxPointDifference (tournament : HockeyTournament) : ℕ :=
  tournament.n

/-- Theorem: The maximum point difference between consecutively ranked teams is n -/
theorem max_point_difference_is_n (tournament : HockeyTournament) :
  maxPointDifference tournament = tournament.n := by
  sorry

end NUMINAMATH_CALUDE_max_point_difference_is_n_l3674_367430


namespace NUMINAMATH_CALUDE_M_is_real_l3674_367406

-- Define the set M
def M : Set ℂ := {z : ℂ | (z - 1)^2 = Complex.abs (z - 1)^2}

-- Theorem stating that M is equal to the set of real numbers
theorem M_is_real : M = {z : ℂ | z.im = 0} := by sorry

end NUMINAMATH_CALUDE_M_is_real_l3674_367406


namespace NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l3674_367422

-- Define the expressions
def expression1 (a : ℝ) : ℝ := 5 * a^2 - 7 + 4 * a - 2 * a^2 - 9 * a + 3
def expression2 (x : ℝ) : ℝ := (5 * x^2 - 6 * x) - 3 * (2 * x^2 - 3 * x)

-- State the theorems
theorem simplify_expression1 : ∀ a : ℝ, expression1 a = 3 * a^2 - 5 * a - 4 := by sorry

theorem simplify_expression2 : ∀ x : ℝ, expression2 x = -x^2 + 3 * x := by sorry

end NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l3674_367422


namespace NUMINAMATH_CALUDE_problem_statement_l3674_367475

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + 2 * a + b = 16) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 16 ∧ x * y > a * b) →
    a * b ≤ 8 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y + 2 * x + y = 16 → 2 * x + y ≥ 8) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y + 2 * x + y = 16 → x + y ≥ 6 * Real.sqrt 2 - 3) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y + 2 * x + y = 16 → 1 / (x + 1) + 1 / (y + 2) > Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3674_367475


namespace NUMINAMATH_CALUDE_cookie_cost_l3674_367465

theorem cookie_cost (initial_amount : ℚ) (hat_cost : ℚ) (pencil_cost : ℚ) (num_cookies : ℕ) (remaining_amount : ℚ)
  (h1 : initial_amount = 20)
  (h2 : hat_cost = 10)
  (h3 : pencil_cost = 2)
  (h4 : num_cookies = 4)
  (h5 : remaining_amount = 3)
  : (initial_amount - hat_cost - pencil_cost - remaining_amount) / num_cookies = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_l3674_367465


namespace NUMINAMATH_CALUDE_equidistant_point_x_value_l3674_367460

/-- A point in a 2D rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

/-- Distance from a point to the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- A point is equidistant from x-axis and y-axis -/
def isEquidistant (p : Point) : Prop :=
  distToXAxis p = distToYAxis p

/-- The main theorem -/
theorem equidistant_point_x_value (x : ℝ) :
  let p := Point.mk (-2*x) (x-6)
  isEquidistant p → x = 2 ∨ x = -6 := by
  sorry


end NUMINAMATH_CALUDE_equidistant_point_x_value_l3674_367460


namespace NUMINAMATH_CALUDE_hypercoplanar_iff_b_eq_plusminus_one_over_sqrt_two_l3674_367435

/-- A point in 4D space -/
def Point4D := Fin 4 → ℝ

/-- The determinant of a 4x4 matrix -/
def det4 (m : Fin 4 → Fin 4 → ℝ) : ℝ := sorry

/-- Check if five points in 4D space are hypercoplanar -/
def are_hypercoplanar (p1 p2 p3 p4 p5 : Point4D) : Prop :=
  det4 (λ i j => match i, j with
    | 0, _ => p2 j - p1 j
    | 1, _ => p3 j - p1 j
    | 2, _ => p4 j - p1 j
    | 3, _ => p5 j - p1 j) = 0

/-- The given points in 4D space -/
def p1 : Point4D := λ _ => 0
def p2 (b : ℝ) : Point4D := λ i => match i with | 0 => 1 | 1 => b | _ => 0
def p3 (b : ℝ) : Point4D := λ i => match i with | 1 => 1 | 2 => b | _ => 0
def p4 (b : ℝ) : Point4D := λ i => match i with | 0 => b | 2 => 1 | _ => 0
def p5 (b : ℝ) : Point4D := λ i => match i with | 1 => b | 3 => 1 | _ => 0

theorem hypercoplanar_iff_b_eq_plusminus_one_over_sqrt_two :
  ∀ b : ℝ, are_hypercoplanar (p1) (p2 b) (p3 b) (p4 b) (p5 b) ↔ b = 1 / Real.sqrt 2 ∨ b = -1 / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hypercoplanar_iff_b_eq_plusminus_one_over_sqrt_two_l3674_367435


namespace NUMINAMATH_CALUDE_meal_distribution_theorem_l3674_367402

/-- The number of ways to derange 8 items -/
def derangement_8 : ℕ := 14833

/-- The number of ways to choose 2 items from 10 -/
def choose_2_from_10 : ℕ := 45

/-- The number of ways to distribute 10 meals of 4 types to 10 people
    such that exactly 2 people receive the correct meal type -/
def distribute_meals (d₈ : ℕ) (c₁₀₂ : ℕ) : ℕ := d₈ * c₁₀₂

theorem meal_distribution_theorem :
  distribute_meals derangement_8 choose_2_from_10 = 666885 := by
  sorry

end NUMINAMATH_CALUDE_meal_distribution_theorem_l3674_367402


namespace NUMINAMATH_CALUDE_min_value_expression_l3674_367446

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 5/4)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3674_367446


namespace NUMINAMATH_CALUDE_relatively_prime_powers_l3674_367434

theorem relatively_prime_powers (a n m : ℕ) :
  Odd a → n > 0 → m > 0 → n ≠ m →
  Nat.gcd (a^(2^n) + 2^(2^n)) (a^(2^m) + 2^(2^m)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_relatively_prime_powers_l3674_367434


namespace NUMINAMATH_CALUDE_max_clocks_in_workshop_l3674_367418

/-- Represents a digital clock with hours and minutes -/
structure DigitalClock where
  hours : Nat
  minutes : Nat

/-- Represents the state of all clocks in the workshop -/
structure ClockWorkshop where
  clocks : List DigitalClock

/-- Checks if all clocks in the workshop show different times -/
def allDifferentTimes (workshop : ClockWorkshop) : Prop :=
  ∀ c1 c2 : DigitalClock, c1 ∈ workshop.clocks → c2 ∈ workshop.clocks → c1 ≠ c2 →
    (c1.hours ≠ c2.hours ∨ c1.minutes ≠ c2.minutes)

/-- Calculates the sum of hours displayed on all clocks -/
def sumHours (workshop : ClockWorkshop) : Nat :=
  workshop.clocks.foldl (fun sum clock => sum + clock.hours) 0

/-- Calculates the sum of minutes displayed on all clocks -/
def sumMinutes (workshop : ClockWorkshop) : Nat :=
  workshop.clocks.foldl (fun sum clock => sum + clock.minutes) 0

/-- Represents the state of the workshop after some time has passed -/
def advanceTime (workshop : ClockWorkshop) : ClockWorkshop := sorry

theorem max_clocks_in_workshop :
  ∀ (workshop : ClockWorkshop),
    workshop.clocks.length > 1 →
    (∀ clock ∈ workshop.clocks, clock.hours ≥ 1 ∧ clock.hours ≤ 12) →
    (∀ clock ∈ workshop.clocks, clock.minutes ≥ 0 ∧ clock.minutes < 60) →
    allDifferentTimes workshop →
    sumHours (advanceTime workshop) + 1 = sumHours workshop →
    sumMinutes (advanceTime workshop) + 1 = sumMinutes workshop →
    workshop.clocks.length ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_clocks_in_workshop_l3674_367418


namespace NUMINAMATH_CALUDE_parabola_focus_l3674_367431

/-- For a parabola y = ax^2 with focus at (0, 1), a = 1/4 -/
theorem parabola_focus (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Parabola equation
  (0, 1) = (0, 1 / (4 * a)) →  -- Focus at (0, 1)
  a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l3674_367431


namespace NUMINAMATH_CALUDE_intersection_product_constant_l3674_367442

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  (a_pos : 0 < a)
  (b_pos : 0 < b)

/-- A point (x₀, y₀) on the hyperbola -/
structure PointOnHyperbola (H : Hyperbola a b) where
  x₀ : ℝ
  y₀ : ℝ
  on_hyperbola : x₀^2 / a^2 - y₀^2 / b^2 = 1

/-- The theorem stating that the product of x-coordinates of intersections is constant -/
theorem intersection_product_constant
  (H : Hyperbola a b) (P : PointOnHyperbola H) :
  ∃ (x₁ x₂ : ℝ),
    (x₁ * (b / a) = (P.x₀ * x₁) / a^2 - (P.y₀ * (b / a) * x₁) / b^2) ∧
    (x₂ * (-b / a) = (P.x₀ * x₂) / a^2 - (P.y₀ * (-b / a) * x₂) / b^2) ∧
    x₁ * x₂ = a^4 :=
sorry

end NUMINAMATH_CALUDE_intersection_product_constant_l3674_367442


namespace NUMINAMATH_CALUDE_concrete_mixture_theorem_l3674_367415

/-- The amount of 80% cement mixture used in tons -/
def amount_80_percent : ℝ := 7.0

/-- The percentage of cement in the final mixture -/
def final_cement_percentage : ℝ := 0.62

/-- The percentage of cement in the first mixture -/
def first_mixture_percentage : ℝ := 0.20

/-- The percentage of cement in the second mixture -/
def second_mixture_percentage : ℝ := 0.80

/-- The total amount of concrete made in tons -/
def total_concrete : ℝ := 10.0

theorem concrete_mixture_theorem :
  ∃ (x : ℝ),
    x ≥ 0 ∧
    x * first_mixture_percentage + amount_80_percent * second_mixture_percentage =
      final_cement_percentage * (x + amount_80_percent) ∧
    x + amount_80_percent = total_concrete :=
by sorry

end NUMINAMATH_CALUDE_concrete_mixture_theorem_l3674_367415


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3674_367499

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = n * (a 0 + a (n - 1)) / 2

/-- Theorem stating that if S_2 = 3 and S_4 = 15, then S_6 = 63 for an arithmetic sequence -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
    (h1 : seq.S 2 = 3) (h2 : seq.S 4 = 15) : seq.S 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3674_367499
