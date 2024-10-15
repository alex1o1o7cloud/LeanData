import Mathlib

namespace NUMINAMATH_CALUDE_school_location_minimizes_distance_l2675_267587

/-- Represents the distance between two villages in kilometers -/
def village_distance : ℝ := 3

/-- Represents the number of students in village A -/
def students_A : ℕ := 300

/-- Represents the number of students in village B -/
def students_B : ℕ := 200

/-- Represents the distance from village A to the school -/
def school_distance (x : ℝ) : ℝ := x

/-- Calculates the total distance traveled by all students -/
def total_distance (x : ℝ) : ℝ :=
  students_A * school_distance x + students_B * (village_distance - school_distance x)

/-- Theorem: The total distance is minimized when the school is built in village A -/
theorem school_location_minimizes_distance :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ village_distance →
    total_distance 0 ≤ total_distance x :=
by sorry

end NUMINAMATH_CALUDE_school_location_minimizes_distance_l2675_267587


namespace NUMINAMATH_CALUDE_composite_number_l2675_267525

theorem composite_number (n : ℕ+) : ∃ (p : ℕ), Prime p ∧ p ∣ (19 * 8^n.val + 17) ∧ 1 < p ∧ p < 19 * 8^n.val + 17 := by
  sorry

end NUMINAMATH_CALUDE_composite_number_l2675_267525


namespace NUMINAMATH_CALUDE_value_calculation_l2675_267502

theorem value_calculation (initial_number : ℕ) (h : initial_number = 26) : 
  ((((initial_number + 20) * 2) / 2) - 2) * 2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l2675_267502


namespace NUMINAMATH_CALUDE_complex_modulus_l2675_267538

theorem complex_modulus (z : ℂ) (h : (1 + 2*I)*z = 3 - 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2675_267538


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l2675_267527

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 300670

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 3.0067
    exponent := 5
    h1 := by sorry }

/-- Theorem stating that the proposed notation correctly represents the original number -/
theorem correct_scientific_notation :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = original_number := by
  sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l2675_267527


namespace NUMINAMATH_CALUDE_michelle_gas_problem_l2675_267583

/-- Michelle's gas problem -/
theorem michelle_gas_problem (gas_left gas_used : ℚ) 
  (h1 : gas_left = 0.17)
  (h2 : gas_used = 0.33) : 
  gas_left + gas_used = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_michelle_gas_problem_l2675_267583


namespace NUMINAMATH_CALUDE_arc_length_45_degrees_l2675_267560

/-- Given a circle with circumference 90 meters, prove that an arc subtending a 45° angle at the center has a length of 11.25 meters. -/
theorem arc_length_45_degrees (D : Real) (EF : Real) :
  D = 90 →  -- circumference of the circle
  EF = (45 / 360) * D →  -- arc length is proportional to the angle it subtends
  EF = 11.25 := by
sorry

end NUMINAMATH_CALUDE_arc_length_45_degrees_l2675_267560


namespace NUMINAMATH_CALUDE_inequality_proof_l2675_267543

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2675_267543


namespace NUMINAMATH_CALUDE_inequality_theorem_largest_constant_l2675_267539

theorem inequality_theorem (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 :=
by sorry

theorem largest_constant :
  ∀ m : ℝ, (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
    Real.sqrt (e / (a + b + c + d)) > m) → m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_largest_constant_l2675_267539


namespace NUMINAMATH_CALUDE_prob_king_ace_standard_deck_l2675_267547

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (kings : ℕ)
  (aces : ℕ)

/-- Calculates the probability of drawing a King first and an Ace second without replacement -/
def prob_king_ace (d : Deck) : ℚ :=
  (d.kings : ℚ) / d.total_cards * d.aces / (d.total_cards - 1)

/-- Theorem: The probability of drawing a King first and an Ace second from a standard deck is 4/663 -/
theorem prob_king_ace_standard_deck :
  let standard_deck : Deck := ⟨52, 4, 4⟩
  prob_king_ace standard_deck = 4 / 663 := by
sorry

end NUMINAMATH_CALUDE_prob_king_ace_standard_deck_l2675_267547


namespace NUMINAMATH_CALUDE_rotation_150_degrees_l2675_267534

-- Define the shapes
inductive Shape
  | Square
  | Triangle
  | Pentagon

-- Define the positions
inductive Position
  | Top
  | Right
  | Bottom

-- Define the circular arrangement
structure CircularArrangement :=
  (top : Shape)
  (right : Shape)
  (bottom : Shape)

-- Define the rotation function
def rotate150 (arr : CircularArrangement) : CircularArrangement :=
  { top := arr.right
  , right := arr.bottom
  , bottom := arr.top }

-- Theorem statement
theorem rotation_150_degrees (initial : CircularArrangement) 
  (h1 : initial.top = Shape.Square)
  (h2 : initial.right = Shape.Triangle)
  (h3 : initial.bottom = Shape.Pentagon) :
  let final := rotate150 initial
  final.top = Shape.Pentagon ∧ 
  final.right = Shape.Square ∧ 
  final.bottom = Shape.Triangle := by
  sorry

end NUMINAMATH_CALUDE_rotation_150_degrees_l2675_267534


namespace NUMINAMATH_CALUDE_sum_of_facing_angles_l2675_267507

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  vertex_angle : ℝ
  is_isosceles : vertex_angle > 0 ∧ vertex_angle < 180

-- Define the configuration of two isosceles triangles
structure TwoTrianglesConfig where
  triangle1 : IsoscelesTriangle
  triangle2 : IsoscelesTriangle
  distance : ℝ
  same_base_line : Bool
  facing_equal_sides : Bool

-- Theorem statement
theorem sum_of_facing_angles (config : TwoTrianglesConfig) :
  config.triangle1 = config.triangle2 →
  config.triangle1.vertex_angle = 40 →
  config.distance = 4 →
  config.same_base_line = true →
  config.facing_equal_sides = true →
  (180 - config.triangle1.vertex_angle) + (180 - config.triangle2.vertex_angle) = 80 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_facing_angles_l2675_267507


namespace NUMINAMATH_CALUDE_least_integer_x_l2675_267558

theorem least_integer_x (x : ℤ) : (∀ y : ℤ, |3 * y + 5| ≤ 21 → y ≥ -8) ∧ |3 * (-8) + 5| ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_x_l2675_267558


namespace NUMINAMATH_CALUDE_no_four_distinct_naturals_power_sum_equality_l2675_267569

theorem no_four_distinct_naturals_power_sum_equality :
  ¬∃ (x y z t : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t ∧ x^x + y^y = z^z + t^t :=
by sorry

end NUMINAMATH_CALUDE_no_four_distinct_naturals_power_sum_equality_l2675_267569


namespace NUMINAMATH_CALUDE_campers_rowing_total_l2675_267550

/-- The total number of campers who went rowing throughout the day -/
def total_campers (morning afternoon evening : ℕ) : ℕ :=
  morning + afternoon + evening

/-- Theorem stating that the total number of campers who went rowing is 764 -/
theorem campers_rowing_total :
  total_campers 235 387 142 = 764 := by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_total_l2675_267550


namespace NUMINAMATH_CALUDE_pie_distribution_problem_l2675_267509

theorem pie_distribution_problem :
  ∃! (p b a h : ℕ),
    p + b + a + h = 30 ∧
    b + p = a + h ∧
    p + a = 6 * (b + h) ∧
    h < p ∧ h < b ∧ h < a ∧
    h ≥ 1 ∧ p ≥ 1 ∧ b ≥ 1 ∧ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_pie_distribution_problem_l2675_267509


namespace NUMINAMATH_CALUDE_sequence_sum_l2675_267594

theorem sequence_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l2675_267594


namespace NUMINAMATH_CALUDE_tangent_line_circle_l2675_267544

theorem tangent_line_circle (R : ℝ) : 
  R > 0 → 
  (∃ x y : ℝ, x + y = 2 * R ∧ (x - 1)^2 + y^2 = R ∧ 
    ∀ x' y' : ℝ, x' + y' = 2 * R → (x' - 1)^2 + y'^2 ≥ R) →
  R = (3 + Real.sqrt 5) / 4 ∨ R = (3 - Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l2675_267544


namespace NUMINAMATH_CALUDE_height_difference_l2675_267528

def pine_height : ℚ := 12 + 4/5
def birch_height : ℚ := 18 + 1/2
def maple_height : ℚ := 14 + 3/5

def tallest_height : ℚ := max (max pine_height birch_height) maple_height
def shortest_height : ℚ := min (min pine_height birch_height) maple_height

theorem height_difference :
  tallest_height - shortest_height = 7 + 7/10 := by sorry

end NUMINAMATH_CALUDE_height_difference_l2675_267528


namespace NUMINAMATH_CALUDE_right_triangle_parity_l2675_267576

theorem right_triangle_parity (a b c : ℕ) (h_right : a^2 + b^2 = c^2) :
  (Even a ∧ Even b ∧ Even c) ∨
  ((Odd a ∧ Even b ∧ Odd c) ∨ (Even a ∧ Odd b ∧ Odd c)) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_parity_l2675_267576


namespace NUMINAMATH_CALUDE_soda_preference_l2675_267592

/-- Given a survey of 520 people and a central angle of 144° for "Soda" preference
    in a pie chart, prove that 208 people favor "Soda". -/
theorem soda_preference (total : ℕ) (angle : ℝ) (h1 : total = 520) (h2 : angle = 144) :
  (angle / 360 : ℝ) * total = 208 := by
  sorry

end NUMINAMATH_CALUDE_soda_preference_l2675_267592


namespace NUMINAMATH_CALUDE_sum_powers_i_2047_l2675_267563

def imaginary_unit_sum (i : ℂ) : ℕ → ℂ
  | 0 => 1
  | n + 1 => i^(n + 1) + imaginary_unit_sum i n

theorem sum_powers_i_2047 (i : ℂ) (h : i^2 = -1) :
  imaginary_unit_sum i 2047 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_powers_i_2047_l2675_267563


namespace NUMINAMATH_CALUDE_greatest_b_value_l2675_267521

theorem greatest_b_value (a b : ℤ) (h : a * b + 7 * a + 6 * b = -6) : 
  ∀ c : ℤ, (∃ d : ℤ, d * c + 7 * d + 6 * c = -6) → c ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2675_267521


namespace NUMINAMATH_CALUDE_grid_ball_probability_l2675_267590

theorem grid_ball_probability
  (a b c r : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_a_gt_b : a > b)
  (h_r_lt_b_half : r < b / 2)
  (h_strip_width : 2 * c = Real.sqrt ((a + b)^2 / 4 + a * b) - (a + b) / 2)
  : (a - 2 * r) * (b - 2 * r) / ((a + 2 * c) * (b + 2 * c)) ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_grid_ball_probability_l2675_267590


namespace NUMINAMATH_CALUDE_car_value_decrease_l2675_267505

theorem car_value_decrease (original_value current_value : ℝ) 
  (h1 : original_value = 4000)
  (h2 : current_value = 2800) :
  (original_value - current_value) / original_value * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_car_value_decrease_l2675_267505


namespace NUMINAMATH_CALUDE_julia_played_with_12_on_monday_l2675_267504

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 19 - 7

/-- The total number of kids Julia played with -/
def total_kids : ℕ := 19

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 7

theorem julia_played_with_12_on_monday :
  monday_kids = 12 :=
sorry

end NUMINAMATH_CALUDE_julia_played_with_12_on_monday_l2675_267504


namespace NUMINAMATH_CALUDE_solution_exists_l2675_267585

theorem solution_exists (x : ℝ) (h : x = 5) : ∃ some_number : ℝ, (x / 5) + some_number = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l2675_267585


namespace NUMINAMATH_CALUDE_arrangements_count_l2675_267506

def number_of_people : ℕ := 7
def number_of_gaps : ℕ := number_of_people - 1

theorem arrangements_count :
  (number_of_people - 2).factorial * number_of_gaps.choose 2 = 3600 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l2675_267506


namespace NUMINAMATH_CALUDE_female_elementary_students_l2675_267574

theorem female_elementary_students (total_students : ℕ) (non_elementary_girls : ℕ) 
  (h1 : total_students = 30)
  (h2 : non_elementary_girls = 7) :
  total_students / 2 - non_elementary_girls = 8 := by
  sorry

end NUMINAMATH_CALUDE_female_elementary_students_l2675_267574


namespace NUMINAMATH_CALUDE_area_bounded_region_area_is_four_l2675_267552

/-- The area of the region bounded by x = 2, y = 2, x = 0, and y = 0 is 4 -/
theorem area_bounded_region : ℝ :=
  let x_bound : ℝ := 2
  let y_bound : ℝ := 2
  x_bound * y_bound

#check area_bounded_region

theorem area_is_four : area_bounded_region = 4 := by sorry

end NUMINAMATH_CALUDE_area_bounded_region_area_is_four_l2675_267552


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2675_267588

theorem pure_imaginary_complex_number (a : ℝ) :
  (Complex.I * (a - 2) : ℂ).re = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2675_267588


namespace NUMINAMATH_CALUDE_equation_solution_l2675_267517

theorem equation_solution (x : ℝ) : 1 / x + x / 80 = 7 / 30 → x = 12 ∨ x = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2675_267517


namespace NUMINAMATH_CALUDE_parallel_condition_l2675_267500

def a : ℝ × ℝ := (1, -4)
def b (x : ℝ) : ℝ × ℝ := (-1, x)
def c (x : ℝ) : ℝ × ℝ := a + 3 • (b x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v = k • w

theorem parallel_condition (x : ℝ) :
  parallel a (c x) ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_parallel_condition_l2675_267500


namespace NUMINAMATH_CALUDE_remainder_3_pow_210_mod_17_l2675_267595

theorem remainder_3_pow_210_mod_17 : (3^210 : ℕ) % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_210_mod_17_l2675_267595


namespace NUMINAMATH_CALUDE_books_movies_difference_l2675_267575

def books_read : ℕ := 17
def movies_watched : ℕ := 21

theorem books_movies_difference :
  (books_read : ℤ) - movies_watched = -4 :=
sorry

end NUMINAMATH_CALUDE_books_movies_difference_l2675_267575


namespace NUMINAMATH_CALUDE_exists_natural_sqrt_nested_root_l2675_267551

theorem exists_natural_sqrt_nested_root : ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, (n : ℝ)^(5/4) = m := by
  sorry

end NUMINAMATH_CALUDE_exists_natural_sqrt_nested_root_l2675_267551


namespace NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l2675_267598

/-- Given a company's yearly magazine subscription cost and desired budget cut,
    calculate the percentage reduction in the budget. -/
theorem magazine_budget_cut_percentage
  (original_cost : ℝ)
  (budget_cut : ℝ)
  (h_original_cost : original_cost = 940)
  (h_budget_cut : budget_cut = 611) :
  (budget_cut / original_cost) * 100 = 65 := by
sorry

end NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l2675_267598


namespace NUMINAMATH_CALUDE_ceiling_sqrt_169_l2675_267597

theorem ceiling_sqrt_169 : ⌈Real.sqrt 169⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_169_l2675_267597


namespace NUMINAMATH_CALUDE_joan_seashells_l2675_267562

/-- Given 245 initial seashells, prove that after giving 3/5 to Mike and 2/5 of the remainder to Lisa, Joan is left with 59 seashells. -/
theorem joan_seashells (initial_seashells : ℕ) (mike_fraction : ℚ) (lisa_fraction : ℚ) :
  initial_seashells = 245 →
  mike_fraction = 3 / 5 →
  lisa_fraction = 2 / 5 →
  initial_seashells - (initial_seashells * mike_fraction).floor -
    ((initial_seashells - (initial_seashells * mike_fraction).floor) * lisa_fraction).floor = 59 := by
  sorry


end NUMINAMATH_CALUDE_joan_seashells_l2675_267562


namespace NUMINAMATH_CALUDE_candy_bar_count_l2675_267537

theorem candy_bar_count (num_bags : ℕ) (bars_per_bag : ℕ) (h1 : num_bags = 5) (h2 : bars_per_bag = 3) :
  num_bags * bars_per_bag = 15 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_count_l2675_267537


namespace NUMINAMATH_CALUDE_pyramid_cone_properties_l2675_267535

/-- Represents a square pyramid with a cone resting on its base --/
structure PyramidWithCone where
  pyramid_height : ℝ
  cone_base_radius : ℝ
  -- The cone is tangent to the other four faces of the pyramid
  is_tangent : Bool

/-- Calculates the edge length of the pyramid's base --/
def calculate_edge_length (p : PyramidWithCone) : ℝ := sorry

/-- Calculates the surface area of the cone not in contact with the pyramid --/
def calculate_cone_surface_area (p : PyramidWithCone) : ℝ := sorry

/-- Theorem stating the properties of the specific pyramid and cone configuration --/
theorem pyramid_cone_properties :
  let p : PyramidWithCone := {
    pyramid_height := 9,
    cone_base_radius := 3,
    is_tangent := true
  }
  calculate_edge_length p = 9 ∧
  calculate_cone_surface_area p = 30 * Real.pi := by sorry

end NUMINAMATH_CALUDE_pyramid_cone_properties_l2675_267535


namespace NUMINAMATH_CALUDE_elizabeth_study_time_l2675_267519

/-- Given that Elizabeth studied for a total of 60 minutes, including 35 minutes for math,
    prove that she studied for 25 minutes for science. -/
theorem elizabeth_study_time (total_time math_time science_time : ℕ) : 
  total_time = 60 ∧ math_time = 35 ∧ total_time = math_time + science_time →
  science_time = 25 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_study_time_l2675_267519


namespace NUMINAMATH_CALUDE_hadley_walk_distance_l2675_267584

/-- The distance Hadley walked to the pet store -/
def distance_to_pet_store : ℝ := 1

/-- The distance Hadley walked to the grocery store -/
def distance_to_grocery : ℝ := 2

/-- The distance Hadley walked back home -/
def distance_back_home : ℝ := 4 - 1

/-- The total distance Hadley walked -/
def total_distance : ℝ := 6

theorem hadley_walk_distance :
  distance_to_grocery + distance_to_pet_store + distance_back_home = total_distance :=
by sorry

end NUMINAMATH_CALUDE_hadley_walk_distance_l2675_267584


namespace NUMINAMATH_CALUDE_intersection_condition_l2675_267518

-- Define the quadratic function
def f (a x : ℝ) := a * x^2 - 4 * a * x - 2

-- Define the solution set of the inequality
def solution_set (a : ℝ) := {x : ℝ | f a x > 0}

-- Define the given set
def given_set := {x : ℝ | 3 < x ∧ x < 4}

-- Theorem statement
theorem intersection_condition (a : ℝ) : 
  (∃ x, x ∈ solution_set a ∧ x ∈ given_set) ↔ a < -2/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l2675_267518


namespace NUMINAMATH_CALUDE_sum_a_b_equals_three_l2675_267530

theorem sum_a_b_equals_three (a b : ℝ) (h : |a - 4| + (b + 1)^2 = 0) : a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_three_l2675_267530


namespace NUMINAMATH_CALUDE_complement_of_union_equals_zero_five_l2675_267549

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 4}

theorem complement_of_union_equals_zero_five :
  (U \ (A ∪ B)) = {0, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_zero_five_l2675_267549


namespace NUMINAMATH_CALUDE_max_operation_value_l2675_267573

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def operation (n : ℕ) : ℕ := 3 * (300 - n)

theorem max_operation_value :
  ∃ (m : ℕ), (∀ (n : ℕ), is_two_digit n → operation n ≤ m) ∧ (∃ (k : ℕ), is_two_digit k ∧ operation k = m) ∧ m = 870 :=
sorry

end NUMINAMATH_CALUDE_max_operation_value_l2675_267573


namespace NUMINAMATH_CALUDE_triangle_area_with_angle_bisector_l2675_267555

/-- The area of a triangle given two sides and the angle bisector between them -/
theorem triangle_area_with_angle_bisector (a b l : ℝ) (ha : a > 0) (hb : b > 0) (hl : l > 0) :
  let area := l * (a + b) / (4 * a * b) * Real.sqrt (4 * a^2 * b^2 - l^2 * (a + b)^2)
  ∃ (α : ℝ), α > 0 ∧ α < π/2 ∧
    (l * (a + b) / (2 * a * b) = Real.cos α) ∧
    area = (1/2) * a * b * Real.sin (2 * α) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_with_angle_bisector_l2675_267555


namespace NUMINAMATH_CALUDE_extremum_implies_slope_l2675_267586

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := (x - 2) * (x^2 + c)

-- Define the derivative of f(x)
def f' (c : ℝ) (x : ℝ) : ℝ := (x^2 + c) + (x - 2) * (2 * x)

theorem extremum_implies_slope (c : ℝ) :
  (∃ k, f' c 1 = k ∧ k = 0) → f' c (-1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_slope_l2675_267586


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l2675_267531

theorem largest_divisor_of_expression (n : ℕ+) : 
  ∃ (m : ℕ), m = 2448 ∧ 
  (∀ k : ℕ+, (9^(2*k.val) - 8^(2*k.val) - 17) % m = 0) ∧
  (∀ m' : ℕ, m' > m → ∃ k : ℕ+, (9^(2*k.val) - 8^(2*k.val) - 17) % m' ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l2675_267531


namespace NUMINAMATH_CALUDE_equation_solution_l2675_267540

theorem equation_solution :
  let f (x : ℝ) := (x^3 - x^2 - 4*x) / (x^2 + 5*x + 6) + x
  ∀ x : ℝ, f x = -4 ↔ x = (3 + Real.sqrt 105) / 4 ∨ x = (3 - Real.sqrt 105) / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2675_267540


namespace NUMINAMATH_CALUDE_magical_gate_diameter_l2675_267523

theorem magical_gate_diameter :
  let circle_equation := fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y + 3 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    2 * radius = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_magical_gate_diameter_l2675_267523


namespace NUMINAMATH_CALUDE_area_of_smaller_circle_l2675_267548

/-- Given two externally tangent circles with common tangent lines,
    where the tangent segment length is 6 and the radius of the larger circle
    is 3 times that of the smaller circle, prove that the area of the
    smaller circle is 12π/5. -/
theorem area_of_smaller_circle (r : ℝ) : 
  r > 0 →  -- radius of smaller circle is positive
  6^2 + r^2 = (4*r)^2 →  -- Pythagorean theorem applied to the tangent-radius triangle
  π * r^2 = 12*π/5 :=
by sorry

end NUMINAMATH_CALUDE_area_of_smaller_circle_l2675_267548


namespace NUMINAMATH_CALUDE_child_b_share_l2675_267579

theorem child_b_share (total_money : ℝ) (ratios : Fin 5 → ℝ) : 
  total_money = 10800 ∧ 
  ratios 0 = 0.5 ∧ 
  ratios 1 = 1.5 ∧ 
  ratios 2 = 2.25 ∧ 
  ratios 3 = 3.5 ∧ 
  ratios 4 = 4.25 → 
  (ratios 1 * total_money) / (ratios 0 + ratios 1 + ratios 2 + ratios 3 + ratios 4) = 1350 := by
sorry

end NUMINAMATH_CALUDE_child_b_share_l2675_267579


namespace NUMINAMATH_CALUDE_exists_five_threes_equal_100_l2675_267514

/-- An arithmetic expression using only the number 3, parentheses, and arithmetic operations. -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression. -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of threes in an expression. -/
def countThrees : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

/-- There exists an arithmetic expression using five threes that evaluates to 100. -/
theorem exists_five_threes_equal_100 : ∃ e : Expr, countThrees e = 5 ∧ eval e = 100 := by
  sorry

end NUMINAMATH_CALUDE_exists_five_threes_equal_100_l2675_267514


namespace NUMINAMATH_CALUDE_sqrt_31_between_5_and_6_l2675_267511

theorem sqrt_31_between_5_and_6 : 5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_31_between_5_and_6_l2675_267511


namespace NUMINAMATH_CALUDE_bridge_length_l2675_267571

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 235 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2675_267571


namespace NUMINAMATH_CALUDE_a_plus_b_equals_seven_l2675_267582

theorem a_plus_b_equals_seven (a b : ℝ) (h : ∀ x, a * (x + b) = 3 * x + 12) : a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_seven_l2675_267582


namespace NUMINAMATH_CALUDE_perfect_square_prime_exponents_l2675_267596

theorem perfect_square_prime_exponents (p q r : Nat) : 
  Prime p ∧ Prime q ∧ Prime r → 
  (∃ (n : Nat), p^q + p^r = n^2) ↔ 
  ((p = 2 ∧ q = 2 ∧ r = 5) ∨ 
   (p = 2 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 3) ∨ 
   (p = 3 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = r ∧ q ≥ 3 ∧ Odd q)) := by
  sorry

#check perfect_square_prime_exponents

end NUMINAMATH_CALUDE_perfect_square_prime_exponents_l2675_267596


namespace NUMINAMATH_CALUDE_total_bread_and_treats_l2675_267501

/-- The number of treats Jane brings -/
def jane_treats : ℕ := sorry

/-- The number of pieces of bread Jane brings -/
def jane_bread : ℕ := sorry

/-- The number of treats Wanda brings -/
def wanda_treats : ℕ := sorry

/-- The number of pieces of bread Wanda brings -/
def wanda_bread : ℕ := 90

theorem total_bread_and_treats :
  (jane_treats : ℚ) * (3 / 4) = jane_bread ∧
  (jane_treats : ℚ) / 2 = wanda_treats ∧
  3 * wanda_treats = wanda_bread ∧
  jane_treats + jane_bread + wanda_treats + wanda_bread = 225 := by sorry

end NUMINAMATH_CALUDE_total_bread_and_treats_l2675_267501


namespace NUMINAMATH_CALUDE_moles_of_ch4_combined_l2675_267541

-- Define the chemical reaction
structure Reaction where
  ch4 : ℝ
  cl2 : ℝ
  ch3cl : ℝ
  hcl : ℝ

-- Define the stoichiometric coefficients
def stoichiometric_ratio : Reaction → Prop :=
  fun r => r.ch4 = r.cl2 ∧ r.ch4 = r.ch3cl ∧ r.ch4 = r.hcl

-- Define the theorem
theorem moles_of_ch4_combined 
  (r : Reaction) 
  (h1 : stoichiometric_ratio r) 
  (h2 : r.ch3cl = 2) 
  (h3 : r.cl2 = 2) : 
  r.ch4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_moles_of_ch4_combined_l2675_267541


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_l2675_267589

def smallest_two_digit_prime_1 : Nat := 11
def smallest_two_digit_prime_2 : Nat := 13
def smallest_three_digit_prime : Nat := 101

theorem product_of_smallest_primes :
  smallest_two_digit_prime_1 * smallest_two_digit_prime_2 * smallest_three_digit_prime = 14443 := by
  sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_l2675_267589


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_arithmetic_sequence_has_671_terms_l2675_267515

/-- An arithmetic sequence starting at 2, with common difference 3, and last term 2014 -/
def ArithmeticSequence : ℕ → ℤ := fun n ↦ 2 + 3 * (n - 1)

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ ArithmeticSequence n = 2014 ∧ ∀ m : ℕ, m > n → ArithmeticSequence m > 2014 :=
by
  sorry

theorem arithmetic_sequence_has_671_terms :
  ∃! n : ℕ, n > 0 ∧ ArithmeticSequence n = 2014 ∧ n = 671 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_arithmetic_sequence_has_671_terms_l2675_267515


namespace NUMINAMATH_CALUDE_four_inch_cube_multi_painted_l2675_267570

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_faces : ℕ
  hsl : side_length = n
  hpf : painted_faces = 6

/-- Represents a smaller cube cut from a larger cube -/
structure SmallCube where
  painted_faces : ℕ

/-- Function to count cubes with at least two painted faces -/
def count_multi_painted_cubes (n : ℕ) : ℕ :=
  8 + 12 * (n - 2)

/-- Theorem statement -/
theorem four_inch_cube_multi_painted (c : Cube 4) :
  count_multi_painted_cubes c.side_length = 40 :=
sorry

end NUMINAMATH_CALUDE_four_inch_cube_multi_painted_l2675_267570


namespace NUMINAMATH_CALUDE_snowball_partition_l2675_267542

/-- A directed graph where each vertex has an out-degree of exactly 1 -/
structure SnowballGraph (V : Type) :=
  (edges : V → V)

/-- A partition of vertices into three sets -/
def ThreeTeamPartition (V : Type) := V → Fin 3

theorem snowball_partition {V : Type} (G : SnowballGraph V) :
  ∃ (partition : ThreeTeamPartition V),
    ∀ (v w : V), G.edges v = w → partition v ≠ partition w :=
sorry

end NUMINAMATH_CALUDE_snowball_partition_l2675_267542


namespace NUMINAMATH_CALUDE_bucket_capacity_l2675_267516

theorem bucket_capacity (tank_volume : ℝ) (bucket_count1 bucket_count2 : ℕ) 
  (bucket_capacity2 : ℝ) (h1 : bucket_count1 * bucket_capacity1 = tank_volume) 
  (h2 : bucket_count2 * bucket_capacity2 = tank_volume) 
  (h3 : bucket_count1 = 26) (h4 : bucket_count2 = 39) (h5 : bucket_capacity2 = 9) : 
  bucket_capacity1 = 13.5 :=
by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_l2675_267516


namespace NUMINAMATH_CALUDE_area_of_triangle_DBC_l2675_267591

/-- Given points A, B, C, D, and E in a coordinate plane, where D and E are midpoints of AB and BC respectively, prove that the area of triangle DBC is 30 square units. -/
theorem area_of_triangle_DBC (A B C D E : ℝ × ℝ) : 
  A = (0, 10) → 
  B = (0, 0) → 
  C = (12, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) → 
  (1 / 2) * (C.1 - B.1) * D.2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_DBC_l2675_267591


namespace NUMINAMATH_CALUDE_circle_area_tripled_l2675_267533

theorem circle_area_tripled (r n : ℝ) : 
  (r > 0) → (n > 0) → (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l2675_267533


namespace NUMINAMATH_CALUDE_cost_reduction_proof_l2675_267510

theorem cost_reduction_proof (total_reduction : ℝ) (years : ℕ) (annual_reduction : ℝ) : 
  total_reduction = 0.36 ∧ years = 2 → 
  (1 - annual_reduction) ^ years = 1 - total_reduction →
  annual_reduction = 0.2 := by
sorry

end NUMINAMATH_CALUDE_cost_reduction_proof_l2675_267510


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2675_267577

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_value (a b c : ℝ) (h : a ≠ 0) :
  f a b c (-3) = 7 →
  f a b c (-2) = 0 →
  f a b c 0 = -8 →
  f a b c 1 = -9 →
  f a b c 3 = -5 →
  f a b c 5 = 7 →
  f a b c 2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2675_267577


namespace NUMINAMATH_CALUDE_base_b_subtraction_divisibility_other_bases_divisible_l2675_267554

theorem base_b_subtraction_divisibility (b : ℤ) : 
  b = 6 ↔ (b^3 - 3*b^2 + 3*b - 2) % 5 ≠ 0 := by sorry

theorem other_bases_divisible : 
  ∀ b ∈ ({5, 7, 9, 10} : Set ℤ), (b^3 - 3*b^2 + 3*b - 2) % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_base_b_subtraction_divisibility_other_bases_divisible_l2675_267554


namespace NUMINAMATH_CALUDE_total_popsicle_sticks_popsicle_sum_l2675_267556

theorem total_popsicle_sticks : ℕ → ℕ → ℕ → ℕ
  | gino_sticks, your_sticks, nick_sticks =>
    gino_sticks + your_sticks + nick_sticks

theorem popsicle_sum (gino_sticks your_sticks nick_sticks : ℕ) 
  (h1 : gino_sticks = 63)
  (h2 : your_sticks = 50)
  (h3 : nick_sticks = 82) :
  total_popsicle_sticks gino_sticks your_sticks nick_sticks = 195 :=
by
  sorry

end NUMINAMATH_CALUDE_total_popsicle_sticks_popsicle_sum_l2675_267556


namespace NUMINAMATH_CALUDE_selling_price_with_gain_l2675_267564

/-- Given an article with a cost price where a $10 gain represents a 10% gain, 
    prove that the selling price is $110. -/
theorem selling_price_with_gain (cost_price : ℝ) 
  (h1 : cost_price > 0)
  (h2 : 10 / cost_price = 0.1) : 
  cost_price + 10 = 110 := by
  sorry

#check selling_price_with_gain

end NUMINAMATH_CALUDE_selling_price_with_gain_l2675_267564


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_dot_product_l2675_267508

/-- The ellipse with equation x²/2 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- A line with inclination angle 45° passing through a focus of the ellipse -/
def Line (f : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = p.1 - f.1}

/-- The dot product of two points in ℝ² -/
def dotProduct (p q : ℝ × ℝ) : ℝ :=
  p.1 * q.1 + p.2 * q.2

theorem ellipse_line_intersection_dot_product :
  ∀ f A B : ℝ × ℝ,
  f ∈ Ellipse →
  f.2 = 0 →
  A ∈ Ellipse →
  B ∈ Ellipse →
  A ∈ Line f →
  B ∈ Line f →
  A ≠ B →
  dotProduct A B = -1/3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_dot_product_l2675_267508


namespace NUMINAMATH_CALUDE_inheritance_calculation_l2675_267593

theorem inheritance_calculation (inheritance : ℝ) : 
  inheritance * 0.25 + (inheritance - inheritance * 0.25) * 0.15 = 15000 → 
  inheritance = 41379 := by
sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l2675_267593


namespace NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l2675_267503

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the center and right focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (1, 0)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- State the theorem
theorem min_dot_product_on_ellipse :
  ∃ (min : ℝ), min = 1/2 ∧
  ∀ (P : ℝ × ℝ), is_on_ellipse P.1 P.2 →
    dot_product (P.1 - O.1, P.2 - O.2) (P.1 - F.1, P.2 - F.2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l2675_267503


namespace NUMINAMATH_CALUDE_second_quadrant_trig_identity_l2675_267546

/-- For any angle α in the second quadrant, (sin α / cos α) * √(1 / sin²α - 1) = -1 -/
theorem second_quadrant_trig_identity (α : Real) (h : π / 2 < α ∧ α < π) :
  (Real.sin α / Real.cos α) * Real.sqrt (1 / Real.sin α ^ 2 - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_trig_identity_l2675_267546


namespace NUMINAMATH_CALUDE_constraint_implies_sum_equals_nine_l2675_267581

open Real

/-- The maximum value of xy + xz + yz given the constraint -/
noncomputable def N : ℝ := sorry

/-- The minimum value of xy + xz + yz given the constraint -/
noncomputable def n : ℝ := sorry

/-- Theorem stating that N + 8n = 9 given the constraint -/
theorem constraint_implies_sum_equals_nine :
  ∀ x y z : ℝ, 3 * (x + y + z) = x^2 + y^2 + z^2 → N + 8 * n = 9 := by
  sorry

end NUMINAMATH_CALUDE_constraint_implies_sum_equals_nine_l2675_267581


namespace NUMINAMATH_CALUDE_correct_investment_equation_l2675_267532

/-- Represents the investment scenario over two years -/
def investment_scenario (initial_investment : ℝ) (total_investment : ℝ) (growth_rate : ℝ) : Prop :=
  initial_investment * (1 + growth_rate) + initial_investment * (1 + growth_rate)^2 = total_investment

/-- Theorem stating that the given equation correctly represents the investment scenario -/
theorem correct_investment_equation :
  investment_scenario 2500 6600 x = true :=
by
  sorry

end NUMINAMATH_CALUDE_correct_investment_equation_l2675_267532


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_equals_four_l2675_267567

/-- The function f with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * x^2 - a^2 * x

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * x - a^2

theorem local_minimum_implies_a_equals_four :
  ∀ a : ℝ, (∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → f a x ≥ f a 1) →
  f_derivative a 1 = 0 →
  a = 4 := by
  sorry

#check local_minimum_implies_a_equals_four

end NUMINAMATH_CALUDE_local_minimum_implies_a_equals_four_l2675_267567


namespace NUMINAMATH_CALUDE_satisfactory_grade_fraction_l2675_267561

-- Define the grade categories
inductive Grade
| A
| B
| C
| D
| F

-- Define a function to check if a grade is satisfactory
def isSatisfactory (g : Grade) : Bool :=
  match g with
  | Grade.A => true
  | Grade.B => true
  | Grade.C => true
  | _ => false

-- Define the distribution of grades
def gradeDistribution : List (Grade × Nat) :=
  [(Grade.A, 6), (Grade.B, 5), (Grade.C, 7), (Grade.D, 4), (Grade.F, 6)]

-- Theorem to prove
theorem satisfactory_grade_fraction :
  let totalStudents := (gradeDistribution.map (·.2)).sum
  let satisfactoryStudents := (gradeDistribution.filter (isSatisfactory ·.1)).map (·.2) |>.sum
  (satisfactoryStudents : Rat) / totalStudents = 9 / 14 := by
  sorry


end NUMINAMATH_CALUDE_satisfactory_grade_fraction_l2675_267561


namespace NUMINAMATH_CALUDE_knicks_win_probability_l2675_267572

/-- The probability of winning a single game for the Heat -/
def p : ℚ := 3/5

/-- The probability of winning a single game for the Knicks -/
def q : ℚ := 1 - p

/-- The number of games needed to win the tournament -/
def games_to_win : ℕ := 3

/-- The total number of games in the tournament -/
def total_games : ℕ := 5

/-- The probability of the Knicks winning the tournament in exactly 5 games -/
def knicks_win_prob : ℚ :=
  (Nat.choose 4 2 : ℚ) * q^2 * p^2 * q

theorem knicks_win_probability :
  knicks_win_prob = 432/3125 :=
sorry

end NUMINAMATH_CALUDE_knicks_win_probability_l2675_267572


namespace NUMINAMATH_CALUDE_crystal_barrettes_count_l2675_267524

/-- The cost of one set of barrettes in dollars -/
def barrette_cost : ℕ := 3

/-- The cost of one comb in dollars -/
def comb_cost : ℕ := 1

/-- The total amount spent by both girls in dollars -/
def total_spent : ℕ := 14

/-- The number of sets of barrettes Kristine bought -/
def kristine_barrettes : ℕ := 1

/-- The number of combs Kristine bought -/
def kristine_combs : ℕ := 1

/-- The number of combs Crystal bought -/
def crystal_combs : ℕ := 1

/-- 
Given the costs of barrettes and combs, and the purchasing information for Kristine and Crystal,
prove that Crystal bought 3 sets of barrettes.
-/
theorem crystal_barrettes_count : 
  ∃ (x : ℕ), 
    barrette_cost * (kristine_barrettes + x) + 
    comb_cost * (kristine_combs + crystal_combs) = 
    total_spent ∧ x = 3 := by
  sorry


end NUMINAMATH_CALUDE_crystal_barrettes_count_l2675_267524


namespace NUMINAMATH_CALUDE_monkey_climb_l2675_267520

theorem monkey_climb (tree_height : ℝ) (climb_rate : ℝ) (total_time : ℕ) 
  (h1 : tree_height = 19)
  (h2 : climb_rate = 3)
  (h3 : total_time = 17) :
  ∃ (slip_back : ℝ), 
    slip_back = 2 ∧ 
    (total_time - 1 : ℝ) * (climb_rate - slip_back) + climb_rate = tree_height :=
by sorry

end NUMINAMATH_CALUDE_monkey_climb_l2675_267520


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l2675_267568

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_angle : c / a = Real.sqrt 3 / 2
  h_dist : a + c = 2 + Real.sqrt 3

/-- The line passing through a focus of the ellipse -/
structure FocusLine where
  m : ℝ

/-- The theorem statement -/
theorem special_ellipse_properties (e : SpecialEllipse) (l : FocusLine) :
  (e.a = 2 ∧ e.b = 1) ∧
  (l.m = Real.sqrt 2 ∨ l.m = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l2675_267568


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2675_267513

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧
  y₀ = f x₀ ∧
  (Real.log x₀ + 1) * 0 - (-1) = (Real.log x₀ + 1) * x₀ - y₀ ∧
  ∀ (x y : ℝ), y = Real.log x₀ + 1 * (x - x₀) + y₀ ↔ x - y - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2675_267513


namespace NUMINAMATH_CALUDE_max_m_inequality_l2675_267580

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y ≥ m/(2*x + y)) ∧
  (∀ (n : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y ≥ n/(2*x + y)) → n ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l2675_267580


namespace NUMINAMATH_CALUDE_gcd_72_120_l2675_267565

theorem gcd_72_120 : Nat.gcd 72 120 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_72_120_l2675_267565


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2675_267526

theorem z_in_first_quadrant : 
  ∀ z : ℂ, z / (1 + Complex.I) = 2 - Complex.I → 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2675_267526


namespace NUMINAMATH_CALUDE_fence_construction_l2675_267599

/-- A fence construction problem -/
theorem fence_construction (panels : ℕ) (sheets_per_panel : ℕ) (beams_per_panel : ℕ) 
  (rods_per_sheet : ℕ) (total_rods : ℕ) :
  panels = 10 →
  sheets_per_panel = 3 →
  beams_per_panel = 2 →
  rods_per_sheet = 10 →
  total_rods = 380 →
  (total_rods - panels * sheets_per_panel * rods_per_sheet) / (panels * beams_per_panel) = 4 :=
by sorry

end NUMINAMATH_CALUDE_fence_construction_l2675_267599


namespace NUMINAMATH_CALUDE_total_pears_picked_l2675_267522

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17

theorem total_pears_picked :
  alyssa_pears + nancy_pears = 59 := by sorry

end NUMINAMATH_CALUDE_total_pears_picked_l2675_267522


namespace NUMINAMATH_CALUDE_larger_solid_volume_is_seven_halves_l2675_267512

-- Define the rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the function to calculate the volume of the larger solid
def largerSolidVolume (prism : RectangularPrism) (plane : Plane3D) : ℝ := sorry

-- Theorem statement
theorem larger_solid_volume_is_seven_halves :
  let prism := RectangularPrism.mk 2 3 1
  let A := Point3D.mk 0 0 0
  let B := Point3D.mk 3 0 0
  let E := Point3D.mk 0 3 0
  let F := Point3D.mk 0 3 1
  let G := Point3D.mk 3 3 1
  let P := Point3D.mk 1.5 (3/2) (1/2)
  let Q := Point3D.mk 0 (3/2) (1/2)
  let plane := Plane3D.mk 1 1 1 0  -- Placeholder plane equation
  largerSolidVolume prism plane = 7/2 := by
  sorry


end NUMINAMATH_CALUDE_larger_solid_volume_is_seven_halves_l2675_267512


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2675_267557

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ x = 2 - Real.sqrt 5 ∨ x = 2 + Real.sqrt 5) ∧
  (∀ x : ℝ, 3*x^2 - 5*x + 1 = 0 ↔ x = (5 - Real.sqrt 13) / 6 ∨ x = (5 + Real.sqrt 13) / 6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2675_267557


namespace NUMINAMATH_CALUDE_mistaken_quotient_l2675_267536

theorem mistaken_quotient (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 21 = 32) : D / 12 = 56 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_quotient_l2675_267536


namespace NUMINAMATH_CALUDE_smallest_cube_multiple_l2675_267545

theorem smallest_cube_multiple : 
  ∃ (x : ℕ+) (M : ℤ), 
    (1890 : ℤ) * (x : ℤ) = M^3 ∧ 
    (∀ (y : ℕ+) (N : ℤ), (1890 : ℤ) * (y : ℤ) = N^3 → x ≤ y) ∧
    x = 4900 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_multiple_l2675_267545


namespace NUMINAMATH_CALUDE_equation_solution_l2675_267559

theorem equation_solution (a b x : ℤ) : 
  (a * x^2 + b * x + 14)^2 + (b * x^2 + a * x + 8)^2 = 0 →
  a = -6 ∧ b = -5 ∧ x = -2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2675_267559


namespace NUMINAMATH_CALUDE_orange_sales_l2675_267553

theorem orange_sales (alice_oranges emily_oranges total_oranges : ℕ) : 
  alice_oranges = 120 →
  alice_oranges = 2 * emily_oranges →
  total_oranges = alice_oranges + emily_oranges →
  total_oranges = 180 := by
  sorry

end NUMINAMATH_CALUDE_orange_sales_l2675_267553


namespace NUMINAMATH_CALUDE_parabola_equation_l2675_267529

/-- A parabola with focus at (5,0) has the standard equation y^2 = 20x -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (F : ℝ × ℝ), F = (5, 0) ∧ 
   ∀ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y → 
   (P.1 - F.1)^2 + P.2^2 = (P.1 - 2.5)^2) → 
  y^2 = 20 * x := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2675_267529


namespace NUMINAMATH_CALUDE_clock_synchronization_l2675_267566

/-- Represents the chiming behavior of a clock -/
structure Clock where
  strikes_per_hour : ℕ
  chime_rate : ℚ

/-- The scenario of the King's and Queen's clocks -/
def clock_scenario (h : ℕ) : Prop :=
  let king_clock : Clock := { strikes_per_hour := h, chime_rate := 3/2 }
  let queen_clock : Clock := { strikes_per_hour := h, chime_rate := 1 }
  (king_clock.chime_rate * queen_clock.strikes_per_hour : ℚ) + 2 = h

/-- The theorem stating that the synchronization occurs at 5 o'clock -/
theorem clock_synchronization : 
  clock_scenario 5 := by sorry

end NUMINAMATH_CALUDE_clock_synchronization_l2675_267566


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l2675_267578

/-- A trinomial ax^2 + bxy + cy^2 is a perfect square if and only if b^2 = 4ac -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  b^2 = 4*a*c

/-- The value of k for which 9x^2 - kxy + 4y^2 is a perfect square trinomial -/
theorem perfect_square_trinomial_k : 
  ∃ (k : ℝ), is_perfect_square_trinomial 9 (-k) 4 ∧ (k = 12 ∨ k = -12) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l2675_267578
