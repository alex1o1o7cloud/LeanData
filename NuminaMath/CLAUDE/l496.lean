import Mathlib

namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l496_49684

theorem fourth_root_equation_solution (x : ℝ) :
  (x^3)^(1/4) = 81 * 81^(1/16) → x = 243 * 9^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l496_49684


namespace NUMINAMATH_CALUDE_parabola_vertex_l496_49609

/-- The equation of a parabola is x^2 - 4x + 3y + 8 = 0. -/
def parabola_equation (x y : ℝ) : Prop := x^2 - 4*x + 3*y + 8 = 0

/-- The vertex of a parabola is the point where it reaches its maximum or minimum y-value. -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x' y', eq x' y' → y ≤ y' ∨ y ≥ y'

/-- The vertex of the parabola defined by x^2 - 4x + 3y + 8 = 0 is (2, -4/3). -/
theorem parabola_vertex : is_vertex 2 (-4/3) parabola_equation := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l496_49609


namespace NUMINAMATH_CALUDE_sara_grew_four_onions_l496_49617

/-- The number of onions grown by Sara, given the total number of onions and the numbers grown by Sally and Fred. -/
def saras_onions (total : ℕ) (sallys : ℕ) (freds : ℕ) : ℕ :=
  total - sallys - freds

/-- Theorem stating that Sara grew 4 onions given the conditions of the problem. -/
theorem sara_grew_four_onions :
  let total := 18
  let sallys := 5
  let freds := 9
  saras_onions total sallys freds = 4 := by
  sorry

end NUMINAMATH_CALUDE_sara_grew_four_onions_l496_49617


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l496_49659

theorem magnitude_of_complex_power (z : ℂ) (n : ℕ) (h : z = 4/5 + 3/5 * I) :
  Complex.abs (z^n) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l496_49659


namespace NUMINAMATH_CALUDE_combined_salaries_l496_49691

/-- The combined salaries of A, C, D, and E given B's salary and the average salary of all five. -/
theorem combined_salaries 
  (salary_B : ℕ) 
  (average_salary : ℕ) 
  (num_individuals : ℕ) 
  (h1 : salary_B = 5000)
  (h2 : average_salary = 8400)
  (h3 : num_individuals = 5) :
  average_salary * num_individuals - salary_B = 37000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l496_49691


namespace NUMINAMATH_CALUDE_inequality_proof_l496_49645

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π / 2) : 
  π / 2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2 * x) + Real.sin (2 * y) + Real.sin (2 * z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l496_49645


namespace NUMINAMATH_CALUDE_range_of_c_l496_49646

def A (c : ℝ) := {x : ℝ | |x - 1| < c}
def B := {x : ℝ | |x - 3| > 4}

theorem range_of_c :
  ∀ c : ℝ, (A c ∩ B = ∅) ↔ c ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l496_49646


namespace NUMINAMATH_CALUDE_max_min_difference_l496_49619

theorem max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (heq : x + 2 * y = 4) :
  ∃ (max min : ℝ), 
    (∀ z w : ℝ, z ≠ 0 → w ≠ 0 → z + 2 * w = 4 → |2 * z - w| / (|z| + |w|) ≤ max) ∧
    (∀ z w : ℝ, z ≠ 0 → w ≠ 0 → z + 2 * w = 4 → min ≤ |2 * z - w| / (|z| + |w|)) ∧
    max - min = 5 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_l496_49619


namespace NUMINAMATH_CALUDE_negation_of_existence_sqrt_leq_x_minus_one_negation_l496_49640

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) := by sorry

theorem sqrt_leq_x_minus_one_negation :
  (¬ ∃ x > 0, Real.sqrt x ≤ x - 1) ↔ (∀ x > 0, Real.sqrt x > x - 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_sqrt_leq_x_minus_one_negation_l496_49640


namespace NUMINAMATH_CALUDE_shared_edge_angle_l496_49604

-- Define the angle of a regular decagon
def decagon_angle : ℝ := 144

-- Define the angle of a square
def square_angle : ℝ := 90

-- Theorem statement
theorem shared_edge_angle (x : ℝ) : 
  x + 36 + (360 - decagon_angle) + square_angle = 360 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_shared_edge_angle_l496_49604


namespace NUMINAMATH_CALUDE_curve_symmetry_l496_49630

/-- A curve f is symmetric with respect to the line x - y - 3 = 0 if and only if
    it can be expressed as f(y+3, x-3) = 0 for all x and y. -/
theorem curve_symmetry (f : ℝ → ℝ → ℝ) :
  (∀ x y, f x y = 0 ↔ f (y + 3) (x - 3) = 0) ↔
  (∀ x y, (x - y = 3) → (f x y = 0 ↔ f y x = 0)) :=
sorry

end NUMINAMATH_CALUDE_curve_symmetry_l496_49630


namespace NUMINAMATH_CALUDE_sedan_count_l496_49626

theorem sedan_count (trucks sedans motorcycles : ℕ) : 
  trucks * 7 = sedans * 3 →
  sedans * 2 = motorcycles * 7 →
  motorcycles = 2600 →
  sedans = 9100 := by
sorry

end NUMINAMATH_CALUDE_sedan_count_l496_49626


namespace NUMINAMATH_CALUDE_fast_area_scientific_notation_l496_49689

/-- The area of the reflecting surface of the FAST radio telescope in square meters -/
def fast_area : ℝ := 250000

/-- Scientific notation representation of the FAST area -/
def fast_area_scientific : ℝ := 2.5 * (10 ^ 5)

/-- Theorem stating that the FAST area is equal to its scientific notation representation -/
theorem fast_area_scientific_notation : fast_area = fast_area_scientific := by
  sorry

end NUMINAMATH_CALUDE_fast_area_scientific_notation_l496_49689


namespace NUMINAMATH_CALUDE_robert_basic_salary_l496_49697

/-- Represents Robert's financial situation --/
structure RobertFinances where
  basic_salary : ℝ
  total_sales : ℝ
  monthly_expenses : ℝ

/-- Calculates Robert's total earnings --/
def total_earnings (r : RobertFinances) : ℝ :=
  r.basic_salary + 0.1 * r.total_sales

/-- Theorem stating Robert's basic salary --/
theorem robert_basic_salary :
  ∃ (r : RobertFinances),
    r.total_sales = 23600 ∧
    r.monthly_expenses = 2888 ∧
    0.8 * (total_earnings r) = r.monthly_expenses ∧
    r.basic_salary = 1250 := by
  sorry


end NUMINAMATH_CALUDE_robert_basic_salary_l496_49697


namespace NUMINAMATH_CALUDE_flu_virus_diameter_scientific_notation_l496_49631

theorem flu_virus_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000054 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 5.4 ∧ n = -6 :=
sorry

end NUMINAMATH_CALUDE_flu_virus_diameter_scientific_notation_l496_49631


namespace NUMINAMATH_CALUDE_octahedron_tetrahedron_volume_ratio_l496_49614

/-- The volume of a regular tetrahedron with edge length a -/
noncomputable def tetrahedron_volume (a : ℝ) : ℝ := sorry

/-- The volume of a regular octahedron with edge length a -/
noncomputable def octahedron_volume (a : ℝ) : ℝ := sorry

/-- Theorem stating that the volume of a regular octahedron is 4 times 
    the volume of a regular tetrahedron with the same edge length -/
theorem octahedron_tetrahedron_volume_ratio (a : ℝ) (h : a > 0) : 
  octahedron_volume a = 4 * tetrahedron_volume a := by sorry

end NUMINAMATH_CALUDE_octahedron_tetrahedron_volume_ratio_l496_49614


namespace NUMINAMATH_CALUDE_power_of_fraction_l496_49620

theorem power_of_fraction :
  (5 : ℚ) / 6 ^ 4 = 625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_l496_49620


namespace NUMINAMATH_CALUDE_julie_lettuce_purchase_l496_49608

/-- The total pounds of lettuce Julie bought -/
def total_lettuce (green_cost red_cost price_per_pound : ℚ) : ℚ :=
  green_cost / price_per_pound + red_cost / price_per_pound

/-- Proof that Julie bought 7 pounds of lettuce -/
theorem julie_lettuce_purchase : 
  total_lettuce 8 6 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_julie_lettuce_purchase_l496_49608


namespace NUMINAMATH_CALUDE_fly_path_on_cone_l496_49661

/-- A right circular cone -/
structure Cone where
  base_radius : ℝ
  height : ℝ

/-- A point on the surface of a cone -/
structure ConePoint where
  distance_from_vertex : ℝ

/-- The shortest distance between two points on the surface of a cone -/
def shortest_distance (c : Cone) (p1 p2 : ConePoint) : ℝ := sorry

/-- The theorem statement -/
theorem fly_path_on_cone :
  let c : Cone := { base_radius := 600, height := 200 * Real.sqrt 7 }
  let p1 : ConePoint := { distance_from_vertex := 125 }
  let p2 : ConePoint := { distance_from_vertex := 375 * Real.sqrt 2 }
  shortest_distance c p1 p2 = 625 := by sorry

end NUMINAMATH_CALUDE_fly_path_on_cone_l496_49661


namespace NUMINAMATH_CALUDE_no_natural_number_with_sum_of_squared_divisors_perfect_square_l496_49642

theorem no_natural_number_with_sum_of_squared_divisors_perfect_square :
  ¬ ∃ (n : ℕ), ∃ (d₁ d₂ d₃ d₄ d₅ : ℕ), 
    (∀ d : ℕ, d ∣ n → d ≥ d₅) ∧ 
    (d₁ ∣ n) ∧ (d₂ ∣ n) ∧ (d₃ ∣ n) ∧ (d₄ ∣ n) ∧ (d₅ ∣ n) ∧
    (d₁ < d₂) ∧ (d₂ < d₃) ∧ (d₃ < d₄) ∧ (d₄ < d₅) ∧
    ∃ (m : ℕ), d₁^2 + d₂^2 + d₃^2 + d₄^2 + d₅^2 = m^2 :=
by
  sorry


end NUMINAMATH_CALUDE_no_natural_number_with_sum_of_squared_divisors_perfect_square_l496_49642


namespace NUMINAMATH_CALUDE_probability_of_specific_colors_l496_49622

def black_balls : ℕ := 5
def white_balls : ℕ := 7
def green_balls : ℕ := 2
def blue_balls : ℕ := 3
def red_balls : ℕ := 4

def total_balls : ℕ := black_balls + white_balls + green_balls + blue_balls + red_balls

def favorable_outcomes : ℕ := black_balls * green_balls * red_balls

def total_outcomes : ℕ := (total_balls.choose 3)

theorem probability_of_specific_colors : 
  (favorable_outcomes : ℚ) / total_outcomes = 4 / 133 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_colors_l496_49622


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l496_49664

theorem parabola_y_intercepts :
  let f (y : ℝ) := 3 * y^2 - 6 * y + 1
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ f y₁ = 0 ∧ f y₂ = 0 ∧ ∀ y, f y = 0 → y = y₁ ∨ y = y₂ :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l496_49664


namespace NUMINAMATH_CALUDE_ball_piles_problem_l496_49693

theorem ball_piles_problem (x y z a : ℕ) : 
  x + y + z = 2012 →
  y - a = 17 →
  x - a = 2 * (z - a) →
  z = 665 := by
sorry

end NUMINAMATH_CALUDE_ball_piles_problem_l496_49693


namespace NUMINAMATH_CALUDE_h_value_at_4_l496_49613

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x - 5

-- Define the properties of h
def is_valid_h (h : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ),
    (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    (∀ x, h x = 0 ↔ x = a^2 ∨ x = b^2 ∨ x = c^2) ∧
    (h 1 = 2)

-- Theorem statement
theorem h_value_at_4 (h : ℝ → ℝ) (hvalid : is_valid_h h) : h 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_h_value_at_4_l496_49613


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l496_49635

theorem quadratic_equation_problem (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*(a-1)*x + a^2 - 7*a - 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁*x₂ - 3*x₁ - 3*x₂ - 2 = 0 →
  (1 + 4/(a^2 - 4)) * (a + 2)/a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l496_49635


namespace NUMINAMATH_CALUDE_negation_of_p_l496_49668

-- Define the proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

-- State the theorem
theorem negation_of_p (f : ℝ → ℝ) :
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_p_l496_49668


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l496_49627

theorem boat_speed_in_still_water : 
  ∀ (v_b v_c v_w : ℝ),
    v_b - v_c - v_w = 4 →
    v_c ≤ 4 →
    v_w ≥ -1 →
    v_b = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l496_49627


namespace NUMINAMATH_CALUDE_roberts_chocolates_l496_49658

theorem roberts_chocolates (nickel_chocolates : ℕ) (difference : ℕ) : nickel_chocolates = 2 → difference = 7 → nickel_chocolates + difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_roberts_chocolates_l496_49658


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l496_49650

theorem complex_expression_equals_negative_two :
  (2023 * Real.pi) ^ 0 + (-1/2)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (Real.pi / 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l496_49650


namespace NUMINAMATH_CALUDE_eighth_term_value_l496_49624

/-- An arithmetic sequence with 30 terms, first term 5, and last term 80 -/
def arithmeticSequence (n : ℕ) : ℚ :=
  let d := (80 - 5) / 29
  5 + (n - 1) * d

/-- The 8th term of the arithmetic sequence -/
def eighthTerm : ℚ := arithmeticSequence 8

theorem eighth_term_value : eighthTerm = 670 / 29 := by sorry

end NUMINAMATH_CALUDE_eighth_term_value_l496_49624


namespace NUMINAMATH_CALUDE_parent_son_age_ratio_l496_49669

/-- The ratio of a parent's age to their son's age -/
def age_ratio (parent_age : ℕ) (son_age : ℕ) : ℚ :=
  parent_age / son_age

theorem parent_son_age_ratio :
  let parent_age : ℕ := 35
  let son_age : ℕ := 7
  age_ratio parent_age son_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_parent_son_age_ratio_l496_49669


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l496_49655

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Calculates the number of the selected student in a given group. -/
def selected_number_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.selected_number + s.students_per_group * (group - s.selected_group)

/-- Theorem stating that if student 12 is selected from group 3, 
    then student 37 will be selected from group 8 in a systematic sampling of 50 students. -/
theorem systematic_sampling_theorem (s : SystematicSampling) :
  s.total_students = 50 ∧ 
  s.num_groups = 10 ∧ 
  s.students_per_group = 5 ∧ 
  s.selected_number = 12 ∧ 
  s.selected_group = 3 →
  selected_number_in_group s 8 = 37 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l496_49655


namespace NUMINAMATH_CALUDE_odd_function_log_property_l496_49637

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_log_property (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_pos : ∀ x > 0, f x = Real.log (x + 1)) : 
  ∀ x < 0, f x = -Real.log (1 - x) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_log_property_l496_49637


namespace NUMINAMATH_CALUDE_triangle_circumcircle_radius_l496_49633

theorem triangle_circumcircle_radius 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π) -- Sum of angles in a triangle
  (h5 : Real.sin C + Real.sin B = 4 * Real.sin A) -- Given condition
  (h6 : a = 2) -- Given condition
  (h7 : a = 2 * Real.sin (A/2) * R) -- Relation between side and circumradius
  (h8 : b = 2 * Real.sin (B/2) * R) -- Relation between side and circumradius
  (h9 : c = 2 * Real.sin (C/2) * R) -- Relation between side and circumradius
  (h10 : ∀ R' > 0, R ≤ R') -- R is the minimum possible radius
  : R = 8 * Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_radius_l496_49633


namespace NUMINAMATH_CALUDE_matrix_power_eight_l496_49649

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 1, 1]

theorem matrix_power_eight :
  A^8 = !![16, 0; 0, 16] := by sorry

end NUMINAMATH_CALUDE_matrix_power_eight_l496_49649


namespace NUMINAMATH_CALUDE_monthly_cost_correct_l496_49696

/-- Represents the monthly cost for online access -/
def monthly_cost : ℝ := 8

/-- Represents the initial app cost -/
def app_cost : ℝ := 5

/-- Represents the number of months of online access -/
def months : ℝ := 2

/-- Represents the total cost for the app and online access -/
def total_cost : ℝ := 21

/-- Proves that the monthly cost for online access is correct -/
theorem monthly_cost_correct : app_cost + months * monthly_cost = total_cost := by
  sorry

end NUMINAMATH_CALUDE_monthly_cost_correct_l496_49696


namespace NUMINAMATH_CALUDE_triangle_operation_result_l496_49695

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a * b / (-6)

-- State the theorem
theorem triangle_operation_result :
  triangle 4 (triangle 3 2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_operation_result_l496_49695


namespace NUMINAMATH_CALUDE_divisibility_by_101_l496_49603

theorem divisibility_by_101 : ∃! (x y : Nat), x < 10 ∧ y < 10 ∧ (201300 + 10 * x + y) % 101 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_101_l496_49603


namespace NUMINAMATH_CALUDE_discount_calculation_l496_49687

/-- Calculates the discounted price for a purchase with a percentage discount on amounts over a threshold --/
def discountedPrice (itemCount : ℕ) (itemPrice : ℚ) (discountPercentage : ℚ) (discountThreshold : ℚ) : ℚ :=
  let totalPrice := itemCount * itemPrice
  let amountOverThreshold := max (totalPrice - discountThreshold) 0
  let discountAmount := discountPercentage * amountOverThreshold
  totalPrice - discountAmount

/-- Proves that for a purchase of 7 items at $200 each, with a 10% discount on amounts over $1000, the final cost is $1360 --/
theorem discount_calculation :
  discountedPrice 7 200 0.1 1000 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l496_49687


namespace NUMINAMATH_CALUDE_ace_diamond_probability_l496_49605

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of Diamonds in a standard deck -/
def NumDiamonds : ℕ := 13

/-- Probability of drawing an Ace as the first card and a Diamond as the second card -/
def prob_ace_then_diamond (deck : ℕ) (aces : ℕ) (diamonds : ℕ) : ℚ :=
  (aces : ℚ) / (deck : ℚ) * (diamonds : ℚ) / ((deck - 1) : ℚ)

theorem ace_diamond_probability :
  prob_ace_then_diamond StandardDeck NumAces NumDiamonds = 1 / StandardDeck :=
sorry

end NUMINAMATH_CALUDE_ace_diamond_probability_l496_49605


namespace NUMINAMATH_CALUDE_lukes_trips_l496_49662

/-- Luke's tray-carrying problem -/
theorem lukes_trips (trays_per_trip : ℕ) (trays_table1 : ℕ) (trays_table2 : ℕ)
  (h1 : trays_per_trip = 4)
  (h2 : trays_table1 = 20)
  (h3 : trays_table2 = 16) :
  (trays_table1 + trays_table2) / trays_per_trip = 9 :=
by sorry

end NUMINAMATH_CALUDE_lukes_trips_l496_49662


namespace NUMINAMATH_CALUDE_certain_number_plus_two_l496_49615

theorem certain_number_plus_two (x : ℝ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_plus_two_l496_49615


namespace NUMINAMATH_CALUDE_stevens_height_l496_49667

-- Define the building's height and shadow length
def building_height : ℝ := 50
def building_shadow : ℝ := 25

-- Define Steven's shadow length
def steven_shadow : ℝ := 20

-- Define the theorem
theorem stevens_height :
  ∃ (h : ℝ), h = (building_height / building_shadow) * steven_shadow ∧ h = 40 :=
by sorry

end NUMINAMATH_CALUDE_stevens_height_l496_49667


namespace NUMINAMATH_CALUDE_cos_half_angle_l496_49688

theorem cos_half_angle (θ : ℝ) (h1 : |Real.cos θ| = (1 : ℝ) / 5) 
  (h2 : (7 : ℝ) * Real.pi / 2 < θ) (h3 : θ < 4 * Real.pi) : 
  Real.cos (θ / 2) = Real.sqrt 15 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_half_angle_l496_49688


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_theorem_l496_49686

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Predicate to check if a quadrilateral is inscribed -/
def is_inscribed (q : Quadrilateral) : Prop := sorry

/-- Function to get the radius of the circumscribed circle of a quadrilateral -/
def circumscribed_radius (q : Quadrilateral) : ℝ := sorry

/-- Predicate to check if a point is inside a quadrilateral -/
def is_inside (P : Point) (q : Quadrilateral) : Prop := sorry

/-- Function to divide a quadrilateral into four parts given an internal point -/
def divide_quadrilateral (q : Quadrilateral) (P : Point) : 
  (Quadrilateral × Quadrilateral × Quadrilateral × Quadrilateral) := sorry

theorem inscribed_quadrilateral_theorem 
  (ABCD : Quadrilateral) 
  (P : Point) 
  (h_inscribed : is_inscribed ABCD) 
  (h_inside : is_inside P ABCD) :
  let (APB, BPC, CPD, APD) := divide_quadrilateral ABCD P
  (is_inscribed APB ∧ is_inscribed BPC ∧ is_inscribed CPD) →
  (circumscribed_radius APB = circumscribed_radius BPC) →
  (circumscribed_radius BPC = circumscribed_radius CPD) →
  (is_inscribed APD ∧ circumscribed_radius APD = circumscribed_radius APB) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_theorem_l496_49686


namespace NUMINAMATH_CALUDE_g_2010_equals_one_l496_49677

/-- A function satisfying the given properties -/
def g_function (g : ℝ → ℝ) : Prop :=
  (∀ x > 0, g x > 0) ∧ 
  (∀ x y, x > y ∧ y > 0 → g (x - y) = (g (x * y) + 1) ^ (1/3)) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x - y = x * y ∧ x * y = 2010)

/-- The main theorem stating that g(2010) = 1 -/
theorem g_2010_equals_one (g : ℝ → ℝ) (h : g_function g) : g 2010 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_2010_equals_one_l496_49677


namespace NUMINAMATH_CALUDE_hyperbola_equation_l496_49685

theorem hyperbola_equation (x y : ℝ) :
  (∀ t : ℝ, y = (2/3) * x ∨ y = -(2/3) * x) →  -- asymptotes condition
  (∃ x₀ y₀ : ℝ, x₀ = 3 ∧ y₀ = 4 ∧ (y₀^2 / 12 - x₀^2 / 27 = 1)) →  -- point condition
  (y^2 / 12 - x^2 / 27 = 1) :=  -- equation of the hyperbola
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l496_49685


namespace NUMINAMATH_CALUDE_exists_a_C_is_line_C_passes_through_origin_L_intersects_C_l496_49610

-- Define the curve C
def C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1^2 + a * p.2^2 - 2 * p.1 - 2 * p.2 = 0}

-- Define a straight line
def isLine (S : Set (ℝ × ℝ)) : Prop :=
  ∃ A B C : ℝ, ∀ p : ℝ × ℝ, p ∈ S ↔ A * p.1 + B * p.2 + C = 0

-- Statement 1: C is a straight line for some a
theorem exists_a_C_is_line : ∃ a : ℝ, isLine (C a) := by sorry

-- Statement 2: C passes through (0, 0) for all a
theorem C_passes_through_origin : ∀ a : ℝ, (0, 0) ∈ C a := by sorry

-- Define the line x + 2y = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + 2 * p.2 = 0}

-- Statement 3: When a = 1, L intersects C
theorem L_intersects_C : (C 1) ∩ L ≠ ∅ := by sorry

end NUMINAMATH_CALUDE_exists_a_C_is_line_C_passes_through_origin_L_intersects_C_l496_49610


namespace NUMINAMATH_CALUDE_student_committee_candidates_l496_49625

theorem student_committee_candidates (n : ℕ) : n * (n - 1) = 72 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_student_committee_candidates_l496_49625


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l496_49602

theorem three_digit_number_proof :
  ∃! (a b c : ℕ),
    0 < a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    a + b + c = 16 ∧
    100 * b + 10 * a + c = 100 * a + 10 * b + c - 360 ∧
    100 * a + 10 * c + b = 100 * a + 10 * b + c + 54 ∧
    100 * a + 10 * b + c = 628 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l496_49602


namespace NUMINAMATH_CALUDE_no_more_permutations_than_value_l496_49644

theorem no_more_permutations_than_value (b n : ℕ) : b > 1 → n > 1 → 
  let r := (Nat.log b n).succ
  let digits := Nat.digits b n
  (List.permutations digits).length ≤ n := by
  sorry

end NUMINAMATH_CALUDE_no_more_permutations_than_value_l496_49644


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_line_relationships_l496_49698

/-- Two lines are different if they are not equal -/
def different_lines (m n : Line) : Prop := m ≠ n

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two lines are parallel -/
def lines_parallel (m n : Line) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_to_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perpendicular (α β : Plane) : Prop := sorry

theorem perpendicular_planes_from_line_relationships 
  (m n : Line) (α β : Plane) 
  (h1 : different_lines m n)
  (h2 : different_planes α β)
  (h3 : line_perpendicular_to_plane m α)
  (h4 : lines_parallel m n)
  (h5 : line_parallel_to_plane n β) :
  planes_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_line_relationships_l496_49698


namespace NUMINAMATH_CALUDE_remaining_flavors_to_try_l496_49694

def ice_cream_flavors (total : ℕ) (tried_two_years_ago : ℕ) (tried_last_year : ℕ) : Prop :=
  tried_two_years_ago = total / 4 ∧
  tried_last_year = 2 * tried_two_years_ago ∧
  tried_two_years_ago + tried_last_year + 25 = total

theorem remaining_flavors_to_try
  (total : ℕ)
  (tried_two_years_ago : ℕ)
  (tried_last_year : ℕ)
  (h : ice_cream_flavors total tried_two_years_ago tried_last_year)
  (h_total : total = 100) :
  25 = total - (tried_two_years_ago + tried_last_year) :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_flavors_to_try_l496_49694


namespace NUMINAMATH_CALUDE_sum_one_to_fortyfive_base6_l496_49636

/-- Represents a number in base 6 --/
def Base6 := ℕ

/-- Converts a natural number to its base 6 representation --/
def to_base6 (n : ℕ) : Base6 := sorry

/-- Converts a base 6 number to its natural number representation --/
def from_base6 (b : Base6) : ℕ := sorry

/-- Adds two base 6 numbers --/
def add_base6 (a b : Base6) : Base6 := sorry

/-- Multiplies two base 6 numbers --/
def mul_base6 (a b : Base6) : Base6 := sorry

/-- Divides a base 6 number by 2 --/
def div2_base6 (a : Base6) : Base6 := sorry

/-- Calculates the sum of an arithmetic sequence in base 6 --/
def sum_arithmetic_base6 (first last : Base6) : Base6 :=
  let n := add_base6 (from_base6 last) (to_base6 1)
  div2_base6 (mul_base6 n (add_base6 first last))

/-- The main theorem to be proved --/
theorem sum_one_to_fortyfive_base6 :
  sum_arithmetic_base6 (to_base6 1) (to_base6 45) = to_base6 2003 := by sorry

end NUMINAMATH_CALUDE_sum_one_to_fortyfive_base6_l496_49636


namespace NUMINAMATH_CALUDE_x_range_proof_l496_49621

theorem x_range_proof (x : ℝ) : 
  (∀ θ : ℝ, 0 < θ ∧ θ < π/2 → 1/(Real.sin θ)^2 + 4/(Real.cos θ)^2 ≥ |2*x - 1|) 
  ↔ -4 ≤ x ∧ x ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_x_range_proof_l496_49621


namespace NUMINAMATH_CALUDE_books_about_science_l496_49607

theorem books_about_science 
  (total_books : ℕ) 
  (school_books : ℕ) 
  (sports_books : ℕ) 
  (h1 : total_books = 85) 
  (h2 : school_books = 19) 
  (h3 : sports_books = 35) :
  total_books - (school_books + sports_books) = 31 :=
by sorry

end NUMINAMATH_CALUDE_books_about_science_l496_49607


namespace NUMINAMATH_CALUDE_sector_radius_l496_49612

/-- Given a circular sector with central angle 5π/7 and perimeter 5π + 14, its radius is 7. -/
theorem sector_radius (r : ℝ) : 
  (5 / 7 : ℝ) * π * r + 2 * r = 5 * π + 14 → r = 7 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l496_49612


namespace NUMINAMATH_CALUDE_sandwich_shop_jalapeno_requirement_l496_49606

/-- Represents the number of jalapeno peppers required for a day's operation --/
def jalapeno_peppers_required (strips_per_sandwich : ℕ) (slices_per_pepper : ℕ) 
  (minutes_per_sandwich : ℕ) (hours_of_operation : ℕ) : ℕ :=
  let peppers_per_sandwich := strips_per_sandwich / slices_per_pepper
  let sandwiches_per_hour := 60 / minutes_per_sandwich
  let peppers_per_hour := peppers_per_sandwich * sandwiches_per_hour
  peppers_per_hour * hours_of_operation

/-- Theorem stating the number of jalapeno peppers required for the Sandwich Shop's 8-hour day --/
theorem sandwich_shop_jalapeno_requirement :
  jalapeno_peppers_required 4 8 5 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_shop_jalapeno_requirement_l496_49606


namespace NUMINAMATH_CALUDE_circumcircle_area_of_triangle_l496_49690

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that the area of its circumcircle is π/2 under certain conditions. -/
theorem circumcircle_area_of_triangle (a b c : Real) (S : Real) :
  a = 1 →
  4 * S = b^2 + c^2 - 1 →
  (∃ A B C : Real, 
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    A + B + C = π ∧
    S = (1/2) * b * c * Real.sin A) →
  (∃ R : Real, R > 0 ∧ π * R^2 = π/2) :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_area_of_triangle_l496_49690


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l496_49660

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 5) * (Real.sqrt 3 / Real.sqrt 6) * (Real.sqrt 4 / Real.sqrt 8) * (Real.sqrt 5 / Real.sqrt 9) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l496_49660


namespace NUMINAMATH_CALUDE_wire_cutting_l496_49628

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) :
  total_length = 30 →
  difference = 2 →
  total_length = shorter_piece + (shorter_piece + difference) →
  shorter_piece = 14 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l496_49628


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l496_49672

theorem three_digit_number_problem :
  ∃! x : ℕ, 100 ≤ x ∧ x < 1000 ∧ (x : ℚ) - (x : ℚ) / 10 = 201.6 :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l496_49672


namespace NUMINAMATH_CALUDE_seeds_per_row_in_top_bed_l496_49671

theorem seeds_per_row_in_top_bed (
  top_beds : Nat
  ) (bottom_beds : Nat)
  (rows_per_top_bed : Nat)
  (rows_per_bottom_bed : Nat)
  (seeds_per_row_bottom : Nat)
  (total_seeds : Nat)
  (h1 : top_beds = 2)
  (h2 : bottom_beds = 2)
  (h3 : rows_per_top_bed = 4)
  (h4 : rows_per_bottom_bed = 3)
  (h5 : seeds_per_row_bottom = 20)
  (h6 : total_seeds = 320) :
  (total_seeds - (bottom_beds * rows_per_bottom_bed * seeds_per_row_bottom)) / (top_beds * rows_per_top_bed) = 25 := by
  sorry

end NUMINAMATH_CALUDE_seeds_per_row_in_top_bed_l496_49671


namespace NUMINAMATH_CALUDE_video_game_players_l496_49638

theorem video_game_players (lives_per_player : ℕ) (total_lives : ℕ) (h1 : lives_per_player = 8) (h2 : total_lives = 64) :
  total_lives / lives_per_player = 8 :=
by sorry

end NUMINAMATH_CALUDE_video_game_players_l496_49638


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l496_49623

theorem fractional_equation_solution :
  ∃ (x : ℚ), (1 - x) / (2 - x) - 1 = (3 * x - 4) / (x - 2) ∧ x = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l496_49623


namespace NUMINAMATH_CALUDE_quadratic_inequality_l496_49639

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  solution_set : ∀ x : ℝ, (x < -2 ∨ x > 4) ↔ (a * x^2 + b * x + c > 0)

/-- The main theorem stating the inequality for specific x values -/
theorem quadratic_inequality (f : QuadraticFunction) :
  f.a * 2^2 + f.b * 2 + f.c < f.a * (-1)^2 + f.b * (-1) + f.c ∧
  f.a * (-1)^2 + f.b * (-1) + f.c < f.a * 5^2 + f.b * 5 + f.c :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l496_49639


namespace NUMINAMATH_CALUDE_largest_six_digit_number_l496_49600

/-- Represents a six-digit number where each digit, starting from the third,
    is the sum of the two preceding digits. -/
structure SixDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  h1 : c = a + b
  h2 : d = b + c
  h3 : e = c + d
  h4 : f = d + e
  h5 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10

/-- Converts a SixDigitNumber to its numerical value -/
def toNumber (n : SixDigitNumber) : Nat :=
  100000 * n.a + 10000 * n.b + 1000 * n.c + 100 * n.d + 10 * n.e + n.f

/-- The largest SixDigitNumber is 303369 -/
theorem largest_six_digit_number :
  ∀ n : SixDigitNumber, toNumber n ≤ 303369 := by
  sorry

end NUMINAMATH_CALUDE_largest_six_digit_number_l496_49600


namespace NUMINAMATH_CALUDE_shooting_scores_mode_and_variance_l496_49643

def scores : List ℕ := [8, 9, 9, 10, 10, 7, 8, 9, 10, 10]

def mode (l : List ℕ) : ℕ := 
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

def mean (l : List ℕ) : ℚ := 
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ := 
  let μ := mean l
  (l.map (λ x => ((x : ℚ) - μ) ^ 2)).sum / l.length

theorem shooting_scores_mode_and_variance :
  mode scores = 10 ∧ variance scores = 1 := by sorry

end NUMINAMATH_CALUDE_shooting_scores_mode_and_variance_l496_49643


namespace NUMINAMATH_CALUDE_complement_of_union_MN_l496_49681

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_of_union_MN : 
  (U \ (M ∪ N)) = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_MN_l496_49681


namespace NUMINAMATH_CALUDE_min_value_exponential_product_l496_49641

theorem min_value_exponential_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 3) :
  Real.exp (1 / a) * Real.exp (1 / b) ≥ Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_product_l496_49641


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_min_value_is_3_plus_2sqrt2_min_value_achieved_l496_49674

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/(2*b) = 1 → x + 4*y ≤ a + 4*b :=
by sorry

theorem min_value_is_3_plus_2sqrt2 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  x + 4*y ≥ 3 + 2*Real.sqrt 2 :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = 1 ∧ a + 4*b = 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_min_value_is_3_plus_2sqrt2_min_value_achieved_l496_49674


namespace NUMINAMATH_CALUDE_solution_set_f_gt_2_min_a_for_full_solution_set_l496_49670

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |2*x + 3|

-- Theorem for part (I)
theorem solution_set_f_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | -2 < x ∧ x < -4/3} :=
sorry

-- Theorem for part (II)
theorem min_a_for_full_solution_set (a : ℝ) :
  (∀ x, f x ≤ (3/2)*a^2 - a) ↔ a ≥ 5/3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_2_min_a_for_full_solution_set_l496_49670


namespace NUMINAMATH_CALUDE_roses_cut_l496_49647

theorem roses_cut (initial_roses vase_roses garden_roses : ℕ) 
  (h1 : initial_roses = 7)
  (h2 : vase_roses = 20)
  (h3 : garden_roses = 59) :
  vase_roses - initial_roses = 13 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l496_49647


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l496_49648

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (3 + Real.sqrt (4 * x - 5)) = Real.sqrt 10 → x = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l496_49648


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l496_49651

-- Define the function
def f (x : ℝ) : ℝ := (2*x - 1)^3

-- State the theorem
theorem tangent_slope_at_zero : 
  (deriv f) 0 = 6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l496_49651


namespace NUMINAMATH_CALUDE_P_bounds_l496_49699

/-- A convex n-gon divided into triangles by non-intersecting diagonals -/
structure ConvexNGon (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- Transformation that replaces triangles ABC and ACD with ABD and BCD -/
def transformation (n : ℕ) (g : ConvexNGon n) : ConvexNGon n := sorry

/-- P(n) is the minimum number of transformations required to convert any partition into any other partition -/
def P (n : ℕ) : ℕ := sorry

/-- Main theorem about bounds on P(n) -/
theorem P_bounds (n : ℕ) (g : ConvexNGon n) :
  P n ≥ n - 3 ∧
  P n ≤ 2*n - 7 ∧
  (n ≥ 13 → P n ≤ 2*n - 10) :=
sorry

end NUMINAMATH_CALUDE_P_bounds_l496_49699


namespace NUMINAMATH_CALUDE_inequality_equivalence_l496_49665

theorem inequality_equivalence (x : ℝ) : 
  (|(x^2 - 9) / 3| < 3) ↔ (-Real.sqrt 18 < x ∧ x < Real.sqrt 18) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l496_49665


namespace NUMINAMATH_CALUDE_cubic_identity_l496_49652

theorem cubic_identity (a b c : ℝ) : 
  a^3*(b^3 - c^3) + b^3*(c^3 - a^3) + c^3*(a^3 - b^3) = 
  (a - b)*(b - c)*(c - a) * ((a^2 + a*b + b^2)*(b^2 + b*c + c^2)*(c^2 + c*a + a^2)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_identity_l496_49652


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l496_49657

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 + 1994*m + 7 = 0 → 
  n^2 + 1994*n + 7 = 0 → 
  (m^2 + 1993*m + 6) * (n^2 + 1995*n + 8) = 1986 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l496_49657


namespace NUMINAMATH_CALUDE_circle_diameter_l496_49653

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 64 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l496_49653


namespace NUMINAMATH_CALUDE_similar_triangles_not_necessarily_equal_sides_l496_49663

-- Define a structure for triangles
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ ∧
    t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c ∧
    t1.a / t2.a = k

-- Define a property for equal corresponding sides
def equal_sides (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Theorem statement
theorem similar_triangles_not_necessarily_equal_sides :
  ¬ (∀ t1 t2 : Triangle, similar t1 t2 → equal_sides t1 t2) :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_not_necessarily_equal_sides_l496_49663


namespace NUMINAMATH_CALUDE_inequalities_hold_l496_49632

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (a^2 + b^2 ≥ 8) ∧ (1/(a*b) ≥ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l496_49632


namespace NUMINAMATH_CALUDE_soda_costs_80_cents_l496_49682

/-- The cost of a burger in cents -/
def burger_cost : ℕ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

/-- The cost of fries in cents -/
def fries_cost : ℕ := sorry

/-- Alice's purchase equation -/
axiom alice_purchase : 5 * burger_cost + 3 * soda_cost + 2 * fries_cost = 520

/-- Bill's purchase equation -/
axiom bill_purchase : 3 * burger_cost + 2 * soda_cost + fries_cost = 340

/-- Theorem stating that a soda costs 80 cents -/
theorem soda_costs_80_cents : soda_cost = 80 := by sorry

end NUMINAMATH_CALUDE_soda_costs_80_cents_l496_49682


namespace NUMINAMATH_CALUDE_zero_in_interval_l496_49611

def f (x : ℝ) : ℝ := x^5 + 8*x^3 - 1

theorem zero_in_interval :
  (f 0 < 0) → (f 0.5 > 0) → ∃ x, x ∈ Set.Ioo 0 0.5 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l496_49611


namespace NUMINAMATH_CALUDE_no_real_solution_log_equation_l496_49680

theorem no_real_solution_log_equation :
  ¬ ∃ x : ℝ, (Real.log (x + 5) + Real.log (x - 2) = Real.log (x^2 - 7*x + 10)) ∧ 
             (x + 5 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 7*x + 10 > 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_log_equation_l496_49680


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_square_area_l496_49656

theorem shaded_region_perimeter_square_area (PS PQ QR RS : ℝ) : 
  PS = 4 ∧ PQ + QR + RS = PS →
  let shaded_perimeter := (π/2) * (PS + PQ + QR + RS)
  let square_side := shaded_perimeter / 4
  square_side ^ 2 = π ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_square_area_l496_49656


namespace NUMINAMATH_CALUDE_specific_triangle_area_l496_49676

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The altitude to the base
  altitude : ℝ
  -- The perimeter of the triangle
  perimeter : ℝ
  -- The ratio of equal sides to base (represented as two integers)
  ratio_equal_to_base : ℕ × ℕ
  -- Condition: altitude is positive
  altitude_pos : altitude > 0
  -- Condition: perimeter is positive
  perimeter_pos : perimeter > 0
  -- Condition: ratio components are positive
  ratio_pos : ratio_equal_to_base.1 > 0 ∧ ratio_equal_to_base.2 > 0

/-- The area of an isosceles triangle with given properties -/
def triangle_area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles triangle is 75 -/
theorem specific_triangle_area :
  ∃ t : IsoscelesTriangle,
    t.altitude = 10 ∧
    t.perimeter = 40 ∧
    t.ratio_equal_to_base = (5, 3) ∧
    triangle_area t = 75 :=
  sorry

end NUMINAMATH_CALUDE_specific_triangle_area_l496_49676


namespace NUMINAMATH_CALUDE_digit_150_of_1_13_l496_49683

/-- The decimal representation of 1/13 as a sequence of digits -/
def decimal_rep_1_13 : ℕ → Fin 10 := fun n => 
  match n % 6 with
  | 0 => 0
  | 1 => 7
  | 2 => 6
  | 3 => 9
  | 4 => 2
  | 5 => 3
  | _ => 0 -- This case is unreachable, but needed for exhaustiveness

/-- The 150th digit after the decimal point in the decimal representation of 1/13 is 3 -/
theorem digit_150_of_1_13 : decimal_rep_1_13 150 = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_1_13_l496_49683


namespace NUMINAMATH_CALUDE_selected_student_in_range_l496_49601

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) (n : ℕ) : ℕ :=
  firstSelected + (n - 1) * (totalStudents / sampleSize)

/-- Theorem: The selected student number in the range 33 to 48 is 39 -/
theorem selected_student_in_range (totalStudents : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) :
  totalStudents = 800 →
  sampleSize = 50 →
  firstSelected = 7 →
  ∃ n : ℕ, systematicSample totalStudents sampleSize firstSelected n ∈ Set.Icc 33 48 ∧
           systematicSample totalStudents sampleSize firstSelected n = 39 :=
by
  sorry


end NUMINAMATH_CALUDE_selected_student_in_range_l496_49601


namespace NUMINAMATH_CALUDE_log_xy_value_l496_49618

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^4) = 2) (h2 : Real.log (x^3 * y) = 2) :
  Real.log (x * y) = 10 / 11 := by
sorry

end NUMINAMATH_CALUDE_log_xy_value_l496_49618


namespace NUMINAMATH_CALUDE_quadratic_condition_l496_49678

/-- The condition for a quadratic equation in x with parameter m -/
def is_quadratic_in_x (m : ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, m * x^2 - 3*x = x^2 - m*x + 2 ↔ a * x^2 + b * x + c = 0

/-- Theorem stating that for the given equation to be quadratic in x, m must not equal 1 -/
theorem quadratic_condition (m : ℝ) : is_quadratic_in_x m → m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l496_49678


namespace NUMINAMATH_CALUDE_inequality_solution_l496_49675

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) - 4 / (x + 8) > 1 / 2) ↔ (x > -4 ∧ x ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l496_49675


namespace NUMINAMATH_CALUDE_z_in_terms_of_x_and_y_l496_49679

theorem z_in_terms_of_x_and_y (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : y ≠ 2*x) 
  (h : 1/x - 2/y = 1/z) : z = x*y/(y - 2*x) := by
  sorry

end NUMINAMATH_CALUDE_z_in_terms_of_x_and_y_l496_49679


namespace NUMINAMATH_CALUDE_function_value_at_two_l496_49634

/-- Given a function f : ℝ → ℝ satisfying f(x) + 2f(1/x) = 3x for all x ≠ 0,
    prove that f(2) = -1 -/
theorem function_value_at_two (f : ℝ → ℝ) 
    (h : ∀ (x : ℝ), x ≠ 0 → f x + 2 * f (1 / x) = 3 * x) : 
    f 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l496_49634


namespace NUMINAMATH_CALUDE_probability_of_opening_classroom_door_l496_49654

/-- Represents a keychain with a total number of keys and a number of keys that can open a specific door. -/
structure Keychain where
  total_keys : ℕ
  opening_keys : ℕ
  h_opening_keys_le_total : opening_keys ≤ total_keys

/-- Calculates the probability of randomly selecting a key that can open the door. -/
def probability_of_opening (k : Keychain) : ℚ :=
  k.opening_keys / k.total_keys

/-- The class monitor's keychain. -/
def class_monitor_keychain : Keychain :=
  { total_keys := 5
    opening_keys := 2
    h_opening_keys_le_total := by norm_num }

theorem probability_of_opening_classroom_door :
  probability_of_opening class_monitor_keychain = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_opening_classroom_door_l496_49654


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l496_49666

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 4*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 4

-- Define the point of interest
def point : ℝ × ℝ := (1, -3)

-- Theorem statement
theorem tangent_slope_angle :
  let slope := f' point.1
  let angle := Real.arctan slope
  angle = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l496_49666


namespace NUMINAMATH_CALUDE_intersection_slope_l496_49629

/-- Given two lines that intersect at a specific point, prove the slope of one line. -/
theorem intersection_slope (m : ℝ) : 
  (∀ x y, y = 5 * x + 3 → (x = 2 ∧ y = 13)) →  -- Line p passes through (2, 13)
  (∀ x y, y = m * x + 1 → (x = 2 ∧ y = 13)) →  -- Line q passes through (2, 13)
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_slope_l496_49629


namespace NUMINAMATH_CALUDE_parallelogram_area_error_l496_49616

/-- Calculates the percentage error in the area of a parallelogram given measurement errors -/
theorem parallelogram_area_error (x y : ℝ) (z : Real) (hx : x > 0) (hy : y > 0) (hz : 0 < z ∧ z < pi) :
  let actual_area := x * y * Real.sin z
  let measured_area := (1.05 * x) * (1.07 * y) * Real.sin z
  (measured_area - actual_area) / actual_area * 100 = 12.35 := by
sorry


end NUMINAMATH_CALUDE_parallelogram_area_error_l496_49616


namespace NUMINAMATH_CALUDE_split_2017_implies_45_l496_49692

-- Define the sum of consecutive integers from 2 to n
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2 - 1

-- Define the property that 2017 is in the split of m³
def split_contains_2017 (m : ℕ) : Prop :=
  m > 1 ∧ sum_to_n m ≥ 1008 ∧ sum_to_n (m - 1) < 1008

theorem split_2017_implies_45 :
  ∀ m : ℕ, split_contains_2017 m → m = 45 :=
by sorry

end NUMINAMATH_CALUDE_split_2017_implies_45_l496_49692


namespace NUMINAMATH_CALUDE_tangent_lines_to_parabola_l496_49673

/-- The curve function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- Point B -/
def B : ℝ × ℝ := (3, 5)

/-- Tangent line equation type -/
structure TangentLine where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ℝ → ℝ → Prop := fun x y => a * x + b * y + c = 0

/-- Theorem: The equations of the lines that pass through B and are tangent to f are 2x - y - 1 = 0 and 10x - y - 25 = 0 -/
theorem tangent_lines_to_parabola :
  ∃ (l₁ l₂ : TangentLine),
    (l₁.equation 3 5 ∧ l₂.equation 3 5) ∧
    (∀ x y, y = f x → (l₁.equation x y ∨ l₂.equation x y) → 
      ∃ ε > 0, ∀ h ∈ Set.Ioo (x - ε) (x + ε), h ≠ x → f h > (l₁.a * h + l₁.c) ∧ f h > (l₂.a * h + l₂.c)) ∧
    l₁.equation = fun x y => 2 * x - y - 1 = 0 ∧
    l₂.equation = fun x y => 10 * x - y - 25 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_parabola_l496_49673
