import Mathlib

namespace NUMINAMATH_CALUDE_fq_length_l386_38623

-- Define the triangle DEF
structure RightTriangle where
  DE : ℝ
  DF : ℝ
  rightAngleAtE : True

-- Define the circle
structure TangentCircle where
  centerOnDE : True
  tangentToDF : True
  tangentToEF : True

-- Define the theorem
theorem fq_length
  (triangle : RightTriangle)
  (circle : TangentCircle)
  (h1 : triangle.DF = Real.sqrt 85)
  (h2 : triangle.DE = 7)
  : ∃ Q : ℝ × ℝ, ∃ F : ℝ × ℝ, ‖F - Q‖ = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_fq_length_l386_38623


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_inscribed_parallelepiped_l386_38620

/-- The surface area of a sphere that circumscribes a rectangular parallelepiped with edge lengths 3, 4, and 5 is equal to 50π. -/
theorem sphere_surface_area_of_inscribed_parallelepiped (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diameter := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diameter / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 50 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_inscribed_parallelepiped_l386_38620


namespace NUMINAMATH_CALUDE_range_of_a_l386_38663

def proposition_p (a : ℝ) : Prop :=
  ∀ m : ℝ, m ∈ Set.Icc (-1) 1 → a^2 - 5*a + 7 ≥ m + 2

def proposition_q (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a*x₁ = 2 ∧ x₂^2 + a*x₂ = 2

theorem range_of_a :
  ∃ S : Set ℝ, (∀ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ↔ a ∈ S) ∧
  S = {a : ℝ | -2*Real.sqrt 2 ≤ a ∧ a ≤ 1 ∨ 2*Real.sqrt 2 < a ∧ a < 4} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l386_38663


namespace NUMINAMATH_CALUDE_no_double_composition_inverse_l386_38699

-- Define the quadratic function g
def g (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_double_composition_inverse
  (a b c : ℝ)
  (h1 : ∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
                         g a b c (g a b c x₁) = x₁ ∧
                         g a b c (g a b c x₂) = x₂ ∧
                         g a b c (g a b c x₃) = x₃ ∧
                         g a b c (g a b c x₄) = x₄) :
  ¬∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = g a b c x :=
by sorry

end NUMINAMATH_CALUDE_no_double_composition_inverse_l386_38699


namespace NUMINAMATH_CALUDE_unattainable_y_value_l386_38660

theorem unattainable_y_value (x : ℝ) (hx : x ≠ -4/3) :
  ¬∃y : ℝ, y = (2 - x) / (3 * x + 4) ↔ y = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l386_38660


namespace NUMINAMATH_CALUDE_arc_length_300_degrees_l386_38622

/-- The length of an arc with radius 2 and central angle 300° is 10π/3 -/
theorem arc_length_300_degrees (r : Real) (θ : Real) : 
  r = 2 → θ = 300 * Real.pi / 180 → r * θ = 10 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_300_degrees_l386_38622


namespace NUMINAMATH_CALUDE_bicycle_sale_profit_l386_38682

/-- Profit percentage calculation for a bicycle sale chain --/
theorem bicycle_sale_profit (cost_price_A : ℝ) (profit_percent_A : ℝ) (price_C : ℝ)
  (h1 : cost_price_A = 150)
  (h2 : profit_percent_A = 20)
  (h3 : price_C = 225) :
  let price_B := cost_price_A * (1 + profit_percent_A / 100)
  let profit_B := price_C - price_B
  let profit_percent_B := (profit_B / price_B) * 100
  profit_percent_B = 25 := by
sorry

end NUMINAMATH_CALUDE_bicycle_sale_profit_l386_38682


namespace NUMINAMATH_CALUDE_books_per_shelf_l386_38636

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) (h1 : total_books = 315) (h2 : num_shelves = 7) :
  total_books / num_shelves = 45 := by
sorry

end NUMINAMATH_CALUDE_books_per_shelf_l386_38636


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l386_38606

theorem trigonometric_expression_evaluation :
  (Real.sin (20 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l386_38606


namespace NUMINAMATH_CALUDE_rocky_training_ratio_l386_38621

/-- Rocky's training schedule over three days -/
structure TrainingSchedule where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- Conditions for Rocky's training -/
def validSchedule (s : TrainingSchedule) : Prop :=
  s.day1 = 4 ∧ 
  s.day2 = 2 * s.day1 ∧ 
  s.day3 > s.day2 ∧
  s.day1 + s.day2 + s.day3 = 36

/-- The ratio of miles run on day 3 to day 2 is 3 -/
theorem rocky_training_ratio (s : TrainingSchedule) 
  (h : validSchedule s) : s.day3 / s.day2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_rocky_training_ratio_l386_38621


namespace NUMINAMATH_CALUDE_breakable_iff_composite_l386_38618

def is_breakable (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ), a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ a + b = n ∧ (x : ℚ) / a + (y : ℚ) / b = 1

theorem breakable_iff_composite (n : ℕ) : is_breakable n ↔ ¬ Nat.Prime n ∧ n > 1 :=
sorry

end NUMINAMATH_CALUDE_breakable_iff_composite_l386_38618


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l386_38666

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 2 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + m*y - 2 = 0 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l386_38666


namespace NUMINAMATH_CALUDE_tangent_slope_point_coordinates_l386_38616

theorem tangent_slope_point_coordinates :
  ∀ (x y : ℝ), 
    y = 1 / x →  -- The curve equation
    (-1 / x^2) = -4 →  -- The slope of the tangent line
    ((x = 1/2 ∧ y = 2) ∨ (x = -1/2 ∧ y = -2)) := by sorry

end NUMINAMATH_CALUDE_tangent_slope_point_coordinates_l386_38616


namespace NUMINAMATH_CALUDE_central_symmetry_line_symmetry_two_lines_max_distance_l386_38602

-- Define the curve C
def C (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + a * p.1 * p.2 = 1}

-- Statement 1: C is centrally symmetric about the origin for all a
theorem central_symmetry (a : ℝ) : ∀ p : ℝ × ℝ, p ∈ C a → (-p.1, -p.2) ∈ C a := by sorry

-- Statement 2: C is symmetric about the lines y = x and y = -x for all a
theorem line_symmetry (a : ℝ) : 
  (∀ p : ℝ × ℝ, p ∈ C a → (p.2, p.1) ∈ C a) ∧ 
  (∀ p : ℝ × ℝ, p ∈ C a → (-p.2, -p.1) ∈ C a) := by sorry

-- Statement 3: There exist at least two distinct values of a for which C represents two lines
theorem two_lines : ∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧ 
  (∃ l₁ l₂ m₁ m₂ : ℝ → ℝ, C a₁ = {p : ℝ × ℝ | p.2 = l₁ p.1 ∨ p.2 = l₂ p.1} ∧ 
                          C a₂ = {p : ℝ × ℝ | p.2 = m₁ p.1 ∨ p.2 = m₂ p.1}) := by sorry

-- Statement 4: When a = 1, the maximum distance between any two points on C is 2√2
theorem max_distance : 
  (∀ p q : ℝ × ℝ, p ∈ C 1 → q ∈ C 1 → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ 2 * Real.sqrt 2) ∧
  (∃ p q : ℝ × ℝ, p ∈ C 1 ∧ q ∈ C 1 ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_central_symmetry_line_symmetry_two_lines_max_distance_l386_38602


namespace NUMINAMATH_CALUDE_odd_prime_non_divisibility_l386_38684

theorem odd_prime_non_divisibility (p r : ℕ) : 
  Prime p → Odd p → Odd r → ¬(p * r + 1 ∣ p^p - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_non_divisibility_l386_38684


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_min_value_is_2_plus_sqrt2_min_value_achieved_l386_38651

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/(2*b) = 2 → x + 4*y ≤ a + 4*b :=
by sorry

theorem min_value_is_2_plus_sqrt2 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) :
  x + 4*y ≥ 2 + Real.sqrt 2 :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = 2 ∧ a + 4*b = 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_min_value_is_2_plus_sqrt2_min_value_achieved_l386_38651


namespace NUMINAMATH_CALUDE_prob_both_blue_is_25_64_l386_38688

/-- Represents the contents of a jar --/
structure JarContents where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a blue button from a jar --/
def prob_blue (jar : JarContents) : ℚ :=
  jar.blue / (jar.red + jar.blue)

/-- The initial contents of Jar C --/
def initial_jar_c : JarContents :=
  { red := 6, blue := 10 }

/-- The number of buttons removed from Jar C --/
def removed : JarContents :=
  { red := 3, blue := 5 }

/-- The contents of Jar C after removal --/
def final_jar_c : JarContents :=
  { red := initial_jar_c.red - removed.red,
    blue := initial_jar_c.blue - removed.blue }

/-- The contents of Jar D after removal --/
def jar_d : JarContents := removed

theorem prob_both_blue_is_25_64 :
  (prob_blue final_jar_c * prob_blue jar_d = 25 / 64) ∧
  (final_jar_c.red + final_jar_c.blue = (initial_jar_c.red + initial_jar_c.blue) / 2) :=
sorry

end NUMINAMATH_CALUDE_prob_both_blue_is_25_64_l386_38688


namespace NUMINAMATH_CALUDE_wall_width_calculation_l386_38670

/-- Calculates the width of a wall given its other dimensions and the number and size of bricks used. -/
theorem wall_width_calculation 
  (wall_length wall_height : ℝ) 
  (brick_length brick_width brick_height : ℝ)
  (num_bricks : ℕ) : 
  wall_length = 800 ∧ 
  wall_height = 600 ∧
  brick_length = 125 ∧ 
  brick_width = 11.25 ∧ 
  brick_height = 6 ∧
  num_bricks = 1280 →
  ∃ (wall_width : ℝ), 
    wall_width = 22.5 ∧
    wall_length * wall_height * wall_width = 
      num_bricks * (brick_length * brick_width * brick_height) := by
  sorry


end NUMINAMATH_CALUDE_wall_width_calculation_l386_38670


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l386_38669

theorem function_satisfies_equation (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x - 3) / (x^2 - x + 4)
  2 * f (1 - x) + 1 = x * f x := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l386_38669


namespace NUMINAMATH_CALUDE_correct_quotient_l386_38629

theorem correct_quotient (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 35) : D / 21 = 20 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_l386_38629


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l386_38644

theorem binomial_12_choose_6 : Nat.choose 12 6 = 1848 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l386_38644


namespace NUMINAMATH_CALUDE_sum_fraction_inequality_l386_38650

theorem sum_fraction_inequality (x y z : ℝ) (h : x + y + z = x*y + y*z + z*x) :
  x / (x^2 + 1) + y / (y^2 + 1) + z / (z^2 + 1) ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_inequality_l386_38650


namespace NUMINAMATH_CALUDE_initial_cards_count_l386_38654

/-- The number of cards Jennifer had initially -/
def initial_cards : ℕ := sorry

/-- The number of cards eaten by the hippopotamus -/
def eaten_cards : ℕ := 61

/-- The number of cards remaining after some were eaten -/
def remaining_cards : ℕ := 11

/-- Theorem stating that the initial number of cards is 72 -/
theorem initial_cards_count : initial_cards = 72 := by sorry

end NUMINAMATH_CALUDE_initial_cards_count_l386_38654


namespace NUMINAMATH_CALUDE_question_1_question_2_question_3_l386_38676

-- Define the functions f and g
def f (k : ℝ) (x : ℝ) : ℝ := 8 * x^2 + 16 * x - k
def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x

-- Define the interval [-3, 3]
def I : Set ℝ := Set.Icc (-3) 3

-- Statement for question 1
theorem question_1 (k : ℝ) : 
  (∀ x ∈ I, f k x ≤ g x) ↔ k ≥ 45 := by sorry

-- Statement for question 2
theorem question_2 (k : ℝ) : 
  (∃ x ∈ I, f k x ≤ g x) ↔ k ≥ -7 := by sorry

-- Statement for question 3
theorem question_3 (k : ℝ) : 
  (∀ x₁ ∈ I, ∀ x₂ ∈ I, f k x₁ ≤ g x₂) ↔ k ≥ 141 := by sorry

end NUMINAMATH_CALUDE_question_1_question_2_question_3_l386_38676


namespace NUMINAMATH_CALUDE_difference_of_sum_and_difference_of_squares_l386_38645

theorem difference_of_sum_and_difference_of_squares 
  (x y : ℝ) 
  (h1 : x + y = 6) 
  (h2 : x^2 - y^2 = 24) : 
  x - y = 4 := by
sorry

end NUMINAMATH_CALUDE_difference_of_sum_and_difference_of_squares_l386_38645


namespace NUMINAMATH_CALUDE_playlist_song_length_l386_38686

theorem playlist_song_length 
  (n_unknown : ℕ) 
  (n_known : ℕ) 
  (known_length : ℕ) 
  (total_duration : ℕ) : 
  n_unknown = 10 → 
  n_known = 15 → 
  known_length = 2 → 
  total_duration = 60 → 
  ∃ (unknown_length : ℕ), 
    unknown_length = 3 ∧ 
    n_unknown * unknown_length + n_known * known_length = total_duration :=
by sorry

end NUMINAMATH_CALUDE_playlist_song_length_l386_38686


namespace NUMINAMATH_CALUDE_max_individual_score_l386_38608

theorem max_individual_score (n : ℕ) (total_score : ℕ) (min_score : ℕ) 
  (h1 : n = 12)
  (h2 : total_score = 100)
  (h3 : min_score = 7)
  (h4 : ∀ player, player ∈ Finset.range n → player ≥ min_score) :
  (total_score - (n - 1) * min_score) = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_individual_score_l386_38608


namespace NUMINAMATH_CALUDE_evaluate_expression_l386_38657

theorem evaluate_expression : 
  2000 * 1995 * 0.1995 - 10 = 0.2 * 1995^2 - 10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l386_38657


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_l386_38674

theorem polynomial_roots_sum (a b c d m n : ℝ) : 
  (∃ (z : ℂ), z^3 + a*z + b = 0 ∧ z^3 + c*z^2 + d = 0) →
  (-20 : ℂ)^3 + a*(-20 : ℂ) + b = 0 →
  (-21 : ℂ)^3 + c*(-21 : ℂ)^2 + d = 0 →
  m > 0 →
  n > 0 →
  (m + Complex.I * Real.sqrt n : ℂ)^3 + a*(m + Complex.I * Real.sqrt n : ℂ) + b = 0 →
  (m + Complex.I * Real.sqrt n : ℂ)^3 + c*(m + Complex.I * Real.sqrt n : ℂ)^2 + d = 0 →
  m + n = 330 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_l386_38674


namespace NUMINAMATH_CALUDE_digit_sum_proof_l386_38609

theorem digit_sum_proof (A B : ℕ) :
  A ≤ 9 ∧ B ≤ 9 ∧ 
  111 * A + 110 * A + B + 100 * A + 11 * B + 111 * B = 1503 →
  A = 2 ∧ B = 7 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_proof_l386_38609


namespace NUMINAMATH_CALUDE_inverse_function_of_log_l386_38680

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_of_log (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f a (2 : ℝ) = -1) :
  ∀ x, f⁻¹ a x = (1/2 : ℝ) ^ x :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_of_log_l386_38680


namespace NUMINAMATH_CALUDE_fencing_required_l386_38693

/-- Calculates the fencing required for a rectangular field with given area and one uncovered side. -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 720 ∧ uncovered_side = 20 →
  uncovered_side + 2 * (area / uncovered_side) = 92 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l386_38693


namespace NUMINAMATH_CALUDE_parallelogram_cross_section_exists_l386_38613

/-- A cuboid in 3D space -/
structure Cuboid where
  -- Define the cuboid structure (you may need to add more fields)
  dummy : Unit

/-- A plane in 3D space -/
structure Plane where
  -- Define the plane structure (you may need to add more fields)
  dummy : Unit

/-- The cross-section resulting from a plane intersecting a cuboid -/
def crossSection (c : Cuboid) (p : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry -- Define the cross-section

/-- A predicate to check if a set of points forms a parallelogram -/
def isParallelogram (s : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry -- Define the conditions for a parallelogram

/-- Theorem stating that there exists a plane that intersects a cuboid to form a parallelogram cross-section -/
theorem parallelogram_cross_section_exists :
  ∃ (c : Cuboid) (p : Plane), isParallelogram (crossSection c p) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_cross_section_exists_l386_38613


namespace NUMINAMATH_CALUDE_crayon_selection_proof_l386_38646

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem crayon_selection_proof : choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_proof_l386_38646


namespace NUMINAMATH_CALUDE_units_digit_of_j_squared_plus_three_to_j_l386_38683

def j : ℕ := 19^2 + 3^10

theorem units_digit_of_j_squared_plus_three_to_j (j : ℕ := 19^2 + 3^10) : 
  (j^2 + 3^j) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_j_squared_plus_three_to_j_l386_38683


namespace NUMINAMATH_CALUDE_percentage_difference_l386_38601

theorem percentage_difference : 
  (0.6 * 50 + 0.45 * 30) - (0.4 * 30 + 0.25 * 20) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l386_38601


namespace NUMINAMATH_CALUDE_hot_dog_price_is_two_l386_38630

/-- Calculates the price of a single hot dog given the hourly sales rate, operating hours, and total sales. -/
def hot_dog_price (hourly_rate : ℕ) (hours : ℕ) (total_sales : ℕ) : ℚ :=
  total_sales / (hourly_rate * hours)

/-- Theorem stating that the price of each hot dog is $2 under given conditions. -/
theorem hot_dog_price_is_two :
  hot_dog_price 10 10 200 = 2 := by
  sorry

#eval hot_dog_price 10 10 200

end NUMINAMATH_CALUDE_hot_dog_price_is_two_l386_38630


namespace NUMINAMATH_CALUDE_complex_multiplication_l386_38653

theorem complex_multiplication (R S T : ℂ) : 
  R = 3 + 4*I ∧ S = 2*I ∧ T = 3 - 4*I → R * S * T = 50 * I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l386_38653


namespace NUMINAMATH_CALUDE_alloy_mixture_problem_l386_38635

/-- Represents the composition of an alloy --/
structure Alloy where
  component1 : ℝ
  component2 : ℝ
  ratio : ℚ

/-- Represents the mixture of two alloys --/
structure Mixture where
  alloyA : Alloy
  alloyB : Alloy
  massA : ℝ
  massB : ℝ
  tinTotal : ℝ

/-- The theorem to be proved --/
theorem alloy_mixture_problem (m : Mixture) : 
  m.alloyA.ratio = 1/3 ∧ 
  m.alloyB.ratio = 3/5 ∧ 
  m.massA = 170 ∧ 
  m.tinTotal = 221.25 → 
  m.massB = 250 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_problem_l386_38635


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_geq_two_l386_38662

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem set_inclusion_implies_a_geq_two (a : ℝ) :
  A ⊆ B a → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_geq_two_l386_38662


namespace NUMINAMATH_CALUDE_intersection_radius_l386_38687

/-- A sphere intersecting planes -/
structure IntersectingSphere where
  /-- Center of the circle in xz-plane -/
  xz_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in xz-plane -/
  xz_radius : ℝ
  /-- Center of the circle in xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in xy-plane -/
  xy_radius : ℝ

/-- The theorem stating the radius of the xy-plane intersection -/
theorem intersection_radius (sphere : IntersectingSphere) 
  (h1 : sphere.xz_center = (3, 0, 3))
  (h2 : sphere.xz_radius = 2)
  (h3 : sphere.xy_center = (3, 3, 0)) :
  sphere.xy_radius = 3 := by
  sorry


end NUMINAMATH_CALUDE_intersection_radius_l386_38687


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l386_38692

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l386_38692


namespace NUMINAMATH_CALUDE_increasing_square_neg_func_l386_38678

/-- Given an increasing function f: ℝ → ℝ with f(x) < 0 for all x,
    the function g(x) = x^2 * f(x) is increasing on (-∞, 0) -/
theorem increasing_square_neg_func
  (f : ℝ → ℝ)
  (h_incr : ∀ x y, x < y → f x < f y)
  (h_neg : ∀ x, f x < 0) :
  ∀ x y, x < y → x < 0 → y < 0 → x^2 * f x < y^2 * f y :=
by sorry

end NUMINAMATH_CALUDE_increasing_square_neg_func_l386_38678


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l386_38694

theorem inequality_not_always_true (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ¬ (∀ a b, a > b ∧ b > 0 → a + b < 2 * Real.sqrt (a * b)) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l386_38694


namespace NUMINAMATH_CALUDE_complex_equation_solution_l386_38647

theorem complex_equation_solution (a : ℝ) : (Complex.I + a) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l386_38647


namespace NUMINAMATH_CALUDE_shaded_area_is_four_thirds_l386_38614

/-- Rectangle with specific dimensions and lines forming a shaded region --/
structure ShadedRectangle where
  J : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  M : ℝ × ℝ
  h_rectangle : J.1 = 0 ∧ J.2 = 0 ∧ K.1 = 4 ∧ K.2 = 0 ∧ L.1 = 4 ∧ L.2 = 5 ∧ M.1 = 0 ∧ M.2 = 5
  h_mj : M.2 - J.2 = 2
  h_jk : K.1 - J.1 = 1
  h_kl : L.2 - K.2 = 1
  h_lm : M.1 - L.1 = 1

/-- The area of the shaded region in the rectangle --/
def shadedArea (r : ShadedRectangle) : ℝ := sorry

/-- Theorem stating that the shaded area is 4/3 --/
theorem shaded_area_is_four_thirds (r : ShadedRectangle) : shadedArea r = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_four_thirds_l386_38614


namespace NUMINAMATH_CALUDE_basketball_shots_l386_38668

theorem basketball_shots (t h f : ℕ) : 
  (2 * t = 3 * h) →  -- Two-point shots scored double the points of three-point shots
  (f = h - 4) →      -- Number of free throws is four fewer than three-point shots
  (t + h + f = 40) → -- Total shots is 40
  (2 * t + 3 * h + f = 76) → -- Total score is 76
  h = 8 := by sorry

end NUMINAMATH_CALUDE_basketball_shots_l386_38668


namespace NUMINAMATH_CALUDE_water_lost_is_eight_gallons_l386_38671

/-- Represents the water filling and leaking scenario of a pool --/
structure PoolFilling where
  hour1_rate : ℝ
  hour2_3_rate : ℝ
  hour4_rate : ℝ
  final_amount : ℝ

/-- Calculates the amount of water lost due to the leak --/
def water_lost (p : PoolFilling) : ℝ :=
  p.hour1_rate * 1 + p.hour2_3_rate * 2 + p.hour4_rate * 1 - p.final_amount

/-- Theorem stating that for the given scenario, the water lost is 8 gallons --/
theorem water_lost_is_eight_gallons : 
  ∀ (p : PoolFilling), 
  p.hour1_rate = 8 ∧ 
  p.hour2_3_rate = 10 ∧ 
  p.hour4_rate = 14 ∧ 
  p.final_amount = 34 → 
  water_lost p = 8 := by
  sorry


end NUMINAMATH_CALUDE_water_lost_is_eight_gallons_l386_38671


namespace NUMINAMATH_CALUDE_train_speed_l386_38665

/-- Proves that a train with given length and time to cross a pole has a specific speed -/
theorem train_speed (train_length : Real) (crossing_time : Real) (speed : Real) : 
  train_length = 200 → 
  crossing_time = 12 → 
  speed = (train_length / 1000) / (crossing_time / 3600) → 
  speed = 60 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l386_38665


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l386_38625

/-- Given a rectangular plot where the length is thrice the breadth 
    and the area is 972 sq m, prove that the breadth is 18 meters. -/
theorem rectangular_plot_breadth : 
  ∀ (breadth length area : ℝ),
  length = 3 * breadth →
  area = length * breadth →
  area = 972 →
  breadth = 18 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l386_38625


namespace NUMINAMATH_CALUDE_triangle_altitude_median_equations_l386_38626

/-- Triangle ABC with given coordinates -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given triangle ABC, return the equation of the altitude from C to AB -/
def altitude (t : Triangle) : LineEquation :=
  sorry

/-- Given triangle ABC, return the equation of the median from C to AB -/
def median (t : Triangle) : LineEquation :=
  sorry

theorem triangle_altitude_median_equations :
  let t : Triangle := { A := (3, 3), B := (2, -2), C := (-7, 1) }
  (altitude t = { a := 1, b := 5, c := 2 }) ∧
  (median t = { a := 1, b := 19, c := -12 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_median_equations_l386_38626


namespace NUMINAMATH_CALUDE_cos_210_degrees_l386_38689

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l386_38689


namespace NUMINAMATH_CALUDE_lowest_unique_score_above_100_l386_38612

/-- Represents the scoring system and conditions of the math competition. -/
structure MathCompetition where
  total_questions : Nat
  base_score : Nat
  correct_points : Nat
  wrong_points : Nat
  score : Nat

/-- Checks if a given score is valid for the math competition. -/
def is_valid_score (comp : MathCompetition) (correct wrong : Nat) : Prop :=
  correct + wrong ≤ comp.total_questions ∧
  comp.score = comp.base_score + comp.correct_points * correct - comp.wrong_points * wrong

/-- Checks if a score has a unique solution for correct and wrong answers. -/
def has_unique_solution (comp : MathCompetition) : Prop :=
  ∃! (correct wrong : Nat), is_valid_score comp correct wrong

/-- The main theorem stating that 150 is the lowest score above 100 with a unique solution. -/
theorem lowest_unique_score_above_100 : 
  let comp : MathCompetition := {
    total_questions := 50,
    base_score := 50,
    correct_points := 5,
    wrong_points := 2,
    score := 150
  }
  (comp.score > 100) ∧ 
  has_unique_solution comp ∧
  ∀ (s : Nat), 100 < s ∧ s < comp.score → 
    ¬(has_unique_solution {comp with score := s}) := by
  sorry

end NUMINAMATH_CALUDE_lowest_unique_score_above_100_l386_38612


namespace NUMINAMATH_CALUDE_inequality_proof_l386_38661

theorem inequality_proof (a b c d e : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_prod : a * b * c * d * e = 1) : 
  (a + a*b*c) / (1 + a*b + a*b*c*d) + 
  (b + b*c*d) / (1 + b*c + b*c*d*e) + 
  (c + c*d*e) / (1 + c*d + c*d*e*a) + 
  (d + d*e*a) / (1 + d*e + d*e*a*b) + 
  (e + e*a*b) / (1 + e*a + e*a*b*c) ≥ 10/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l386_38661


namespace NUMINAMATH_CALUDE_solar_panel_installation_time_l386_38677

/-- Calculates the number of hours needed to install solar panels given the costs of various items --/
def solar_panel_installation_hours (land_acres : ℕ) (land_cost_per_acre : ℕ) 
  (house_cost : ℕ) (cow_count : ℕ) (cow_cost : ℕ) (chicken_count : ℕ) 
  (chicken_cost : ℕ) (solar_panel_hourly_rate : ℕ) (solar_panel_equipment_fee : ℕ) 
  (total_cost : ℕ) : ℕ :=
  let land_cost := land_acres * land_cost_per_acre
  let cows_cost := cow_count * cow_cost
  let chickens_cost := chicken_count * chicken_cost
  let costs_before_solar := land_cost + house_cost + cows_cost + chickens_cost
  let solar_panel_total_cost := total_cost - costs_before_solar
  let installation_cost := solar_panel_total_cost - solar_panel_equipment_fee
  installation_cost / solar_panel_hourly_rate

theorem solar_panel_installation_time : 
  solar_panel_installation_hours 30 20 120000 20 1000 100 5 100 6000 147700 = 6 := by
  sorry

end NUMINAMATH_CALUDE_solar_panel_installation_time_l386_38677


namespace NUMINAMATH_CALUDE_pigeon_count_l386_38615

/-- The number of pigeons in the pigeon house -/
def num_pigeons : ℕ := 600

/-- The number of days the feed lasts if 75 pigeons are sold -/
def days_after_selling : ℕ := 20

/-- The number of days the feed lasts if 100 pigeons are bought -/
def days_after_buying : ℕ := 15

/-- The number of pigeons sold -/
def pigeons_sold : ℕ := 75

/-- The number of pigeons bought -/
def pigeons_bought : ℕ := 100

/-- Theorem stating that the number of pigeons in the pigeon house is 600 -/
theorem pigeon_count : 
  (num_pigeons - pigeons_sold) * days_after_selling = (num_pigeons + pigeons_bought) * days_after_buying :=
by sorry

end NUMINAMATH_CALUDE_pigeon_count_l386_38615


namespace NUMINAMATH_CALUDE_min_value_of_expression_l386_38681

theorem min_value_of_expression (x y : ℝ) :
  (2 * x * y - 1)^2 + (x - y)^2 ≥ 0 ∧
  ∃ a b : ℝ, (2 * a * b - 1)^2 + (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l386_38681


namespace NUMINAMATH_CALUDE_no_square_cut_with_250_remaining_l386_38617

theorem no_square_cut_with_250_remaining : ¬∃ (n m : ℕ), n > m ∧ n^2 - m^2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_no_square_cut_with_250_remaining_l386_38617


namespace NUMINAMATH_CALUDE_prob_at_least_two_heads_l386_38633

-- Define the number of coins
def n : ℕ := 5

-- Define the probability of getting heads on a single coin toss
def p : ℚ := 1/2

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the probability of getting exactly k heads in n tosses
def prob_exactly (k : ℕ) : ℚ := (binomial n k : ℚ) * p^n

-- State the theorem
theorem prob_at_least_two_heads :
  1 - (prob_exactly 0 + prob_exactly 1) = 13/16 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_heads_l386_38633


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l386_38628

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with a₂ = 2 and S₄ = 9, a₆ = 4 -/
theorem arithmetic_sequence_a6 (seq : ArithmeticSequence) 
    (h1 : seq.a 2 = 2) 
    (h2 : seq.S 4 = 9) : 
  seq.a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l386_38628


namespace NUMINAMATH_CALUDE_root_sequence_difference_l386_38640

theorem root_sequence_difference (m n : ℝ) : 
  (∃ a b c d : ℝ, 
    (a = 1) ∧
    (a * d = b * c) ∧
    ({a, b, c, d} = {x : ℝ | (x^2 - m*x + 27 = 0) ∨ (x^2 - n*x + 27 = 0)}) ∧
    (∃ q : ℝ, b = a*q ∧ c = b*q ∧ d = c*q)) →
  |m - n| = 16 :=
by sorry

end NUMINAMATH_CALUDE_root_sequence_difference_l386_38640


namespace NUMINAMATH_CALUDE_regular_star_polygon_points_l386_38642

-- Define the structure of a regular star polygon
structure RegularStarPolygon where
  n : ℕ  -- number of points
  A : ℝ  -- measure of each Aᵢ angle in degrees
  B : ℝ  -- measure of each Bᵢ angle in degrees

-- Define the properties of the regular star polygon
def is_valid_regular_star_polygon (p : RegularStarPolygon) : Prop :=
  p.A > 0 ∧ p.B > 0 ∧  -- angles are positive
  p.A = p.B + 15 ∧     -- Aᵢ is 15° more than Bᵢ
  p.n * (p.A + p.B) = 360  -- sum of external angles is 360°

-- Theorem: A regular star polygon with the given conditions has 24 points
theorem regular_star_polygon_points (p : RegularStarPolygon) :
  is_valid_regular_star_polygon p → p.n = 24 :=
by sorry

end NUMINAMATH_CALUDE_regular_star_polygon_points_l386_38642


namespace NUMINAMATH_CALUDE_soccer_ball_cost_is_6_l386_38639

/-- The cost of a soccer ball purchased by four friends -/
def soccer_ball_cost (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 = 2.30 ∧
  x2 = (1/3) * (x1 + x3 + x4) ∧
  x3 = (1/4) * (x1 + x2 + x4) ∧
  x4 = (1/5) * (x1 + x2 + x3) ∧
  x1 + x2 + x3 + x4 = 6

theorem soccer_ball_cost_is_6 :
  ∃ x1 x2 x3 x4 : ℝ, soccer_ball_cost x1 x2 x3 x4 :=
sorry

end NUMINAMATH_CALUDE_soccer_ball_cost_is_6_l386_38639


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l386_38685

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x + 1 > 2 ∧ 2*x - 4 < x) ↔ (1 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l386_38685


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l386_38643

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -14/27
  let a₃ : ℚ := 56/216
  let r : ℚ := a₂ / a₁
  r = -16/27 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l386_38643


namespace NUMINAMATH_CALUDE_triangle_problem_l386_38673

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the values of b and cos(2B - π/3) under specific conditions. -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  b * Real.sin A = 3 * c * Real.sin B →
  a = 3 →
  Real.cos B = 2/3 →
  b = Real.sqrt 6 ∧ Real.cos (2*B - π/3) = (4 * Real.sqrt 15 - 1) / 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l386_38673


namespace NUMINAMATH_CALUDE_solution_in_interval_l386_38619

open Real

theorem solution_in_interval : ∃ (x₀ : ℝ), ∃ (k : ℤ),
  (log x₀ = 5 - 2 * x₀) ∧ 
  (x₀ > k) ∧ (x₀ < k + 1) ∧
  (k = 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l386_38619


namespace NUMINAMATH_CALUDE_tom_customers_per_hour_l386_38638

/-- The number of customers Tom served per hour -/
def customers_per_hour : ℝ := 10

/-- The number of hours Tom worked -/
def hours_worked : ℝ := 8

/-- The bonus point percentage (20% = 0.2) -/
def bonus_percentage : ℝ := 0.2

/-- The total bonus points Tom earned -/
def total_bonus_points : ℝ := 16

theorem tom_customers_per_hour :
  customers_per_hour * hours_worked * bonus_percentage = total_bonus_points :=
by sorry

end NUMINAMATH_CALUDE_tom_customers_per_hour_l386_38638


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_parts_l386_38648

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def first_digit (n : ℕ) : ℕ := n / 100

def second_digit (n : ℕ) : ℕ := (n / 10) % 10

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem smallest_three_digit_divisible_by_parts : 
  ∃ (n : ℕ), is_three_digit n ∧ 
  first_digit n ≠ 0 ∧
  n % (n / 10) = 0 ∧ 
  n % (last_two_digits n) = 0 ∧
  ∀ m, is_three_digit m ∧ 
       first_digit m ≠ 0 ∧ 
       m % (m / 10) = 0 ∧ 
       m % (last_two_digits m) = 0 → 
       n ≤ m ∧
  n = 110 := by
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_parts_l386_38648


namespace NUMINAMATH_CALUDE_compute_expression_l386_38605

theorem compute_expression : 6 * (2/3)^4 - 1/6 = 55/54 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l386_38605


namespace NUMINAMATH_CALUDE_alternating_sequences_20_l386_38603

/-- A function that computes the number of alternating sequences -/
def A : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => A (n + 1) + A n

/-- The number of alternating sequences for n = 20 is 10946 -/
theorem alternating_sequences_20 : A 20 = 10946 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sequences_20_l386_38603


namespace NUMINAMATH_CALUDE_square_area_error_l386_38611

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0404 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l386_38611


namespace NUMINAMATH_CALUDE_largest_integer_solution_2x_plus_3_lt_0_l386_38604

theorem largest_integer_solution_2x_plus_3_lt_0 :
  ∀ x : ℤ, 2 * x + 3 < 0 → x ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_solution_2x_plus_3_lt_0_l386_38604


namespace NUMINAMATH_CALUDE_alok_order_cost_l386_38696

def chapati_quantity : ℕ := 16
def chapati_price : ℕ := 6
def rice_quantity : ℕ := 5
def rice_price : ℕ := 45
def vegetable_quantity : ℕ := 7
def vegetable_price : ℕ := 70

def total_cost : ℕ := chapati_quantity * chapati_price + 
                      rice_quantity * rice_price + 
                      vegetable_quantity * vegetable_price

theorem alok_order_cost : total_cost = 811 := by
  sorry

end NUMINAMATH_CALUDE_alok_order_cost_l386_38696


namespace NUMINAMATH_CALUDE_longFurredBrownCount_l386_38667

/-- Represents the number of dogs in a kennel with specific characteristics. -/
structure DogKennel where
  total : ℕ
  longFurred : ℕ
  brown : ℕ
  neither : ℕ

/-- Calculates the number of long-furred brown dogs in the kennel. -/
def longFurredBrown (k : DogKennel) : ℕ :=
  k.longFurred + k.brown - (k.total - k.neither)

/-- Theorem stating the number of long-furred brown dogs in a specific kennel configuration. -/
theorem longFurredBrownCount :
  let k : DogKennel := {
    total := 45,
    longFurred := 26,
    brown := 30,
    neither := 8
  }
  longFurredBrown k = 27 := by sorry

end NUMINAMATH_CALUDE_longFurredBrownCount_l386_38667


namespace NUMINAMATH_CALUDE_bird_triangle_theorem_l386_38690

/-- A bird's position on a regular n-gon --/
structure BirdPosition (n : ℕ) where
  vertex : Fin n

/-- The type of a triangle --/
inductive TriangleType
  | Acute
  | Obtuse
  | RightAngled

/-- Determine the type of a triangle formed by three birds on a regular n-gon --/
def triangleType (n : ℕ) (a b c : BirdPosition n) : TriangleType := sorry

/-- A permutation of birds --/
def BirdPermutation (n : ℕ) := Fin n → Fin n

/-- The main theorem --/
theorem bird_triangle_theorem (n : ℕ) (h : n ≥ 3 ∧ n ≠ 5) :
  ∀ (perm : BirdPermutation n),
  ∃ (a b c : Fin n),
    triangleType n ⟨a⟩ ⟨b⟩ ⟨c⟩ = triangleType n ⟨perm a⟩ ⟨perm b⟩ ⟨perm c⟩ :=
sorry

end NUMINAMATH_CALUDE_bird_triangle_theorem_l386_38690


namespace NUMINAMATH_CALUDE_evaluate_expression_l386_38655

theorem evaluate_expression (a : ℝ) (h : a = 2) : 
  8^3 + 4*a*(8^2) + 6*(a^2)*8 + a^3 = 1224 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l386_38655


namespace NUMINAMATH_CALUDE_triangle_theorem_l386_38634

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides opposite to angles A, B, C respectively
  (S : Real)      -- Area

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b + t.c = 2 * t.a * Real.cos t.B)  -- Given condition
  (h2 : t.S = t.a^2 / 4)                     -- Given area condition
  : t.A = 2 * t.B ∧ (t.A = Real.pi / 2 ∨ t.A = Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l386_38634


namespace NUMINAMATH_CALUDE_divisible_by_27_l386_38695

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, (10 ^ n : ℤ) + 18 * n - 1 = 27 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_27_l386_38695


namespace NUMINAMATH_CALUDE_sarahs_journey_length_l386_38600

theorem sarahs_journey_length :
  ∀ (total : ℚ),
    (1 / 4 : ℚ) * total +   -- First part
    30 +                    -- Second part
    (1 / 6 : ℚ) * total =   -- Third part
    total →                 -- Sum of parts equals total
    total = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_journey_length_l386_38600


namespace NUMINAMATH_CALUDE_greatest_missable_problems_l386_38641

theorem greatest_missable_problems (total_problems : ℕ) (passing_percentage : ℚ) 
  (h1 : total_problems = 50)
  (h2 : passing_percentage = 85 / 100) :
  ∃ (max_missable : ℕ), 
    max_missable = 7 ∧ 
    (total_problems - max_missable : ℚ) / total_problems ≥ passing_percentage ∧
    ∀ (n : ℕ), n > max_missable → (total_problems - n : ℚ) / total_problems < passing_percentage :=
by sorry

end NUMINAMATH_CALUDE_greatest_missable_problems_l386_38641


namespace NUMINAMATH_CALUDE_airline_capacity_example_l386_38631

/-- Calculates the total number of passengers an airline can accommodate daily --/
def airline_capacity (num_airplanes : ℕ) (rows_per_airplane : ℕ) (seats_per_row : ℕ) (flights_per_day : ℕ) : ℕ :=
  num_airplanes * rows_per_airplane * seats_per_row * flights_per_day

/-- Theorem: An airline with 5 airplanes, 20 rows per airplane, 7 seats per row, and 2 flights per day can accommodate 1400 passengers daily --/
theorem airline_capacity_example : airline_capacity 5 20 7 2 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_airline_capacity_example_l386_38631


namespace NUMINAMATH_CALUDE_tom_batteries_in_toys_l386_38697

/-- The number of batteries Tom used in his toys -/
def batteries_in_toys (total batteries_in_flashlights batteries_in_controllers : ℕ) : ℕ :=
  total - (batteries_in_flashlights + batteries_in_controllers)

/-- Theorem stating that Tom used 15 batteries in his toys -/
theorem tom_batteries_in_toys :
  batteries_in_toys 19 2 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_tom_batteries_in_toys_l386_38697


namespace NUMINAMATH_CALUDE_committee_arrangement_count_l386_38664

def committee_size : ℕ := 10
def num_men : ℕ := 3
def num_women : ℕ := 7

theorem committee_arrangement_count :
  (committee_size.choose num_men) = 120 := by
  sorry

end NUMINAMATH_CALUDE_committee_arrangement_count_l386_38664


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l386_38610

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_condition
  (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) :
  (is_increasing_sequence a → a 1 < a 2 ∧ a 2 < a 3) ∧
  ¬(a 1 < a 2 ∧ a 2 < a 3 → is_increasing_sequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l386_38610


namespace NUMINAMATH_CALUDE_garden_vegetable_ratio_l386_38691

theorem garden_vegetable_ratio :
  let potatoes : ℕ := 237
  let cucumbers : ℕ := potatoes - 60
  let total_vegetables : ℕ := 768
  let peppers : ℕ := total_vegetables - potatoes - cucumbers
  peppers = 2 * cucumbers :=
by sorry

end NUMINAMATH_CALUDE_garden_vegetable_ratio_l386_38691


namespace NUMINAMATH_CALUDE_movie_ticket_price_l386_38675

/-- The price of a 3D movie ticket --/
def price_3d : ℕ := sorry

/-- The price of a matinee ticket --/
def price_matinee : ℕ := 5

/-- The price of an evening ticket --/
def price_evening : ℕ := 12

/-- The number of matinee tickets sold --/
def num_matinee : ℕ := 200

/-- The number of evening tickets sold --/
def num_evening : ℕ := 300

/-- The number of 3D tickets sold --/
def num_3d : ℕ := 100

/-- The total revenue from all ticket sales --/
def total_revenue : ℕ := 6600

theorem movie_ticket_price :
  price_3d = 20 ∧
  price_matinee * num_matinee +
  price_evening * num_evening +
  price_3d * num_3d = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_movie_ticket_price_l386_38675


namespace NUMINAMATH_CALUDE_correct_distribution_l386_38632

/-- Represents the distribution of sampled students across three camps -/
structure CampDistribution where
  camp1 : Nat
  camp2 : Nat
  camp3 : Nat

/-- Parameters for the systematic sampling -/
structure SamplingParams where
  totalStudents : Nat
  sampleSize : Nat
  startNumber : Nat

/-- Function to perform systematic sampling and calculate camp distribution -/
def systematicSampling (params : SamplingParams) : CampDistribution :=
  sorry

/-- Theorem stating the correct distribution for the given problem -/
theorem correct_distribution :
  let params : SamplingParams := {
    totalStudents := 300,
    sampleSize := 20,
    startNumber := 3
  }
  let result : CampDistribution := systematicSampling params
  result.camp1 = 14 ∧ result.camp2 = 3 ∧ result.camp3 = 3 :=
sorry

end NUMINAMATH_CALUDE_correct_distribution_l386_38632


namespace NUMINAMATH_CALUDE_f_at_seven_l386_38627

/-- The polynomial f(x) = 7x^5 + 12x^4 - 5x^3 - 6x^2 + 3x - 5 -/
def f (x : ℝ) : ℝ := 7*x^5 + 12*x^4 - 5*x^3 - 6*x^2 + 3*x - 5

/-- Theorem stating that f(7) = 144468 -/
theorem f_at_seven : f 7 = 144468 := by
  sorry

end NUMINAMATH_CALUDE_f_at_seven_l386_38627


namespace NUMINAMATH_CALUDE_car_distance_proof_l386_38656

theorem car_distance_proof (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 125 → time = 3 → distance = speed * time → distance = 375 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l386_38656


namespace NUMINAMATH_CALUDE_inequality_proof_l386_38637

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l386_38637


namespace NUMINAMATH_CALUDE_annual_interest_proof_l386_38659

/-- Calculates the simple interest for a loan -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proves that the annual interest for a $9,000 loan at 9% simple interest is $810 -/
theorem annual_interest_proof :
  let principal : ℝ := 9000
  let rate : ℝ := 0.09
  let time : ℝ := 1
  simple_interest principal rate time = 810 := by
sorry


end NUMINAMATH_CALUDE_annual_interest_proof_l386_38659


namespace NUMINAMATH_CALUDE_marching_band_theorem_l386_38672

def marching_band_ratio (total_members brass_players : ℕ) : Prop :=
  ∃ (percussion woodwind : ℕ),
    -- Total members condition
    total_members = percussion + woodwind + brass_players ∧
    -- Woodwind is twice brass
    woodwind = 2 * brass_players ∧
    -- Percussion is a multiple of woodwind
    ∃ (k : ℕ), percussion = k * woodwind ∧
    -- Ratio of percussion to woodwind is 4:1
    percussion = 4 * woodwind

theorem marching_band_theorem :
  marching_band_ratio 110 10 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_theorem_l386_38672


namespace NUMINAMATH_CALUDE_greatest_multiple_of_12_with_unique_digits_M_mod_1000_l386_38658

/-- A function that checks if a natural number has all unique digits -/
def has_unique_digits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 12 with all unique digits -/
def M : ℕ := sorry

theorem greatest_multiple_of_12_with_unique_digits : 
  M % 12 = 0 ∧ 
  has_unique_digits M ∧ 
  ∀ k, k % 12 = 0 → has_unique_digits k → k ≤ M :=
sorry

theorem M_mod_1000 : M % 1000 = 320 := sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_12_with_unique_digits_M_mod_1000_l386_38658


namespace NUMINAMATH_CALUDE_essay_writing_speed_l386_38679

/-- Represents the essay writing scenario -/
structure EssayWriting where
  total_words : ℕ
  initial_speed : ℕ
  initial_hours : ℕ
  total_hours : ℕ

/-- Calculates the words written per hour after the initial period -/
def words_per_hour_after (e : EssayWriting) : ℕ :=
  (e.total_words - e.initial_speed * e.initial_hours) / (e.total_hours - e.initial_hours)

/-- Theorem stating that under the given conditions, the writing speed after
    the first two hours is 200 words per hour -/
theorem essay_writing_speed (e : EssayWriting) 
    (h1 : e.total_words = 1200)
    (h2 : e.initial_speed = 400)
    (h3 : e.initial_hours = 2)
    (h4 : e.total_hours = 4) : 
  words_per_hour_after e = 200 := by
  sorry

#eval words_per_hour_after { total_words := 1200, initial_speed := 400, initial_hours := 2, total_hours := 4 }

end NUMINAMATH_CALUDE_essay_writing_speed_l386_38679


namespace NUMINAMATH_CALUDE_petya_wins_2021_petya_wins_l386_38652

/-- Represents the game state -/
structure GameState :=
  (piles : ℕ)

/-- Represents a player in the game -/
inductive Player
  | Petya
  | Vasya

/-- Defines a valid move in the game -/
def valid_move (state : GameState) : Prop :=
  state.piles ≥ 3

/-- Applies a move to the game state -/
def apply_move (state : GameState) : GameState :=
  { piles := state.piles - 2 }

/-- Determines the winner of the game -/
def winner (initial_piles : ℕ) : Player :=
  if initial_piles % 2 = 0 then Player.Vasya else Player.Petya

/-- Theorem stating that Petya wins the game with 2021 initial piles -/
theorem petya_wins_2021 : winner 2021 = Player.Petya := by
  sorry

/-- Main theorem proving Petya's victory -/
theorem petya_wins :
  ∀ (initial_state : GameState),
    initial_state.piles = 2021 →
    winner initial_state.piles = Player.Petya := by
  sorry

end NUMINAMATH_CALUDE_petya_wins_2021_petya_wins_l386_38652


namespace NUMINAMATH_CALUDE_red_tetrahedron_volume_l386_38698

/-- The volume of a tetrahedron formed by red vertices in a cube with alternately colored vertices -/
theorem red_tetrahedron_volume (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume := cube_side_length ^ 3
  let green_tetrahedron_volume := (1 / 3) * (1 / 2 * cube_side_length ^ 2) * cube_side_length
  let red_tetrahedron_volume := cube_volume - 4 * green_tetrahedron_volume
  red_tetrahedron_volume = 512 / 3 := by
  sorry

end NUMINAMATH_CALUDE_red_tetrahedron_volume_l386_38698


namespace NUMINAMATH_CALUDE_amy_biking_distance_l386_38624

def miles_yesterday : ℕ := 12

def miles_today (y : ℕ) : ℕ := 2 * y - 3

def total_miles (y t : ℕ) : ℕ := y + t

theorem amy_biking_distance :
  total_miles miles_yesterday (miles_today miles_yesterday) = 33 :=
by sorry

end NUMINAMATH_CALUDE_amy_biking_distance_l386_38624


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_neg_one_l386_38607

theorem sin_cos_sum_equals_neg_one : 
  Real.sin (315 * π / 180) - Real.cos (135 * π / 180) + 2 * Real.sin (570 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_neg_one_l386_38607


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l386_38649

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_property
  (seq : ArithmeticSequence)
  (h1 : seq.S 3 = 9)
  (h2 : seq.S 6 = 36) :
  seq.a 7 + seq.a 8 + seq.a 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l386_38649
