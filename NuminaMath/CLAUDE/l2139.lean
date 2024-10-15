import Mathlib

namespace NUMINAMATH_CALUDE_estimated_probability_one_hit_l2139_213990

-- Define the type for a throw result
inductive ThrowResult
| Hit
| Miss

-- Define a type for a set of two throws
def TwoThrows := (ThrowResult × ThrowResult)

-- Define the simulation data
def simulation_data : List TwoThrows := sorry

-- Define the function to count sets with exactly one hit
def count_one_hit (data : List TwoThrows) : Nat := sorry

-- Theorem statement
theorem estimated_probability_one_hit 
  (h1 : simulation_data.length = 20)
  (h2 : count_one_hit simulation_data = 10) :
  (count_one_hit simulation_data : ℚ) / simulation_data.length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_estimated_probability_one_hit_l2139_213990


namespace NUMINAMATH_CALUDE_similar_radical_expressions_l2139_213996

-- Define the concept of similar radical expressions
def are_similar_radical_expressions (x y : ℝ) : Prop :=
  ∃ (k : ℝ) (n : ℕ), k > 0 ∧ x = k * (y^(1/n))

theorem similar_radical_expressions :
  ∀ (a : ℝ), a > 0 →
  (are_similar_radical_expressions (a^(1/3) * (3^(1/3))) 3) ∧
  ¬(are_similar_radical_expressions a (3*a/2)) ∧
  ¬(are_similar_radical_expressions (2*a) (a^(1/2))) ∧
  ¬(are_similar_radical_expressions (2*a) ((3*a^2)^(1/2))) :=
by sorry

end NUMINAMATH_CALUDE_similar_radical_expressions_l2139_213996


namespace NUMINAMATH_CALUDE_storks_and_birds_difference_l2139_213904

theorem storks_and_birds_difference : 
  ∀ (initial_birds initial_storks additional_storks : ℕ),
    initial_birds = 4 →
    initial_storks = 3 →
    additional_storks = 6 →
    (initial_storks + additional_storks) - initial_birds = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_and_birds_difference_l2139_213904


namespace NUMINAMATH_CALUDE_min_printers_purchase_l2139_213965

def printer_cost_a : ℕ := 350
def printer_cost_b : ℕ := 200

theorem min_printers_purchase :
  ∃ (n_a n_b : ℕ),
    n_a * printer_cost_a = n_b * printer_cost_b ∧
    n_a + n_b = 11 ∧
    ∀ (m_a m_b : ℕ),
      m_a * printer_cost_a = m_b * printer_cost_b →
      m_a + m_b ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_min_printers_purchase_l2139_213965


namespace NUMINAMATH_CALUDE_cubic_equation_properties_l2139_213926

/-- The cubic equation (x-1)(x^2-3x+m) = 0 -/
def cubic_equation (x m : ℝ) : Prop := (x - 1) * (x^2 - 3*x + m) = 0

/-- The discriminant of the quadratic part x^2 - 3x + m -/
def discriminant (m : ℝ) : ℝ := 9 - 4*m

theorem cubic_equation_properties :
  /- When m = 4, the equation has only one real root x = 1 -/
  (∀ x : ℝ, cubic_equation x 4 ↔ x = 1) ∧
  /- The equation has exactly two equal roots when m = 2 or m = 9/4 -/
  (∀ x₁ x₂ x₃ : ℝ, (cubic_equation x₁ 2 ∧ cubic_equation x₂ 2 ∧ cubic_equation x₃ 2 ∧
    ((x₁ = x₂ ∧ x₁ ≠ x₃) ∨ (x₁ = x₃ ∧ x₁ ≠ x₂) ∨ (x₂ = x₃ ∧ x₁ ≠ x₂))) ∨
   (cubic_equation x₁ (9/4) ∧ cubic_equation x₂ (9/4) ∧ cubic_equation x₃ (9/4) ∧
    ((x₁ = x₂ ∧ x₁ ≠ x₃) ∨ (x₁ = x₃ ∧ x₁ ≠ x₂) ∨ (x₂ = x₃ ∧ x₁ ≠ x₂)))) ∧
  /- The three real roots form a triangle if and only if 2 < m ≤ 9/4 -/
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, cubic_equation x₁ m ∧ cubic_equation x₂ m ∧ cubic_equation x₃ m ∧
    x₁ + x₂ > x₃ ∧ x₁ + x₃ > x₂ ∧ x₂ + x₃ > x₁) ↔ (2 < m ∧ m ≤ 9/4)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_properties_l2139_213926


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2139_213925

theorem cubic_equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x^3 - 2*x^2 = 0) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2139_213925


namespace NUMINAMATH_CALUDE_sqrt_three_minus_sqrt_one_third_l2139_213920

theorem sqrt_three_minus_sqrt_one_third : 
  Real.sqrt 3 - Real.sqrt (1/3) = (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_sqrt_one_third_l2139_213920


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_y_l2139_213933

noncomputable def y (x : ℝ) : ℝ := (1/3) * Real.cos x - 1

theorem sum_of_max_and_min_y : 
  (⨆ (x : ℝ), y x) + (⨅ (x : ℝ), y x) = -2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_y_l2139_213933


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2139_213907

theorem arithmetic_geometric_sequence (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- distinct real numbers
  (b - a = c - b) →  -- arithmetic sequence
  (a / c = b / a) →  -- geometric sequence
  (a + 3*b + c = 10) →  -- sum condition
  a = -4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2139_213907


namespace NUMINAMATH_CALUDE_triangle_inequality_l2139_213911

theorem triangle_inequality (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_sum : a + f = b + c ∧ b + c = d + e) : 
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (c^2 - c*d + d^2) > Real.sqrt (e^2 - e*f + f^2) ∧
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (e^2 - e*f + f^2) > Real.sqrt (c^2 - c*d + d^2) ∧
  Real.sqrt (c^2 - c*d + d^2) + Real.sqrt (e^2 - e*f + f^2) > Real.sqrt (a^2 - a*b + b^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2139_213911


namespace NUMINAMATH_CALUDE_pencil_case_combinations_l2139_213908

theorem pencil_case_combinations :
  let n : ℕ := 6
  2^n = 64 :=
by sorry

end NUMINAMATH_CALUDE_pencil_case_combinations_l2139_213908


namespace NUMINAMATH_CALUDE_octahedron_tetrahedron_combination_l2139_213900

/-- Represents a regular octahedron --/
structure RegularOctahedron :=
  (edge_length : ℝ)

/-- Represents a regular tetrahedron --/
structure RegularTetrahedron :=
  (edge_length : ℝ)

/-- Theorem stating that it's possible to combine six regular octahedrons and eight regular tetrahedrons
    to form a larger regular octahedron with twice the edge length --/
theorem octahedron_tetrahedron_combination
  (small_octahedrons : Fin 6 → RegularOctahedron)
  (tetrahedrons : Fin 8 → RegularTetrahedron)
  (h1 : ∀ i j, small_octahedrons i = small_octahedrons j)  -- All small octahedrons are congruent
  (h2 : ∀ i, (tetrahedrons i).edge_length = (small_octahedrons 0).edge_length)  -- Tetrahedron edges equal octahedron edges
  : ∃ (large_octahedron : RegularOctahedron),
    large_octahedron.edge_length = 2 * (small_octahedrons 0).edge_length :=
by sorry

end NUMINAMATH_CALUDE_octahedron_tetrahedron_combination_l2139_213900


namespace NUMINAMATH_CALUDE_zero_vector_length_l2139_213916

theorem zero_vector_length (n : Type*) [NormedAddCommGroup n] : ‖(0 : n)‖ = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_vector_length_l2139_213916


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l2139_213943

-- Define M as a positive integer
def M : ℕ+ := sorry

-- Define the condition that M^2 = 36^50 * 50^36
axiom M_squared : (M : ℕ).pow 2 = (36 : ℕ).pow 50 * (50 : ℕ).pow 36

-- Define a function to calculate the sum of digits of a natural number
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_M : sum_of_digits (M : ℕ) = 344 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l2139_213943


namespace NUMINAMATH_CALUDE_inequality_bound_l2139_213903

theorem inequality_bound (m : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) → m ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_bound_l2139_213903


namespace NUMINAMATH_CALUDE_stamp_collection_value_l2139_213966

/-- Given a collection of stamps with equal individual value, 
    calculate the total value of the collection. -/
theorem stamp_collection_value 
  (total_stamps : ℕ) 
  (sample_stamps : ℕ) 
  (sample_value : ℕ) 
  (h1 : total_stamps = 21)
  (h2 : sample_stamps = 7)
  (h3 : sample_value = 28) :
  (total_stamps : ℚ) * (sample_value : ℚ) / (sample_stamps : ℚ) = 84 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l2139_213966


namespace NUMINAMATH_CALUDE_unique_alpha_beta_pair_l2139_213962

theorem unique_alpha_beta_pair :
  ∃! (α β : ℝ), α > 0 ∧ β > 0 ∧
  (∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 →
    x + y^2 + z^3 + w^6 ≥ α * (x*y*z*w)^β) ∧
  (∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    x + y^2 + z^3 + w^6 = α * (x*y*z*w)^β) ∧
  α = 2^(4/3) * 3^(1/4) ∧ β = 1/2 := by
sorry

end NUMINAMATH_CALUDE_unique_alpha_beta_pair_l2139_213962


namespace NUMINAMATH_CALUDE_incorrect_reasonings_l2139_213956

-- Define the type for analogical reasoning
inductive AnalogicalReasoning
  | addition_subtraction
  | vector_complex_square
  | quadratic_equation
  | geometric_addition

-- Define a function to check if a reasoning is correct
def is_correct_reasoning (r : AnalogicalReasoning) : Prop :=
  match r with
  | AnalogicalReasoning.addition_subtraction => True
  | AnalogicalReasoning.vector_complex_square => False
  | AnalogicalReasoning.quadratic_equation => False
  | AnalogicalReasoning.geometric_addition => True

-- Theorem statement
theorem incorrect_reasonings :
  ∃ (incorrect : List AnalogicalReasoning),
    incorrect.length = 2 ∧
    (∀ r ∈ incorrect, ¬(is_correct_reasoning r)) ∧
    (∀ r, r ∉ incorrect → is_correct_reasoning r) :=
  sorry

end NUMINAMATH_CALUDE_incorrect_reasonings_l2139_213956


namespace NUMINAMATH_CALUDE_monthly_payment_difference_l2139_213934

/-- The cost of the house in dollars -/
def house_cost : ℕ := 480000

/-- The cost of the trailer in dollars -/
def trailer_cost : ℕ := 120000

/-- The number of months over which the loans are paid -/
def loan_duration_months : ℕ := 240

/-- The monthly payment for the house -/
def house_monthly_payment : ℚ := house_cost / loan_duration_months

/-- The monthly payment for the trailer -/
def trailer_monthly_payment : ℚ := trailer_cost / loan_duration_months

/-- Theorem stating the difference in monthly payments -/
theorem monthly_payment_difference :
  house_monthly_payment - trailer_monthly_payment = 1500 := by
  sorry


end NUMINAMATH_CALUDE_monthly_payment_difference_l2139_213934


namespace NUMINAMATH_CALUDE_profit_calculation_l2139_213992

/-- Calculates the actual percent profit given the markup percentage and discount percentage -/
def actualPercentProfit (markup : ℝ) (discount : ℝ) : ℝ :=
  let labeledPrice := 1 + markup
  let sellingPrice := labeledPrice * (1 - discount)
  (sellingPrice - 1) * 100

/-- Theorem stating that a 40% markup with a 5% discount results in a 33% profit -/
theorem profit_calculation (markup discount : ℝ) 
  (h1 : markup = 0.4) 
  (h2 : discount = 0.05) : 
  actualPercentProfit markup discount = 33 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l2139_213992


namespace NUMINAMATH_CALUDE_sixteen_radii_ten_circles_regions_l2139_213927

/-- Calculates the number of regions created by radii and concentric circles within a larger circle -/
def regions_in_circle (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles create 176 regions -/
theorem sixteen_radii_ten_circles_regions :
  regions_in_circle 16 10 = 176 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_radii_ten_circles_regions_l2139_213927


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2139_213957

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : 
  |x - y| = 22 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2139_213957


namespace NUMINAMATH_CALUDE_exam_score_calculation_l2139_213953

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) 
  (correct_score : ℤ) (wrong_score : ℤ) : 
  total_questions = 120 →
  correct_answers = 75 →
  correct_score = 3 →
  wrong_score = -1 →
  (correct_answers * correct_score + (total_questions - correct_answers) * wrong_score : ℤ) = 180 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l2139_213953


namespace NUMINAMATH_CALUDE_largest_constant_K_l2139_213998

theorem largest_constant_K : ∃ (K : ℝ), K > 0 ∧
  (∀ (k : ℝ) (a b c : ℝ), 0 ≤ k ∧ k ≤ K ∧ 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
    a^2 + b^2 + c^2 + k*a*b*c = k + 3 →
    a + b + c ≤ 3) ∧
  (∀ (K' : ℝ), K' > K →
    ∃ (k : ℝ) (a b c : ℝ), 0 ≤ k ∧ k ≤ K' ∧ 
      a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
      a^2 + b^2 + c^2 + k*a*b*c = k + 3 ∧
      a + b + c > 3) ∧
  K = 1 := by
sorry

end NUMINAMATH_CALUDE_largest_constant_K_l2139_213998


namespace NUMINAMATH_CALUDE_product_of_numbers_l2139_213914

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 16) (h2 : x^2 + y^2 = 200) : x * y = 28 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2139_213914


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2139_213958

theorem mod_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 500000 [ZMOD 9] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2139_213958


namespace NUMINAMATH_CALUDE_problem_building_has_20_stories_l2139_213930

/-- A building with specific height properties -/
structure Building where
  first_stories : ℕ
  first_story_height : ℕ
  remaining_story_height : ℕ
  total_height : ℕ

/-- The number of stories in the building -/
def Building.total_stories (b : Building) : ℕ :=
  b.first_stories + (b.total_height - b.first_stories * b.first_story_height) / b.remaining_story_height

/-- The specific building described in the problem -/
def problem_building : Building := {
  first_stories := 10
  first_story_height := 12
  remaining_story_height := 15
  total_height := 270
}

/-- Theorem stating that the problem building has 20 stories -/
theorem problem_building_has_20_stories :
  problem_building.total_stories = 20 := by
  sorry

end NUMINAMATH_CALUDE_problem_building_has_20_stories_l2139_213930


namespace NUMINAMATH_CALUDE_initial_water_percentage_l2139_213994

/-- Proves that the initial water percentage in a mixture is 60% given the specified conditions --/
theorem initial_water_percentage (initial_volume : ℝ) (added_water : ℝ) (final_water_percentage : ℝ) :
  initial_volume = 300 →
  added_water = 100 →
  final_water_percentage = 70 →
  (initial_volume * x / 100 + added_water) / (initial_volume + added_water) * 100 = final_water_percentage →
  x = 60 := by
  sorry

#check initial_water_percentage

end NUMINAMATH_CALUDE_initial_water_percentage_l2139_213994


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2139_213982

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that for a geometric sequence with positive terms, if the third term is 8 and the seventh term is 18, then the fifth term is 12. -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_third : a 3 = 8)
  (h_seventh : a 7 = 18) :
  a 5 = 12 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2139_213982


namespace NUMINAMATH_CALUDE_smallest_total_books_satisfying_conditions_l2139_213922

/-- Represents the number of books for each subject -/
structure BookCounts where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Checks if the given book counts satisfy the required ratios -/
def satisfiesRatios (books : BookCounts) : Prop :=
  3 * books.chemistry = 2 * books.physics ∧
  4 * books.biology = 3 * books.chemistry

/-- Calculates the total number of books -/
def totalBooks (books : BookCounts) : ℕ :=
  books.physics + books.chemistry + books.biology

/-- Theorem stating the smallest possible total number of books satisfying the conditions -/
theorem smallest_total_books_satisfying_conditions :
  ∃ (books : BookCounts),
    satisfiesRatios books ∧
    totalBooks books > 3000 ∧
    ∀ (other : BookCounts),
      satisfiesRatios other → totalBooks other > 3000 →
      totalBooks books ≤ totalBooks other :=
by sorry

end NUMINAMATH_CALUDE_smallest_total_books_satisfying_conditions_l2139_213922


namespace NUMINAMATH_CALUDE_ellipse_chord_theorem_l2139_213901

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Define a chord passing through the left focus
def chord_through_left_focus (x1 y1 x2 y2 : ℝ) : Prop :=
  is_on_ellipse x1 y1 ∧ is_on_ellipse x2 y2 ∧
  ∃ t : ℝ, (1 - t) * x1 + t * x2 = -3 ∧ (1 - t) * y1 + t * y2 = 0

-- Define the incircle circumference condition
def incircle_circumference_2pi (x1 y1 x2 y2 : ℝ) : Prop :=
  ∃ r : ℝ, r * (Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) +
               Real.sqrt ((x1 - 3)^2 + y1^2) +
               Real.sqrt ((x2 - 3)^2 + y2^2)) = 10 ∧
           2 * Real.pi * r = 2 * Real.pi

theorem ellipse_chord_theorem (x1 y1 x2 y2 : ℝ) :
  chord_through_left_focus x1 y1 x2 y2 →
  incircle_circumference_2pi x1 y1 x2 y2 →
  |y1 - y2| = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_chord_theorem_l2139_213901


namespace NUMINAMATH_CALUDE_initial_deposit_calculation_l2139_213974

/-- Proves that the initial deposit is 8000 given the conditions of the problem -/
theorem initial_deposit_calculation (P R : ℝ) 
  (h1 : P * (1 + 3 * R / 100) = 9200)
  (h2 : P * (1 + 3 * (R + 1) / 100) = 9440) : 
  P = 8000 := by
  sorry

end NUMINAMATH_CALUDE_initial_deposit_calculation_l2139_213974


namespace NUMINAMATH_CALUDE_problem_solution_l2139_213967

theorem problem_solution :
  (∀ a b : ℝ, 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b) ∧
  (∀ x y : ℝ, (3*x - y)^2 - (3*x + 2*y) * (3*x - 2*y) = -6*x*y + 5*y^2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2139_213967


namespace NUMINAMATH_CALUDE_meena_cookies_left_l2139_213989

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of cookies Meena bakes -/
def meena_bakes : ℕ := 5

/-- The number of dozens of cookies sold to the biology teacher -/
def sold_to_teacher : ℕ := 2

/-- The number of cookies Brock buys -/
def brock_buys : ℕ := 7

/-- Katy buys twice as many cookies as Brock -/
def katy_buys : ℕ := 2 * brock_buys

/-- The total number of cookies Meena initially bakes -/
def total_baked : ℕ := meena_bakes * dozen

/-- The number of cookies sold to the biology teacher -/
def teacher_cookies : ℕ := sold_to_teacher * dozen

/-- The total number of cookies sold -/
def total_sold : ℕ := teacher_cookies + brock_buys + katy_buys

/-- The number of cookies Meena has left -/
def cookies_left : ℕ := total_baked - total_sold

theorem meena_cookies_left : cookies_left = 15 := by
  sorry

end NUMINAMATH_CALUDE_meena_cookies_left_l2139_213989


namespace NUMINAMATH_CALUDE_raul_initial_money_l2139_213979

def initial_money (comics_bought : ℕ) (comic_price : ℕ) (money_left : ℕ) : ℕ :=
  comics_bought * comic_price + money_left

theorem raul_initial_money :
  initial_money 8 4 55 = 87 := by
  sorry

end NUMINAMATH_CALUDE_raul_initial_money_l2139_213979


namespace NUMINAMATH_CALUDE_power_function_linear_intersection_min_value_l2139_213924

theorem power_function_linear_intersection_min_value (m n k b : ℝ) : 
  (2 * m - 1 = 1) →  -- Condition for power function
  (n - 2 = 0) →      -- Condition for power function
  (k > 0) →          -- Given condition for k
  (b > 0) →          -- Given condition for b
  (k * m + b = n) →  -- Linear function passes through (m, n)
  (∀ k' b' : ℝ, k' > 0 → b' > 0 → k' * m + b' = n → 4 / k' + 1 / b' ≥ 9 / 2) ∧ 
  (∃ k' b' : ℝ, k' > 0 ∧ b' > 0 ∧ k' * m + b' = n ∧ 4 / k' + 1 / b' = 9 / 2) :=
by sorry

end NUMINAMATH_CALUDE_power_function_linear_intersection_min_value_l2139_213924


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2139_213972

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, k*x^2 - 3*x + 1 = 0 ∧ 
   ∀ y : ℝ, k*y^2 - 3*y + 1 = 0 → y = x) → 
  k = 9/4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2139_213972


namespace NUMINAMATH_CALUDE_percentage_problem_l2139_213940

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 690 →
  (0.5 * x) = (P / 100 * 1500 - 30) →
  P = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2139_213940


namespace NUMINAMATH_CALUDE_minimum_additional_amount_l2139_213980

def current_order : ℝ := 49.90
def discount_rate : ℝ := 0.10
def target_amount : ℝ := 50.00

theorem minimum_additional_amount :
  ∃ (x : ℝ), x ≥ 0 ∧
  (current_order + x) * (1 - discount_rate) = target_amount ∧
  ∀ (y : ℝ), y ≥ 0 → (current_order + y) * (1 - discount_rate) ≥ target_amount → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_minimum_additional_amount_l2139_213980


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2139_213937

theorem fraction_sum_equality (a b c : ℝ) : 
  (a / (30 - a) + b / (70 - b) + c / (55 - c) = 9) →
  (6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 5.08) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2139_213937


namespace NUMINAMATH_CALUDE_extra_flowers_l2139_213973

def tulips : ℕ := 4
def roses : ℕ := 11
def used_flowers : ℕ := 11

theorem extra_flowers :
  tulips + roses - used_flowers = 4 :=
by sorry

end NUMINAMATH_CALUDE_extra_flowers_l2139_213973


namespace NUMINAMATH_CALUDE_expression_evaluation_l2139_213969

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = -0.5) :
  2 * (2 * x - 3 * y) - (3 * x + 2 * y + 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2139_213969


namespace NUMINAMATH_CALUDE_order_mnpq_l2139_213986

theorem order_mnpq (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n :=
by sorry

end NUMINAMATH_CALUDE_order_mnpq_l2139_213986


namespace NUMINAMATH_CALUDE_sister_age_problem_l2139_213976

theorem sister_age_problem (younger_current_age older_current_age : ℕ) 
  (h1 : younger_current_age = 18)
  (h2 : older_current_age = 26)
  (h3 : ∃ k : ℕ, younger_current_age - k + older_current_age - k = 20) :
  ∃ k : ℕ, older_current_age - k = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_sister_age_problem_l2139_213976


namespace NUMINAMATH_CALUDE_six_integers_mean_double_mode_l2139_213963

def is_valid_list (l : List Int) : Prop :=
  l.length = 6 ∧ l.all (λ x => x > 0 ∧ x ≤ 150)

def mean (l : List Int) : Rat :=
  (l.sum : Rat) / l.length

def mode (l : List Int) : Int :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem six_integers_mean_double_mode :
  ∀ y z : Int,
    let l := [45, 76, y, y, z, z]
    is_valid_list l →
    mean l = 2 * (mode l : Rat) →
    y = 49 ∧ z = 21 := by
  sorry

end NUMINAMATH_CALUDE_six_integers_mean_double_mode_l2139_213963


namespace NUMINAMATH_CALUDE_blue_fish_ratio_l2139_213983

theorem blue_fish_ratio (total_fish : ℕ) (blue_spotted_fish : ℕ) : 
  total_fish = 30 →
  blue_spotted_fish = 5 →
  (blue_spotted_fish : ℚ) / (total_fish : ℚ) = 1/6 →
  (2 * blue_spotted_fish : ℚ) / (total_fish : ℚ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_blue_fish_ratio_l2139_213983


namespace NUMINAMATH_CALUDE_sufficient_to_necessary_contrapositive_l2139_213995

theorem sufficient_to_necessary_contrapositive (a b : Prop) :
  (a → b) → (¬b → ¬a) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_to_necessary_contrapositive_l2139_213995


namespace NUMINAMATH_CALUDE_denise_removed_five_bananas_l2139_213981

/-- The number of bananas Denise removed from the jar -/
def bananas_removed (original : ℕ) (remaining : ℕ) : ℕ := original - remaining

/-- Theorem stating that Denise removed 5 bananas -/
theorem denise_removed_five_bananas :
  bananas_removed 46 41 = 5 := by
  sorry

end NUMINAMATH_CALUDE_denise_removed_five_bananas_l2139_213981


namespace NUMINAMATH_CALUDE_probability_all_even_simplified_l2139_213984

def total_slips : ℕ := 49
def even_slips : ℕ := 9
def draws : ℕ := 8

def probability_all_even : ℚ :=
  (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2) / (49 * 48 * 47 * 46 * 45 * 44 * 43 * 42)

theorem probability_all_even_simplified (h : probability_all_even = 5 / (6 * 47 * 23 * 45 * 11 * 43 * 7)) :
  probability_all_even = 5 / (6 * 47 * 23 * 45 * 11 * 43 * 7) := by
  sorry

end NUMINAMATH_CALUDE_probability_all_even_simplified_l2139_213984


namespace NUMINAMATH_CALUDE_james_tylenol_intake_l2139_213923

/-- Calculates the total milligrams of Tylenol taken per day given the number of tablets per dose,
    milligrams per tablet, hours between doses, and hours in a day. -/
def tylenolPerDay (tabletsPerDose : ℕ) (mgPerTablet : ℕ) (hoursBetweenDoses : ℕ) (hoursInDay : ℕ) : ℕ :=
  let mgPerDose := tabletsPerDose * mgPerTablet
  let dosesPerDay := hoursInDay / hoursBetweenDoses
  mgPerDose * dosesPerDay

/-- Proves that James takes 3000 mg of Tylenol per day given the specified conditions. -/
theorem james_tylenol_intake : tylenolPerDay 2 375 6 24 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_james_tylenol_intake_l2139_213923


namespace NUMINAMATH_CALUDE_at_least_one_negative_l2139_213946

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l2139_213946


namespace NUMINAMATH_CALUDE_green_ball_probability_l2139_213988

/-- Represents a container with balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from two containers -/
def prob_green (a b : Container) : ℚ :=
  let total_a := a.red + a.green
  let total_b := b.red + b.green
  let prob_a := (a.green : ℚ) / (2 * total_a)
  let prob_b := (b.green : ℚ) / (2 * total_b)
  prob_a + prob_b

theorem green_ball_probability :
  let a := Container.mk 5 5
  let b := Container.mk 7 3
  prob_green a b = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2139_213988


namespace NUMINAMATH_CALUDE_total_limes_is_57_l2139_213917

/-- The number of limes Alyssa picked -/
def alyssa_limes : ℕ := 25

/-- The number of limes Mike picked -/
def mike_limes : ℕ := 32

/-- The total number of limes picked -/
def total_limes : ℕ := alyssa_limes + mike_limes

/-- Theorem: The total number of limes picked is 57 -/
theorem total_limes_is_57 : total_limes = 57 := by
  sorry

end NUMINAMATH_CALUDE_total_limes_is_57_l2139_213917


namespace NUMINAMATH_CALUDE_ding_score_is_97_l2139_213964

-- Define the average score of Jia, Yi, and Bing
def avg_three : ℝ := 89

-- Define Ding's score
def ding_score : ℝ := 97

-- Define the average score of all four people
def avg_four : ℝ := avg_three + 2

-- Theorem statement
theorem ding_score_is_97 :
  ding_score = 4 * avg_four - 3 * avg_three :=
by sorry

end NUMINAMATH_CALUDE_ding_score_is_97_l2139_213964


namespace NUMINAMATH_CALUDE_problem_solution_l2139_213961

theorem problem_solution (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : 6 * x^3 + 12 * x^2 * y = 2 * x^4 + 3 * x^3 * y) 
  (h4 : x + y = 3) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2139_213961


namespace NUMINAMATH_CALUDE_prob_six_largest_is_two_sevenths_l2139_213905

/-- A function that calculates the probability of selecting 6 as the largest value
    when drawing 4 cards from a set of 7 cards numbered 1 to 7 without replacement -/
def prob_six_largest (n : ℕ) (k : ℕ) : ℚ :=
  if n = 7 ∧ k = 4 then 2/7 else 0

/-- Theorem stating that the probability of selecting 6 as the largest value
    when drawing 4 cards from a set of 7 cards numbered 1 to 7 without replacement is 2/7 -/
theorem prob_six_largest_is_two_sevenths :
  prob_six_largest 7 4 = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_six_largest_is_two_sevenths_l2139_213905


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2139_213951

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 - 3 / (x + 2)) / ((x^2 - 2*x + 1) / (3*x + 6)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2139_213951


namespace NUMINAMATH_CALUDE_road_repair_workers_l2139_213999

/-- The number of persons in the first group -/
def first_group : ℕ := 63

/-- The number of days the first group works -/
def first_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_hours : ℕ := 5

/-- The number of days the second group works -/
def second_days : ℕ := 21

/-- The number of hours per day the second group works -/
def second_hours : ℕ := 6

/-- The total man-hours required to complete the work -/
def total_man_hours : ℕ := first_group * first_days * first_hours

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (second_days * second_hours)

theorem road_repair_workers :
  second_group = 30 :=
by sorry

end NUMINAMATH_CALUDE_road_repair_workers_l2139_213999


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_odd_integers_l2139_213960

theorem largest_of_five_consecutive_odd_integers (a b c d e : ℤ) : 
  (∃ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5 ∧ d = 2*n + 7 ∧ e = 2*n + 9) →
  a + b + c + d + e = 255 →
  max a (max b (max c (max d e))) = 55 :=
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_odd_integers_l2139_213960


namespace NUMINAMATH_CALUDE_unique_solution_l2139_213942

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, 
    (∀ x y z : ℝ, f x y + f y z + f z x = max x (max y z) - min x (min y z)) ∧
    (∀ x : ℝ, f x a = f a x)

/-- The theorem stating the unique solution -/
theorem unique_solution : 
  ∃! f : ℝ → ℝ → ℝ, satisfies_conditions f ∧ ∀ x y : ℝ, f x y = |x - y| / 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2139_213942


namespace NUMINAMATH_CALUDE_min_value_a_l2139_213959

theorem min_value_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x : ℝ, x ≥ 1 → a^x ≥ a*x) :
  ∀ b : ℝ, (b > 0 ∧ b ≠ 1 ∧ (∀ x : ℝ, x ≥ 1 → b^x ≥ b*x)) → a ≤ b → a = Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l2139_213959


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2139_213978

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = -5) :
  a 1 - a 2 - a 3 - a 4 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2139_213978


namespace NUMINAMATH_CALUDE_graph_transformation_l2139_213910

/-- Given a function f(x) = sin(x - π/3), prove that after stretching its x-coordinates
    to twice their original length and shifting the resulting graph to the right by π/3 units,
    the equation of the resulting graph is y = sin(x/2 - π/2). -/
theorem graph_transformation (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ Real.sin (x - π/3)
  let g : ℝ → ℝ := fun x ↦ f (x/2)
  let h : ℝ → ℝ := fun x ↦ g (x - π/3)
  h x = Real.sin (x/2 - π/2) := by
  sorry

end NUMINAMATH_CALUDE_graph_transformation_l2139_213910


namespace NUMINAMATH_CALUDE_triangle_properties_l2139_213919

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.a - t.b) * Real.cos t.C + 2 * t.c * Real.sin (t.B / 2) ^ 2 = t.c)
  (h2 : t.a + t.b = 4)
  (h3 : t.c = Real.sqrt 7) :
  t.C = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2139_213919


namespace NUMINAMATH_CALUDE_miriam_initial_marbles_l2139_213906

/-- The number of marbles Miriam initially had --/
def initial_marbles : ℕ := sorry

/-- The number of marbles Miriam currently has --/
def current_marbles : ℕ := 30

/-- The number of marbles Miriam gave to her brother --/
def brother_marbles : ℕ := 60

/-- The number of marbles Miriam gave to her sister --/
def sister_marbles : ℕ := 2 * brother_marbles

/-- The number of marbles Miriam gave to her friend Savanna --/
def savanna_marbles : ℕ := 3 * current_marbles

theorem miriam_initial_marbles :
  initial_marbles = current_marbles + brother_marbles + sister_marbles + savanna_marbles ∧
  initial_marbles = 300 := by
  sorry

end NUMINAMATH_CALUDE_miriam_initial_marbles_l2139_213906


namespace NUMINAMATH_CALUDE_roots_of_equation_l2139_213915

theorem roots_of_equation : 
  let f : ℝ → ℝ := fun x => x * (x - 1) + 3 * (x - 1)
  (f (-3) = 0 ∧ f 1 = 0) ∧ ∀ x : ℝ, f x = 0 → (x = -3 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2139_213915


namespace NUMINAMATH_CALUDE_inequality_implication_l2139_213948

theorem inequality_implication (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)) : 
  a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2139_213948


namespace NUMINAMATH_CALUDE_complex_roots_power_sum_l2139_213918

theorem complex_roots_power_sum (α β : ℂ) (p : ℕ) : 
  (2 * α^4 - 6 * α^3 + 11 * α^2 - 6 * α - 4 = 0) →
  (2 * β^4 - 6 * β^3 + 11 * β^2 - 6 * β - 4 = 0) →
  p ≥ 5 →
  α^p + β^p = (α + β)^p := by sorry

end NUMINAMATH_CALUDE_complex_roots_power_sum_l2139_213918


namespace NUMINAMATH_CALUDE_calculation_proof_l2139_213947

theorem calculation_proof : ((-4)^2 * (-1/2)^3 - (-4+1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2139_213947


namespace NUMINAMATH_CALUDE_light_bulb_state_l2139_213929

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def toggle_light (n : ℕ) (i : ℕ) : Bool := i % n = 0

def final_state (n : ℕ) : Bool :=
  (List.range n).foldl (fun acc i => acc ≠ toggle_light (i + 1) n) false

theorem light_bulb_state (n : ℕ) (hn : n ≤ 100) :
  final_state n = true ↔ is_perfect_square n :=
sorry

end NUMINAMATH_CALUDE_light_bulb_state_l2139_213929


namespace NUMINAMATH_CALUDE_salary_increase_l2139_213952

theorem salary_increase (num_employees : ℕ) (avg_salary : ℝ) (manager_salary : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  manager_salary = 12000 →
  let total_salary := num_employees * avg_salary
  let new_total := total_salary + manager_salary
  let new_avg := new_total / (num_employees + 1)
  new_avg - avg_salary = 500 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l2139_213952


namespace NUMINAMATH_CALUDE_congruent_to_one_mod_seven_l2139_213968

theorem congruent_to_one_mod_seven (n : ℕ) : 
  (Finset.filter (fun k => k % 7 = 1) (Finset.range 300)).card = 43 := by
  sorry

end NUMINAMATH_CALUDE_congruent_to_one_mod_seven_l2139_213968


namespace NUMINAMATH_CALUDE_valid_sequences_10_l2139_213902

def T : ℕ → ℕ
  | 0 => 0  -- We define T(0) as 0 for completeness
  | 1 => 2
  | 2 => 4
  | (n + 3) => T (n + 2) + T (n + 1)

def valid_sequences (n : ℕ) : ℕ := T n

theorem valid_sequences_10 : valid_sequences 10 = 178 := by
  sorry

#eval valid_sequences 10

end NUMINAMATH_CALUDE_valid_sequences_10_l2139_213902


namespace NUMINAMATH_CALUDE_johns_number_l2139_213912

theorem johns_number : ∃! n : ℕ, 
  200 ∣ n ∧ 
  18 ∣ n ∧ 
  1000 < n ∧ 
  n < 2500 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_johns_number_l2139_213912


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2139_213944

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2139_213944


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2139_213977

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2139_213977


namespace NUMINAMATH_CALUDE_remaining_land_to_clean_l2139_213945

theorem remaining_land_to_clean (total_land area_lizzie area_other : ℕ) :
  total_land = 900 ∧ area_lizzie = 250 ∧ area_other = 265 →
  total_land - (area_lizzie + area_other) = 385 := by
  sorry

end NUMINAMATH_CALUDE_remaining_land_to_clean_l2139_213945


namespace NUMINAMATH_CALUDE_completing_square_l2139_213932

theorem completing_square (x : ℝ) : x^2 + 4*x + 1 = 0 ↔ (x + 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l2139_213932


namespace NUMINAMATH_CALUDE_function_inequality_and_zero_relation_l2139_213909

noncomputable section

variables (a : ℝ) (x x₀ x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + Real.log x

def g (x : ℝ) : ℝ := 2 * x + (a / 2) * Real.log x

theorem function_inequality_and_zero_relation 
  (h₁ : ∀ x > 0, f x ≥ g x)
  (h₂ : f x₁ = 0)
  (h₃ : f x₂ = 0)
  (h₄ : x₁ < x₂)
  (h₅ : x₀ = -a/4) :
  a ≥ (4 + 4 * Real.log 2) / (1 + 2 * Real.log 2) ∧ 
  x₁ / x₂ > 4 * Real.exp x₀ :=
sorry

end NUMINAMATH_CALUDE_function_inequality_and_zero_relation_l2139_213909


namespace NUMINAMATH_CALUDE_daughters_return_days_l2139_213971

/-- Represents the return frequency of each daughter in days -/
structure DaughterReturnFrequency where
  eldest : Nat
  middle : Nat
  youngest : Nat

/-- Calculates the number of days at least one daughter returns home -/
def daysAtLeastOneDaughterReturns (freq : DaughterReturnFrequency) (period : Nat) : Nat :=
  sorry

theorem daughters_return_days (freq : DaughterReturnFrequency) (period : Nat) :
  freq.eldest = 5 →
  freq.middle = 4 →
  freq.youngest = 3 →
  period = 100 →
  daysAtLeastOneDaughterReturns freq period = 60 := by
  sorry

end NUMINAMATH_CALUDE_daughters_return_days_l2139_213971


namespace NUMINAMATH_CALUDE_negative_one_to_zero_power_equals_one_l2139_213997

theorem negative_one_to_zero_power_equals_one : (-1 : ℤ) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_to_zero_power_equals_one_l2139_213997


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l2139_213987

theorem solution_implies_m_value (x m : ℝ) : 
  x = 2 → 4 * x + 2 * m - 14 = 0 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l2139_213987


namespace NUMINAMATH_CALUDE_problem_solution_l2139_213970

theorem problem_solution (x y : ℝ) 
  (h1 : (1/2) * (x - 2)^3 + 32 = 0)
  (h2 : 3*x - 2*y = 6^2) :
  Real.sqrt (x^2 - y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2139_213970


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2139_213950

theorem inequality_and_equality_condition (α β a b : ℝ) (h_pos_α : 0 < α) (h_pos_β : 0 < β)
  (h_a_range : α ≤ a ∧ a ≤ β) (h_b_range : α ≤ b ∧ b ≤ β) :
  b / a + a / b ≤ β / α + α / β ∧
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β) ∨ (a = β ∧ b = α)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2139_213950


namespace NUMINAMATH_CALUDE_parallelepiped_net_squares_l2139_213949

/-- Represents a paper parallelepiped -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the net of an unfolded parallelepiped -/
structure Net where
  squares : ℕ

/-- The function that unfolds a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := 2 * (p.length * p.width + p.length * p.height + p.width * p.height) }

/-- The theorem to be proved -/
theorem parallelepiped_net_squares (p : Parallelepiped) (n : Net) :
  p.length = 2 ∧ p.width = 1 ∧ p.height = 1 →
  unfold p = n →
  n.squares - 1 = 9 →
  n.squares = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_net_squares_l2139_213949


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l2139_213955

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) ↔
  (x^2 / 25 + y^2 / 21 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l2139_213955


namespace NUMINAMATH_CALUDE_outfit_choices_l2139_213991

/-- The number of shirts -/
def num_shirts : ℕ := 5

/-- The number of pants -/
def num_pants : ℕ := 5

/-- The number of hats -/
def num_hats : ℕ := 7

/-- The number of colors shared by shirts and pants -/
def num_shared_colors : ℕ := 5

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of undesired outfit combinations (shirt and pants same color) -/
def undesired_combinations : ℕ := num_shared_colors * num_hats

/-- The number of valid outfit choices -/
def valid_outfit_choices : ℕ := total_combinations - undesired_combinations

theorem outfit_choices :
  valid_outfit_choices = 140 := by sorry

end NUMINAMATH_CALUDE_outfit_choices_l2139_213991


namespace NUMINAMATH_CALUDE_curve_is_ellipse_l2139_213928

/-- Given m ∈ ℝ, the curve C is represented by the equation (2-m)x² + (m+1)y² = 1.
    This theorem states that when m is between 1/2 and 2 (exclusive),
    the curve C represents an ellipse with foci on the x-axis. -/
theorem curve_is_ellipse (m : ℝ) (h1 : 1/2 < m) (h2 : m < 2) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  ∀ (x y : ℝ), (2-m)*x^2 + (m+1)*y^2 = 1 ↔ x^2/a^2 + y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_l2139_213928


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2139_213954

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / a = Real.sqrt 3) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 4) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 / 12 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2139_213954


namespace NUMINAMATH_CALUDE_algebraic_equality_l2139_213935

theorem algebraic_equality (m n : ℝ) : 4*m + 2*n - (n - m) = 5*m + n := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equality_l2139_213935


namespace NUMINAMATH_CALUDE_equal_sum_product_quotient_l2139_213936

theorem equal_sum_product_quotient :
  ∃! (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a + b = a * b ∧ a + b = a / b ∧ a = 1/2 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_product_quotient_l2139_213936


namespace NUMINAMATH_CALUDE_mothers_age_is_50_point_5_l2139_213975

def allen_age (mother_age : ℝ) : ℝ := mother_age - 30

theorem mothers_age_is_50_point_5 (mother_age : ℝ) :
  allen_age mother_age = mother_age - 30 →
  allen_age mother_age + 7 + (mother_age + 7) = 85 →
  mother_age = 50.5 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_is_50_point_5_l2139_213975


namespace NUMINAMATH_CALUDE_gcf_of_180_240_45_l2139_213993

theorem gcf_of_180_240_45 : Nat.gcd 180 (Nat.gcd 240 45) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_240_45_l2139_213993


namespace NUMINAMATH_CALUDE_peach_apple_ratio_l2139_213913

/-- Given that Mr. Connell harvested 60 apples and the difference between
    the number of peaches and apples is 120, prove that the ratio of
    peaches to apples is 3:1. -/
theorem peach_apple_ratio :
  ∀ (peaches : ℕ),
  peaches - 60 = 120 →
  (peaches : ℚ) / 60 = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_peach_apple_ratio_l2139_213913


namespace NUMINAMATH_CALUDE_f_properties_l2139_213938

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem f_properties :
  (∃ (x_max : ℝ), x_max = ℯ ∧ ∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  (f 4 < f π ∧ f π < f 3) ∧
  (π^4 < 4^π) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2139_213938


namespace NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l2139_213941

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 2 * 8753 * k) :
  Int.gcd (4 * b^2 + 27 * b + 100) (3 * b + 7) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l2139_213941


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l2139_213921

/-- Represents a card color -/
inductive CardColor
| Red
| Black
| Blue
| White

/-- Represents a person -/
inductive Person
| A
| B
| C
| D

/-- Represents the distribution of cards to people -/
def Distribution := Person → CardColor

/-- The event "A receives the red card" -/
def event_A_red (d : Distribution) : Prop := d Person.A = CardColor.Red

/-- The event "B receives the red card" -/
def event_B_red (d : Distribution) : Prop := d Person.B = CardColor.Red

/-- The set of all possible distributions -/
def all_distributions : Set Distribution :=
  {d | ∀ c : CardColor, ∃! p : Person, d p = c}

theorem events_mutually_exclusive_but_not_opposite :
  (∀ d : Distribution, d ∈ all_distributions →
    ¬(event_A_red d ∧ event_B_red d)) ∧
  (∃ d : Distribution, d ∈ all_distributions ∧
    ¬event_A_red d ∧ ¬event_B_red d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l2139_213921


namespace NUMINAMATH_CALUDE_bird_problem_equations_l2139_213931

/-- Represents the cost of each type of bird in coins -/
structure BirdCosts where
  rooster : ℚ
  hen : ℚ
  chick : ℚ

/-- Represents the quantities of each type of bird -/
structure BirdQuantities where
  roosters : ℕ
  hens : ℕ
  chicks : ℕ

/-- The problem constraints -/
def bird_problem (costs : BirdCosts) (quantities : BirdQuantities) : Prop :=
  costs.rooster = 5 ∧
  costs.hen = 3 ∧
  costs.chick = 1/3 ∧
  quantities.roosters = 8 ∧
  quantities.roosters + quantities.hens + quantities.chicks = 100

/-- The system of equations representing the problem -/
def problem_equations (costs : BirdCosts) (quantities : BirdQuantities) : Prop :=
  costs.rooster * quantities.roosters + costs.hen * quantities.hens + costs.chick * quantities.chicks = 100 ∧
  quantities.roosters + quantities.hens + quantities.chicks = 100

/-- Theorem stating that the problem constraints imply the system of equations -/
theorem bird_problem_equations (costs : BirdCosts) (quantities : BirdQuantities) :
  bird_problem costs quantities → problem_equations costs quantities :=
by
  sorry


end NUMINAMATH_CALUDE_bird_problem_equations_l2139_213931


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2139_213985

theorem unique_solution_condition (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ k = 35 / 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2139_213985


namespace NUMINAMATH_CALUDE_cos_shift_equals_sin_l2139_213939

theorem cos_shift_equals_sin (x : ℝ) : 
  Real.cos (2 * x - π / 4) = Real.sin (2 * (x + π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_equals_sin_l2139_213939
