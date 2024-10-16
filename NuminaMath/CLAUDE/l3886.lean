import Mathlib

namespace NUMINAMATH_CALUDE_base_10_to_base_7_l3886_388660

theorem base_10_to_base_7 : 
  (1 * 7^3 + 5 * 7^2 + 1 * 7^1 + 5 * 7^0 : ℕ) = 600 := by
  sorry

#eval 1 * 7^3 + 5 * 7^2 + 1 * 7^1 + 5 * 7^0

end NUMINAMATH_CALUDE_base_10_to_base_7_l3886_388660


namespace NUMINAMATH_CALUDE_savings_calculation_l3886_388644

/-- Calculates a person's savings given their income and income-to-expenditure ratio -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem stating that for an income of 15000 and an income-to-expenditure ratio of 15:8, the savings are 7000 -/
theorem savings_calculation :
  calculate_savings 15000 15 8 = 7000 := by
  sorry

#eval calculate_savings 15000 15 8

end NUMINAMATH_CALUDE_savings_calculation_l3886_388644


namespace NUMINAMATH_CALUDE_paintable_area_is_1520_l3886_388684

/-- Calculates the total paintable area of walls in multiple bedrooms. -/
def total_paintable_area (num_bedrooms length width height unpaintable_area : ℝ) : ℝ :=
  num_bedrooms * ((2 * (length * height + width * height)) - unpaintable_area)

/-- Proves that the total paintable area of walls in 4 bedrooms is 1520 square feet. -/
theorem paintable_area_is_1520 :
  total_paintable_area 4 14 11 9 70 = 1520 := by
  sorry

end NUMINAMATH_CALUDE_paintable_area_is_1520_l3886_388684


namespace NUMINAMATH_CALUDE_division_problem_l3886_388628

theorem division_problem (a b : ℕ) (h1 : a = 555) (h2 : b = 445) : 
  220050 % (a + b) = 50 ∧ 220050 / (a + b) = 2 * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3886_388628


namespace NUMINAMATH_CALUDE_bar_chart_clarity_l3886_388647

/-- Represents a bar chart --/
structure BarChart where
  data : List (String × ℝ)

/-- Represents the clarity of quantity representation in a chart --/
def ClearQuantityRepresentation : Prop := True

/-- Theorem: A bar chart clearly shows the amount of each quantity it represents --/
theorem bar_chart_clarity (chart : BarChart) : ClearQuantityRepresentation := by
  sorry

end NUMINAMATH_CALUDE_bar_chart_clarity_l3886_388647


namespace NUMINAMATH_CALUDE_expression_factorization_l3886_388603

theorem expression_factorization (y : ℝ) : 
  (12 * y^6 + 35 * y^4 - 5) - (2 * y^6 - 4 * y^4 + 5) = 10 * (y^6 + 3.9 * y^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3886_388603


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l3886_388677

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 418 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l3886_388677


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3886_388604

theorem quadratic_minimum_value (x y : ℝ) :
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 ≥ -13/5 ∧
  ∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * x₀ * y₀ + 2 * y₀^2 - 6 * x₀ + 8 * y₀ + 10 = -13/5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3886_388604


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3886_388696

theorem hyperbola_vertices_distance (x y : ℝ) :
  (x^2 / 121 - y^2 / 49 = 1) →
  (∃ v₁ v₂ : ℝ × ℝ, v₁.1 = -11 ∧ v₁.2 = 0 ∧ v₂.1 = 11 ∧ v₂.2 = 0 ∧
    (v₁.1 - v₂.1)^2 + (v₁.2 - v₂.2)^2 = 22^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3886_388696


namespace NUMINAMATH_CALUDE_geometric_sequence_theorem_l3886_388627

/-- A geometric sequence with positive common ratio -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_theorem (a : ℕ → ℝ) :
  GeometricSequence a →
  a 2 * a 10 = 2 * (a 5)^2 →
  a 2 = 1 →
  ∀ n : ℕ, a n = 2^((n - 2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_theorem_l3886_388627


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3886_388614

theorem min_sum_of_squares (x y z : ℝ) : 
  (x + 5) * (y - 5) = 0 →
  (y + 5) * (z - 5) = 0 →
  (z + 5) * (x - 5) = 0 →
  x^2 + y^2 + z^2 ≥ 75 ∧ ∃ (x' y' z' : ℝ), 
    (x' + 5) * (y' - 5) = 0 ∧
    (y' + 5) * (z' - 5) = 0 ∧
    (z' + 5) * (x' - 5) = 0 ∧
    x'^2 + y'^2 + z'^2 = 75 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3886_388614


namespace NUMINAMATH_CALUDE_commute_time_difference_l3886_388626

theorem commute_time_difference (x y : ℝ) : 
  (x + y + 10 + 11 + 9) / 5 = 10 →
  ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2 →
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_commute_time_difference_l3886_388626


namespace NUMINAMATH_CALUDE_problem_statement_l3886_388618

theorem problem_statement (x : ℝ) : 
  x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10 →
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 289/8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3886_388618


namespace NUMINAMATH_CALUDE_rational_equation_solution_l3886_388699

theorem rational_equation_solution : 
  ∃ (x : ℚ), (x + 11) / (x - 4) = (x - 3) / (x + 7) ∧ x = -13/5 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l3886_388699


namespace NUMINAMATH_CALUDE_janabel_widget_sales_l3886_388639

def arithmeticSequenceSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem janabel_widget_sales : 
  let a₁ : ℕ := 2  -- First day sales
  let d : ℕ := 3   -- Daily increase
  let n : ℕ := 15  -- Number of days
  let bonus : ℕ := 1  -- Bonus widget on last day
  arithmeticSequenceSum a₁ d n + bonus = 346 :=
by
  sorry

#check janabel_widget_sales

end NUMINAMATH_CALUDE_janabel_widget_sales_l3886_388639


namespace NUMINAMATH_CALUDE_volume_ratio_cubes_l3886_388600

/-- Given two cubes with edge lengths in the ratio 3:1, if the volume of the smaller cube is 1 unit,
    then the volume of the larger cube is 27 units. -/
theorem volume_ratio_cubes (e : ℝ) (h1 : e > 0) (h2 : e^3 = 1) :
  (3*e)^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_cubes_l3886_388600


namespace NUMINAMATH_CALUDE_quiz_score_problem_l3886_388662

theorem quiz_score_problem (total_questions : ℕ) 
  (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 20 ∧ 
  correct_points = 7 ∧ 
  incorrect_points = -4 ∧ 
  total_score = 100 → 
  ∃ (correct incorrect blank : ℕ), 
    correct + incorrect + blank = total_questions ∧ 
    correct_points * correct + incorrect_points * incorrect = total_score ∧ 
    blank = 1 :=
by sorry

end NUMINAMATH_CALUDE_quiz_score_problem_l3886_388662


namespace NUMINAMATH_CALUDE_investment_problem_l3886_388685

theorem investment_problem (x y : ℝ) : 
  x * 0.10 - y * 0.08 = 83 →
  y = 650 →
  x + y = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3886_388685


namespace NUMINAMATH_CALUDE_inequality_range_l3886_388691

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3886_388691


namespace NUMINAMATH_CALUDE_parking_ticket_ratio_l3886_388666

/-- Represents the number of tickets for each person -/
structure Tickets where
  parking : ℕ
  speeding : ℕ

/-- The problem setup -/
def ticketProblem (mark sarah : Tickets) : Prop :=
  mark.speeding = sarah.speeding ∧
  sarah.speeding = 6 ∧
  mark.parking = 8 ∧
  mark.parking + mark.speeding + sarah.parking + sarah.speeding = 24

/-- The theorem to prove -/
theorem parking_ticket_ratio (mark sarah : Tickets) 
  (h : ticketProblem mark sarah) : 
  mark.parking * 1 = sarah.parking * 2 := by
  sorry


end NUMINAMATH_CALUDE_parking_ticket_ratio_l3886_388666


namespace NUMINAMATH_CALUDE_square_inequality_l3886_388652

theorem square_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 2 / (x - y)) : x^2 > y^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l3886_388652


namespace NUMINAMATH_CALUDE_probability_is_one_seventh_l3886_388650

/-- Represents the total number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of socks per color -/
def socks_per_color : ℕ := 2

/-- Represents the number of socks drawn -/
def socks_drawn : ℕ := 6

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the probability of drawing exactly two pairs of different colors -/
def probability_two_pairs : ℚ :=
  let total_outcomes := choose total_socks socks_drawn
  let favorable_outcomes := choose num_colors 2 * choose (num_colors - 2) 2
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_is_one_seventh :
  probability_two_pairs = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_seventh_l3886_388650


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l3886_388687

theorem divisible_by_twelve : ∃! n : ℕ, n < 10 ∧ 12 ∣ (3150 + n) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l3886_388687


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3886_388608

/-- The total surface area of a rectangular solid -/
def total_surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: The total surface area of a rectangular solid with length 5 meters, width 4 meters, and depth 1 meter is 58 square meters -/
theorem rectangular_solid_surface_area :
  total_surface_area 5 4 1 = 58 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3886_388608


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l3886_388679

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem twentieth_term_of_sequence :
  let a₁ := 8
  let d := -3
  arithmetic_sequence a₁ d 20 = -49 := by sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l3886_388679


namespace NUMINAMATH_CALUDE_sin_cos_roots_quadratic_l3886_388690

theorem sin_cos_roots_quadratic (θ : Real) (m : Real) : 
  (4 * (Real.sin θ)^2 + 2 * m * (Real.sin θ) + m = 0) ∧ 
  (4 * (Real.cos θ)^2 + 2 * m * (Real.cos θ) + m = 0) →
  m = 1 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_roots_quadratic_l3886_388690


namespace NUMINAMATH_CALUDE_max_value_theorem_min_value_theorem_l3886_388668

/-- Given a > b > 0 and 7a² + 8ab + 4b² = 24, the maximum value of 3a + 2b occurs when b = √2/2 -/
theorem max_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 7 * a^2 + 8 * a * b + 4 * b^2 = 24) :
  (∀ a' b', a' > b' ∧ b' > 0 ∧ 7 * a'^2 + 8 * a' * b' + 4 * b'^2 = 24 → 3 * a' + 2 * b' ≤ 3 * a + 2 * b) →
  b = Real.sqrt 2 / 2 :=
sorry

/-- Given a > b > 0 and 1/(a - b) + 1/b = 1, the minimum value of a + 3b is 9 -/
theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 1 / (a - b) + 1 / b = 1) :
  a + 3 * b ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_min_value_theorem_l3886_388668


namespace NUMINAMATH_CALUDE_survey_dislike_radio_and_music_l3886_388665

theorem survey_dislike_radio_and_music :
  ∀ (total : ℕ) (dislike_radio_percent : ℚ) (dislike_both_percent : ℚ),
    total = 2000 →
    dislike_radio_percent = 25 / 100 →
    dislike_both_percent = 15 / 100 →
    (total * dislike_radio_percent * dislike_both_percent : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_survey_dislike_radio_and_music_l3886_388665


namespace NUMINAMATH_CALUDE_square_perimeter_l3886_388638

theorem square_perimeter (s : Real) : 
  s > 0 → 
  (5 * s / 2 = 32) → 
  (4 * s = 51.2) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3886_388638


namespace NUMINAMATH_CALUDE_divisor_product_sum_theorem_l3886_388649

/-- The type of positive divisors of n -/
def Divisor (n : ℕ) := { d : ℕ // d > 0 ∧ n % d = 0 }

/-- The list of all positive divisors of n in ascending order -/
def divisors (n : ℕ) : List (Divisor n) := sorry

/-- The sum of products of consecutive divisors -/
def D (n : ℕ) : ℕ :=
  let ds := divisors n
  (List.zip ds (List.tail ds)).map (fun (d₁, d₂) => d₁.val * d₂.val) |>.sum

/-- The main theorem -/
theorem divisor_product_sum_theorem (n : ℕ) (h : n > 1) :
  D n < n^2 ∧ (D n ∣ n^2 ↔ Nat.Prime n) := by sorry

end NUMINAMATH_CALUDE_divisor_product_sum_theorem_l3886_388649


namespace NUMINAMATH_CALUDE_largest_n_is_correct_l3886_388624

/-- The largest positive integer n for which the system of equations has integer solutions -/
def largest_n : ℕ := 3

/-- Predicate to check if a given n has integer solutions for the system of equations -/
def has_integer_solution (n : ℕ) : Prop :=
  ∃ x : ℤ, ∃ y : Fin n → ℤ,
    ∀ i j : Fin n, (x + i.val + 1)^2 + y i^2 = (x + j.val + 1)^2 + y j^2

/-- Theorem stating that largest_n is indeed the largest n with integer solutions -/
theorem largest_n_is_correct :
  (has_integer_solution largest_n) ∧
  (∀ m : ℕ, m > largest_n → ¬(has_integer_solution m)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_is_correct_l3886_388624


namespace NUMINAMATH_CALUDE_div_power_equals_power_diff_l3886_388621

theorem div_power_equals_power_diff (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_div_power_equals_power_diff_l3886_388621


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l3886_388655

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a3 : a 3 = 2)
  (h_d : ∃ d : ℚ, d = -1/2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l3886_388655


namespace NUMINAMATH_CALUDE_fraction_equality_l3886_388623

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 10 * b) / (b + 10 * a) = 2) : 
  a / b = 0.8 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3886_388623


namespace NUMINAMATH_CALUDE_wire_around_square_field_l3886_388612

/-- Proves that a wire of length 15840 m goes around a square field of area 69696 m^2 exactly 15 times -/
theorem wire_around_square_field (field_area : ℝ) (wire_length : ℝ) : 
  field_area = 69696 → wire_length = 15840 → 
  (wire_length / (4 * Real.sqrt field_area) : ℝ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_wire_around_square_field_l3886_388612


namespace NUMINAMATH_CALUDE_eliminate_denominators_l3886_388615

theorem eliminate_denominators (x : ℝ) :
  (2*x - 1) / 3 - (3*x - 4) / 4 = 1 ↔ 4*(2*x - 1) - 3*(3*x - 4) = 12 :=
by sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l3886_388615


namespace NUMINAMATH_CALUDE_bracelet_count_l3886_388605

/-- Calculates the number of sets that can be made from a given number of beads -/
def sets_from_beads (beads : ℕ) : ℕ := beads / 2

/-- Represents the number of beads Nancy and Rose have -/
structure BeadCounts where
  metal : ℕ
  pearl : ℕ
  crystal : ℕ
  stone : ℕ

/-- Calculates the maximum number of bracelets that can be made -/
def max_bracelets (counts : BeadCounts) : ℕ :=
  min (min (sets_from_beads counts.metal) (sets_from_beads counts.pearl))
      (min (sets_from_beads counts.crystal) (sets_from_beads counts.stone))

theorem bracelet_count (counts : BeadCounts)
  (h1 : counts.metal = 40)
  (h2 : counts.pearl = 60)
  (h3 : counts.crystal = 20)
  (h4 : counts.stone = 40) :
  max_bracelets counts = 10 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_count_l3886_388605


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l3886_388692

theorem opposite_of_negative_two : -(-(2 : ℤ)) = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l3886_388692


namespace NUMINAMATH_CALUDE_class_size_l3886_388697

theorem class_size (poor_vision_percentage : ℝ) (glasses_percentage : ℝ) (glasses_count : ℕ) :
  poor_vision_percentage = 0.4 →
  glasses_percentage = 0.7 →
  glasses_count = 21 →
  ∃ total_students : ℕ, 
    (poor_vision_percentage * glasses_percentage * total_students : ℝ) = glasses_count ∧
    total_students = 75 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l3886_388697


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3886_388651

theorem condition_neither_sufficient_nor_necessary 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (ha₁ : a₁ ≠ 0) (hb₁ : b₁ ≠ 0) (hc₁ : c₁ ≠ 0)
  (ha₂ : a₂ ≠ 0) (hb₂ : b₂ ≠ 0) (hc₂ : c₂ ≠ 0)
  (M : Set ℝ) (hM : M = {x : ℝ | a₁ * x^2 + b₁ * x + c₁ > 0})
  (N : Set ℝ) (hN : N = {x : ℝ | a₂ * x^2 + b₂ * x + c₂ > 0}) :
  ¬(((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)) → (M = N)) ∧ 
  ¬((M = N) → ((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂))) :=
sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3886_388651


namespace NUMINAMATH_CALUDE_prob_equals_two_thirteenths_l3886_388673

-- Define the deck
def total_cards : ℕ := 52
def num_queens : ℕ := 4
def num_jacks : ℕ := 4

-- Define the event
def prob_two_jacks_or_at_least_one_queen : ℚ :=
  (num_jacks * (num_jacks - 1)) / (total_cards * (total_cards - 1)) +
  (num_queens * (total_cards - num_queens)) / (total_cards * (total_cards - 1)) +
  (num_queens * (num_queens - 1)) / (total_cards * (total_cards - 1))

-- State the theorem
theorem prob_equals_two_thirteenths :
  prob_two_jacks_or_at_least_one_queen = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_equals_two_thirteenths_l3886_388673


namespace NUMINAMATH_CALUDE_binary_101_to_decimal_l3886_388617

def binary_to_decimal (b₂ : ℕ) (b₁ : ℕ) (b₀ : ℕ) : ℕ :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_101_to_decimal :
  binary_to_decimal 1 0 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_to_decimal_l3886_388617


namespace NUMINAMATH_CALUDE_birdseed_mix_proportion_l3886_388625

/-- Proves that the proportion of Brand A in a birdseed mix is 60% when the mix is 50% sunflower -/
theorem birdseed_mix_proportion :
  ∀ (x : ℝ), 
  x ≥ 0 ∧ x ≤ 1 →  -- x represents the proportion of Brand A in the mix
  0.60 * x + 0.35 * (1 - x) = 0.50 →  -- The mix is 50% sunflower
  x = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_birdseed_mix_proportion_l3886_388625


namespace NUMINAMATH_CALUDE_x_intercepts_count_l3886_388698

theorem x_intercepts_count : 
  (⌊(100000 : ℝ) / Real.pi⌋ : ℤ) - (⌊(10000 : ℝ) / Real.pi⌋ : ℤ) = 28648 := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l3886_388698


namespace NUMINAMATH_CALUDE_inequality_implies_a_value_l3886_388646

theorem inequality_implies_a_value (a : ℝ) 
  (h : ∀ x : ℝ, x > 0 → (x^2 + a*x - 5)*(a*x - 1) ≥ 0) : 
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_value_l3886_388646


namespace NUMINAMATH_CALUDE_janice_overtime_shifts_l3886_388640

/-- Proves that Janice worked 3 overtime shifts given her work schedule and earnings --/
theorem janice_overtime_shifts :
  let regular_days : ℕ := 5
  let regular_daily_pay : ℕ := 30
  let overtime_pay : ℕ := 15
  let total_earnings : ℕ := 195
  let regular_earnings := regular_days * regular_daily_pay
  let overtime_earnings := total_earnings - regular_earnings
  overtime_earnings / overtime_pay = 3 := by sorry

end NUMINAMATH_CALUDE_janice_overtime_shifts_l3886_388640


namespace NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l3886_388656

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 6
  let small_hole_diameter : ℝ := 1.5
  let large_hole_diameter : ℝ := 2.5
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2)^3
  let small_hole_volume := π * (small_hole_diameter / 2)^2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter / 2)^2 * hole_depth
  sphere_volume - 2 * small_hole_volume - large_hole_volume = 2287.875 * π :=
by sorry

end NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l3886_388656


namespace NUMINAMATH_CALUDE_vehicle_distance_theorem_l3886_388629

/-- Represents a vehicle with two wheels of different perimeters -/
structure Vehicle where
  back_wheel_perimeter : ℝ
  front_wheel_perimeter : ℝ

/-- Calculates the distance traveled by the vehicle given the number of revolutions of the back wheel -/
def distance_traveled (v : Vehicle) (back_wheel_revolutions : ℝ) : ℝ :=
  back_wheel_revolutions * v.back_wheel_perimeter

/-- Theorem stating that the vehicle travels 315 feet under the given conditions -/
theorem vehicle_distance_theorem (v : Vehicle) 
    (h1 : v.back_wheel_perimeter = 9)
    (h2 : v.front_wheel_perimeter = 7)
    (h3 : ∃ (r : ℝ), r * v.back_wheel_perimeter = (r + 10) * v.front_wheel_perimeter) :
    ∃ (r : ℝ), distance_traveled v r = 315 := by
  sorry


end NUMINAMATH_CALUDE_vehicle_distance_theorem_l3886_388629


namespace NUMINAMATH_CALUDE_correct_tile_count_l3886_388682

/-- The dimensions of the room --/
def room_width : ℝ := 8
def room_height : ℝ := 12

/-- The dimensions of a tile --/
def tile_width : ℝ := 1.5
def tile_height : ℝ := 2

/-- The number of tiles needed to cover the room --/
def tiles_needed : ℕ := 32

/-- Theorem stating that the number of tiles needed is correct --/
theorem correct_tile_count : 
  (room_width * room_height) / (tile_width * tile_height) = tiles_needed := by
  sorry

end NUMINAMATH_CALUDE_correct_tile_count_l3886_388682


namespace NUMINAMATH_CALUDE_inequality_proof_l3886_388610

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ 1) (h3 : 1 ≥ c) (h4 : c ≥ 0) 
  (h5 : a + b + c = 3) : 
  2 ≤ a * b + b * c + c * a ∧ 
  a * b + b * c + c * a ≤ 3 ∧
  24 / (a^3 + b^3 + c^3) + 25 / (a * b + b * c + c * a) ≥ 14 := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l3886_388610


namespace NUMINAMATH_CALUDE_binomial_factorial_product_l3886_388686

theorem binomial_factorial_product : Nat.choose 10 4 * Nat.factorial 6 = 151200 := by sorry

end NUMINAMATH_CALUDE_binomial_factorial_product_l3886_388686


namespace NUMINAMATH_CALUDE_room_width_calculation_l3886_388636

/-- Given a rectangular room with length 12 feet and width w feet, 
    with a carpet placed leaving a 2-foot wide border all around, 
    if the area of the border is 72 square feet, 
    then the width of the room is 10 feet. -/
theorem room_width_calculation (w : ℝ) : 
  w > 0 →  -- width is positive
  12 * w - 8 * (w - 4) = 72 →  -- area of border is 72 sq ft
  w = 10 := by
sorry

end NUMINAMATH_CALUDE_room_width_calculation_l3886_388636


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3886_388632

theorem quadratic_equation_solution (t s : ℝ) : t = 15 * s^2 + 5 → t = 20 → s = 1 ∨ s = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3886_388632


namespace NUMINAMATH_CALUDE_base4_77_last_digit_l3886_388661

def base4LastDigit (n : Nat) : Nat :=
  n % 4

theorem base4_77_last_digit :
  base4LastDigit 77 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base4_77_last_digit_l3886_388661


namespace NUMINAMATH_CALUDE_dot_product_condition_l3886_388622

/-- Given vectors a and b, if a · (2a - b) = 0, then k = 12 -/
theorem dot_product_condition (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (2, 1))
  (h2 : b = (-1, k))
  (h3 : a • (2 • a - b) = 0) :
  k = 12 := by sorry

end NUMINAMATH_CALUDE_dot_product_condition_l3886_388622


namespace NUMINAMATH_CALUDE_children_not_enrolled_l3886_388689

theorem children_not_enrolled (total children_basketball children_robotics children_both : ℕ) 
  (h_total : total = 150)
  (h_basketball : children_basketball = 85)
  (h_robotics : children_robotics = 58)
  (h_both : children_both = 18) :
  total - (children_basketball + children_robotics - children_both) = 25 := by
  sorry

end NUMINAMATH_CALUDE_children_not_enrolled_l3886_388689


namespace NUMINAMATH_CALUDE_container_problem_l3886_388620

theorem container_problem :
  ∃! (x y : ℕ), 130 * x + 160 * y = 3000 ∧ x = 12 ∧ y = 9 :=
by sorry

end NUMINAMATH_CALUDE_container_problem_l3886_388620


namespace NUMINAMATH_CALUDE_modular_arithmetic_equivalence_l3886_388611

theorem modular_arithmetic_equivalence : 144 * 20 - 17^2 + 5 ≡ 4 [ZMOD 16] := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_equivalence_l3886_388611


namespace NUMINAMATH_CALUDE_fraction_equality_l3886_388676

theorem fraction_equality (q r s t v : ℝ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : v / t = 4)
  (h4 : s / v = 1 / 3) :
  t / q = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3886_388676


namespace NUMINAMATH_CALUDE_inscribed_circle_tangent_sum_l3886_388671

/-- A point on the inscribed circle of a square -/
structure InscribedCirclePoint (α β : ℝ) where
  -- P is on the inscribed circle of square ABCD
  on_inscribed_circle : True
  -- Angle APC = α
  angle_apc : True
  -- Angle BPD = β
  angle_bpd : True

/-- The sum of squared tangents of angles α and β is 8 -/
theorem inscribed_circle_tangent_sum (α β : ℝ) (p : InscribedCirclePoint α β) : 
  Real.tan α ^ 2 + Real.tan β ^ 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_tangent_sum_l3886_388671


namespace NUMINAMATH_CALUDE_y_percent_of_y_is_9_l3886_388637

theorem y_percent_of_y_is_9 (y : ℝ) (h1 : y > 0) (h2 : y / 100 * y = 9) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_y_percent_of_y_is_9_l3886_388637


namespace NUMINAMATH_CALUDE_square_of_gcd_product_l3886_388606

theorem square_of_gcd_product (x y z : ℕ) (h : x > 0 ∧ y > 0 ∧ z > 0) 
  (eq : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) : 
  ∃ (k : ℕ), Nat.gcd x (Nat.gcd y z) * x * y * z = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_square_of_gcd_product_l3886_388606


namespace NUMINAMATH_CALUDE_two_forty_is_eighty_supplement_of_half_forty_is_onesixtey_half_forty_is_twenty_l3886_388669

-- Define the given angle
def given_angle : ℝ := 40

-- Theorem 1: Two 40° angles form an 80° angle
theorem two_forty_is_eighty : given_angle + given_angle = 80 := by sorry

-- Theorem 2: The supplement of half of a 40° angle is 160°
theorem supplement_of_half_forty_is_onesixtey : 180 - (given_angle / 2) = 160 := by sorry

-- Theorem 3: Half of a 40° angle is 20°
theorem half_forty_is_twenty : given_angle / 2 = 20 := by sorry

end NUMINAMATH_CALUDE_two_forty_is_eighty_supplement_of_half_forty_is_onesixtey_half_forty_is_twenty_l3886_388669


namespace NUMINAMATH_CALUDE_mia_money_l3886_388616

/-- Given that Darwin has $45 and Mia has $20 more than twice as much money as Darwin,
    prove that Mia has $110. -/
theorem mia_money (darwin_money : ℕ) (mia_money : ℕ) : 
  darwin_money = 45 → 
  mia_money = 2 * darwin_money + 20 → 
  mia_money = 110 := by
sorry

end NUMINAMATH_CALUDE_mia_money_l3886_388616


namespace NUMINAMATH_CALUDE_larger_number_proof_l3886_388658

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1355)
  (h2 : L = 6 * S + 15) : 
  L = 1623 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3886_388658


namespace NUMINAMATH_CALUDE_painter_problem_l3886_388653

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculates the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Proves that for the given scenario, the time to paint the remaining rooms is 49 hours. -/
theorem painter_problem :
  let total_rooms : ℕ := 12
  let time_per_room : ℕ := 7
  let painted_rooms : ℕ := 5
  time_to_paint_remaining total_rooms time_per_room painted_rooms = 49 := by
  sorry


end NUMINAMATH_CALUDE_painter_problem_l3886_388653


namespace NUMINAMATH_CALUDE_common_sum_is_negative_fifteen_l3886_388609

def is_valid_arrangement (arr : Matrix (Fin 5) (Fin 5) ℤ) : Prop :=
  ∀ i j, -15 ≤ arr i j ∧ arr i j ≤ 9

def row_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) (i : Fin 5) : ℤ :=
  (Finset.range 5).sum (λ j => arr i j)

def col_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) (j : Fin 5) : ℤ :=
  (Finset.range 5).sum (λ i => arr i j)

def main_diagonal_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) : ℤ :=
  (Finset.range 5).sum (λ i => arr i i)

def anti_diagonal_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) : ℤ :=
  (Finset.range 5).sum (λ i => arr i (4 - i))

def all_sums_equal (arr : Matrix (Fin 5) (Fin 5) ℤ) : Prop :=
  ∃ s, (∀ i, row_sum arr i = s) ∧
       (∀ j, col_sum arr j = s) ∧
       (main_diagonal_sum arr = s) ∧
       (anti_diagonal_sum arr = s)

theorem common_sum_is_negative_fifteen
  (arr : Matrix (Fin 5) (Fin 5) ℤ)
  (h1 : is_valid_arrangement arr)
  (h2 : all_sums_equal arr) :
  ∃ s, s = -15 ∧ all_sums_equal arr ∧ (∀ i j, row_sum arr i = s ∧ col_sum arr j = s) :=
sorry

end NUMINAMATH_CALUDE_common_sum_is_negative_fifteen_l3886_388609


namespace NUMINAMATH_CALUDE_carpenter_woodblocks_needed_l3886_388663

/-- Calculates the total number of woodblocks needed by a carpenter to build a house. -/
theorem carpenter_woodblocks_needed 
  (initial_logs : ℕ) 
  (woodblocks_per_log : ℕ) 
  (additional_logs_needed : ℕ) : 
  (initial_logs + additional_logs_needed) * woodblocks_per_log = 80 :=
by
  sorry

#check carpenter_woodblocks_needed 8 5 8

end NUMINAMATH_CALUDE_carpenter_woodblocks_needed_l3886_388663


namespace NUMINAMATH_CALUDE_sum_of_squares_l3886_388683

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 70) → (a + b + c = 17) → (a^2 + b^2 + c^2 = 149) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3886_388683


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3886_388602

theorem fraction_decomposition (n : ℕ) 
  (h1 : ∀ n, 1 / (n * (n + 1)) = 1 / n - 1 / (n + 1))
  (h2 : ∀ n, 1 / (n * (n + 1) * (n + 2)) = 1 / (2 * n * (n + 1)) - 1 / (2 * (n + 1) * (n + 2))) :
  1 / (n * (n + 1) * (n + 2) * (n + 3)) = 
    1 / (3 * n * (n + 1) * (n + 2)) - 1 / (3 * (n + 1) * (n + 2) * (n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3886_388602


namespace NUMINAMATH_CALUDE_smallest_marked_cells_for_unique_tiling_l3886_388688

/-- Represents a grid of size 2n × 2n -/
structure Grid (n : ℕ) where
  size : ℕ := 2 * n

/-- Represents a set of marked cells in the grid -/
def MarkedCells (n : ℕ) := Finset (Fin (2 * n) × Fin (2 * n))

/-- Represents a domino tiling of the grid -/
def Tiling (n : ℕ) := Finset (Fin (2 * n) × Fin (2 * n) × Bool)

/-- Checks if a tiling is valid for a given set of marked cells -/
def isValidTiling (n : ℕ) (marked : MarkedCells n) (tiling : Tiling n) : Prop :=
  sorry

/-- Checks if there exists a unique valid tiling for a given set of marked cells -/
def hasUniqueTiling (n : ℕ) (marked : MarkedCells n) : Prop :=
  sorry

/-- The main theorem: The smallest number of marked cells that ensures a unique tiling is 2n -/
theorem smallest_marked_cells_for_unique_tiling (n : ℕ) (h : 0 < n) :
  ∃ (marked : MarkedCells n),
    marked.card = 2 * n ∧
    hasUniqueTiling n marked ∧
    ∀ (marked' : MarkedCells n),
      marked'.card < 2 * n → ¬(hasUniqueTiling n marked') :=
  sorry

end NUMINAMATH_CALUDE_smallest_marked_cells_for_unique_tiling_l3886_388688


namespace NUMINAMATH_CALUDE_vector_dot_product_properties_l3886_388607

/-- Given two vectors in ℝ², prove dot product properties --/
theorem vector_dot_product_properties (a b : ℝ × ℝ) 
    (h1 : a = (1, 2)) 
    (h2 : b = (2, -3)) : 
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 9) ∧ 
  ((a.1 + b.1) * (a.1 - (1/9) * b.1) + (a.2 + b.2) * (a.2 - (1/9) * b.2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_properties_l3886_388607


namespace NUMINAMATH_CALUDE_slope_of_solutions_l3886_388641

theorem slope_of_solutions (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : (5 / x₁) + (4 / y₁) = 0) (h₃ : (5 / x₂) + (4 / y₂) = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_solutions_l3886_388641


namespace NUMINAMATH_CALUDE_als_original_portion_l3886_388613

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  a - 150 + 3 * b + 3 * c = 1800 →
  c = 2 * b →
  a = 825 :=
by sorry

end NUMINAMATH_CALUDE_als_original_portion_l3886_388613


namespace NUMINAMATH_CALUDE_percentage_women_red_hair_men_dark_hair_l3886_388694

theorem percentage_women_red_hair_men_dark_hair (
  women_fair_hair : Real) (women_dark_hair : Real) (women_red_hair : Real)
  (men_fair_hair : Real) (men_dark_hair : Real) (men_red_hair : Real)
  (h1 : women_fair_hair = 30)
  (h2 : women_dark_hair = 28)
  (h3 : women_red_hair = 12)
  (h4 : men_fair_hair = 20)
  (h5 : men_dark_hair = 35)
  (h6 : men_red_hair = 5)
  : women_red_hair + men_dark_hair = 47 := by
  sorry

end NUMINAMATH_CALUDE_percentage_women_red_hair_men_dark_hair_l3886_388694


namespace NUMINAMATH_CALUDE_scale_division_l3886_388674

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 125

/-- Represents the length of each part in inches -/
def part_length : ℕ := 25

/-- Theorem stating that the scale is divided into 5 equal parts -/
theorem scale_division :
  scale_length / part_length = 5 := by sorry

end NUMINAMATH_CALUDE_scale_division_l3886_388674


namespace NUMINAMATH_CALUDE_music_shop_total_cost_l3886_388670

/-- Calculates the total cost of CDs purchased from a music shop --/
theorem music_shop_total_cost 
  (life_journey_price : ℝ) 
  (life_journey_discount : ℝ) 
  (day_life_price : ℝ) 
  (rescind_price : ℝ) 
  (life_journey_quantity : ℕ) 
  (day_life_quantity : ℕ) 
  (rescind_quantity : ℕ) : 
  life_journey_price = 100 →
  life_journey_discount = 0.2 →
  day_life_price = 50 →
  rescind_price = 85 →
  life_journey_quantity = 3 →
  day_life_quantity = 4 →
  rescind_quantity = 2 →
  (life_journey_quantity * (life_journey_price * (1 - life_journey_discount))) +
  ((day_life_quantity / 2) * day_life_price) +
  (rescind_quantity * rescind_price) = 510 := by
sorry

end NUMINAMATH_CALUDE_music_shop_total_cost_l3886_388670


namespace NUMINAMATH_CALUDE_valid_paths_count_l3886_388678

/-- Represents a point on a 2D grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points on a grid --/
def numPaths (start finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- Calculates the number of paths between two points passing through an intermediate point --/
def numPathsThrough (start mid finish : Point) : ℕ :=
  (numPaths start mid) * (numPaths mid finish)

/-- The main theorem stating the number of valid paths --/
theorem valid_paths_count :
  let start := Point.mk 0 0
  let finish := Point.mk 5 3
  let risky := Point.mk 2 2
  (numPaths start finish) - (numPathsThrough start risky finish) = 32 := by
  sorry

end NUMINAMATH_CALUDE_valid_paths_count_l3886_388678


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l3886_388659

/-- The set T of points (x,y) in the coordinate plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b : ℝ), 
    ((a = 7 ∧ b = p.1 - 3) ∨ 
     (a = 7 ∧ b = p.2 + 5) ∨ 
     (a = p.1 - 3 ∧ b = p.2 + 5)) ∧
    (a = b) ∧
    (7 ≥ a ∧ p.1 - 3 ≥ a ∧ p.2 + 5 ≥ a)}

/-- A ray in the plane, defined by its starting point and direction -/
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- The property that T consists of three rays with a common point -/
def isThreeRaysWithCommonPoint (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ) (r₁ r₂ r₃ : Ray),
    r₁.start = p ∧ r₂.start = p ∧ r₃.start = p ∧
    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    s = {q : ℝ × ℝ | ∃ (t : ℝ), t ≥ 0 ∧
      (q = r₁.start + t • r₁.direction ∨
       q = r₂.start + t • r₂.direction ∨
       q = r₃.start + t • r₃.direction)}

theorem T_is_three_rays_with_common_point : isThreeRaysWithCommonPoint T := by
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l3886_388659


namespace NUMINAMATH_CALUDE_quadratic_root_scaling_l3886_388675

theorem quadratic_root_scaling (a b c n : ℝ) (h : a ≠ 0) :
  let original_eq := fun x : ℝ => a * x^2 + b * x + c
  let scaled_eq := fun x : ℝ => a * x^2 + n * b * x + n^2 * c
  let roots := { x : ℝ | original_eq x = 0 }
  let scaled_roots := { x : ℝ | ∃ y ∈ roots, x = n * y }
  scaled_roots = { x : ℝ | scaled_eq x = 0 } :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_scaling_l3886_388675


namespace NUMINAMATH_CALUDE_gym_time_zero_l3886_388619

/-- Represents the exercise plan with yoga and exercise components -/
structure ExercisePlan where
  yoga_time : ℕ
  exercise_time : ℕ
  bike_time : ℕ
  gym_time : ℕ
  yoga_exercise_ratio : yoga_time * 3 = exercise_time * 2
  exercise_components : exercise_time = bike_time + gym_time

/-- 
Given an exercise plan where the bike riding time equals the total exercise time,
prove that the gym workout time is zero.
-/
theorem gym_time_zero (plan : ExercisePlan) 
  (h : plan.bike_time = plan.exercise_time) : plan.gym_time = 0 := by
  sorry

end NUMINAMATH_CALUDE_gym_time_zero_l3886_388619


namespace NUMINAMATH_CALUDE_object_distance_in_one_hour_l3886_388693

/-- Proves that an object traveling at 3 feet per second will cover 10800 feet in one hour. -/
theorem object_distance_in_one_hour 
  (speed : ℝ) 
  (seconds_per_hour : ℕ) 
  (h1 : speed = 3) 
  (h2 : seconds_per_hour = 3600) : 
  speed * seconds_per_hour = 10800 := by
  sorry

end NUMINAMATH_CALUDE_object_distance_in_one_hour_l3886_388693


namespace NUMINAMATH_CALUDE_school_population_theorem_l3886_388631

theorem school_population_theorem (b g t : ℕ) : 
  b = 4 * g → g = 10 * t → b + g + t = 51 * t := by sorry

end NUMINAMATH_CALUDE_school_population_theorem_l3886_388631


namespace NUMINAMATH_CALUDE_rearrangement_time_proof_l3886_388654

/-- The number of hours required to write all rearrangements of a 12-letter name -/
def rearrangement_hours : ℕ := 798336

/-- The number of letters in the name -/
def name_length : ℕ := 12

/-- The number of arrangements written per minute -/
def arrangements_per_minute : ℕ := 10

/-- Theorem stating the time required to write all rearrangements -/
theorem rearrangement_time_proof :
  rearrangement_hours = (name_length.factorial / arrangements_per_minute) / 60 := by
  sorry


end NUMINAMATH_CALUDE_rearrangement_time_proof_l3886_388654


namespace NUMINAMATH_CALUDE_modular_inverse_35_mod_37_l3886_388634

theorem modular_inverse_35_mod_37 : ∃ x : ℕ, x ≤ 36 ∧ (35 * x) % 37 = 1 :=
by
  use 18
  sorry

end NUMINAMATH_CALUDE_modular_inverse_35_mod_37_l3886_388634


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3886_388664

theorem root_sum_theorem (x : ℝ) (a b c d : ℝ) : 
  (1/x + 1/(x+4) - 1/(x+6) - 1/(x+10) + 1/(x+12) + 1/(x+16) - 1/(x+18) - 1/(x+20) = 0) →
  (∃ (sign1 sign2 : Bool), x = -a + (-1)^(sign1.toNat : ℕ) * Real.sqrt (b + (-1)^(sign2.toNat : ℕ) * c * Real.sqrt d)) →
  a + b + c + d = 27 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3886_388664


namespace NUMINAMATH_CALUDE_coronavirus_spread_rate_l3886_388695

/-- The number of people infected after two rounds of novel coronavirus spread -/
def total_infected : ℕ := 121

/-- The number of people initially infected -/
def initial_infected : ℕ := 1

/-- The average number of people infected by one person in each round -/
def m : ℕ := 10

/-- Theorem stating that m = 10 given the conditions of the coronavirus spread -/
theorem coronavirus_spread_rate :
  (initial_infected + m)^2 = total_infected :=
sorry

end NUMINAMATH_CALUDE_coronavirus_spread_rate_l3886_388695


namespace NUMINAMATH_CALUDE_distinct_pairs_count_l3886_388657

/-- Represents the colors of marbles --/
inductive Color
  | Red
  | Green
  | Blue
  | Yellow

/-- Represents a marble with a color and quantity --/
structure Marble where
  color : Color
  quantity : Nat

/-- Calculates the number of distinct pairs of marbles that can be chosen --/
def countDistinctPairs (marbles : List Marble) : Nat :=
  sorry

/-- Theorem: Given the specific set of marbles, the number of distinct pairs is 7 --/
theorem distinct_pairs_count :
  let marbles : List Marble := [
    ⟨Color.Red, 1⟩,
    ⟨Color.Green, 1⟩,
    ⟨Color.Blue, 2⟩,
    ⟨Color.Yellow, 2⟩
  ]
  countDistinctPairs marbles = 7 := by
  sorry

end NUMINAMATH_CALUDE_distinct_pairs_count_l3886_388657


namespace NUMINAMATH_CALUDE_value_of_n_l3886_388667

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters on the left side of the equation -/
def left_quarters : ℕ := 15

/-- The number of nickels on the left side of the equation -/
def left_nickels : ℕ := 18

/-- The number of quarters on the right side of the equation -/
def right_quarters : ℕ := 7

/-- Theorem stating that the value of n is 58 -/
theorem value_of_n : 
  ∃ n : ℕ, 
    left_quarters * quarter_value + left_nickels * nickel_value = 
    right_quarters * quarter_value + n * nickel_value ∧ 
    n = 58 := by
  sorry

end NUMINAMATH_CALUDE_value_of_n_l3886_388667


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3886_388635

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + c = 60 →
  a + b + c = 70 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3886_388635


namespace NUMINAMATH_CALUDE_line_not_through_point_l3886_388648

theorem line_not_through_point (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (2*m+1)*x₁ + m^2 + 4 = 0 ∧ x₂^2 + (2*m+1)*x₂ + m^2 + 4 = 0) →
  ¬((2*m-3)*(-2) - 4*m + 7 = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_not_through_point_l3886_388648


namespace NUMINAMATH_CALUDE_odd_function_composition_even_l3886_388601

-- Define an odd function
def OddFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem odd_function_composition_even
  (g : ℝ → ℝ)
  (h : OddFunction g) :
  EvenFunction (fun x ↦ g (g (g (g x)))) :=
sorry

end NUMINAMATH_CALUDE_odd_function_composition_even_l3886_388601


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l3886_388633

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l3886_388633


namespace NUMINAMATH_CALUDE_brush_square_ratio_l3886_388630

theorem brush_square_ratio (s w : ℝ) (h_pos_s : 0 < s) (h_pos_w : 0 < w) :
  w^2 + ((s - w)^2) / 2 = (1/3) * s^2 →
  s / w = 2 * Real.sqrt 3 - 2 := by
sorry

end NUMINAMATH_CALUDE_brush_square_ratio_l3886_388630


namespace NUMINAMATH_CALUDE_attention_index_properties_l3886_388672

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 10 then 100 * a^(x/10) - 60
  else if 10 < x ∧ x ≤ 20 then 340
  else if 20 < x ∧ x ≤ 40 then 640 - 15*x
  else 0

theorem attention_index_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 5 = 140) :
  a = 4 ∧ 
  f a 5 > f a 35 ∧ 
  (Set.Icc 5 (100/3) : Set ℝ) = {x | 0 ≤ x ∧ x ≤ 40 ∧ f a x ≥ 140} :=
by sorry

end NUMINAMATH_CALUDE_attention_index_properties_l3886_388672


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3886_388643

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the right vertex A
def right_vertex (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the circle centered at A
def circle_at_A (a r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = r^2

-- Define the angle PAQ
def angle_PAQ (p q : ℝ × ℝ) : ℝ := sorry

-- Define the distance PQ
def distance_PQ (p q : ℝ × ℝ) : ℝ := sorry

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x

theorem hyperbola_asymptote
  (a b : ℝ)
  (p q : ℝ × ℝ)
  (h1 : hyperbola a b p.1 p.2)
  (h2 : hyperbola a b q.1 q.2)
  (h3 : ∃ r, circle_at_A a r p.1 p.2 ∧ circle_at_A a r q.1 q.2)
  (h4 : angle_PAQ p q = Real.pi / 3)
  (h5 : distance_PQ p q = Real.sqrt 3 / 3 * a) :
  asymptote_equation p.1 p.2 ∧ asymptote_equation q.1 q.2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3886_388643


namespace NUMINAMATH_CALUDE_tangent_parallel_to_given_line_l3886_388642

-- Define the curve
def f (x : ℝ) := x^4

-- Define the derivative of the curve
def f' (x : ℝ) := 4 * x^3

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the given line
def givenLine (x y : ℝ) : Prop := 4 * x - y + 1 = 0

-- Define parallel lines
def parallel (m₁ b₁ m₂ b₂ : ℝ) : Prop := m₁ = m₂ ∧ b₁ ≠ b₂

theorem tangent_parallel_to_given_line :
  let m := f' P.1  -- Slope of tangent line
  let b := P.2 - m * P.1  -- y-intercept of tangent line
  parallel m b 4 (-1) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_given_line_l3886_388642


namespace NUMINAMATH_CALUDE_circles_intersect_l3886_388645

theorem circles_intersect : ∃ (x y : ℝ),
  (x^2 + y^2 - 8*x + 6*y - 11 = 0) ∧ (x^2 + y^2 = 100) := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3886_388645


namespace NUMINAMATH_CALUDE_sofa_price_calculation_l3886_388680

def living_room_set_price (sofa_price armchair_price coffee_table_price : ℝ) : ℝ :=
  sofa_price + 2 * armchair_price + coffee_table_price

theorem sofa_price_calculation (armchair_price coffee_table_price total_price : ℝ)
  (h1 : armchair_price = 425)
  (h2 : coffee_table_price = 330)
  (h3 : total_price = 2430)
  (h4 : living_room_set_price (total_price - 2 * armchair_price - coffee_table_price) armchair_price coffee_table_price = total_price) :
  total_price - 2 * armchair_price - coffee_table_price = 1250 := by
  sorry

#check sofa_price_calculation

end NUMINAMATH_CALUDE_sofa_price_calculation_l3886_388680


namespace NUMINAMATH_CALUDE_circle_equation_range_l3886_388681

/-- A circle in the xy-plane can be represented by an equation of the form
    (x - h)^2 + (y - k)^2 = r^2, where (h, k) is the center and r is the radius. -/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ h k r, r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The equation x^2 + y^2 + 2kx + 4y + 3k + 8 = 0 represents a circle for some real k -/
def equation (k : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8

/-- The range of k for which the equation represents a circle -/
def k_range (k : ℝ) : Prop :=
  k < -1 ∨ k > 4

theorem circle_equation_range :
  ∀ k, is_circle (equation k) ↔ k_range k :=
sorry

end NUMINAMATH_CALUDE_circle_equation_range_l3886_388681
