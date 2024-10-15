import Mathlib

namespace NUMINAMATH_CALUDE_line_CD_passes_through_fixed_point_l3205_320538

-- Define the Cartesian plane
variable (x y : ℝ)

-- Define points E and F
def E : ℝ × ℝ := (0, 1)
def F : ℝ × ℝ := (0, -1)

-- Define the trajectory of point G
def trajectory (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define line l
def line_l (x : ℝ) : Prop := x = 4

-- Define point P on line l
def P (y₀ : ℝ) : ℝ × ℝ := (4, y₀)

-- Define points A and B (vertices of the trajectory)
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem line_CD_passes_through_fixed_point (x y y₀ : ℝ) 
  (h1 : trajectory x y) 
  (h2 : y₀ ≠ 0) :
  ∃ (xc yc xd yd : ℝ), 
    trajectory xc yc ∧ 
    trajectory xd yd ∧ 
    (yc - yd) / (xc - xd) * (1 - xc) + yc = 0 := 
sorry


end NUMINAMATH_CALUDE_line_CD_passes_through_fixed_point_l3205_320538


namespace NUMINAMATH_CALUDE_price_reduction_equation_l3205_320516

theorem price_reduction_equation (x : ℝ) : 
  (∃ (original_price final_price : ℝ),
    original_price = 28 ∧ 
    final_price = 16 ∧ 
    final_price = original_price * (1 - x)^2) →
  28 * (1 - x)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l3205_320516


namespace NUMINAMATH_CALUDE_power_multiplication_l3205_320581

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3205_320581


namespace NUMINAMATH_CALUDE_complex_sum_zero_l3205_320505

theorem complex_sum_zero : 
  let z : ℂ := (1 / 2 : ℂ) + (Complex.I * Real.sqrt 3 / 2)
  z + z^2 + z^3 + z^4 + z^5 + z^6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l3205_320505


namespace NUMINAMATH_CALUDE_inequality_proof_l3205_320580

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / ((a * b * c) ^ (1/3) * (1 + (a * b * c) ^ (1/3))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3205_320580


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l3205_320551

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 2020 * a + 2024 * b = 2025)
  (eq2 : 2022 * a + 2026 * b = 2030) : 
  a - b = 1515 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l3205_320551


namespace NUMINAMATH_CALUDE_inequality_solution_l3205_320596

theorem inequality_solution (x : ℝ) : 
  (x / 4 ≤ 3 + x ∧ 3 + x ≤ -3 * (1 + x)) ↔ -4 ≤ x ∧ x ≤ -3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3205_320596


namespace NUMINAMATH_CALUDE_fraction_power_product_l3205_320586

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by sorry

end NUMINAMATH_CALUDE_fraction_power_product_l3205_320586


namespace NUMINAMATH_CALUDE_square_sum_from_means_l3205_320568

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20) 
  (h_geometric : Real.sqrt (a * b) = 16) : 
  a^2 + b^2 = 1088 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l3205_320568


namespace NUMINAMATH_CALUDE_distributive_law_l3205_320515

theorem distributive_law (a b c : ℝ) : (a + b) * c = a * c + b * c := by
  sorry

end NUMINAMATH_CALUDE_distributive_law_l3205_320515


namespace NUMINAMATH_CALUDE_shortest_path_on_specific_frustum_l3205_320542

/-- Represents a truncated circular right cone (frustum) -/
structure Frustum where
  lower_circumference : ℝ
  upper_circumference : ℝ
  inclination_angle : ℝ

/-- Calculates the shortest path on the surface of a frustum -/
def shortest_path (f : Frustum) (upper_travel : ℝ) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem shortest_path_on_specific_frustum :
  let f : Frustum := {
    lower_circumference := 10,
    upper_circumference := 9,
    inclination_angle := Real.pi / 3  -- 60 degrees in radians
  }
  shortest_path f 3 = 5 * Real.sqrt 3 / Real.pi :=
sorry

end NUMINAMATH_CALUDE_shortest_path_on_specific_frustum_l3205_320542


namespace NUMINAMATH_CALUDE_infinitely_many_twin_pretty_numbers_l3205_320544

-- Define what it means for a number to be "pretty"
def isPrettyNumber (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (∃ k : ℕ, k ≥ 2 ∧ p^k ∣ n)

-- Define a pair of twin pretty numbers
def isTwinPrettyPair (n m : ℕ) : Prop :=
  isPrettyNumber n ∧ isPrettyNumber m ∧ m = n + 1

-- Theorem statement
theorem infinitely_many_twin_pretty_numbers :
  ∀ k : ℕ, ∃ n m : ℕ, n > k ∧ isTwinPrettyPair n m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_twin_pretty_numbers_l3205_320544


namespace NUMINAMATH_CALUDE_expression_equals_three_l3205_320547

theorem expression_equals_three :
  (-1)^2023 + Real.sqrt 9 - π^0 + Real.sqrt (1/8) * Real.sqrt 32 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l3205_320547


namespace NUMINAMATH_CALUDE_total_supervisors_count_l3205_320592

/-- The number of buses -/
def num_buses : ℕ := 7

/-- The number of supervisors per bus -/
def supervisors_per_bus : ℕ := 3

/-- The total number of supervisors -/
def total_supervisors : ℕ := num_buses * supervisors_per_bus

theorem total_supervisors_count : total_supervisors = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_supervisors_count_l3205_320592


namespace NUMINAMATH_CALUDE_rearrangements_without_substring_l3205_320559

def word : String := "HMMTHMMT"

def total_permutations : ℕ := 420

def permutations_with_substring : ℕ := 60

theorem rearrangements_without_substring :
  (total_permutations - permutations_with_substring + 1 : ℕ) = 361 := by sorry

end NUMINAMATH_CALUDE_rearrangements_without_substring_l3205_320559


namespace NUMINAMATH_CALUDE_selling_price_correct_l3205_320536

/-- Calculates the selling price of a television after applying discounts -/
def selling_price (a : ℝ) : ℝ :=
  0.9 * (a - 100)

/-- Theorem stating that the selling price function correctly applies the discounts -/
theorem selling_price_correct (a : ℝ) : 
  selling_price a = 0.9 * (a - 100) := by
  sorry

end NUMINAMATH_CALUDE_selling_price_correct_l3205_320536


namespace NUMINAMATH_CALUDE_mean_temperature_l3205_320512

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 0]

theorem mean_temperature :
  (List.sum temperatures : ℚ) / temperatures.length = -10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3205_320512


namespace NUMINAMATH_CALUDE_prakash_copies_five_pages_l3205_320550

/-- Represents the number of pages a person can copy in a given time -/
structure CopyingRate where
  pages : ℕ
  hours : ℕ

/-- Subash's copying rate -/
def subash_rate : CopyingRate := ⟨50, 10⟩

/-- Combined copying rate of Subash and Prakash -/
def combined_rate : CopyingRate := ⟨300, 40⟩

/-- Calculate the number of pages Prakash can copy in 2 hours -/
def prakash_pages : ℕ :=
  let subash_40_hours := (subash_rate.pages * combined_rate.hours) / subash_rate.hours
  let prakash_40_hours := combined_rate.pages - subash_40_hours
  (prakash_40_hours * 2) / combined_rate.hours

theorem prakash_copies_five_pages : prakash_pages = 5 := by
  sorry

end NUMINAMATH_CALUDE_prakash_copies_five_pages_l3205_320550


namespace NUMINAMATH_CALUDE_function_transformation_l3205_320555

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_transformation (x : ℝ) :
  (∀ y : ℝ, f (y + 1) = 3 * y + 4) →
  f x = 3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l3205_320555


namespace NUMINAMATH_CALUDE_three_digit_five_times_smaller_l3205_320537

/-- A three-digit number -/
def ThreeDigitNumber (a b c : ℕ) : Prop :=
  100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c < 1000

/-- The condition that a number becomes five times smaller when the first digit is removed -/
def FiveTimesSmallerWithoutFirstDigit (a b c : ℕ) : Prop :=
  5 * (b * 10 + c) = a * 100 + b * 10 + c

/-- The theorem stating that 125, 250, and 375 are the only three-digit numbers
    that become five times smaller when the first digit is removed -/
theorem three_digit_five_times_smaller :
  ∀ a b c : ℕ,
  ThreeDigitNumber a b c ∧ FiveTimesSmallerWithoutFirstDigit a b c ↔
  (a = 1 ∧ b = 2 ∧ c = 5) ∨ (a = 2 ∧ b = 5 ∧ c = 0) ∨ (a = 3 ∧ b = 7 ∧ c = 5) :=
by sorry


end NUMINAMATH_CALUDE_three_digit_five_times_smaller_l3205_320537


namespace NUMINAMATH_CALUDE_hours_per_day_l3205_320519

/-- Given that there are 8760 hours in a year and 365 days in a year,
    prove that there are 24 hours in a day. -/
theorem hours_per_day :
  let hours_per_year : ℕ := 8760
  let days_per_year : ℕ := 365
  (hours_per_year / days_per_year : ℚ) = 24 := by
sorry

end NUMINAMATH_CALUDE_hours_per_day_l3205_320519


namespace NUMINAMATH_CALUDE_parentheses_placement_l3205_320533

theorem parentheses_placement :
  (1 : ℚ) / (2 / (3 / (4 / (5 / (6 / (7 / (8 / (9 / 10)))))))) = 7 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_placement_l3205_320533


namespace NUMINAMATH_CALUDE_unique_score_with_three_combinations_l3205_320500

/-- Represents a scoring combination for the test -/
structure ScoringCombination where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- Calculates the score for a given scoring combination -/
def calculateScore (sc : ScoringCombination) : ℕ :=
  6 * sc.correct + 3 * sc.unanswered

/-- Checks if a scoring combination is valid (sums to 25 questions) -/
def isValidCombination (sc : ScoringCombination) : Prop :=
  sc.correct + sc.unanswered + sc.incorrect = 25

/-- Theorem: 78 is the only score achievable in exactly three ways -/
theorem unique_score_with_three_combinations :
  ∃! score : ℕ,
    (∃ (combinations : Finset ScoringCombination),
      combinations.card = 3 ∧
      (∀ sc ∈ combinations, isValidCombination sc ∧ calculateScore sc = score) ∧
      (∀ sc : ScoringCombination, isValidCombination sc ∧ calculateScore sc = score → sc ∈ combinations)) ∧
    score = 78 := by
  sorry

end NUMINAMATH_CALUDE_unique_score_with_three_combinations_l3205_320500


namespace NUMINAMATH_CALUDE_binomial_coefficient_product_l3205_320554

theorem binomial_coefficient_product (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_product_l3205_320554


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3205_320507

/-- The distance between the foci of a hyperbola with equation xy = 4 is 8 -/
theorem hyperbola_foci_distance :
  ∀ (x y : ℝ), x * y = 4 →
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 * f₁.2 = 4 ∧ f₂.1 * f₂.2 = 4) ∧
    (Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 8) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3205_320507


namespace NUMINAMATH_CALUDE_complement_union_and_intersection_range_of_a_l3205_320582

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem for part (1)
theorem complement_union_and_intersection :
  (Set.univ \ (A ∪ B) = {x : ℝ | x ≤ 2 ∨ x ≥ 10}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) := by sorry

-- Theorem for part (2)
theorem range_of_a (h : A ∩ C a ≠ ∅) : a > 3 := by sorry

end NUMINAMATH_CALUDE_complement_union_and_intersection_range_of_a_l3205_320582


namespace NUMINAMATH_CALUDE_book_arrangement_l3205_320561

theorem book_arrangement (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) :
  (n.factorial / k.factorial : ℕ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_l3205_320561


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3205_320591

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem quadratic_function_properties :
  (∀ x, f x = x^2 - 2*x + 3) ∧
  (f 0 = 3) ∧
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∀ x ≤ 1, ∀ y ≥ x, f x ≥ f y) ∧
  (∀ x ≥ 1, ∀ y ≥ x, f x ≤ f y) ∧
  (∀ x, f x ≥ 2) ∧
  (f 1 = 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3205_320591


namespace NUMINAMATH_CALUDE_potato_fetching_time_l3205_320522

/-- Represents the problem of calculating how long it takes a dog to fetch a launched potato. -/
theorem potato_fetching_time 
  (football_fields : ℕ) -- number of football fields the potato is launched
  (yards_per_field : ℕ) -- length of a football field in yards
  (dog_speed : ℕ) -- dog's speed in feet per minute
  (h1 : football_fields = 6)
  (h2 : yards_per_field = 200)
  (h3 : dog_speed = 400) :
  (football_fields * yards_per_field * 3) / dog_speed = 9 := by
  sorry

#check potato_fetching_time

end NUMINAMATH_CALUDE_potato_fetching_time_l3205_320522


namespace NUMINAMATH_CALUDE_product_divisibility_l3205_320599

theorem product_divisibility : ∃ k : ℕ, 86 * 87 * 88 * 89 * 90 * 91 * 92 = 7 * k := by
  sorry

#check product_divisibility

end NUMINAMATH_CALUDE_product_divisibility_l3205_320599


namespace NUMINAMATH_CALUDE_dans_marbles_l3205_320508

/-- The number of marbles Dan has after giving some away -/
def remaining_marbles (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Dan's remaining marbles is the difference between initial and given away -/
theorem dans_marbles (initial : ℕ) (given_away : ℕ) :
  remaining_marbles initial given_away = initial - given_away :=
by sorry

end NUMINAMATH_CALUDE_dans_marbles_l3205_320508


namespace NUMINAMATH_CALUDE_min_value_sum_l3205_320509

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - x - 2 * y = 0) :
  x + y ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_l3205_320509


namespace NUMINAMATH_CALUDE_three_solutions_condition_l3205_320583

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  ((x - 5) * Real.sin a - (y - 5) * Real.cos a = 0) ∧
  (((x + 1)^2 + (y + 1)^2 - 4) * ((x + 1)^2 + (y + 1)^2 - 16) = 0)

-- Define the condition for three solutions
def has_three_solutions (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    system x₁ y₁ a ∧ system x₂ y₂ a ∧ system x₃ y₃ a ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    ∀ (x y : ℝ), system x y a → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃)

-- Theorem statement
theorem three_solutions_condition (a : ℝ) :
  has_three_solutions a ↔ ∃ (n : ℤ), a = π/4 + Real.arcsin (Real.sqrt 2 / 6) + n * π ∨
                                     a = π/4 - Real.arcsin (Real.sqrt 2 / 6) + n * π :=
sorry

end NUMINAMATH_CALUDE_three_solutions_condition_l3205_320583


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l3205_320566

theorem rectangular_parallelepiped_volume
  (l α β : ℝ)
  (h_l : l > 0)
  (h_α : 0 < α ∧ α < π / 2)
  (h_β : 0 < β ∧ β < π / 2) :
  ∃ V : ℝ,
    V = l^3 * Real.sin α * Real.sin β * Real.sqrt (Real.cos (α + β) * Real.cos (α - β)) ∧
    V > 0 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l3205_320566


namespace NUMINAMATH_CALUDE_fourth_square_area_l3205_320525

-- Define the triangles and their properties
structure Triangle :=
  (P Q R : ℝ × ℝ)
  (isRightAngle : Bool)

-- Define the squares on the sides
structure Square :=
  (side : ℝ)
  (area : ℝ)

-- Theorem statement
theorem fourth_square_area
  (PQR PRM : Triangle)
  (square1 square2 square3 : Square)
  (h1 : PQR.isRightAngle = true)
  (h2 : PRM.isRightAngle = true)
  (h3 : square1.area = 25)
  (h4 : square2.area = 81)
  (h5 : square3.area = 64)
  : ∃ (square4 : Square), square4.area = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_square_area_l3205_320525


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l3205_320503

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 104 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l3205_320503


namespace NUMINAMATH_CALUDE_roberto_outfits_l3205_320579

def trousers : ℕ := 5
def shirts : ℕ := 6
def jackets : ℕ := 3
def ties : ℕ := 2

theorem roberto_outfits : trousers * shirts * jackets * ties = 180 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l3205_320579


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3205_320530

/-- The atomic weight of Carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in the compound -/
def carbon_count : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 1

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ := carbon_weight * carbon_count + oxygen_weight * oxygen_count

theorem compound_molecular_weight :
  molecular_weight = 28.01 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3205_320530


namespace NUMINAMATH_CALUDE_derivative_limit_equality_l3205_320571

theorem derivative_limit_equality (f : ℝ → ℝ) (h : HasDerivAt f (-2) 2) :
  Filter.Tendsto (fun x => (f x - f 2) / (x - 2)) (Filter.atTop.comap (fun x => |x - 2|)) (nhds (-2)) := by
  sorry

end NUMINAMATH_CALUDE_derivative_limit_equality_l3205_320571


namespace NUMINAMATH_CALUDE_carol_carrots_l3205_320552

-- Define the variables
def total_carrots : ℕ := 38 + 7
def mom_carrots : ℕ := 16

-- State the theorem
theorem carol_carrots : total_carrots - mom_carrots = 29 := by
  sorry

end NUMINAMATH_CALUDE_carol_carrots_l3205_320552


namespace NUMINAMATH_CALUDE_parallel_lines_solution_l3205_320567

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ : ℝ} : 
  (∀ x y : ℝ, m₁ * x - y = 0 ↔ m₂ * x - y = 0) ↔ m₁ = m₂

/-- Given two parallel lines ax - y + a = 0 and (2a - 3)x + ay - a = 0, prove that a = -3 -/
theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a * x - y + a = 0 ↔ (2 * a - 3) * x + a * y - a = 0) → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_solution_l3205_320567


namespace NUMINAMATH_CALUDE_not_yellow_houses_l3205_320514

/-- Represents the number of houses Isabella has of each color --/
structure Houses where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Conditions for Isabella's houses --/
def isabellas_houses (h : Houses) : Prop :=
  h.green = 3 * h.yellow ∧
  h.yellow = h.red - 40 ∧
  h.green = 90

/-- Theorem stating the number of houses that are not yellow --/
theorem not_yellow_houses (h : Houses) (hcond : isabellas_houses h) :
  h.green + h.red = 160 :=
sorry

end NUMINAMATH_CALUDE_not_yellow_houses_l3205_320514


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3205_320556

theorem floor_plus_self_unique_solution :
  ∃! r : ℝ, ⌊r⌋ + r = 14.5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3205_320556


namespace NUMINAMATH_CALUDE_min_value_problem_l3205_320545

/-- Given positive real numbers a, b, c, and a function f with minimum value 4,
    prove that a + b + c = 4 and the minimum value of (1/4)a² + (1/9)b² + c² is 8/7 -/
theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 →
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3205_320545


namespace NUMINAMATH_CALUDE_trapezoid_EN_squared_l3205_320541

/-- Trapezoid ABCD with given side lengths and point N -/
structure Trapezoid :=
  (A B C D E M N : ℝ × ℝ)
  (AB_parallel_CD : (A.2 - B.2) / (A.1 - B.1) = (C.2 - D.2) / (C.1 - D.1))
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5)
  (BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 9)
  (CD_length : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 10)
  (DA_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 7)
  (E_on_BC : ∃ t, E = (1 - t) • B + t • C)
  (E_on_DA : ∃ s, E = (1 - s) • D + s • A)
  (M_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
  (N_on_BMC : (N.1 - B.1)^2 + (N.2 - B.2)^2 = (N.1 - M.1)^2 + (N.2 - M.2)^2 ∧
               (N.1 - M.1)^2 + (N.2 - M.2)^2 = (N.1 - C.1)^2 + (N.2 - C.2)^2)
  (N_on_DMA : (N.1 - D.1)^2 + (N.2 - D.2)^2 = (N.1 - M.1)^2 + (N.2 - M.2)^2 ∧
               (N.1 - M.1)^2 + (N.2 - M.2)^2 = (N.1 - A.1)^2 + (N.2 - A.2)^2)
  (N_not_M : N ≠ M)

/-- The main theorem -/
theorem trapezoid_EN_squared (t : Trapezoid) : 
  (t.E.1 - t.N.1)^2 + (t.E.2 - t.N.2)^2 = 900 / 11 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_EN_squared_l3205_320541


namespace NUMINAMATH_CALUDE_pyramid_sum_l3205_320502

theorem pyramid_sum (x : ℝ) : 
  let row2_left : ℝ := 11
  let row2_middle : ℝ := 6 + x
  let row2_right : ℝ := x + 7
  let row3_left : ℝ := row2_left + row2_middle
  let row3_right : ℝ := row2_middle + row2_right
  let row4 : ℝ := row3_left + row3_right
  row4 = 60 → x = 10 := by
sorry

end NUMINAMATH_CALUDE_pyramid_sum_l3205_320502


namespace NUMINAMATH_CALUDE_calculation_proof_l3205_320527

def mixed_to_improper (whole : Int) (num : Int) (denom : Int) : Rat :=
  (whole * denom + num) / denom

theorem calculation_proof :
  let a := mixed_to_improper 2 3 7
  let b := mixed_to_improper 5 1 3
  let c := mixed_to_improper 3 1 5
  let d := mixed_to_improper 2 1 6
  75 * (a - b) / (c + d) = -208 - 7/9 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_l3205_320527


namespace NUMINAMATH_CALUDE_geography_history_difference_l3205_320587

/-- Represents the number of pages in each textbook --/
structure TextbookPages where
  history : ℕ
  geography : ℕ
  math : ℕ
  science : ℕ

/-- Conditions for Suzanna's textbooks --/
def suzanna_textbooks (t : TextbookPages) : Prop :=
  t.history = 160 ∧
  t.geography > t.history ∧
  t.math = (t.history + t.geography) / 2 ∧
  t.science = 2 * t.history ∧
  t.history + t.geography + t.math + t.science = 905

/-- Theorem stating the difference in pages between geography and history textbooks --/
theorem geography_history_difference (t : TextbookPages) 
  (h : suzanna_textbooks t) : t.geography - t.history = 70 := by
  sorry


end NUMINAMATH_CALUDE_geography_history_difference_l3205_320587


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3205_320577

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_condition
  (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (a 3 * a 5 = 16 → a 4 = 4) ∧ 
  ¬(a 4 = 4 → a 3 * a 5 = 16) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3205_320577


namespace NUMINAMATH_CALUDE_abs_fraction_less_than_one_l3205_320518

theorem abs_fraction_less_than_one (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : 
  |((x - y) / (1 - x * y))| < 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_fraction_less_than_one_l3205_320518


namespace NUMINAMATH_CALUDE_souvenir_walk_distance_l3205_320549

theorem souvenir_walk_distance (total : ℝ) (hotel_to_postcard : ℝ) (postcard_to_tshirt : ℝ)
  (h1 : total = 0.89)
  (h2 : hotel_to_postcard = 0.11)
  (h3 : postcard_to_tshirt = 0.11) :
  total - (hotel_to_postcard + postcard_to_tshirt) = 0.67 := by
sorry

end NUMINAMATH_CALUDE_souvenir_walk_distance_l3205_320549


namespace NUMINAMATH_CALUDE_quadrilateral_sine_equality_l3205_320539

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)  -- Interior angles
  (AB BC CD DA : ℝ)  -- Side lengths

-- Define the property of being convex
def is_convex (q : Quadrilateral) : Prop :=
  q.A > 0 ∧ q.B > 0 ∧ q.C > 0 ∧ q.D > 0 ∧ q.A + q.B + q.C + q.D = 2 * Real.pi

-- State the theorem
theorem quadrilateral_sine_equality (q : Quadrilateral) (h : is_convex q) :
  (Real.sin q.A) / (q.BC * q.CD) + (Real.sin q.C) / (q.DA * q.AB) =
  (Real.sin q.B) / (q.CD * q.DA) + (Real.sin q.D) / (q.AB * q.BC) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_sine_equality_l3205_320539


namespace NUMINAMATH_CALUDE_entree_dessert_cost_difference_l3205_320569

/-- Given Hannah's restaurant bill, prove the cost difference between entree and dessert -/
theorem entree_dessert_cost_difference 
  (total_cost : ℕ) 
  (entree_cost : ℕ) 
  (h1 : total_cost = 23)
  (h2 : entree_cost = 14) :
  entree_cost - (total_cost - entree_cost) = 5 := by
  sorry

#check entree_dessert_cost_difference

end NUMINAMATH_CALUDE_entree_dessert_cost_difference_l3205_320569


namespace NUMINAMATH_CALUDE_triangle_side_length_l3205_320524

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that if b = 6, c = 4, and A = 2B, then a = 2√15 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  b = 6 → c = 4 → A = 2 * B → 
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = Real.pi) →
  (a / Real.sin A = b / Real.sin B) →
  (a / Real.sin A = c / Real.sin C) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  a = 2 * Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3205_320524


namespace NUMINAMATH_CALUDE_jame_practice_weeks_l3205_320573

def regular_cards_per_tear : ℕ := 30
def thick_cards_per_tear : ℕ := 25
def cards_per_regular_deck : ℕ := 52
def cards_per_thick_deck : ℕ := 55
def tears_per_week : ℕ := 4
def regular_decks_bought : ℕ := 27
def thick_decks_bought : ℕ := 14

def total_cards : ℕ := regular_decks_bought * cards_per_regular_deck + thick_decks_bought * cards_per_thick_deck

def cards_torn_per_week : ℕ := (regular_cards_per_tear + thick_cards_per_tear) * (tears_per_week / 2)

theorem jame_practice_weeks :
  (total_cards / cards_torn_per_week : ℕ) = 19 := by sorry

end NUMINAMATH_CALUDE_jame_practice_weeks_l3205_320573


namespace NUMINAMATH_CALUDE_gigi_mushrooms_l3205_320523

/-- The number of pieces each mushroom is cut into -/
def pieces_per_mushroom : ℕ := 4

/-- The number of mushroom pieces Kenny used -/
def kenny_pieces : ℕ := 38

/-- The number of mushroom pieces Karla used -/
def karla_pieces : ℕ := 42

/-- The number of mushroom pieces left on the cutting board -/
def leftover_pieces : ℕ := 8

/-- The total number of mushroom pieces -/
def total_pieces : ℕ := kenny_pieces + karla_pieces + leftover_pieces

/-- The number of whole mushrooms GiGi cut up -/
def whole_mushrooms : ℕ := total_pieces / pieces_per_mushroom

theorem gigi_mushrooms : whole_mushrooms = 22 := by
  sorry

end NUMINAMATH_CALUDE_gigi_mushrooms_l3205_320523


namespace NUMINAMATH_CALUDE_tan_negative_225_degrees_l3205_320578

theorem tan_negative_225_degrees : Real.tan (-(225 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_225_degrees_l3205_320578


namespace NUMINAMATH_CALUDE_marble_selection_with_blue_l3205_320521

def total_marbles : ℕ := 10
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 4
def green_marbles : ℕ := 3
def selection_size : ℕ := 4

theorem marble_selection_with_blue (total_marbles red_marbles blue_marbles green_marbles selection_size : ℕ) 
  (h1 : total_marbles = red_marbles + blue_marbles + green_marbles)
  (h2 : total_marbles = 10)
  (h3 : red_marbles = 3)
  (h4 : blue_marbles = 4)
  (h5 : green_marbles = 3)
  (h6 : selection_size = 4) :
  (Nat.choose total_marbles selection_size) - (Nat.choose (total_marbles - blue_marbles) selection_size) = 195 :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_with_blue_l3205_320521


namespace NUMINAMATH_CALUDE_halfway_between_fractions_average_of_fractions_l3205_320528

theorem halfway_between_fractions :
  (1/8 : ℚ) + (1/10 : ℚ) = (9/40 : ℚ) :=
by sorry

theorem average_of_fractions :
  ((1/8 : ℚ) + (1/10 : ℚ)) / 2 = (9/80 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_average_of_fractions_l3205_320528


namespace NUMINAMATH_CALUDE_simplify_expression_calculate_expression_find_expression_value_l3205_320506

-- Question 1
theorem simplify_expression (a b : ℝ) :
  2 * (a - b)^2 - 4 * (a - b)^2 + 7 * (a - b)^2 = 5 * (a - b)^2 := by sorry

-- Question 2
theorem calculate_expression (a b : ℝ) (h : a^2 - 2*b^2 - 3 = 0) :
  -3*a^2 + 6*b^2 + 2032 = 2023 := by sorry

-- Question 3
theorem find_expression_value (a b : ℝ) (h1 : a^2 + 2*a*b = 15) (h2 : b^2 + 2*a*b = 6) :
  2*a^2 - 4*b^2 - 4*a*b = 6 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_calculate_expression_find_expression_value_l3205_320506


namespace NUMINAMATH_CALUDE_card_game_problem_l3205_320557

/-- The card game problem -/
theorem card_game_problem (T : ℚ) :
  -- Initial ratios
  let initial_aldo : ℚ := 7 / 18 * T
  let initial_bernardo : ℚ := 6 / 18 * T
  let initial_carlos : ℚ := 5 / 18 * T
  -- Final ratios
  let final_aldo : ℚ := 6 / 15 * T
  let final_bernardo : ℚ := 5 / 15 * T
  let final_carlos : ℚ := 4 / 15 * T
  -- One player won 12 reais
  (∃ (winner : ℚ), winner - (winner - 12) = 12) →
  -- The changes in amounts
  (final_aldo - initial_aldo = 12 ∨
   final_bernardo - initial_bernardo = 12 ∨
   final_carlos - initial_carlos = 12) →
  -- Prove the final amounts
  (final_aldo = 432 ∧ final_bernardo = 360 ∧ final_carlos = 288) := by
sorry


end NUMINAMATH_CALUDE_card_game_problem_l3205_320557


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3205_320572

/-- A quadratic function satisfying specific conditions -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- The theorem stating the properties of the quadratic function and the range of m -/
theorem quadratic_function_properties :
  (∀ x ∈ Set.Icc (-3 : ℝ) 1, f x ≤ 0) ∧
  (∀ x ∈ Set.Ioi 1 ∪ Set.Iio (-3 : ℝ), f x > 0) ∧
  (f 2 = 5) ∧
  (∀ m : ℝ, (∃ x : ℝ, f x = 9*m + 3) ↔ m ≥ -7/9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3205_320572


namespace NUMINAMATH_CALUDE_equation_solution_l3205_320511

theorem equation_solution (x y : ℝ) :
  y^2 - 2*y = x^2 + 2*x ↔ y = x + 2 ∨ y = -x := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3205_320511


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3205_320526

/-- The perimeter of a rhombus with diagonals measuring 20 feet and 16 feet is 8√41 feet. -/
theorem rhombus_perimeter (d₁ d₂ : ℝ) (h₁ : d₁ = 20) (h₂ : d₂ = 16) :
  let side := Real.sqrt ((d₁/2)^2 + (d₂/2)^2)
  4 * side = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3205_320526


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3205_320546

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3205_320546


namespace NUMINAMATH_CALUDE_real_part_of_inverse_difference_l3205_320597

theorem real_part_of_inverse_difference (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 2) :
  (1 / (2 - z)).re = 1/4 :=
sorry

end NUMINAMATH_CALUDE_real_part_of_inverse_difference_l3205_320597


namespace NUMINAMATH_CALUDE_runners_speed_difference_l3205_320562

/-- Given two runners starting at the same point, with one going north at 8 miles per hour
    and the other going east, if they are 5 miles apart after 1/2 hour,
    then the difference in their speeds is 2 miles per hour. -/
theorem runners_speed_difference (v : ℝ) : 
  (v ≥ 0) →  -- Ensuring non-negative speed
  ((8 * (1/2))^2 + (v * (1/2))^2 = 5^2) → 
  (8 - v = 2) :=
by sorry

end NUMINAMATH_CALUDE_runners_speed_difference_l3205_320562


namespace NUMINAMATH_CALUDE_cristina_pace_race_scenario_l3205_320588

/-- Cristina's pace in a race with Nicky --/
theorem cristina_pace (head_start : ℝ) (nicky_pace : ℝ) (catch_up_time : ℝ) : ℝ :=
  let nicky_distance := head_start * nicky_pace + catch_up_time * nicky_pace
  nicky_distance / catch_up_time

/-- The race scenario --/
theorem race_scenario : cristina_pace 12 3 30 = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_cristina_pace_race_scenario_l3205_320588


namespace NUMINAMATH_CALUDE_multiple_sum_properties_l3205_320513

theorem multiple_sum_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, a + b = 2 * n) ∧ 
  (¬ ∀ p : ℤ, a + b = 6 * p) ∧
  (¬ ∀ q : ℤ, a + b = 8 * q) ∧
  (∃ r s : ℤ, a = 6 * r ∧ b = 8 * s ∧ ¬ ∃ t : ℤ, a + b = 8 * t) :=
by sorry

end NUMINAMATH_CALUDE_multiple_sum_properties_l3205_320513


namespace NUMINAMATH_CALUDE_shirt_price_theorem_l3205_320532

/-- The price of a shirt when the total cost of the shirt and a coat is $600,
    and the shirt costs one-third the price of the coat. -/
def shirt_price : ℝ := 150

/-- The price of a coat when the total cost of the shirt and the coat is $600,
    and the shirt costs one-third the price of the coat. -/
def coat_price : ℝ := 3 * shirt_price

theorem shirt_price_theorem :
  shirt_price + coat_price = 600 ∧ shirt_price = (1/3) * coat_price →
  shirt_price = 150 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_theorem_l3205_320532


namespace NUMINAMATH_CALUDE_A_is_singleton_floor_sum_property_l3205_320531

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the set A
def A : Set ℝ :=
  {x | x^2 - (floor x : ℝ) - 1 = 0 ∧ -1 < x ∧ x < 2}

-- Theorem 1: A is a singleton set
theorem A_is_singleton : ∃! x, x ∈ A := by sorry

-- Theorem 2: Floor function property
theorem floor_sum_property (x : ℝ) :
  (floor x : ℝ) + (floor (x + 1/2) : ℝ) = (floor (2*x) : ℝ) := by sorry

end NUMINAMATH_CALUDE_A_is_singleton_floor_sum_property_l3205_320531


namespace NUMINAMATH_CALUDE_cookie_distribution_l3205_320520

theorem cookie_distribution (total_cookies : ℕ) (num_children : ℕ) 
  (h1 : total_cookies = 28) (h2 : num_children = 6) : 
  total_cookies - (num_children * (total_cookies / num_children)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l3205_320520


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3205_320510

theorem smallest_sum_of_sequence (X Y Z W : ℕ) : 
  X > 0 → Y > 0 → Z > 0 →
  (∃ d : ℤ, Z - Y = Y - X) →  -- arithmetic sequence condition
  (∃ r : ℚ, Z = r * Y ∧ W = r * Z) →  -- geometric sequence condition
  (Z : ℚ) / Y = 7 / 4 →
  X + Y + Z + W ≥ 97 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3205_320510


namespace NUMINAMATH_CALUDE_third_grade_students_l3205_320553

/-- The number of story books to be distributed -/
def total_books : ℕ := 90

/-- Proves that the number of third-grade students is 60 -/
theorem third_grade_students :
  ∃ n : ℕ, n > 0 ∧ n < total_books ∧ total_books - n = n / 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_third_grade_students_l3205_320553


namespace NUMINAMATH_CALUDE_decimal_power_equivalence_l3205_320590

theorem decimal_power_equivalence : (1 / 10 : ℝ) ^ 2 = 0.010000000000000002 := by
  sorry

end NUMINAMATH_CALUDE_decimal_power_equivalence_l3205_320590


namespace NUMINAMATH_CALUDE_smallest_number_l3205_320565

/-- Converts a number from base b to decimal (base 10) --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The given numbers in their respective bases --/
def number_A : List Nat := [2, 0, 0, 1]  -- 1002 in base 3
def number_B : List Nat := [0, 1, 2]     -- 210 in base 6
def number_C : List Nat := [0, 0, 0, 1]  -- 1000 in base 4
def number_D : List Nat := [1, 1, 1, 1, 1, 1]  -- 111111 in base 2

/-- The bases of the given numbers --/
def base_A : Nat := 3
def base_B : Nat := 6
def base_C : Nat := 4
def base_D : Nat := 2

theorem smallest_number :
  (to_decimal number_A base_A < to_decimal number_B base_B) ∧
  (to_decimal number_A base_A < to_decimal number_C base_C) ∧
  (to_decimal number_A base_A < to_decimal number_D base_D) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3205_320565


namespace NUMINAMATH_CALUDE_heathers_weight_l3205_320535

/-- Given that Emily weighs 9 pounds and Heather is 78 pounds heavier than Emily,
    prove that Heather weighs 87 pounds. -/
theorem heathers_weight (emily_weight : ℕ) (weight_difference : ℕ) 
  (h1 : emily_weight = 9)
  (h2 : weight_difference = 78) :
  emily_weight + weight_difference = 87 := by
  sorry

end NUMINAMATH_CALUDE_heathers_weight_l3205_320535


namespace NUMINAMATH_CALUDE_vector_calculation_l3205_320540

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ m ↦ (4, m)

theorem vector_calculation (m : ℝ) (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  (5 * a.1 - 3 * (b m).1, 5 * a.2 - 3 * (b m).2) = (-7, -16) := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l3205_320540


namespace NUMINAMATH_CALUDE_bugs_meet_time_l3205_320517

/-- The time (in minutes) it takes for two bugs to meet again at the starting point,
    given they start on two tangent circles with radii 7 and 3 inches,
    crawling at speeds of 4π and 3π inches per minute respectively. -/
def meeting_time : ℝ :=
  let r₁ : ℝ := 7  -- radius of larger circle
  let r₂ : ℝ := 3  -- radius of smaller circle
  let v₁ : ℝ := 4 * Real.pi  -- speed of bug on larger circle
  let v₂ : ℝ := 3 * Real.pi  -- speed of bug on smaller circle
  let t₁ : ℝ := (2 * Real.pi * r₁) / v₁  -- time for full circle on larger circle
  let t₂ : ℝ := (2 * Real.pi * r₂) / v₂  -- time for full circle on smaller circle
  14  -- the actual meeting time

theorem bugs_meet_time :
  meeting_time = 14 := by sorry

end NUMINAMATH_CALUDE_bugs_meet_time_l3205_320517


namespace NUMINAMATH_CALUDE_percentage_women_without_retirement_plan_l3205_320560

theorem percentage_women_without_retirement_plan 
  (total_workers : ℕ)
  (workers_without_plan : ℕ)
  (men_with_plan : ℕ)
  (total_men : ℕ)
  (total_women : ℕ)
  (h1 : workers_without_plan = total_workers / 3)
  (h2 : men_with_plan = (total_workers - workers_without_plan) * 2 / 5)
  (h3 : total_men = 120)
  (h4 : total_women = 120)
  (h5 : total_workers = total_men + total_women) :
  (workers_without_plan - (total_men - men_with_plan)) * 100 / total_women = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_women_without_retirement_plan_l3205_320560


namespace NUMINAMATH_CALUDE_divides_power_difference_l3205_320585

theorem divides_power_difference (n : ℕ) : n ∣ 2^(2*n.factorial) - 2^(n.factorial) := by
  sorry

end NUMINAMATH_CALUDE_divides_power_difference_l3205_320585


namespace NUMINAMATH_CALUDE_james_weekly_beats_l3205_320584

/-- The number of beats heard in a week given a music speed and daily listening time -/
def beats_per_week (beats_per_minute : ℕ) (hours_per_day : ℕ) : ℕ :=
  beats_per_minute * (hours_per_day * 60) * 7

/-- Theorem: James hears 168,000 beats per week -/
theorem james_weekly_beats :
  beats_per_week 200 2 = 168000 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_beats_l3205_320584


namespace NUMINAMATH_CALUDE_complex_number_modulus_one_l3205_320534

theorem complex_number_modulus_one (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  Complex.abs z = 1 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_one_l3205_320534


namespace NUMINAMATH_CALUDE_integer_root_values_l3205_320504

def polynomial (a x : ℤ) : ℤ := x^3 + 3*x^2 + a*x + 9

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial a x = 0

theorem integer_root_values :
  {a : ℤ | has_integer_root a} = {-109, -21, -13, 3, 11, 53} := by sorry

end NUMINAMATH_CALUDE_integer_root_values_l3205_320504


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l3205_320570

/-- Given a canoe's upstream speed and the stream speed, calculates the downstream speed. -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  2 * upstream_speed + 3 * stream_speed

/-- Theorem stating that for a canoe with upstream speed 8 km/hr and stream speed 2 km/hr, 
    the downstream speed is 12 km/hr. -/
theorem canoe_downstream_speed :
  let upstream_speed := 8
  let stream_speed := 2
  downstream_speed upstream_speed stream_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_canoe_downstream_speed_l3205_320570


namespace NUMINAMATH_CALUDE_average_run_time_l3205_320593

/-- Represents the average minutes run per day for each grade -/
structure GradeRunTime where
  sixth : ℝ
  seventh : ℝ
  eighth : ℝ

/-- Represents the number of students in each grade -/
structure GradePopulation where
  seventh : ℝ
  sixth : ℝ
  eighth : ℝ

/-- Represents the number of days each grade runs per week -/
structure RunDays where
  sixth : ℕ
  seventh : ℕ
  eighth : ℕ

theorem average_run_time 
  (run_time : GradeRunTime)
  (population : GradePopulation)
  (days : RunDays)
  (h1 : run_time.sixth = 10)
  (h2 : run_time.seventh = 12)
  (h3 : run_time.eighth = 8)
  (h4 : population.sixth = 3 * population.seventh)
  (h5 : population.eighth = population.seventh / 2)
  (h6 : days.sixth = 2)
  (h7 : days.seventh = 2)
  (h8 : days.eighth = 1) :
  (run_time.sixth * population.sixth * days.sixth +
   run_time.seventh * population.seventh * days.seventh +
   run_time.eighth * population.eighth * days.eighth) /
  (population.sixth + population.seventh + population.eighth) /
  7 = 176 / 9 := by
  sorry


end NUMINAMATH_CALUDE_average_run_time_l3205_320593


namespace NUMINAMATH_CALUDE_prime_triple_product_sum_l3205_320576

theorem prime_triple_product_sum : 
  ∀ x y z : ℕ, 
    Prime x → Prime y → Prime z →
    x * y * z = 5 * (x + y + z) →
    ((x = 2 ∧ y = 5 ∧ z = 7) ∨ (x = 5 ∧ y = 2 ∧ z = 7)) :=
by
  sorry

end NUMINAMATH_CALUDE_prime_triple_product_sum_l3205_320576


namespace NUMINAMATH_CALUDE_draw_probability_standard_deck_l3205_320548

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (hearts : Nat)
  (clubs : Nat)
  (spades : Nat)

/-- A standard 52-card deck -/
def standardDeck : Deck :=
  { cards := 52,
    hearts := 13,
    clubs := 13,
    spades := 13 }

/-- The probability of drawing a heart, then a club, then a spade from a standard deck -/
def drawProbability (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.cards *
  (d.clubs : ℚ) / (d.cards - 1) *
  (d.spades : ℚ) / (d.cards - 2)

theorem draw_probability_standard_deck :
  drawProbability standardDeck = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_draw_probability_standard_deck_l3205_320548


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_value_l3205_320575

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a5_value
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_nonzero : ∃ n : ℕ, a n ≠ 0)
  (h_eq : a 5 ^ 2 - a 3 - a 7 = 0) :
  a 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_value_l3205_320575


namespace NUMINAMATH_CALUDE_blood_donation_selection_l3205_320564

theorem blood_donation_selection (m n k : ℕ) (hm : m = 3) (hn : n = 6) (hk : k = 5) :
  (Nat.choose (m + n) k) - (Nat.choose n k) = 120 := by
  sorry

end NUMINAMATH_CALUDE_blood_donation_selection_l3205_320564


namespace NUMINAMATH_CALUDE_electricity_pricing_l3205_320563

/-- Represents the electricity pricing problem -/
theorem electricity_pricing
  (a : ℝ) -- annual electricity consumption in kilowatt-hours
  (x : ℝ) -- new electricity price per kilowatt-hour
  (h1 : 0 < a) -- assumption that consumption is positive
  (h2 : 0.55 ≤ x ∧ x ≤ 0.75) -- new price range
  : ((0.2 * a / (x - 0.40) + a) * (x - 0.30) ≥ 0.60 * a) ↔ (x ≥ 0.60) :=
by sorry

end NUMINAMATH_CALUDE_electricity_pricing_l3205_320563


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l3205_320598

/-- Given a right circular cone with diameter 14 and altitude 16, and an inscribed right circular
cylinder whose height is twice its radius, the radius of the cylinder is 56/15. -/
theorem inscribed_cylinder_radius (cone_diameter : ℝ) (cone_altitude : ℝ) (cylinder_radius : ℝ) :
  cone_diameter = 14 →
  cone_altitude = 16 →
  (∃ (cylinder_height : ℝ), cylinder_height = 2 * cylinder_radius) →
  cylinder_radius = 56 / 15 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l3205_320598


namespace NUMINAMATH_CALUDE_secretary_work_time_l3205_320589

theorem secretary_work_time 
  (ratio : Fin 3 → ℕ)
  (total_time : ℕ) :
  ratio 0 = 2 →
  ratio 1 = 3 →
  ratio 2 = 5 →
  total_time = 110 →
  (ratio 0 + ratio 1 + ratio 2) * (total_time / (ratio 0 + ratio 1 + ratio 2)) = total_time →
  ratio 2 * (total_time / (ratio 0 + ratio 1 + ratio 2)) = 55 :=
by sorry

end NUMINAMATH_CALUDE_secretary_work_time_l3205_320589


namespace NUMINAMATH_CALUDE_youngest_daughter_cost_l3205_320501

/-- Represents the cost of dresses and hats bought by the daughters -/
structure Purchase where
  dresses : ℕ
  hats : ℕ
  cost : ℕ

/-- The problem setup -/
def merchant_problem : Prop :=
  ∃ (dress_cost hat_cost : ℕ),
    let eldest := Purchase.mk 6 3 105
    let second := Purchase.mk 3 5 70
    let youngest := Purchase.mk 1 2 0
    eldest.cost = eldest.dresses * dress_cost + eldest.hats * hat_cost ∧
    second.cost = second.dresses * dress_cost + second.hats * hat_cost ∧
    youngest.dresses * dress_cost + youngest.hats * hat_cost = 25

/-- The theorem to be proved -/
theorem youngest_daughter_cost :
  merchant_problem := by sorry

end NUMINAMATH_CALUDE_youngest_daughter_cost_l3205_320501


namespace NUMINAMATH_CALUDE_some_number_calculation_l3205_320595

theorem some_number_calculation : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 := by
  sorry

end NUMINAMATH_CALUDE_some_number_calculation_l3205_320595


namespace NUMINAMATH_CALUDE_complex_distance_range_l3205_320558

theorem complex_distance_range (z : ℂ) (h : Complex.abs z = 1) :
  0 ≤ Complex.abs (z - (1 - Complex.I * Real.sqrt 3)) ∧
  Complex.abs (z - (1 - Complex.I * Real.sqrt 3)) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_complex_distance_range_l3205_320558


namespace NUMINAMATH_CALUDE_randy_cheese_purchase_l3205_320529

/-- The number of slices in a package of cheddar cheese -/
def cheddar_slices : ℕ := 12

/-- The number of slices in a package of Swiss cheese -/
def swiss_slices : ℕ := 28

/-- The smallest number of slices of each type that Randy could have bought -/
def smallest_equal_slices : ℕ := 84

theorem randy_cheese_purchase :
  smallest_equal_slices = Nat.lcm cheddar_slices swiss_slices ∧
  smallest_equal_slices % cheddar_slices = 0 ∧
  smallest_equal_slices % swiss_slices = 0 ∧
  ∀ n : ℕ, (n % cheddar_slices = 0 ∧ n % swiss_slices = 0) → n ≥ smallest_equal_slices := by
  sorry

end NUMINAMATH_CALUDE_randy_cheese_purchase_l3205_320529


namespace NUMINAMATH_CALUDE_robin_bracelet_cost_l3205_320574

def cost_per_bracelet : ℕ := 2

def friend_names : List String := ["Jessica", "Tori", "Lily", "Patrice"]

def total_letters (names : List String) : ℕ :=
  names.map String.length |>.sum

def total_cost (names : List String) (cost : ℕ) : ℕ :=
  (total_letters names) * cost

theorem robin_bracelet_cost :
  total_cost friend_names cost_per_bracelet = 44 := by
  sorry

end NUMINAMATH_CALUDE_robin_bracelet_cost_l3205_320574


namespace NUMINAMATH_CALUDE_function_bound_l3205_320594

/-- Given real-valued functions f and g defined on the real line,
    if f(x + y) + f(x - y) = 2f(x)g(y) for all x and y,
    f is not identically zero, and |f(x)| ≤ 1 for all x,
    then |g(x)| ≤ 1 for all x. -/
theorem function_bound (f g : ℝ → ℝ)
    (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
    (h2 : ∃ x, f x ≠ 0)
    (h3 : ∀ x, |f x| ≤ 1) :
    ∀ x, |g x| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l3205_320594


namespace NUMINAMATH_CALUDE_vasya_always_wins_l3205_320543

/-- Represents a point on the circle -/
structure Point where
  index : Fin 99

/-- Represents a color (Red or Blue) -/
inductive Color
| Red
| Blue

/-- Represents the state of the game -/
structure GameState where
  coloredPoints : Point → Option Color

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  (p2.index - p1.index) % 33 = 0 ∧
  (p3.index - p2.index) % 33 = 0 ∧
  (p1.index - p3.index) % 33 = 0

/-- Checks if a winning condition is met -/
def isWinningState (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Point) (c : Color),
    isEquilateralTriangle p1 p2 p3 ∧
    state.coloredPoints p1 = some c ∧
    state.coloredPoints p2 = some c ∧
    state.coloredPoints p3 = some c

/-- The main theorem stating that Vasya always has a winning strategy -/
theorem vasya_always_wins :
  ∀ (initialState : GameState),
  ∃ (finalState : GameState),
    (∀ p, Option.isSome (finalState.coloredPoints p)) ∧
    isWinningState finalState :=
sorry

end NUMINAMATH_CALUDE_vasya_always_wins_l3205_320543
