import Mathlib

namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3493_349396

theorem inequality_and_equality_condition (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  ((1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b)) ↔ (a = b ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3493_349396


namespace NUMINAMATH_CALUDE_negation_of_p_l3493_349345

-- Define the proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

-- State the theorem
theorem negation_of_p (f : ℝ → ℝ) : 
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_p_l3493_349345


namespace NUMINAMATH_CALUDE_line_through_M_and_origin_parallel_line_perpendicular_line_main_theorem_l3493_349322

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3*x + 4*y + 5 = 0
def l₂ (x y : ℝ) : Prop := 2*x - 3*y - 8 = 0
def l₃ (x y : ℝ) : Prop := 2*x + y + 5 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, -2)

-- Theorem for the line passing through M and the origin
theorem line_through_M_and_origin :
  ∃ (k : ℝ), ∀ (x y : ℝ), l₁ x y ∧ l₂ x y → (y = k * x) ∧ k = -2 :=
sorry

-- Theorem for the parallel line
theorem parallel_line :
  ∃ (t : ℝ), ∀ (x y : ℝ), l₁ (M.1) (M.2) ∧ l₂ (M.1) (M.2) →
    (2*x + y + t = 0) ∧ t = 0 :=
sorry

-- Theorem for the perpendicular line
theorem perpendicular_line :
  ∃ (s : ℝ), ∀ (x y : ℝ), l₁ (M.1) (M.2) ∧ l₂ (M.1) (M.2) →
    (x - 2*y + s = 0) ∧ s = -5 :=
sorry

-- Main theorem combining all conditions
theorem main_theorem :
  (∀ (x y : ℝ), l₁ x y ∧ l₂ x y → 2*x + y = 0) ∧
  (∀ (x y : ℝ), l₁ x y ∧ l₂ x y → x - 2*y - 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_M_and_origin_parallel_line_perpendicular_line_main_theorem_l3493_349322


namespace NUMINAMATH_CALUDE_number_problem_l3493_349313

theorem number_problem (x : ℤ) : x + 12 - 27 = 24 → x = 39 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3493_349313


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l3493_349309

theorem coefficient_x_cubed_in_expansion : 
  let expansion := (fun x => (2 * x + 1) * (x - 1)^5)
  ∃ a b c d e f, ∀ x, 
    expansion x = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f ∧ 
    c = -10 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l3493_349309


namespace NUMINAMATH_CALUDE_quadratic_monotone_decreasing_m_range_l3493_349369

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 1

-- State the theorem
theorem quadratic_monotone_decreasing_m_range :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 1 → f m x₁ > f m x₂) →
  m ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotone_decreasing_m_range_l3493_349369


namespace NUMINAMATH_CALUDE_suit_price_calculation_l3493_349304

def original_price : ℚ := 200
def increase_rate : ℚ := 0.30
def discount_rate : ℚ := 0.30
def tax_rate : ℚ := 0.07

def increased_price : ℚ := original_price * (1 + increase_rate)
def discounted_price : ℚ := increased_price * (1 - discount_rate)
def final_price : ℚ := discounted_price * (1 + tax_rate)

theorem suit_price_calculation :
  final_price = 194.74 := by sorry

end NUMINAMATH_CALUDE_suit_price_calculation_l3493_349304


namespace NUMINAMATH_CALUDE_clothing_distribution_l3493_349318

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (remaining_loads : ℕ) 
  (h1 : total = 47)
  (h2 : first_load = 17)
  (h3 : remaining_loads = 5)
  : (total - first_load) / remaining_loads = 6 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l3493_349318


namespace NUMINAMATH_CALUDE_mia_egg_decoration_rate_l3493_349391

/-- Mia's egg decoration problem -/
theorem mia_egg_decoration_rate
  (billy_rate : ℕ)
  (total_eggs : ℕ)
  (total_time : ℕ)
  (h1 : billy_rate = 10)
  (h2 : total_eggs = 170)
  (h3 : total_time = 5)
  : ∃ (mia_rate : ℕ), mia_rate = 24 ∧ mia_rate + billy_rate = total_eggs / total_time :=
by sorry

end NUMINAMATH_CALUDE_mia_egg_decoration_rate_l3493_349391


namespace NUMINAMATH_CALUDE_cos_20_minus_cos_40_l3493_349359

theorem cos_20_minus_cos_40 : Real.cos (20 * π / 180) - Real.cos (40 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_20_minus_cos_40_l3493_349359


namespace NUMINAMATH_CALUDE_sum_of_squares_l3493_349332

theorem sum_of_squares (x y z : ℝ) 
  (h1 : x^2 - 6*y = 10)
  (h2 : y^2 - 8*z = -18)
  (h3 : z^2 - 10*x = -40) :
  x^2 + y^2 + z^2 = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3493_349332


namespace NUMINAMATH_CALUDE_three_x_squared_y_squared_l3493_349342

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 
  3*x^2*y^2 = 588 := by
  sorry

end NUMINAMATH_CALUDE_three_x_squared_y_squared_l3493_349342


namespace NUMINAMATH_CALUDE_visible_sides_is_seventeen_l3493_349389

/-- Represents a polygon with a given number of sides. -/
structure Polygon where
  sides : Nat
  sides_positive : sides > 0

/-- The configuration of polygons in the problem. -/
def polygon_configuration : List Polygon :=
  [⟨4, by norm_num⟩, ⟨3, by norm_num⟩, ⟨5, by norm_num⟩, ⟨6, by norm_num⟩, ⟨7, by norm_num⟩]

/-- Calculates the number of visible sides in the configuration. -/
def visible_sides (config : List Polygon) : Nat :=
  (config.map (·.sides)).sum - 2 * (config.length - 1)

/-- Theorem stating that the number of visible sides in the given configuration is 17. -/
theorem visible_sides_is_seventeen :
  visible_sides polygon_configuration = 17 := by
  sorry

#eval visible_sides polygon_configuration

end NUMINAMATH_CALUDE_visible_sides_is_seventeen_l3493_349389


namespace NUMINAMATH_CALUDE_ceiling_product_equation_solution_l3493_349357

theorem ceiling_product_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ (⌈x⌉ : ℝ) * x = 210 ∧ x = 14 := by sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_solution_l3493_349357


namespace NUMINAMATH_CALUDE_division_remainder_l3493_349327

/-- The divisor polynomial -/
def divisor (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

/-- The dividend polynomial -/
def dividend (x : ℂ) : ℂ := x^60 + x^45 + x^30 + x^15 + 1

/-- Theorem stating that the remainder of the division is 5 -/
theorem division_remainder : ∃ (q : ℂ → ℂ), ∀ (x : ℂ), 
  dividend x = (divisor x) * (q x) + 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3493_349327


namespace NUMINAMATH_CALUDE_bears_distribution_l3493_349337

def bears_per_shelf (initial_stock new_shipment num_shelves : ℕ) : ℕ :=
  (initial_stock + new_shipment) / num_shelves

theorem bears_distribution (initial_stock new_shipment num_shelves : ℕ) 
  (h1 : initial_stock = 17)
  (h2 : new_shipment = 10)
  (h3 : num_shelves = 3) :
  bears_per_shelf initial_stock new_shipment num_shelves = 9 := by
  sorry

end NUMINAMATH_CALUDE_bears_distribution_l3493_349337


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3493_349387

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → (x, y) ∈ foci ∨ (∀ f ∈ foci, (x - f.1)^2 + y^2 > 0)) ∧
  (∀ x y, hyperbola x y → 
    let a := 2  -- sqrt(4)
    let c := 4  -- distance from center to focus
    c / a = eccentricity) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3493_349387


namespace NUMINAMATH_CALUDE_students_passed_both_tests_l3493_349381

theorem students_passed_both_tests
  (total_students : ℕ)
  (passed_long_jump : ℕ)
  (passed_shot_put : ℕ)
  (failed_both : ℕ)
  (h1 : total_students = 50)
  (h2 : passed_long_jump = 40)
  (h3 : passed_shot_put = 31)
  (h4 : failed_both = 4) :
  total_students - failed_both = passed_long_jump + passed_shot_put - (passed_long_jump + passed_shot_put - (total_students - failed_both)) :=
by sorry

end NUMINAMATH_CALUDE_students_passed_both_tests_l3493_349381


namespace NUMINAMATH_CALUDE_complex_magnitude_l3493_349328

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 10)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = 6 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3493_349328


namespace NUMINAMATH_CALUDE_reciprocal_sum_greater_than_four_l3493_349307

theorem reciprocal_sum_greater_than_four 
  (a b c : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (sum_of_squares : a^2 + b^2 + c^2 = 1) : 
  1/a + 1/b + 1/c > 4 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_greater_than_four_l3493_349307


namespace NUMINAMATH_CALUDE_man_brother_age_difference_l3493_349353

/-- Represents the age difference between a man and his brother -/
def ageDifference (manAge brotherAge : ℕ) : ℕ := manAge - brotherAge

/-- The problem statement -/
theorem man_brother_age_difference :
  ∀ (manAge brotherAge : ℕ),
    brotherAge = 10 →
    manAge > brotherAge →
    manAge + 2 = 2 * (brotherAge + 2) →
    ageDifference manAge brotherAge = 12 := by
  sorry

end NUMINAMATH_CALUDE_man_brother_age_difference_l3493_349353


namespace NUMINAMATH_CALUDE_conference_married_men_fraction_l3493_349301

theorem conference_married_men_fraction
  (total_women : ℕ)
  (single_women : ℕ)
  (married_women : ℕ)
  (married_men : ℕ)
  (h1 : single_women + married_women = total_women)
  (h2 : married_women = married_men)
  (h3 : (single_women : ℚ) / total_women = 3 / 7) :
  (married_men : ℚ) / (total_women + married_men) = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_conference_married_men_fraction_l3493_349301


namespace NUMINAMATH_CALUDE_banana_permutations_l3493_349323

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2

theorem banana_permutations :
  (word_length.factorial) / (a_count.factorial * n_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_l3493_349323


namespace NUMINAMATH_CALUDE_arjun_initial_investment_l3493_349321

/-- Represents the investment details of a partner in the business --/
structure Investment where
  amount : ℝ
  duration : ℝ

/-- Calculates the share of a partner based on their investment and duration --/
def calculateShare (inv : Investment) : ℝ :=
  inv.amount * inv.duration

/-- Proves that Arjun's initial investment was 2000 given the problem conditions --/
theorem arjun_initial_investment 
  (arjun : Investment)
  (anoop : Investment)
  (h1 : arjun.duration = 12)
  (h2 : anoop.amount = 4000)
  (h3 : anoop.duration = 6)
  (h4 : calculateShare arjun = calculateShare anoop) : 
  arjun.amount = 2000 := by
  sorry

#check arjun_initial_investment

end NUMINAMATH_CALUDE_arjun_initial_investment_l3493_349321


namespace NUMINAMATH_CALUDE_polygon_with_120_degree_interior_angles_has_6_sides_l3493_349384

theorem polygon_with_120_degree_interior_angles_has_6_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    interior_angle = 120 →
    (n : ℝ) * (180 - interior_angle) = 360 →
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_120_degree_interior_angles_has_6_sides_l3493_349384


namespace NUMINAMATH_CALUDE_product_of_numbers_l3493_349371

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 21) (sum_squares_eq : x^2 + y^2 = 527) :
  x * y = -43 := by sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3493_349371


namespace NUMINAMATH_CALUDE_steven_erasers_count_l3493_349390

/-- The number of skittles Steven has -/
def skittles : ℕ := 4502

/-- The number of groups the items are organized into -/
def groups : ℕ := 154

/-- The number of items in each group -/
def items_per_group : ℕ := 57

/-- The total number of items (skittles and erasers) -/
def total_items : ℕ := groups * items_per_group

/-- The number of erasers Steven has -/
def erasers : ℕ := total_items - skittles

theorem steven_erasers_count : erasers = 4276 := by
  sorry

end NUMINAMATH_CALUDE_steven_erasers_count_l3493_349390


namespace NUMINAMATH_CALUDE_student_rank_problem_l3493_349393

/-- Given a group of students, calculates the rank from left based on the rank from right -/
def rankFromLeft (totalStudents : ℕ) (rankFromRight : ℕ) : ℕ :=
  totalStudents - rankFromRight + 1

/-- Theorem stating that in a group of 20 students, 
    if a student is ranked 13th from the right, 
    then their rank from the left is 9th -/
theorem student_rank_problem :
  rankFromLeft 20 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_problem_l3493_349393


namespace NUMINAMATH_CALUDE_sum_equation_l3493_349343

theorem sum_equation : 27474 + 3699 + 1985 - 2047 = 31111 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_l3493_349343


namespace NUMINAMATH_CALUDE_total_average_marks_l3493_349398

theorem total_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 ∧ n2 > 0 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = 
    ((n1 : ℚ) + (n2 : ℚ)) * ((n1 : ℚ) * avg1 + (n2 : ℚ) * avg2) / ((n1 : ℚ) + (n2 : ℚ)) :=
by
  sorry

#eval ((45 : ℚ) * 39 + (70 : ℚ) * 35) / ((39 : ℚ) + (35 : ℚ))

end NUMINAMATH_CALUDE_total_average_marks_l3493_349398


namespace NUMINAMATH_CALUDE_song_length_proof_l3493_349340

/-- Proves that given the conditions, each song on the album is 3.5 minutes long -/
theorem song_length_proof 
  (jumps_per_second : ℕ) 
  (total_songs : ℕ) 
  (total_jumps : ℕ) 
  (h1 : jumps_per_second = 1)
  (h2 : total_songs = 10)
  (h3 : total_jumps = 2100) :
  (total_jumps : ℚ) / (jumps_per_second * 60 * total_songs) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_song_length_proof_l3493_349340


namespace NUMINAMATH_CALUDE_distance_traveled_l3493_349374

/-- Given a speed of 65 km/hr and a time of 3 hr, the distance traveled is 195 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 65) (h2 : time = 3) :
  speed * time = 195 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_l3493_349374


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3493_349386

/-- Given a triangle ABC with vertices A(4,0), B(8,10), and C(0,6),
    the equation of the line passing through A and parallel to BC is x - 2y - 4 = 0 -/
theorem parallel_line_equation (A B C : ℝ × ℝ) : 
  A = (4, 0) → B = (8, 10) → C = (0, 6) → 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (x = 4 ∧ y = 0) ∨ (y - 0 = m * (x - 4)) ↔ x - 2*y - 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3493_349386


namespace NUMINAMATH_CALUDE_curve_k_values_l3493_349335

-- Define the curve equation
def curve_equation (x y k : ℝ) : Prop :=
  5 * x^2 - k * y^2 = 5

-- Define the focal length
def focal_length : ℝ := 4

-- Theorem statement
theorem curve_k_values :
  ∃ k : ℝ, (k = 5/3 ∨ k = -1) ∧
  ∀ x y : ℝ, curve_equation x y k ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ curve_equation x y k) ∧
    (max a b - min a b) / 2 = focal_length) :=
sorry

end NUMINAMATH_CALUDE_curve_k_values_l3493_349335


namespace NUMINAMATH_CALUDE_student_multiplication_factor_l3493_349325

theorem student_multiplication_factor : ∃ (x : ℚ), 121 * x - 138 = 104 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_factor_l3493_349325


namespace NUMINAMATH_CALUDE_expression_simplification_l3493_349331

theorem expression_simplification (x : ℝ) (h : x^2 - 2*x - 2 = 0) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2*x^2 - x) / (x^2 + 2*x + 1)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3493_349331


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_40_l3493_349338

theorem smallest_four_digit_divisible_by_40 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 40 = 0 → n ≥ 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_40_l3493_349338


namespace NUMINAMATH_CALUDE_unique_solution_l3493_349329

/-- The # operation as defined in the problem -/
def hash (a b : ℝ) : ℝ := a * b - 2 * a - 2 * b + 6

/-- Statement of the problem -/
theorem unique_solution : ∃! (x : ℝ), x > 0 ∧ hash (hash x 7) x = 82 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3493_349329


namespace NUMINAMATH_CALUDE_circle_in_rectangle_ratio_l3493_349336

theorem circle_in_rectangle_ratio (r s : ℝ) (h1 : r > 0) (h2 : s > 0) : 
  (π * r^2 = 2 * r * s - π * r^2) → (s / (2 * r) = π / 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_in_rectangle_ratio_l3493_349336


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_product_l3493_349375

theorem consecutive_even_numbers_product (x : ℤ) : 
  (x % 2 = 0) →  -- x is even
  ((x + 2) % 2 = 0) →  -- x + 2 is even (consecutive even number)
  (x * (x + 2) = 224) →  -- their product is 224
  x * (x + 2) = 224 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_product_l3493_349375


namespace NUMINAMATH_CALUDE_prob_four_correct_zero_l3493_349358

/-- Represents the number of people and letters -/
def n : ℕ := 5

/-- The probability of exactly (n-1) people receiving their correct letter
    in a random distribution of n letters to n people -/
def prob_n_minus_one_correct (n : ℕ) : ℝ := 
  if n ≥ 2 then 0 else 1

/-- Theorem stating that the probability of exactly 4 out of 5 people
    receiving their correct letter is 0 -/
theorem prob_four_correct_zero : 
  prob_n_minus_one_correct n = 0 := by sorry

end NUMINAMATH_CALUDE_prob_four_correct_zero_l3493_349358


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3493_349392

theorem tan_alpha_value (α : ℝ) (h : Real.tan (α - π/4) = 1/5) : 
  Real.tan α = 3/2 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3493_349392


namespace NUMINAMATH_CALUDE_smallest_special_is_correct_l3493_349361

/-- A natural number is special if it uses exactly four different digits in its decimal representation -/
def is_special (n : ℕ) : Prop :=
  (n.digits 10).toFinset.card = 4

/-- The smallest special number greater than 3429 -/
def smallest_special : ℕ := 3450

theorem smallest_special_is_correct :
  is_special smallest_special ∧
  smallest_special > 3429 ∧
  ∀ m : ℕ, m > 3429 → is_special m → m ≥ smallest_special :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_is_correct_l3493_349361


namespace NUMINAMATH_CALUDE_cuboid_diagonal_l3493_349388

/-- Given a cuboid with dimensions a, b, and c (in cm), prove that if its
    total surface area is 20 cm² and the sum of all edge lengths is 24 cm,
    then the length of its diagonal is 4 cm. -/
theorem cuboid_diagonal (a b c : ℝ) : 
  (2 * (a * b + b * c + a * c) = 20) →
  (4 * (a + b + c) = 24) →
  Real.sqrt (a^2 + b^2 + c^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_l3493_349388


namespace NUMINAMATH_CALUDE_probability_calculation_l3493_349317

/-- The number of volunteers -/
def num_volunteers : ℕ := 5

/-- The number of venues -/
def num_venues : ℕ := 3

/-- The total number of ways to assign volunteers to venues -/
def total_assignments : ℕ := num_venues ^ num_volunteers

/-- The number of favorable assignments (where each venue has at least one volunteer) -/
def favorable_assignments : ℕ := 150

/-- The probability that each venue has at least one volunteer -/
def probability_all_venues_covered : ℚ := favorable_assignments / total_assignments

theorem probability_calculation :
  probability_all_venues_covered = 50 / 81 :=
sorry

end NUMINAMATH_CALUDE_probability_calculation_l3493_349317


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l3493_349308

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^3 + 1/x^3 = 332 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l3493_349308


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l3493_349383

theorem sqrt_sum_equals_seven (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l3493_349383


namespace NUMINAMATH_CALUDE_second_car_speed_l3493_349367

/-- Given two cars starting from opposite ends of a 333-mile highway at the same time,
    with one car traveling at 54 mph and both cars meeting after 3 hours,
    prove that the speed of the second car is 57 mph. -/
theorem second_car_speed (highway_length : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  highway_length = 333 →
  time = 3 →
  speed1 = 54 →
  speed1 * time + speed2 * time = highway_length →
  speed2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l3493_349367


namespace NUMINAMATH_CALUDE_non_zero_vector_positive_norm_l3493_349334

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem non_zero_vector_positive_norm (a b : V) 
  (h_a : a ≠ 0) (h_b : ‖b‖ = 1) : 
  ‖a‖ > 0 := by sorry

end NUMINAMATH_CALUDE_non_zero_vector_positive_norm_l3493_349334


namespace NUMINAMATH_CALUDE_min_packs_for_126_cans_l3493_349363

/-- Represents the number of cans in each pack size --/
inductive PackSize
| small : PackSize
| medium : PackSize
| large : PackSize

/-- Returns the number of cans for a given pack size --/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | PackSize.small => 15
  | PackSize.medium => 18
  | PackSize.large => 36

/-- Represents a combination of packs --/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination --/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a combination --/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- Defines what it means for a pack combination to be valid --/
def isValidCombination (c : PackCombination) : Prop :=
  totalCans c = 126

/-- Theorem: The minimum number of packs needed to buy exactly 126 cans is 4 --/
theorem min_packs_for_126_cans :
  ∃ (c : PackCombination), isValidCombination c ∧
    totalPacks c = 4 ∧
    (∀ (c' : PackCombination), isValidCombination c' → totalPacks c ≤ totalPacks c') :=
sorry

end NUMINAMATH_CALUDE_min_packs_for_126_cans_l3493_349363


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3493_349314

theorem quadratic_inequality (x : ℝ) : x^2 - 9*x + 14 < 0 ↔ 2 < x ∧ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3493_349314


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3493_349379

theorem reciprocal_of_negative_fraction :
  ((-1 : ℚ) / 2011)⁻¹ = -2011 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3493_349379


namespace NUMINAMATH_CALUDE_not_all_same_color_probability_l3493_349330

def num_people : ℕ := 3
def num_colors : ℕ := 5

theorem not_all_same_color_probability :
  (num_colors^num_people - num_colors) / num_colors^num_people = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_not_all_same_color_probability_l3493_349330


namespace NUMINAMATH_CALUDE_total_distance_is_1734_l3493_349320

/-- The number of trees in the row -/
def num_trees : ℕ := 18

/-- The interval between adjacent trees in meters -/
def tree_interval : ℕ := 3

/-- Calculate the total distance walked to water all trees -/
def total_distance : ℕ :=
  -- Sum of distances for each tree
  (Finset.range num_trees).sum (fun i => 2 * i * tree_interval)

/-- Theorem stating the total distance walked -/
theorem total_distance_is_1734 : total_distance = 1734 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_1734_l3493_349320


namespace NUMINAMATH_CALUDE_opposite_sign_roots_l3493_349316

theorem opposite_sign_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - (a + 3) * x + 2 = 0 ∧ 
               a * y^2 - (a + 3) * y + 2 = 0 ∧ 
               x * y < 0) ↔ 
  a < 0 :=
sorry

end NUMINAMATH_CALUDE_opposite_sign_roots_l3493_349316


namespace NUMINAMATH_CALUDE_symmetry_implies_coordinates_l3493_349385

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

theorem symmetry_implies_coordinates : 
  ∀ (a b : ℝ), 
  symmetric_wrt_origin (a, 1) (5, b) → 
  a = -5 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_symmetry_implies_coordinates_l3493_349385


namespace NUMINAMATH_CALUDE_black_pens_removed_l3493_349364

/-- Proves that 7 black pens were removed from a jar given the initial and final conditions -/
theorem black_pens_removed (initial_blue : ℕ) (initial_black : ℕ) (initial_red : ℕ)
  (blue_removed : ℕ) (final_count : ℕ)
  (h1 : initial_blue = 9)
  (h2 : initial_black = 21)
  (h3 : initial_red = 6)
  (h4 : blue_removed = 4)
  (h5 : final_count = 25) :
  initial_black - (initial_blue + initial_black + initial_red - blue_removed - final_count) = 7 := by
  sorry

#check black_pens_removed

end NUMINAMATH_CALUDE_black_pens_removed_l3493_349364


namespace NUMINAMATH_CALUDE_tuesday_income_l3493_349366

/-- Calculates Lauren's income from her social media channel --/
def laurens_income (commercial_rate : ℚ) (subscription_rate : ℚ) (commercials_viewed : ℕ) (new_subscribers : ℕ) : ℚ :=
  commercial_rate * commercials_viewed + subscription_rate * new_subscribers

/-- Proves that Lauren's income on Tuesday is $77.00 --/
theorem tuesday_income : 
  laurens_income (1/2) 1 100 27 = 77 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_income_l3493_349366


namespace NUMINAMATH_CALUDE_trig_product_value_l3493_349376

theorem trig_product_value : 
  Real.sin (4/3 * Real.pi) * Real.cos (5/6 * Real.pi) * Real.tan (-4/3 * Real.pi) = -3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_value_l3493_349376


namespace NUMINAMATH_CALUDE_set_union_problem_l3493_349326

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {2, 3} →
  B = {1, a} →
  A ∩ B = {2} →
  A ∪ B = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l3493_349326


namespace NUMINAMATH_CALUDE_min_value_ab_l3493_349372

theorem min_value_ab (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 4/b = Real.sqrt (a*b)) : 
  a * b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 4/b₀ = Real.sqrt (a₀*b₀) ∧ a₀ * b₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_ab_l3493_349372


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3493_349394

def A : Set ℝ := {x | x^2 + 2*x = 0}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3493_349394


namespace NUMINAMATH_CALUDE_sqrt_equals_self_l3493_349399

theorem sqrt_equals_self (x : ℝ) : Real.sqrt x = x ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equals_self_l3493_349399


namespace NUMINAMATH_CALUDE_nested_average_equality_l3493_349354

def avg_pair (a b : ℚ) : ℚ := (a + b) / 2

def avg_quad (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem nested_average_equality : 
  avg_quad 
    (avg_quad (avg_pair 2 4) (avg_pair 1 3) (avg_pair 0 2) (avg_pair 1 1))
    (avg_pair 3 3)
    (avg_pair 2 2)
    (avg_pair 4 0) = 35 / 16 := by
  sorry

end NUMINAMATH_CALUDE_nested_average_equality_l3493_349354


namespace NUMINAMATH_CALUDE_a_range_if_increasing_l3493_349300

/-- The sequence defined by a_n = an^2 + n -/
def a_seq (a : ℝ) (n : ℕ) : ℝ := a * n^2 + n

/-- The theorem stating that if the sequence is increasing, then a is non-negative -/
theorem a_range_if_increasing (a : ℝ) :
  (∀ n : ℕ, a_seq a n < a_seq a (n + 1)) → a ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_a_range_if_increasing_l3493_349300


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l3493_349315

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  height : ℝ
  is_isosceles : True

/-- Represents a geometric solid -/
inductive Solid
  | Cylinder
  | Cone
  | Frustum

/-- The result of rotating an isosceles trapezoid -/
def rotate_isosceles_trapezoid (t : IsoscelesTrapezoid) : List Solid :=
  sorry

/-- Theorem stating that rotating an isosceles trapezoid around its longer base
    results in one cylinder and two cones -/
theorem isosceles_trapezoid_rotation 
  (t : IsoscelesTrapezoid) : 
  rotate_isosceles_trapezoid t = [Solid.Cylinder, Solid.Cone, Solid.Cone] :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l3493_349315


namespace NUMINAMATH_CALUDE_max_word_ratio_bound_l3493_349310

/-- Represents a crossword on an n × n grid. -/
structure Crossword (n : ℕ) where
  cells : Set (Fin n × Fin n)
  nonempty : cells.Nonempty

/-- The number of words in a crossword. -/
def num_words (n : ℕ) (c : Crossword n) : ℕ := sorry

/-- The minimum number of words needed to cover a crossword. -/
def min_cover_words (n : ℕ) (c : Crossword n) : ℕ := sorry

/-- Theorem: The maximum ratio of words to minimum cover words is 1 + n/2 -/
theorem max_word_ratio_bound {n : ℕ} (hn : n ≥ 2) (c : Crossword n) :
  (num_words n c : ℚ) / (min_cover_words n c) ≤ 1 + n / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_word_ratio_bound_l3493_349310


namespace NUMINAMATH_CALUDE_star_sum_equals_396_l3493_349306

def star (a b : ℕ) : ℕ := a * a - b * b

theorem star_sum_equals_396 : 
  (List.range 18).foldl (λ acc i => acc + star (i + 3) (i + 2)) 0 = 396 := by
  sorry

end NUMINAMATH_CALUDE_star_sum_equals_396_l3493_349306


namespace NUMINAMATH_CALUDE_crouton_calories_l3493_349324

def salad_calories : ℕ := 350
def lettuce_calories : ℕ := 30
def cucumber_calories : ℕ := 80
def num_croutons : ℕ := 12

theorem crouton_calories : 
  (salad_calories - lettuce_calories - cucumber_calories) / num_croutons = 20 := by
  sorry

end NUMINAMATH_CALUDE_crouton_calories_l3493_349324


namespace NUMINAMATH_CALUDE_cos_sin_15_identity_l3493_349397

theorem cos_sin_15_identity : 
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 + 
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 
  (1 + 2 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_15_identity_l3493_349397


namespace NUMINAMATH_CALUDE_checkout_speed_ratio_l3493_349311

/-- Represents the problem of determining the ratio of cashier checkout speed to the rate of increase in waiting people. -/
theorem checkout_speed_ratio
  (n : ℕ)  -- Initial number of people in line
  (y : ℝ)  -- Rate at which number of people waiting increases (people per minute)
  (x : ℝ)  -- Cashier's checkout speed (people per minute)
  (h1 : 20 * 2 * x = 20 * y + n)  -- Equation for 2 counters open for 20 minutes
  (h2 : 12 * 3 * x = 12 * y + n)  -- Equation for 3 counters open for 12 minutes
  : x = 2 * y :=
sorry

end NUMINAMATH_CALUDE_checkout_speed_ratio_l3493_349311


namespace NUMINAMATH_CALUDE_sara_sent_nine_letters_in_february_l3493_349356

/-- The number of letters Sara sent in February -/
def letters_in_february : ℕ := 33 - (6 + 3 * 6)

/-- Proof that Sara sent 9 letters in February -/
theorem sara_sent_nine_letters_in_february :
  letters_in_february = 9 := by
  sorry

#eval letters_in_february

end NUMINAMATH_CALUDE_sara_sent_nine_letters_in_february_l3493_349356


namespace NUMINAMATH_CALUDE_combination_formula_l3493_349350

/-- The number of combinations of n things taken k at a time -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem combination_formula (n m : ℕ) (h : n ≥ m - 1) :
  binomial n (m - 1) = Nat.factorial n / (Nat.factorial (m - 1) * Nat.factorial (n - m + 1)) := by
  sorry

end NUMINAMATH_CALUDE_combination_formula_l3493_349350


namespace NUMINAMATH_CALUDE_range_and_minimum_l3493_349347

theorem range_and_minimum (x y a : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : x^2 - y^2 = 2)
  (h_ineq : (1 / (2*x^2)) + (2*y/x) < a) :
  (0 < y/x ∧ y/x < 1) ∧ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_and_minimum_l3493_349347


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3493_349349

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  parallel a b → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3493_349349


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3493_349339

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3493_349339


namespace NUMINAMATH_CALUDE_class_size_l3493_349373

/-- The number of students in Yuna's class -/
def total_students : ℕ := 33

/-- The number of students who like Korean -/
def korean_students : ℕ := 28

/-- The number of students who like math -/
def math_students : ℕ := 27

/-- The number of students who like both Korean and math -/
def both_subjects : ℕ := 22

/-- There is no student who does not like both Korean and math -/
axiom no_neither : total_students = korean_students + math_students - both_subjects

theorem class_size : total_students = 33 :=
sorry

end NUMINAMATH_CALUDE_class_size_l3493_349373


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3493_349365

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) -- Sequence of integers indexed by natural numbers
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- Arithmetic sequence condition
  (h_a1 : a 1 = -1) -- First term condition
  (h_a4 : a 4 = 8) -- Fourth term condition
  : ∃ d : ℤ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3493_349365


namespace NUMINAMATH_CALUDE_function_property_l3493_349360

-- Define the function f
variable (f : ℝ → ℝ)
-- Define the point a
variable (a : ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x < y → x < a → y < a → f x < f y)
variable (h2 : ∀ x, f (x + a) = f (a - x))
variable (x₁ x₂ : ℝ)
variable (h3 : x₁ < a ∧ a < x₂)
variable (h4 : |x₁ - a| < |x₂ - a|)

-- State the theorem
theorem function_property : f (2*a - x₁) > f (2*a - x₂) := by sorry

end NUMINAMATH_CALUDE_function_property_l3493_349360


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l3493_349302

theorem trigonometric_expression_value (α : Real) (h : α = -35 * Real.pi / 6) :
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.sin (3 * Real.pi / 2 + α)) /
  (1 + Real.sin α ^ 2 - Real.cos (Real.pi / 2 + α) - Real.cos (Real.pi + α) ^ 2) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l3493_349302


namespace NUMINAMATH_CALUDE_farm_horse_food_calculation_l3493_349344

/-- Calculates the total amount of horse food needed daily on a farm -/
theorem farm_horse_food_calculation 
  (num_sheep : ℕ) 
  (sheep_ratio horse_ratio : ℕ) 
  (food_per_horse : ℕ) 
  (h1 : num_sheep = 48) 
  (h2 : sheep_ratio = 6) 
  (h3 : horse_ratio = 7) 
  (h4 : food_per_horse = 230) : 
  (num_sheep / sheep_ratio) * horse_ratio * food_per_horse = 12880 := by
  sorry

end NUMINAMATH_CALUDE_farm_horse_food_calculation_l3493_349344


namespace NUMINAMATH_CALUDE_tangent_line_correct_l3493_349341

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * x

/-- The point through which the tangent line passes -/
def point : ℝ × ℝ := (1, 1)

/-- The equation of the proposed tangent line -/
def tangent_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_correct :
  (f point.1 = point.2) ∧ 
  (∀ x y : ℝ, tangent_line x y → y - point.2 = f' point.1 * (x - point.1)) ∧
  (∀ x : ℝ, x ≠ point.1 → f x ≠ (f' point.1) * (x - point.1) + point.2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_correct_l3493_349341


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3493_349370

/-- The area of an isosceles triangle with two sides of length 5 and base of length 6 is 12 -/
theorem isosceles_triangle_area : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 6 →
  (∃ (h : ℝ), h^2 = a^2 - (c/2)^2) →
  (1/2) * c * (a^2 - (c/2)^2).sqrt = 12 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l3493_349370


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3493_349303

/-- Given a hyperbola with the general equation y²/a² - x²/b² = 1 where a > 0 and b > 0,
    an asymptote equation of 3x + 4y = 0, and a focus at (0,5),
    prove that the specific equation of the hyperbola is y²/9 - x²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote : ∀ x y : ℝ, 3 * x + 4 * y = 0 → (y / x = -3 / 4 ∨ y / x = 3 / 4))
  (focus : (0 : ℝ) ^ 2 + 5 ^ 2 = (a ^ 2 + b ^ 2)) :
  ∀ x y : ℝ, y ^ 2 / 9 - x ^ 2 / 16 = 1 ↔ y ^ 2 / a ^ 2 - x ^ 2 / b ^ 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3493_349303


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3493_349305

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₆ = 6 and a₉ = 9, prove that a₃ = 3 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_a6 : a 6 = 6) 
  (h_a9 : a 9 = 9) : 
  a 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3493_349305


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3493_349380

/-- Two points are symmetric with respect to the y-axis if their y-coordinates are equal
    and their x-coordinates are opposite in sign and equal in magnitude. -/
def symmetric_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = y₂ ∧ x₁ = -x₂

/-- Given that point A(a,1) is symmetric to point A'(5,b) with respect to the y-axis,
    prove that a + b = -4. -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_y_axis a 1 5 b → a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3493_349380


namespace NUMINAMATH_CALUDE_inequality_proof_l3493_349378

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^6 - a^2 + 4) * (b^6 - b^2 + 4) * (c^6 - c^2 + 4) * (d^6 - d^2 + 4) ≥ (a + b + c + d)^4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3493_349378


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3493_349351

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ 
  (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3493_349351


namespace NUMINAMATH_CALUDE_brick_surface_area_l3493_349319

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm brick is 164 cm² -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l3493_349319


namespace NUMINAMATH_CALUDE_f_decreasing_after_seven_fourths_l3493_349333

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x < 1 then 2 * x^2 - x + 1
  else -2 * x^2 + 7 * x - 7

-- State the theorem
theorem f_decreasing_after_seven_fourths :
  (∀ x, f (x + 1) = -f (-(x + 1))) →
  (∀ x < 1, f x = 2 * x^2 - x + 1) →
  ∀ x > (7/4 : ℝ), ∀ y > x, f y < f x :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_after_seven_fourths_l3493_349333


namespace NUMINAMATH_CALUDE_cubic_roots_fourth_power_sum_l3493_349362

theorem cubic_roots_fourth_power_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 2*x^2 + 3*x - 4 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  a^4 + b^4 + c^4 = 18 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_fourth_power_sum_l3493_349362


namespace NUMINAMATH_CALUDE_fraction_simplification_l3493_349377

theorem fraction_simplification :
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3493_349377


namespace NUMINAMATH_CALUDE_line_slope_product_l3493_349348

/-- Given two lines L₁ and L₂ with equations y = mx and y = nx respectively,
    where L₁ makes twice as large of an angle with the horizontal as L₂,
    L₁ has 3 times the slope of L₂, and L₁ is not horizontal,
    then mn = 1. -/
theorem line_slope_product (m n : ℝ) (hm : m ≠ 0) :
  (∃ θ : ℝ, m = Real.tan (2 * θ) ∧ n = Real.tan θ) →
  m = 3 * n →
  m * n = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_product_l3493_349348


namespace NUMINAMATH_CALUDE_jake_weight_proof_l3493_349346

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 198

/-- Kendra's weight in pounds -/
def kendra_weight : ℝ := 95

/-- The sum of Jake's and Kendra's weights -/
def total_weight : ℝ := 293

theorem jake_weight_proof :
  (jake_weight - 8 = 2 * kendra_weight) ∧
  (jake_weight + kendra_weight = total_weight) →
  jake_weight = 198 := by
sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l3493_349346


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3493_349352

theorem simplify_and_evaluate (x : ℝ) (h : x = 5) :
  (x + 3) / (x^2 - 4) / (2 - (x + 1) / (x + 2)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3493_349352


namespace NUMINAMATH_CALUDE_simplify_expression_l3493_349395

theorem simplify_expression (x : ℝ) : 4*x + 9*x^2 + 8 - (5 - 4*x - 9*x^2) = 18*x^2 + 8*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3493_349395


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3493_349382

theorem cos_alpha_value (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.cos (α + Real.pi / 3) = -2/3) : 
  Real.cos α = (Real.sqrt 15 - 2) / 6 := by sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3493_349382


namespace NUMINAMATH_CALUDE_percentage_equivalence_l3493_349355

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equivalence_l3493_349355


namespace NUMINAMATH_CALUDE_paper_boutique_sales_l3493_349368

theorem paper_boutique_sales (notebook_sales : ℝ) (marker_sales : ℝ) (stapler_sales : ℝ)
  (h1 : notebook_sales = 25)
  (h2 : marker_sales = 40)
  (h3 : stapler_sales = 15)
  (h4 : notebook_sales + marker_sales + stapler_sales + (100 - notebook_sales - marker_sales - stapler_sales) = 100) :
  100 - notebook_sales - marker_sales = 35 := by
sorry

end NUMINAMATH_CALUDE_paper_boutique_sales_l3493_349368


namespace NUMINAMATH_CALUDE_four_solutions_l3493_349312

-- Define the piecewise function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else k * x + 2

-- Theorem statement
theorem four_solutions (k : ℝ) (h : k > 0) :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, |f k x| = 1 :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l3493_349312
