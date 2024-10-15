import Mathlib

namespace NUMINAMATH_CALUDE_stella_profit_is_25_l2356_235625

/-- Represents the profit Stella makes from her antique shop sales -/
def stellas_profit (num_dolls num_clocks num_glasses : ℕ) 
                   (price_doll price_clock price_glass : ℚ) 
                   (cost : ℚ) : ℚ :=
  num_dolls * price_doll + num_clocks * price_clock + num_glasses * price_glass - cost

/-- Theorem stating that Stella's profit is $25 given the specified conditions -/
theorem stella_profit_is_25 : 
  stellas_profit 3 2 5 5 15 4 40 = 25 := by
  sorry

end NUMINAMATH_CALUDE_stella_profit_is_25_l2356_235625


namespace NUMINAMATH_CALUDE_chorus_arrangement_l2356_235687

/-- The maximum number of chorus members that satisfies both arrangements -/
def max_chorus_members : ℕ := 300

/-- The number of columns in the rectangular formation -/
def n : ℕ := 15

/-- The side length of the square formation -/
def k : ℕ := 17

theorem chorus_arrangement :
  (∃ m : ℕ, m = max_chorus_members) ∧
  (∃ k : ℕ, max_chorus_members = k^2 + 11) ∧
  (max_chorus_members = n * (n + 5)) ∧
  (∀ m : ℕ, m > max_chorus_members →
    (¬∃ k : ℕ, m = k^2 + 11) ∨ (¬∃ n : ℕ, m = n * (n + 5))) :=
by sorry

#eval max_chorus_members
#eval n
#eval k

end NUMINAMATH_CALUDE_chorus_arrangement_l2356_235687


namespace NUMINAMATH_CALUDE_product_of_roots_roots_product_of_equation_l2356_235663

theorem product_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  f r₁ = 0 ∧ f r₂ = 0 → r₁ * r₂ = c / a :=
by sorry

theorem roots_product_of_equation :
  let f : ℝ → ℝ := λ x => x^2 + 14*x + 52
  let r₁ := (-14 + Real.sqrt (14^2 - 4*1*52)) / (2*1)
  let r₂ := (-14 - Real.sqrt (14^2 - 4*1*52)) / (2*1)
  f r₁ = 0 ∧ f r₂ = 0 → r₁ * r₂ = 48 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_roots_product_of_equation_l2356_235663


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2356_235617

/-- An arithmetic sequence with first four terms a, x, b, and 2x -/
structure ArithmeticSequence (α : Type) [LinearOrderedField α] where
  a : α
  x : α
  b : α
  arithmetic_property : x - a = 2 * x - b

theorem arithmetic_sequence_ratio 
  {α : Type} [LinearOrderedField α] (seq : ArithmeticSequence α) :
  seq.a / seq.b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2356_235617


namespace NUMINAMATH_CALUDE_arithmetic_sequence_exists_geometric_sequence_not_exists_l2356_235623

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral :=
  (a b c d : ℝ)
  (sum_opposite : a + c = 180 ∧ b + d = 180)
  (angle_bounds : 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 0 < c ∧ c < 180 ∧ 0 < d ∧ d < 180)

-- Theorem for arithmetic sequence
theorem arithmetic_sequence_exists (q : CyclicQuadrilateral) :
  ∃ (α d : ℝ), d ≠ 0 ∧
    q.a = α ∧ q.b = α + d ∧ q.c = α + 2*d ∧ q.d = α + 3*d :=
sorry

-- Theorem for geometric sequence
theorem geometric_sequence_not_exists (q : CyclicQuadrilateral) :
  ¬∃ (α r : ℝ), r ≠ 1 ∧ r > 0 ∧
    q.a = α ∧ q.b = α * r ∧ q.c = α * r^2 ∧ q.d = α * r^3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_exists_geometric_sequence_not_exists_l2356_235623


namespace NUMINAMATH_CALUDE_kg_to_lb_conversion_rate_l2356_235647

/-- Conversion rate from kilograms to pounds -/
def kg_to_lb_rate : ℝ := 2.2

/-- Initial weight in kilograms -/
def initial_weight_kg : ℝ := 80

/-- Weight loss in pounds per hour of exercise -/
def weight_loss_per_hour : ℝ := 1.5

/-- Hours of exercise per day -/
def exercise_hours_per_day : ℝ := 2

/-- Number of days of exercise -/
def exercise_days : ℝ := 14

/-- Final weight in pounds after exercise period -/
def final_weight_lb : ℝ := 134

theorem kg_to_lb_conversion_rate :
  kg_to_lb_rate * initial_weight_kg =
    final_weight_lb + weight_loss_per_hour * exercise_hours_per_day * exercise_days :=
by sorry

end NUMINAMATH_CALUDE_kg_to_lb_conversion_rate_l2356_235647


namespace NUMINAMATH_CALUDE_daisy_solution_l2356_235641

def daisy_problem (day1 day2 day3 day4 total : ℕ) : Prop :=
  day1 = 45 ∧
  day2 = day1 + 20 ∧
  day3 = 2 * day2 - 10 ∧
  day1 + day2 + day3 + day4 = total ∧
  total = 350

theorem daisy_solution :
  ∃ day1 day2 day3 day4 total, daisy_problem day1 day2 day3 day4 total ∧ day4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_daisy_solution_l2356_235641


namespace NUMINAMATH_CALUDE_partition_modular_sum_l2356_235696

theorem partition_modular_sum (p : ℕ) (h_prime : Nat.Prime p) (h_p_ge_5 : p ≥ 5) :
  ∀ (A B C : Set ℕ), 
    (A ∪ B ∪ C = Finset.range (p - 1)) →
    (A ∩ B = ∅) → (B ∩ C = ∅) → (A ∩ C = ∅) →
    ∃ (x y z : ℕ), x ∈ A ∧ y ∈ B ∧ z ∈ C ∧ (x + y) % p = z % p :=
by sorry

end NUMINAMATH_CALUDE_partition_modular_sum_l2356_235696


namespace NUMINAMATH_CALUDE_adult_average_age_l2356_235656

theorem adult_average_age
  (total_members : ℕ)
  (total_average_age : ℚ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (num_adults : ℕ)
  (girls_average_age : ℚ)
  (boys_average_age : ℚ)
  (h1 : total_members = 50)
  (h2 : total_average_age = 18)
  (h3 : num_girls = 25)
  (h4 : num_boys = 20)
  (h5 : num_adults = 5)
  (h6 : girls_average_age = 16)
  (h7 : boys_average_age = 17)
  (h8 : total_members = num_girls + num_boys + num_adults) :
  (total_members * total_average_age - num_girls * girls_average_age - num_boys * boys_average_age) / num_adults = 32 := by
  sorry

end NUMINAMATH_CALUDE_adult_average_age_l2356_235656


namespace NUMINAMATH_CALUDE_tiffany_pies_eaten_l2356_235677

theorem tiffany_pies_eaten (pies_per_day : ℕ) (days : ℕ) (cans_per_pie : ℕ) (remaining_cans : ℕ) : 
  pies_per_day = 3 → days = 11 → cans_per_pie = 2 → remaining_cans = 58 →
  (pies_per_day * days * cans_per_pie - remaining_cans) / cans_per_pie = 4 := by
sorry

end NUMINAMATH_CALUDE_tiffany_pies_eaten_l2356_235677


namespace NUMINAMATH_CALUDE_same_grade_percentage_l2356_235697

/-- Given a class of students who took two tests, this theorem proves
    the percentage of students who received the same grade on both tests. -/
theorem same_grade_percentage
  (total_students : ℕ)
  (same_grade_students : ℕ)
  (h1 : total_students = 30)
  (h2 : same_grade_students = 12) :
  (same_grade_students : ℚ) / total_students * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l2356_235697


namespace NUMINAMATH_CALUDE_square_area_error_l2356_235627

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * 1.04
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error_percentage := (calculated_area - actual_area) / actual_area * 100
  area_error_percentage = 8.16 := by
    sorry

end NUMINAMATH_CALUDE_square_area_error_l2356_235627


namespace NUMINAMATH_CALUDE_wage_increase_result_l2356_235659

/-- Calculates the new wage after a percentage increase -/
def new_wage (original_wage : ℝ) (percent_increase : ℝ) : ℝ :=
  original_wage * (1 + percent_increase)

/-- Theorem stating that a 50% increase on a $28 wage results in $42 -/
theorem wage_increase_result :
  new_wage 28 0.5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_result_l2356_235659


namespace NUMINAMATH_CALUDE_lulu_blueberry_pies_count_l2356_235608

/-- The number of blueberry pies Lulu baked -/
def lulu_blueberry_pies : ℕ := 73 - (13 + 10 + 8 + 16 + 12)

theorem lulu_blueberry_pies_count :
  lulu_blueberry_pies = 14 := by
  sorry

end NUMINAMATH_CALUDE_lulu_blueberry_pies_count_l2356_235608


namespace NUMINAMATH_CALUDE_airport_walk_probability_l2356_235658

/-- Represents an airport with a given number of gates and distance between adjacent gates -/
structure Airport where
  num_gates : ℕ
  distance_between_gates : ℕ

/-- Calculates the number of gate pairs within a given distance -/
def count_pairs_within_distance (a : Airport) (max_distance : ℕ) : ℕ :=
  sorry

/-- The probability of walking at most a given distance between two random gates -/
def probability_within_distance (a : Airport) (max_distance : ℕ) : ℚ :=
  sorry

theorem airport_walk_probability :
  let a : Airport := ⟨15, 90⟩
  probability_within_distance a 360 = 59 / 105 := by
  sorry

end NUMINAMATH_CALUDE_airport_walk_probability_l2356_235658


namespace NUMINAMATH_CALUDE_room_tiling_theorem_l2356_235618

/-- Calculates the number of tiles needed to cover a rectangular room with a border of larger tiles -/
def tilesNeeded (roomLength roomWidth borderTileSize innerTileSize : ℕ) : ℕ :=
  let borderTiles := 2 * (roomLength / borderTileSize + roomWidth / borderTileSize) - 4
  let innerLength := roomLength - 2 * borderTileSize
  let innerWidth := roomWidth - 2 * borderTileSize
  let innerTiles := (innerLength / innerTileSize) * (innerWidth / innerTileSize)
  borderTiles + innerTiles

/-- The theorem stating that 310 tiles are needed for the given room specifications -/
theorem room_tiling_theorem :
  tilesNeeded 24 18 2 1 = 310 := by
  sorry

end NUMINAMATH_CALUDE_room_tiling_theorem_l2356_235618


namespace NUMINAMATH_CALUDE_solve_equation_for_A_l2356_235685

theorem solve_equation_for_A : ∃ A : ℝ,
  (1 / ((5 / (1 + (24 / A))) - 5 / 9)) * (3 / (2 + (5 / 7))) / (2 / (3 + (3 / 4))) + 2.25 = 4 ∧ A = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_for_A_l2356_235685


namespace NUMINAMATH_CALUDE_trig_function_equality_l2356_235619

/-- Given two functions f and g defined on real numbers, prove that g(x) equals f(π/4 + x) for all real x. -/
theorem trig_function_equality (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = Real.sin (2 * x + π / 3))
  (hg : ∀ x, g x = Real.cos (2 * x + π / 3)) :
  ∀ x, g x = f (π / 4 + x) := by
  sorry

end NUMINAMATH_CALUDE_trig_function_equality_l2356_235619


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l2356_235628

/-- Parabola type representing y^2 = 4x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type --/
structure Line where
  passes_through : ℝ × ℝ → Prop

/-- Represents a point on the parabola --/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

theorem parabola_intersection_length 
  (p : Parabola) 
  (l : Line) 
  (A B : ParabolaPoint) 
  (h1 : p.equation = fun x y => y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : l.passes_through p.focus)
  (h4 : p.equation A.x A.y)
  (h5 : p.equation B.x B.y)
  (h6 : l.passes_through (A.x, A.y))
  (h7 : l.passes_through (B.x, B.y))
  (h8 : (A.x + B.x) / 2 = 3)
  : Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l2356_235628


namespace NUMINAMATH_CALUDE_line_AC_equation_circumcircle_equation_l2356_235611

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, 4)

-- Define the line l
def l (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the symmetry condition
def symmetric_about_l (p₁ p₂ : ℝ × ℝ) : Prop :=
  let m := (p₁.1 + p₂.1) / 2
  let n := (p₁.2 + p₂.2) / 2
  l m n

-- Define point C
def C : ℝ × ℝ := (-1, 3)

-- Theorem for the equation of line AC
theorem line_AC_equation (x y : ℝ) : x + y - 2 = 0 ↔ 
  (∃ t : ℝ, x = A.1 + t * (C.1 - A.1) ∧ y = A.2 + t * (C.2 - A.2)) :=
sorry

-- Theorem for the equation of the circumcircle
theorem circumcircle_equation (x y : ℝ) : 
  x^2 + y^2 - 3/2*x + 11/2*y - 17 = 0 ↔ 
  (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
  (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2 :=
sorry

end NUMINAMATH_CALUDE_line_AC_equation_circumcircle_equation_l2356_235611


namespace NUMINAMATH_CALUDE_rectangle_area_minus_hole_l2356_235643

def large_rect_length (x : ℝ) : ℝ := x^2 + 7
def large_rect_width (x : ℝ) : ℝ := x^2 + 5
def hole_rect_length (x : ℝ) : ℝ := 2*x^2 - 3
def hole_rect_width (x : ℝ) : ℝ := x^2 - 2

theorem rectangle_area_minus_hole (x : ℝ) :
  large_rect_length x * large_rect_width x - hole_rect_length x * hole_rect_width x
  = -x^4 + 19*x^2 + 29 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_minus_hole_l2356_235643


namespace NUMINAMATH_CALUDE_ratio_and_closest_whole_number_l2356_235699

theorem ratio_and_closest_whole_number : 
  let ratio := (10^2010 + 10^2013) / (10^2011 + 10^2014)
  ratio = 1/10 ∧ 
  ∀ n : ℤ, |ratio - (n : ℚ)| ≥ |ratio - 0| :=
by sorry

end NUMINAMATH_CALUDE_ratio_and_closest_whole_number_l2356_235699


namespace NUMINAMATH_CALUDE_omar_egg_rolls_l2356_235630

theorem omar_egg_rolls (karen_rolls : ℕ) (total_rolls : ℕ) (omar_rolls : ℕ) : 
  karen_rolls = 229 → total_rolls = 448 → omar_rolls = total_rolls - karen_rolls → omar_rolls = 219 := by
  sorry

end NUMINAMATH_CALUDE_omar_egg_rolls_l2356_235630


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2356_235686

/-- Given a quadratic function f(x) = x^2 - 2x + 3 defined on [0,m], 
    if its maximum value on this interval is 3 and its minimum value is 2, 
    then m is in the closed interval [1,2] -/
theorem quadratic_function_range (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + 3
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 0 m, f x = 2) →
  m ∈ Set.Icc 1 2 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_range_l2356_235686


namespace NUMINAMATH_CALUDE_truck_speed_calculation_l2356_235633

/-- The average speed of Truck X in miles per hour -/
def truck_x_speed : ℝ := 57

/-- The average speed of Truck Y in miles per hour -/
def truck_y_speed : ℝ := 63

/-- The initial distance Truck X is ahead of Truck Y in miles -/
def initial_distance : ℝ := 14

/-- The final distance Truck Y is ahead of Truck X in miles -/
def final_distance : ℝ := 4

/-- The time it takes for Truck Y to overtake Truck X in hours -/
def overtake_time : ℝ := 3

theorem truck_speed_calculation :
  truck_x_speed = (truck_y_speed * overtake_time - initial_distance - final_distance) / overtake_time :=
by
  sorry

#check truck_speed_calculation

end NUMINAMATH_CALUDE_truck_speed_calculation_l2356_235633


namespace NUMINAMATH_CALUDE_factorable_quadratic_b_eq_42_l2356_235638

/-- A quadratic expression that can be factored into two linear binomials with integer coefficients -/
structure FactorableQuadratic where
  b : ℤ
  factored : ∃ (d e f g : ℤ), 28 * x^2 + b * x + 14 = (d * x + e) * (f * x + g)

/-- Theorem stating that for a FactorableQuadratic, b must equal 42 -/
theorem factorable_quadratic_b_eq_42 (q : FactorableQuadratic) : q.b = 42 := by
  sorry

end NUMINAMATH_CALUDE_factorable_quadratic_b_eq_42_l2356_235638


namespace NUMINAMATH_CALUDE_initial_distance_between_cars_l2356_235682

theorem initial_distance_between_cars (speed_A speed_B time_to_overtake distance_ahead : ℝ) 
  (h1 : speed_A = 58)
  (h2 : speed_B = 50)
  (h3 : time_to_overtake = 4.75)
  (h4 : distance_ahead = 8) : 
  (speed_A - speed_B) * time_to_overtake = 30 + distance_ahead := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_between_cars_l2356_235682


namespace NUMINAMATH_CALUDE_square_of_binomial_l2356_235690

theorem square_of_binomial (a : ℚ) : 
  (∃ (r s : ℚ), ∀ (x : ℚ), a * x^2 + 15 * x + 16 = (r * x + s)^2) → 
  a = 225 / 64 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2356_235690


namespace NUMINAMATH_CALUDE_unique_solution_m_value_l2356_235653

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := 16 * x^2 + m * x + 4

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := m^2 - 4 * 16 * 4

-- Theorem statement
theorem unique_solution_m_value :
  ∃! m : ℝ, m > 0 ∧ (∃! x : ℝ, quadratic_equation m x = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_m_value_l2356_235653


namespace NUMINAMATH_CALUDE_equation_solution_pairs_l2356_235661

theorem equation_solution_pairs : 
  ∀ x y : ℕ+, x^(y : ℕ) - y^(x : ℕ) = 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_pairs_l2356_235661


namespace NUMINAMATH_CALUDE_exactly_two_false_l2356_235693

-- Define the types
def Quadrilateral : Type := sorry
def Square : Quadrilateral → Prop := sorry
def Rectangle : Quadrilateral → Prop := sorry

-- Define the propositions
def P1 : Prop := ∀ q : Quadrilateral, Square q → Rectangle q
def P2 : Prop := ∀ q : Quadrilateral, Rectangle q → Square q
def P3 : Prop := ∀ q : Quadrilateral, ¬(Square q) → ¬(Rectangle q)
def P4 : Prop := ∀ q : Quadrilateral, ¬(Rectangle q) → ¬(Square q)

-- The theorem to prove
theorem exactly_two_false : 
  (¬P1 ∧ ¬P2 ∧ P3 ∧ P4) ∨ 
  (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) ∨ 
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) ∨ 
  (P1 ∧ ¬P2 ∧ ¬P3 ∧ P4) ∨ 
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4) ∨ 
  (P1 ∧ P2 ∧ ¬P3 ∧ ¬P4) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_false_l2356_235693


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_l2356_235654

theorem consecutive_odd_sum (n : ℤ) : 
  (∃ k : ℤ, n = 2 * k + 1) →  -- n is odd
  (n + 2 = 9) →              -- middle number is 9
  (n + (n + 2) + (n + 4) - n = 20) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_l2356_235654


namespace NUMINAMATH_CALUDE_two_digit_numbers_product_gcd_l2356_235612

theorem two_digit_numbers_product_gcd (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 1728 ∧ 
  Nat.gcd a b = 12 →
  (a = 36 ∧ b = 48) ∨ (a = 48 ∧ b = 36) := by
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_product_gcd_l2356_235612


namespace NUMINAMATH_CALUDE_palindrome_power_sum_l2356_235605

/-- A function to check if a natural number is a palindrome in decimal representation -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The main theorem stating the condition for 2^n + 2^m + 1 to be a palindrome -/
theorem palindrome_power_sum (m n : ℕ) : 
  isPalindrome (2^n + 2^m + 1) ↔ m ≤ 9 ∨ n ≤ 9 := by sorry

end NUMINAMATH_CALUDE_palindrome_power_sum_l2356_235605


namespace NUMINAMATH_CALUDE_max_z_value_l2356_235648

theorem max_z_value (x y z : ℝ) (sum_eq : x + y + z = 5) (prod_eq : x*y + y*z + z*x = 3) :
  z ≤ 13/3 := by
  sorry

end NUMINAMATH_CALUDE_max_z_value_l2356_235648


namespace NUMINAMATH_CALUDE_problem_solution_l2356_235603

theorem problem_solution (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 16) : y = 64 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2356_235603


namespace NUMINAMATH_CALUDE_alphaBetaArrangementsCount_l2356_235600

/-- The number of distinct arrangements of 9 letters, where one letter appears 4 times
    and six other letters appear once each. -/
def alphaBetaArrangements : ℕ :=
  Nat.factorial 9 / (Nat.factorial 4 * (Nat.factorial 1)^6)

/-- Theorem stating that the number of distinct arrangements of letters in "alpha beta"
    under the given conditions is 15120. -/
theorem alphaBetaArrangementsCount : alphaBetaArrangements = 15120 := by
  sorry

end NUMINAMATH_CALUDE_alphaBetaArrangementsCount_l2356_235600


namespace NUMINAMATH_CALUDE_divisibility_by_six_divisibility_by_120_divisibility_by_48_divisibility_by_1152_not_always_divisible_by_720_l2356_235620

-- Part (a)
theorem divisibility_by_six (a b c : ℤ) (h : 6 ∣ (a + b + c)) : 6 ∣ (a^5 + b^3 + c) := by sorry

-- Part (b)
theorem divisibility_by_120 (n : ℤ) : 120 ∣ (n^5 - 5*n^3 + 4*n) := by sorry

-- Part (c)
theorem divisibility_by_48 (n : ℤ) (h : Odd n) : 48 ∣ (n^3 + 3*n^2 - n - 3) := by sorry

-- Part (d)
theorem divisibility_by_1152 (n : ℤ) (h : Odd n) : 1152 ∣ (n^8 - n^6 - n^4 + n^2) := by sorry

-- Part (e)
theorem not_always_divisible_by_720 : ∃ n : ℤ, ¬(720 ∣ (n*(n^2 - 1)*(n^2 - 4))) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_six_divisibility_by_120_divisibility_by_48_divisibility_by_1152_not_always_divisible_by_720_l2356_235620


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2356_235614

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) 
    and asymptote equations y = ±x, its eccentricity is √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∀ (x y : ℝ), (y = x ∨ y = -x) → (x^2 / a^2 - y^2 / b^2 = 1)) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2356_235614


namespace NUMINAMATH_CALUDE_cakes_served_yesterday_l2356_235640

theorem cakes_served_yesterday (lunch_today dinner_today total : ℕ) 
  (h1 : lunch_today = 5)
  (h2 : dinner_today = 6)
  (h3 : total = 14) :
  total - (lunch_today + dinner_today) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_yesterday_l2356_235640


namespace NUMINAMATH_CALUDE_coin_problem_l2356_235695

theorem coin_problem (x : ℚ) (h : x > 0) : 
  let lost := (2 : ℚ) / 3 * x
  let recovered := (3 : ℚ) / 4 * lost
  x - (x - lost + recovered) = x / 6 := by sorry

end NUMINAMATH_CALUDE_coin_problem_l2356_235695


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2356_235645

theorem complex_equation_solution (i : ℂ) (h_i : i^2 = -1) :
  ∃ z : ℂ, (2 + i) * z = 2 - i ∧ z = 3/5 - 4/5 * i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2356_235645


namespace NUMINAMATH_CALUDE_abs_x_minus_y_equals_four_l2356_235666

theorem abs_x_minus_y_equals_four (x y : ℝ) 
  (h1 : x^3 + y^3 = 26) 
  (h2 : x*y*(x+y) = -6) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_abs_x_minus_y_equals_four_l2356_235666


namespace NUMINAMATH_CALUDE_point_not_in_quadrants_III_IV_l2356_235621

theorem point_not_in_quadrants_III_IV (m : ℝ) : 
  let A : ℝ × ℝ := (m, m^2 + 1)
  ¬(A.1 ≤ 0 ∧ A.2 ≤ 0) ∧ ¬(A.1 ≥ 0 ∧ A.2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_quadrants_III_IV_l2356_235621


namespace NUMINAMATH_CALUDE_max_m_value_max_m_is_optimal_l2356_235650

-- Define the quadratic function
def f (x : ℝ) := x^2 - 4*x

-- State the theorem
theorem max_m_value :
  (∀ x ∈ Set.Ioo 0 1, f x ≥ m) → m ≤ -3 :=
by sorry

-- Define the maximum value of m
def max_m : ℝ := -3

-- Prove that this is indeed the maximum value
theorem max_m_is_optimal :
  (∀ x ∈ Set.Ioo 0 1, f x ≥ max_m) ∧
  ∀ ε > 0, ∃ x ∈ Set.Ioo 0 1, f x < max_m + ε :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_max_m_is_optimal_l2356_235650


namespace NUMINAMATH_CALUDE_log_equation_solution_l2356_235639

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 →
  x = 3^(10/3) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2356_235639


namespace NUMINAMATH_CALUDE_mary_nickels_l2356_235607

/-- The number of nickels Mary has after receiving some from her dad -/
def total_nickels (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Proof that Mary has 12 nickels after receiving 5 from her dad -/
theorem mary_nickels : total_nickels 7 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_l2356_235607


namespace NUMINAMATH_CALUDE_bank_profit_l2356_235698

/-- Bank's profit calculation -/
theorem bank_profit 
  (K : ℝ) (p p₁ : ℝ) (n : ℕ) 
  (h₁ : p₁ > p) 
  (h₂ : p > 0) 
  (h₃ : p₁ > 0) :
  K * ((1 + p₁ / 100) ^ n - (1 + p / 100) ^ n) = 
  K * ((1 + p₁ / 100) ^ n - (1 + p / 100) ^ n) :=
by sorry

end NUMINAMATH_CALUDE_bank_profit_l2356_235698


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2356_235652

theorem polynomial_evaluation (f : ℝ → ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, f x = (1 - 3*x) * (1 + x)^5) →
  (∀ x, f x = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + (1/3)*a₁ + (1/3^2)*a₂ + (1/3^3)*a₃ + (1/3^4)*a₄ + (1/3^5)*a₅ + (1/3^6)*a₆ = 0 := by
sorry


end NUMINAMATH_CALUDE_polynomial_evaluation_l2356_235652


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l2356_235673

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = r1 + r2

theorem tangent_circles_radius (r : ℝ) :
  r > 0 →
  externally_tangent (0, 0) (3, 0) 1 r →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l2356_235673


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l2356_235675

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l2356_235675


namespace NUMINAMATH_CALUDE_gcd_product_l2356_235681

theorem gcd_product (a b a' b' : ℕ+) (d d' : ℕ+) 
  (h1 : d = Nat.gcd a b) (h2 : d' = Nat.gcd a' b') : 
  Nat.gcd (a * a') (Nat.gcd (a * b') (Nat.gcd (b * a') (b * b'))) = d * d' := by
  sorry

end NUMINAMATH_CALUDE_gcd_product_l2356_235681


namespace NUMINAMATH_CALUDE_infinite_congruent_sum_digits_l2356_235644

/-- Sum of digits function -/
def S (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the existence of infinitely many n such that S(n) ≡ n (mod p) for any prime p -/
theorem infinite_congruent_sum_digits (p : ℕ) (hp : Nat.Prime p) :
  ∃ (f : ℕ → ℕ), StrictMono f ∧ ∀ (k : ℕ), S (f k) ≡ f k [MOD p] :=
sorry

end NUMINAMATH_CALUDE_infinite_congruent_sum_digits_l2356_235644


namespace NUMINAMATH_CALUDE_rectangle_area_l2356_235646

/-- The area of a rectangle with length 2x and width 2x-1 is 4x^2 - 2x -/
theorem rectangle_area (x : ℝ) : 
  let length : ℝ := 2 * x
  let width : ℝ := 2 * x - 1
  length * width = 4 * x^2 - 2 * x := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2356_235646


namespace NUMINAMATH_CALUDE_condition_implication_l2356_235637

theorem condition_implication (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (p → ¬q) ∧ ¬(¬q → p) := by
sorry

end NUMINAMATH_CALUDE_condition_implication_l2356_235637


namespace NUMINAMATH_CALUDE_ball_return_to_start_l2356_235626

def circle_size : ℕ := 14
def step_size : ℕ := 3

theorem ball_return_to_start :
  ∀ (start : ℕ),
  start < circle_size →
  (∃ (n : ℕ), n > 0 ∧ (start + n * step_size) % circle_size = start) →
  (∀ (m : ℕ), 0 < m → m < circle_size → (start + m * step_size) % circle_size ≠ start) →
  (start + circle_size * step_size) % circle_size = start :=
by sorry

#check ball_return_to_start

end NUMINAMATH_CALUDE_ball_return_to_start_l2356_235626


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_in_combined_mixture_l2356_235662

/-- Given two containers A and B with alcohol mixtures, this theorem proves
    the ratio of pure alcohol to water in the combined mixture. -/
theorem alcohol_water_ratio_in_combined_mixture
  (v₁ v₂ m₁ n₁ m₂ n₂ : ℝ)
  (hv₁ : v₁ > 0)
  (hv₂ : v₂ > 0)
  (hm₁ : m₁ > 0)
  (hn₁ : n₁ > 0)
  (hm₂ : m₂ > 0)
  (hn₂ : n₂ > 0) :
  let pure_alcohol_A := v₁ * m₁ / (m₁ + n₁)
  let water_A := v₁ * n₁ / (m₁ + n₁)
  let pure_alcohol_B := v₂ * m₂ / (m₂ + n₂)
  let water_B := v₂ * n₂ / (m₂ + n₂)
  let total_pure_alcohol := pure_alcohol_A + pure_alcohol_B
  let total_water := water_A + water_B
  (total_pure_alcohol / total_water) = 
    (v₁*m₁*m₂ + v₁*m₁*n₂ + v₂*m₁*m₂ + v₂*m₂*n₁) / 
    (v₁*m₂*n₁ + v₁*n₁*n₂ + v₂*m₁*n₂ + v₂*n₁*n₂) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_in_combined_mixture_l2356_235662


namespace NUMINAMATH_CALUDE_circle_surrounding_circles_radius_l2356_235655

theorem circle_surrounding_circles_radius (r : ℝ) : 
  r > 0 →  -- r is positive
  (2 + r)^2 = 2 * (2 * r)^2 →  -- Pythagorean theorem for centers
  r = (4 * Real.sqrt 2 + 2) / 7 := by
sorry

end NUMINAMATH_CALUDE_circle_surrounding_circles_radius_l2356_235655


namespace NUMINAMATH_CALUDE_bianca_tulips_l2356_235660

/-- The number of tulips Bianca picked -/
def tulips : ℕ := sorry

/-- The total number of flowers Bianca picked -/
def total_flowers : ℕ := sorry

/-- The number of roses Bianca picked -/
def roses : ℕ := 49

/-- The number of flowers Bianca used -/
def used_flowers : ℕ := 81

/-- The number of extra flowers -/
def extra_flowers : ℕ := 7

theorem bianca_tulips : 
  tulips = 39 ∧ 
  total_flowers = tulips + roses ∧ 
  total_flowers = used_flowers + extra_flowers :=
sorry

end NUMINAMATH_CALUDE_bianca_tulips_l2356_235660


namespace NUMINAMATH_CALUDE_units_digit_of_j_squared_plus_three_to_j_l2356_235602

theorem units_digit_of_j_squared_plus_three_to_j (j : ℕ) : 
  j = 2023^3 + 3^2023 + 2023 → (j^2 + 3^j) % 10 = 6 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_j_squared_plus_three_to_j_l2356_235602


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l2356_235669

/-- A rectangular prism with different dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_ne_width : length ≠ width
  length_ne_height : length ≠ height
  width_ne_height : width ≠ height

/-- The number of face diagonals in a rectangular prism -/
def face_diagonals (prism : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism -/
def space_diagonals (prism : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (prism : RectangularPrism) : ℕ :=
  face_diagonals prism + space_diagonals prism

theorem rectangular_prism_diagonals (prism : RectangularPrism) :
  total_diagonals prism = 16 ∧ space_diagonals prism = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l2356_235669


namespace NUMINAMATH_CALUDE_alpha_beta_ratio_l2356_235629

-- Define the angles
variable (α β x y : ℝ)

-- Define the angle relationships
axiom angle_relation_1 : y = x + β
axiom angle_relation_2 : 2 * y = 2 * x + α

-- Theorem to prove
theorem alpha_beta_ratio : α / β = 2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_ratio_l2356_235629


namespace NUMINAMATH_CALUDE_negation_of_negation_l2356_235672

theorem negation_of_negation : -(-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_negation_l2356_235672


namespace NUMINAMATH_CALUDE_two_different_color_balls_probability_two_different_color_balls_probability_proof_l2356_235631

theorem two_different_color_balls_probability 
  (total_balls : ℕ) 
  (red_balls yellow_balls white_balls : ℕ) 
  (h1 : total_balls = red_balls + yellow_balls + white_balls)
  (h2 : red_balls = 2)
  (h3 : yellow_balls = 2)
  (h4 : white_balls = 1)
  : ℚ :=
4/5

theorem two_different_color_balls_probability_proof 
  (total_balls : ℕ) 
  (red_balls yellow_balls white_balls : ℕ) 
  (h1 : total_balls = red_balls + yellow_balls + white_balls)
  (h2 : red_balls = 2)
  (h3 : yellow_balls = 2)
  (h4 : white_balls = 1)
  : two_different_color_balls_probability total_balls red_balls yellow_balls white_balls h1 h2 h3 h4 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_two_different_color_balls_probability_two_different_color_balls_probability_proof_l2356_235631


namespace NUMINAMATH_CALUDE_percentage_of_juniors_l2356_235616

theorem percentage_of_juniors (total_students : ℕ) (juniors_in_sports : ℕ) 
  (sports_percentage : ℚ) (h1 : total_students = 500) 
  (h2 : juniors_in_sports = 140) (h3 : sports_percentage = 70 / 100) :
  (juniors_in_sports / sports_percentage) / total_students = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_juniors_l2356_235616


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_reciprocal_l2356_235622

theorem max_value_of_x_plus_reciprocal (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_reciprocal_l2356_235622


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2356_235680

/-- The new area of a rectangle after changing its dimensions -/
def new_area (original_area : ℝ) (length_increase : ℝ) (width_decrease : ℝ) : ℝ :=
  original_area * (1 + length_increase) * (1 - width_decrease)

theorem rectangle_area_change :
  new_area 432 0.2 0.1 = 466.56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2356_235680


namespace NUMINAMATH_CALUDE_ben_savings_proof_l2356_235634

/-- Represents the number of days that have elapsed -/
def days : ℕ := 7

/-- Ben's daily starting amount in cents -/
def daily_start : ℕ := 5000

/-- Ben's daily spending in cents -/
def daily_spend : ℕ := 1500

/-- Ben's dad's additional contribution in cents -/
def dad_contribution : ℕ := 1000

/-- Ben's final amount in cents -/
def final_amount : ℕ := 50000

theorem ben_savings_proof :
  2 * (days * (daily_start - daily_spend)) + dad_contribution = final_amount := by
  sorry

end NUMINAMATH_CALUDE_ben_savings_proof_l2356_235634


namespace NUMINAMATH_CALUDE_female_math_only_result_l2356_235601

/-- The number of female students who participated in the math competition but not in the English competition -/
def female_math_only (male_math female_math female_eng male_eng total male_both : ℕ) : ℕ :=
  let male_total := male_math + male_eng - male_both
  let female_total := total - male_total
  let female_both := female_math + female_eng - female_total
  female_math - female_both

/-- Theorem stating the result of the problem -/
theorem female_math_only_result : 
  female_math_only 120 80 120 80 260 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_female_math_only_result_l2356_235601


namespace NUMINAMATH_CALUDE_division_algorithm_l2356_235615

theorem division_algorithm (x y : ℤ) (hx : x ≥ 0) (hy : y > 0) :
  ∃! (q r : ℤ), x = q * y + r ∧ 0 ≤ r ∧ r < y := by
  sorry

end NUMINAMATH_CALUDE_division_algorithm_l2356_235615


namespace NUMINAMATH_CALUDE_becky_lunch_days_proof_l2356_235689

/-- The number of school days in an academic year -/
def school_days : ℕ := 180

/-- The fraction of time Aliyah packs her lunch -/
def aliyah_lunch_fraction : ℚ := 1/2

/-- The fraction of Aliyah's lunch-packing frequency that Becky packs her lunch -/
def becky_lunch_fraction : ℚ := 1/2

/-- The number of days Becky packs her lunch in a school year -/
def becky_lunch_days : ℕ := 45

theorem becky_lunch_days_proof :
  (school_days : ℚ) * aliyah_lunch_fraction * becky_lunch_fraction = becky_lunch_days := by
  sorry

end NUMINAMATH_CALUDE_becky_lunch_days_proof_l2356_235689


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2356_235678

theorem complex_fraction_simplification :
  (5 + 7*I) / (2 + 3*I) = 31/13 - (1/13)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2356_235678


namespace NUMINAMATH_CALUDE_chips_bought_l2356_235671

/-- Given three friends paying $5 each for bags of chips costing $3 per bag,
    prove that they can buy 5 bags of chips. -/
theorem chips_bought (num_friends : ℕ) (payment_per_friend : ℕ) (cost_per_bag : ℕ) :
  num_friends = 3 →
  payment_per_friend = 5 →
  cost_per_bag = 3 →
  (num_friends * payment_per_friend) / cost_per_bag = 5 :=
by sorry

end NUMINAMATH_CALUDE_chips_bought_l2356_235671


namespace NUMINAMATH_CALUDE_zeros_in_Q_l2356_235635

def R (k : ℕ) : ℚ := (10^k - 1) / 9

def Q : ℚ := R 25 / R 5

def count_zeros (q : ℚ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 16 := by sorry

end NUMINAMATH_CALUDE_zeros_in_Q_l2356_235635


namespace NUMINAMATH_CALUDE_system_solution_l2356_235674

theorem system_solution (x y z u : ℚ) : 
  x + y = 12 ∧ 
  x / z = 3 / 2 ∧ 
  z + u = 10 ∧ 
  y * u = 36 →
  x = 6 ∧ y = 6 ∧ z = 4 ∧ u = 6 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2356_235674


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2356_235657

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 + x - 6 > 0 ↔ (x < -3 ∨ x > 2)) ↔
  (∀ x : ℝ, (x ≥ -3 ∧ x ≤ 2) → x^2 + x - 6 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2356_235657


namespace NUMINAMATH_CALUDE_teacher_in_middle_girls_not_adjacent_teacher_flanked_by_girls_l2356_235610

-- Define the class composition
def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_teacher : ℕ := 1

-- Define the total number of people
def total_people : ℕ := num_boys + num_girls + num_teacher

-- Theorem for scenario 1
theorem teacher_in_middle :
  (Nat.factorial (total_people - 1)) = 720 := by sorry

-- Theorem for scenario 2
theorem girls_not_adjacent :
  (Nat.factorial (total_people - num_girls)) * (Nat.factorial (total_people - num_girls - 1)) = 2400 := by sorry

-- Theorem for scenario 3
theorem teacher_flanked_by_girls :
  (Nat.factorial (total_people - num_girls)) * (Nat.factorial num_girls) = 240 := by sorry

end NUMINAMATH_CALUDE_teacher_in_middle_girls_not_adjacent_teacher_flanked_by_girls_l2356_235610


namespace NUMINAMATH_CALUDE_floor_width_calculation_l2356_235692

def tile_length : ℝ := 65
def tile_width : ℝ := 25
def floor_length : ℝ := 150
def max_tiles : ℕ := 36

theorem floor_width_calculation (floor_width : ℝ) 
  (h1 : floor_length = 150)
  (h2 : tile_length = 65)
  (h3 : tile_width = 25)
  (h4 : max_tiles = 36)
  (h5 : 2 * tile_length ≤ floor_length)
  (h6 : floor_width = (max_tiles / 2 : ℝ) * tile_width) :
  floor_width = 450 := by
sorry

end NUMINAMATH_CALUDE_floor_width_calculation_l2356_235692


namespace NUMINAMATH_CALUDE_integral_x_minus_reciprocal_x_l2356_235667

theorem integral_x_minus_reciprocal_x (f : ℝ → ℝ) (hf : ∀ x ∈ Set.Icc 1 2, HasDerivAt f (x - 1/x) x) :
  ∫ x in Set.Icc 1 2, (x - 1/x) = 1 - Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_integral_x_minus_reciprocal_x_l2356_235667


namespace NUMINAMATH_CALUDE_ski_camp_directions_l2356_235691

-- Define the four cardinal directions
inductive Direction
| North
| South
| East
| West

-- Define the four friends
inductive Friend
| Karel
| Mojmir
| Pepa
| Zdenda

-- Define a function that assigns a direction to each friend
def came_from : Friend → Direction := sorry

-- Define the statements made by each friend
def karel_statement : Prop :=
  came_from Friend.Karel ≠ Direction.North ∧ came_from Friend.Karel ≠ Direction.South

def mojmir_statement : Prop :=
  came_from Friend.Mojmir = Direction.South

def pepa_statement : Prop :=
  came_from Friend.Pepa = Direction.North

def zdenda_statement : Prop :=
  came_from Friend.Zdenda ≠ Direction.South

-- Define a function that checks if a statement is true
def is_true_statement : Friend → Prop
| Friend.Karel => karel_statement
| Friend.Mojmir => mojmir_statement
| Friend.Pepa => pepa_statement
| Friend.Zdenda => zdenda_statement

-- Theorem to prove
theorem ski_camp_directions :
  (∃! f : Friend, ¬is_true_statement f) ∧
  (came_from Friend.Zdenda = Direction.North) ∧
  (came_from Friend.Mojmir = Direction.South) ∧
  (¬is_true_statement Friend.Pepa) :=
by sorry

end NUMINAMATH_CALUDE_ski_camp_directions_l2356_235691


namespace NUMINAMATH_CALUDE_m_plus_abs_m_nonnegative_l2356_235649

theorem m_plus_abs_m_nonnegative (m : ℚ) : m + |m| ≥ 0 := by sorry

end NUMINAMATH_CALUDE_m_plus_abs_m_nonnegative_l2356_235649


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2356_235624

def i : ℂ := Complex.I

theorem complex_number_in_second_quadrant :
  let z : ℂ := (1 + 2*i) / (1 + 2*i^3)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2356_235624


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2356_235688

theorem complex_number_quadrant : ∃ (a b : ℝ), (a > 0 ∧ b < 0) ∧ (Complex.mk a b = 5 / (Complex.mk 2 1)) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2356_235688


namespace NUMINAMATH_CALUDE_linear_decreasing_negative_slope_l2356_235665

/-- A linear function f(x) = kx + b that is monotonically decreasing on ℝ has a negative slope k. -/
theorem linear_decreasing_negative_slope (k b : ℝ) : 
  (∀ x y, x < y → (k * x + b) > (k * y + b)) → k < 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_decreasing_negative_slope_l2356_235665


namespace NUMINAMATH_CALUDE_morse_code_symbols_l2356_235609

/-- The number of possible symbols for a given sequence length in Morse code -/
def morse_combinations (n : ℕ) : ℕ := 2^n

/-- The total number of distinct Morse code symbols for sequences up to length 5 -/
def total_morse_symbols : ℕ :=
  (morse_combinations 1) + (morse_combinations 2) + (morse_combinations 3) +
  (morse_combinations 4) + (morse_combinations 5)

theorem morse_code_symbols :
  total_morse_symbols = 62 :=
by sorry

end NUMINAMATH_CALUDE_morse_code_symbols_l2356_235609


namespace NUMINAMATH_CALUDE_inequality_proof_l2356_235664

def M := {x : ℝ | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2 * |a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2356_235664


namespace NUMINAMATH_CALUDE_roots_sum_powers_l2356_235606

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → b^2 - 5*b + 6 = 0 → a^5 + a^4*b + b^5 = -16674 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l2356_235606


namespace NUMINAMATH_CALUDE_ducks_park_solution_l2356_235670

def ducks_park_problem (initial_ducks : ℕ) (geese_arrive : ℕ) (ducks_arrive : ℕ) (geese_leave : ℕ) : Prop :=
  let initial_geese : ℕ := 2 * initial_ducks - 10
  let final_ducks : ℕ := initial_ducks + ducks_arrive
  let final_geese : ℕ := initial_geese - geese_leave
  final_geese - final_ducks = 1

theorem ducks_park_solution :
  ducks_park_problem 25 4 4 10 := by
  sorry

end NUMINAMATH_CALUDE_ducks_park_solution_l2356_235670


namespace NUMINAMATH_CALUDE_divisibility_by_17_l2356_235683

theorem divisibility_by_17 (x y : ℤ) : 
  (∃ k : ℤ, 2*x + 3*y = 17*k) → (∃ m : ℤ, 9*x + 5*y = 17*m) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_17_l2356_235683


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2356_235632

theorem polynomial_factorization (p q : ℝ) :
  ∃ (a b c d e f : ℝ), ∀ (x : ℝ),
    x^4 + p*x^2 + q = (a*x^2 + b*x + c) * (d*x^2 + e*x + f) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2356_235632


namespace NUMINAMATH_CALUDE_x_percent_of_x_squared_is_nine_l2356_235651

theorem x_percent_of_x_squared_is_nine (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x^2 = 9) :
  ∃ (y : ℝ), abs (x - y) < 0.01 ∧ y^3 = 900 ∧ 
  ∀ (z : ℤ), abs (x - ↑z) ≥ abs (x - 10) :=
sorry

end NUMINAMATH_CALUDE_x_percent_of_x_squared_is_nine_l2356_235651


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l2356_235679

theorem rectangle_length_fraction_of_circle_radius : 
  ∀ (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ),
  square_area = 900 →
  rectangle_area = 120 →
  rectangle_breadth = 10 →
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l2356_235679


namespace NUMINAMATH_CALUDE_complex_product_given_pure_imaginary_sum_l2356_235636

theorem complex_product_given_pure_imaginary_sum (a : ℝ) : 
  let z₁ : ℂ := a - 2*I
  let z₂ : ℂ := -1 + a*I
  (∃ (b : ℝ), z₁ + z₂ = b*I) → z₁ * z₂ = 1 + 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_given_pure_imaginary_sum_l2356_235636


namespace NUMINAMATH_CALUDE_min_distance_A_to_E_l2356_235668

/-- Given five points A, B, C, D, and E with specified distances between them,
    prove that the minimum possible distance between A and E is 2 units. -/
theorem min_distance_A_to_E (A B C D E : ℝ) : 
  (∃ (AB BC CD DE : ℝ), 
    AB = 12 ∧ 
    BC = 5 ∧ 
    CD = 3 ∧ 
    DE = 2 ∧ 
    (∀ (AE : ℝ), AE ≥ 2)) → 
  (∃ (AE : ℝ), AE = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_A_to_E_l2356_235668


namespace NUMINAMATH_CALUDE_t_shaped_area_concrete_t_shaped_area_l2356_235613

/-- The area of a T-shaped region formed by subtracting three smaller rectangles from a larger rectangle -/
theorem t_shaped_area (a b c d e f : ℕ) : 
  a * b - (c * d + e * f + c * (b - f)) = 24 :=
by
  sorry

/-- Concrete instance of the T-shaped area theorem -/
theorem concrete_t_shaped_area : 
  8 * 6 - (2 * 2 + 4 * 2 + 2 * 6) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_t_shaped_area_concrete_t_shaped_area_l2356_235613


namespace NUMINAMATH_CALUDE_carla_initial_marbles_l2356_235694

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The total number of marbles Carla has after buying -/
def total_marbles : ℕ := 187

/-- The number of marbles Carla started with -/
def initial_marbles : ℕ := total_marbles - marbles_bought

theorem carla_initial_marbles :
  initial_marbles = 53 := by sorry

end NUMINAMATH_CALUDE_carla_initial_marbles_l2356_235694


namespace NUMINAMATH_CALUDE_clock_hands_opposite_period_l2356_235642

/-- The number of times clock hands are in opposite directions in 12 hours -/
def opposite_directions_per_12_hours : ℕ := 11

/-- The number of hours on a clock -/
def hours_on_clock : ℕ := 12

/-- The number of minutes between opposite directions -/
def minutes_between_opposite : ℕ := 30

/-- The observed number of times the hands are in opposite directions -/
def observed_opposite_directions : ℕ := 22

/-- The period in which the hands show opposite directions 22 times -/
def period : ℕ := 24

theorem clock_hands_opposite_period :
  opposite_directions_per_12_hours * 2 = observed_opposite_directions →
  period = 24 := by sorry

end NUMINAMATH_CALUDE_clock_hands_opposite_period_l2356_235642


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2356_235684

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}
def Q : Set ℝ := {x : ℝ | x > 3}

-- State the theorem
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2356_235684


namespace NUMINAMATH_CALUDE_jackie_phil_same_heads_l2356_235604

/-- The probability of getting heads for a fair coin -/
def fair_coin_prob : ℚ := 1/2

/-- The probability of getting heads for the biased coin -/
def biased_coin_prob : ℚ := 4/7

/-- The probability of getting k heads when flipping the three coins -/
def prob_k_heads (k : ℕ) : ℚ :=
  match k with
  | 0 => (1 - fair_coin_prob)^2 * (1 - biased_coin_prob)
  | 1 => 2 * fair_coin_prob * (1 - fair_coin_prob) * (1 - biased_coin_prob) + 
         (1 - fair_coin_prob)^2 * biased_coin_prob
  | 2 => fair_coin_prob^2 * (1 - biased_coin_prob) + 
         2 * fair_coin_prob * (1 - fair_coin_prob) * biased_coin_prob
  | 3 => fair_coin_prob^2 * biased_coin_prob
  | _ => 0

/-- The probability that Jackie and Phil get the same number of heads -/
def prob_same_heads : ℚ :=
  (prob_k_heads 0)^2 + (prob_k_heads 1)^2 + (prob_k_heads 2)^2 + (prob_k_heads 3)^2

theorem jackie_phil_same_heads : prob_same_heads = 123/392 := by
  sorry

end NUMINAMATH_CALUDE_jackie_phil_same_heads_l2356_235604


namespace NUMINAMATH_CALUDE_cole_average_speed_back_home_l2356_235676

/-- Proves that Cole's average speed back home was 120 km/h given the conditions of his round trip. -/
theorem cole_average_speed_back_home 
  (speed_to_work : ℝ) 
  (total_time : ℝ) 
  (time_to_work : ℝ) 
  (h1 : speed_to_work = 80) 
  (h2 : total_time = 3) 
  (h3 : time_to_work = 108 / 60) : 
  (speed_to_work * time_to_work) / (total_time - time_to_work) = 120 := by
  sorry

#check cole_average_speed_back_home

end NUMINAMATH_CALUDE_cole_average_speed_back_home_l2356_235676
