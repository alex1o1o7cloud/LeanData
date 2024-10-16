import Mathlib

namespace NUMINAMATH_CALUDE_parabola_line_intersection_area_l2948_294831

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  equation : (ℝ × ℝ) → Prop

/-- Line structure -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Triangle structure -/
structure Triangle where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Function to calculate distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Function to calculate area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Main theorem -/
theorem parabola_line_intersection_area 
  (p : Parabola) 
  (l : Line) 
  (A B : ℝ × ℝ) 
  (h1 : p.equation = fun (x, y) ↦ y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : l.point1 = p.focus)
  (h4 : p.equation A ∧ p.equation B)
  (h5 : distance A p.focus = 3) :
  triangleArea { point1 := (0, 0), point2 := A, point3 := B } = 3 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_area_l2948_294831


namespace NUMINAMATH_CALUDE_train_passing_time_train_passes_jogger_in_40_seconds_l2948_294860

/-- Calculates the time for a train to pass a jogger given their initial speeds,
    distances, and speed reduction due to incline. -/
theorem train_passing_time (jogger_speed train_speed : ℝ)
                           (initial_distance train_length : ℝ)
                           (incline_reduction : ℝ) : ℝ :=
  let jogger_effective_speed := jogger_speed * (1 - incline_reduction)
  let train_effective_speed := train_speed * (1 - incline_reduction)
  let relative_speed := train_effective_speed - jogger_effective_speed
  let total_distance := initial_distance + train_length
  total_distance / relative_speed * (3600 / 1000)

/-- The time for the train to pass the jogger is 40 seconds. -/
theorem train_passes_jogger_in_40_seconds :
  train_passing_time 9 45 240 120 0.1 = 40 := by
  sorry


end NUMINAMATH_CALUDE_train_passing_time_train_passes_jogger_in_40_seconds_l2948_294860


namespace NUMINAMATH_CALUDE_sqrt_of_point_zero_nine_equals_point_three_l2948_294842

theorem sqrt_of_point_zero_nine_equals_point_three : 
  Real.sqrt 0.09 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_point_zero_nine_equals_point_three_l2948_294842


namespace NUMINAMATH_CALUDE_base_n_1001_not_prime_l2948_294888

/-- For a positive integer n ≥ 2, 1001_n represents n^3 + 1 in base 10 -/
def base_n_1001 (n : ℕ) : ℕ := n^3 + 1

/-- A number is composite if it has a factor between 1 and itself -/
def is_composite (m : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < m ∧ m % k = 0

theorem base_n_1001_not_prime : 
  ∀ n : ℕ, n ≥ 2 → is_composite (base_n_1001 n) := by
  sorry

end NUMINAMATH_CALUDE_base_n_1001_not_prime_l2948_294888


namespace NUMINAMATH_CALUDE_comic_books_problem_l2948_294858

theorem comic_books_problem (sold : ℕ) (left : ℕ) (h1 : sold = 65) (h2 : left = 25) :
  sold + left = 90 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_problem_l2948_294858


namespace NUMINAMATH_CALUDE_fraction_meaningful_implies_x_not_one_l2948_294862

-- Define a function that represents the meaningfulness of the fraction 1/(x-1)
def is_meaningful (x : ℝ) : Prop := x ≠ 1

-- Theorem statement
theorem fraction_meaningful_implies_x_not_one :
  ∀ x : ℝ, is_meaningful x → x ≠ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_implies_x_not_one_l2948_294862


namespace NUMINAMATH_CALUDE_satisfaction_survey_stats_l2948_294881

def data : List ℝ := [34, 35, 35, 36]

theorem satisfaction_survey_stats (median mode mean variance : ℝ) :
  median = 35 ∧
  mode = 35 ∧
  mean = 35 ∧
  variance = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_satisfaction_survey_stats_l2948_294881


namespace NUMINAMATH_CALUDE_max_digit_difference_l2948_294878

def two_digit_number (tens units : Nat) : Nat :=
  10 * tens + units

theorem max_digit_difference :
  ∃ (a b : Nat),
    a ≠ b ∧
    a ≠ 0 ∧
    b ≠ 0 ∧
    a < 10 ∧
    b < 10 ∧
    ∀ (x y : Nat),
      x ≠ y →
      x ≠ 0 →
      y ≠ 0 →
      x < 10 →
      y < 10 →
      two_digit_number x y - two_digit_number y x ≤ two_digit_number a b - two_digit_number b a ∧
      two_digit_number a b - two_digit_number b a = 72 :=
by sorry

end NUMINAMATH_CALUDE_max_digit_difference_l2948_294878


namespace NUMINAMATH_CALUDE_average_of_first_45_results_l2948_294805

theorem average_of_first_45_results
  (n₁ : ℕ)
  (n₂ : ℕ)
  (a₂ : ℝ)
  (total_avg : ℝ)
  (h₁ : n₁ = 45)
  (h₂ : n₂ = 25)
  (h₃ : a₂ = 45)
  (h₄ : total_avg = 32.142857142857146)
  (h₅ : (n₁ : ℝ) * a₁ + (n₂ : ℝ) * a₂ = (n₁ + n₂ : ℝ) * total_avg) :
  a₁ = 25 :=
by sorry

end NUMINAMATH_CALUDE_average_of_first_45_results_l2948_294805


namespace NUMINAMATH_CALUDE_triangle_ratio_l2948_294836

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  A = Real.pi / 3 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

#check triangle_ratio

end NUMINAMATH_CALUDE_triangle_ratio_l2948_294836


namespace NUMINAMATH_CALUDE_cube_edge_length_l2948_294832

theorem cube_edge_length (box_edge : ℝ) (num_cubes : ℕ) (h1 : box_edge = 1) (h2 : num_cubes = 1000) :
  ∃ (cube_edge : ℝ), cube_edge^3 * num_cubes = box_edge^3 ∧ cube_edge = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2948_294832


namespace NUMINAMATH_CALUDE_badArrangementsCount_l2948_294855

-- Define a type for circular arrangements
def CircularArrangement := List ℕ

-- Define what it means for an arrangement to be valid
def isValidArrangement (arr : CircularArrangement) : Prop :=
  arr.length = 6 ∧ arr.toFinset = {1, 2, 3, 4, 5, 6}

-- Define consecutive subsets in a circular arrangement
def consecutiveSubsets (arr : CircularArrangement) : List (List ℕ) :=
  sorry

-- Define what it means for an arrangement to be "bad"
def isBadArrangement (arr : CircularArrangement) : Prop :=
  ∃ n : ℕ, n ≥ 1 ∧ n ≤ 20 ∧ ∀ subset ∈ consecutiveSubsets arr, (subset.sum ≠ n)

-- Define equivalence of arrangements under rotation and reflection
def areEquivalentArrangements (arr1 arr2 : CircularArrangement) : Prop :=
  sorry

-- The main theorem
theorem badArrangementsCount :
  ∃ badArrs : List CircularArrangement,
    badArrs.length = 3 ∧
    (∀ arr ∈ badArrs, isValidArrangement arr ∧ isBadArrangement arr) ∧
    (∀ arr, isValidArrangement arr → isBadArrangement arr →
      ∃ badArr ∈ badArrs, areEquivalentArrangements arr badArr) :=
  sorry

end NUMINAMATH_CALUDE_badArrangementsCount_l2948_294855


namespace NUMINAMATH_CALUDE_cheerful_not_green_l2948_294821

structure Snake where
  isGreen : Bool
  isCheerful : Bool
  canMultiply : Bool
  canDivide : Bool

def TomCollection : Nat := 15

theorem cheerful_not_green (snakes : Finset Snake) 
  (h1 : snakes.card = TomCollection)
  (h2 : (snakes.filter (fun s => s.isGreen)).card = 5)
  (h3 : (snakes.filter (fun s => s.isCheerful)).card = 6)
  (h4 : ∀ s ∈ snakes, s.isCheerful → s.canMultiply)
  (h5 : ∀ s ∈ snakes, s.isGreen → ¬s.canDivide)
  (h6 : ∀ s ∈ snakes, ¬s.canDivide → ¬s.canMultiply) :
  ∀ s ∈ snakes, s.isCheerful → ¬s.isGreen :=
sorry

end NUMINAMATH_CALUDE_cheerful_not_green_l2948_294821


namespace NUMINAMATH_CALUDE_irrational_difference_representation_l2948_294829

theorem irrational_difference_representation (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  ∃ (α β : ℝ), Irrational α ∧ Irrational β ∧ 0 < α ∧ α < 1 ∧ 0 < β ∧ β < 1 ∧ x = α - β := by
  sorry

end NUMINAMATH_CALUDE_irrational_difference_representation_l2948_294829


namespace NUMINAMATH_CALUDE_book_selling_price_total_selling_price_is_595_l2948_294872

/-- Calculates the total selling price of two books given the following conditions:
    - Total cost of two books is 600
    - First book is sold at a loss of 15%
    - Second book is sold at a gain of 19%
    - Cost of the book sold at a loss is 350
-/
theorem book_selling_price (total_cost : ℝ) (loss_percentage : ℝ) (gain_percentage : ℝ) (loss_book_cost : ℝ) : ℝ :=
  let selling_price_loss_book := loss_book_cost * (1 - loss_percentage / 100)
  let gain_book_cost := total_cost - loss_book_cost
  let selling_price_gain_book := gain_book_cost * (1 + gain_percentage / 100)
  selling_price_loss_book + selling_price_gain_book

theorem total_selling_price_is_595 :
  book_selling_price 600 15 19 350 = 595 := by
  sorry

end NUMINAMATH_CALUDE_book_selling_price_total_selling_price_is_595_l2948_294872


namespace NUMINAMATH_CALUDE_utopia_park_elephants_l2948_294819

/-- The time taken for new elephants to enter Utopia National Park -/
def time_for_new_elephants (initial_elephants : ℕ) (exodus_duration : ℕ) (exodus_rate : ℕ) (entry_rate : ℕ) (final_elephants : ℕ) : ℕ :=
  let elephants_after_exodus := initial_elephants - exodus_duration * exodus_rate
  let new_elephants := final_elephants - elephants_after_exodus
  new_elephants / entry_rate

theorem utopia_park_elephants :
  time_for_new_elephants 30000 4 2880 1500 28980 = 7 :=
by sorry

end NUMINAMATH_CALUDE_utopia_park_elephants_l2948_294819


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2948_294850

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1 / x₀ + 4 / y₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2948_294850


namespace NUMINAMATH_CALUDE_cost_per_pound_mixed_feed_l2948_294806

/-- Calculates the cost per pound of mixed dog feed --/
theorem cost_per_pound_mixed_feed 
  (total_weight : ℝ) 
  (cheap_price : ℝ) 
  (expensive_price : ℝ) 
  (cheap_amount : ℝ) 
  (h1 : total_weight = 35) 
  (h2 : cheap_price = 0.18) 
  (h3 : expensive_price = 0.53) 
  (h4 : cheap_amount = 17) :
  (cheap_amount * cheap_price + (total_weight - cheap_amount) * expensive_price) / total_weight = 0.36 := by
sorry


end NUMINAMATH_CALUDE_cost_per_pound_mixed_feed_l2948_294806


namespace NUMINAMATH_CALUDE_f_comp_f_four_roots_l2948_294811

/-- A quadratic function f(x) = x^2 + 10x + d -/
def f (d : ℝ) (x : ℝ) : ℝ := x^2 + 10*x + d

/-- The composition of f with itself -/
def f_comp_f (d : ℝ) (x : ℝ) : ℝ := f d (f d x)

/-- The theorem stating the condition for f(f(x)) to have exactly 4 distinct real roots -/
theorem f_comp_f_four_roots (d : ℝ) :
  (∃ (a b c e : ℝ), a < b ∧ b < c ∧ c < e ∧
    (∀ x : ℝ, f_comp_f d x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = e)) ↔
  d < 25 :=
sorry

end NUMINAMATH_CALUDE_f_comp_f_four_roots_l2948_294811


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2948_294898

theorem fifteenth_student_age
  (total_students : Nat)
  (average_age : ℝ)
  (group1_count : Nat)
  (group1_average : ℝ)
  (group2_count : Nat)
  (group2_average : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_count = 4)
  (h4 : group1_average = 14)
  (h5 : group2_count = 9)
  (h6 : group2_average = 16)
  (h7 : group1_count + group2_count + 1 = total_students) :
  ∃ (fifteenth_age : ℝ),
    fifteenth_age = total_students * average_age - (group1_count * group1_average + group2_count * group2_average) :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l2948_294898


namespace NUMINAMATH_CALUDE_students_below_50_l2948_294848

/-- Represents the frequency distribution of scores -/
structure ScoreDistribution where
  freq_50_60 : Real
  freq_60_70 : Real
  freq_70_80 : Real
  freq_80_90 : Real
  freq_90_100 : Real

/-- The problem statement -/
theorem students_below_50 
  (total_students : Nat) 
  (selected_students : Nat)
  (score_distribution : ScoreDistribution)
  (h1 : total_students = 600)
  (h2 : selected_students = 60)
  (h3 : score_distribution.freq_50_60 = 0.15)
  (h4 : score_distribution.freq_60_70 = 0.15)
  (h5 : score_distribution.freq_70_80 = 0.30)
  (h6 : score_distribution.freq_80_90 = 0.25)
  (h7 : score_distribution.freq_90_100 = 0.05) :
  (total_students : Real) * (1 - (score_distribution.freq_50_60 + 
                                  score_distribution.freq_60_70 + 
                                  score_distribution.freq_70_80 + 
                                  score_distribution.freq_80_90 + 
                                  score_distribution.freq_90_100)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_students_below_50_l2948_294848


namespace NUMINAMATH_CALUDE_leftover_bread_slices_is_eight_l2948_294852

/-- The number of bread packages Shane buys -/
def bread_packages : ℕ := 2

/-- The number of slices in each bread package -/
def slices_per_bread_package : ℕ := 20

/-- The number of ham packages Shane buys -/
def ham_packages : ℕ := 2

/-- The number of slices in each ham package -/
def slices_per_ham_package : ℕ := 8

/-- The number of bread slices needed for each sandwich -/
def bread_slices_per_sandwich : ℕ := 2

/-- The number of ham slices needed for each sandwich -/
def ham_slices_per_sandwich : ℕ := 1

/-- The total number of bread slices Shane has -/
def total_bread_slices : ℕ := bread_packages * slices_per_bread_package

/-- The total number of ham slices Shane has -/
def total_ham_slices : ℕ := ham_packages * slices_per_ham_package

/-- The number of sandwiches Shane can make -/
def sandwiches_made : ℕ := total_ham_slices / ham_slices_per_sandwich

/-- The number of bread slices used for sandwiches -/
def bread_slices_used : ℕ := sandwiches_made * bread_slices_per_sandwich

/-- The number of leftover bread slices -/
def leftover_bread_slices : ℕ := total_bread_slices - bread_slices_used

theorem leftover_bread_slices_is_eight : leftover_bread_slices = 8 := by
  sorry

end NUMINAMATH_CALUDE_leftover_bread_slices_is_eight_l2948_294852


namespace NUMINAMATH_CALUDE_a_range_characterization_l2948_294818

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 4 = 0

-- Define the set of a values where only one proposition is true
def a_range (a : ℝ) : Prop := (p a ∧ ¬q a) ∨ (¬p a ∧ q a)

-- Theorem statement
theorem a_range_characterization :
  ∀ a : ℝ, a_range a ↔ (a > -2 ∧ a ≤ 1) ∨ (a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_a_range_characterization_l2948_294818


namespace NUMINAMATH_CALUDE_hexagon_puzzle_solution_l2948_294846

/-- Represents the positions in the hexagon puzzle --/
inductive Position
| A | B | C | D | E | F

/-- Represents a valid assignment of digits to positions --/
def Assignment := Position → Fin 6

/-- Checks if an assignment is valid (uses each digit exactly once) --/
def isValidAssignment (a : Assignment) : Prop :=
  ∀ (i : Fin 6), ∃! (p : Position), a p = i

/-- Checks if an assignment satisfies the sum condition for all lines --/
def satisfiesSumCondition (a : Assignment) : Prop :=
  (a Position.A + a Position.C + 9 = 15) ∧
  (a Position.A + 8 + a Position.F = 15) ∧
  (7 + a Position.C + a Position.E = 15) ∧
  (7 + a Position.D + a Position.F = 15) ∧
  (9 + a Position.B + a Position.D = 15) ∧
  (a Position.A + a Position.D + a Position.E = 15)

/-- The main theorem stating the existence and uniqueness of a valid solution --/
theorem hexagon_puzzle_solution :
  ∃! (a : Assignment), isValidAssignment a ∧ satisfiesSumCondition a :=
sorry

end NUMINAMATH_CALUDE_hexagon_puzzle_solution_l2948_294846


namespace NUMINAMATH_CALUDE_class_sizes_correct_l2948_294817

/-- The number of students in Mrs. Finley's class -/
def finley_class : ℕ := 24

/-- The number of students in Mr. Johnson's class -/
def johnson_class : ℕ := finley_class / 2 + 10

/-- The number of students in Ms. Garcia's class -/
def garcia_class : ℕ := 2 * johnson_class

/-- The number of students in Mr. Smith's class -/
def smith_class : ℕ := finley_class / 3

theorem class_sizes_correct :
  finley_class = 24 ∧
  johnson_class = 22 ∧
  garcia_class = 44 ∧
  smith_class = 8 := by
  sorry


end NUMINAMATH_CALUDE_class_sizes_correct_l2948_294817


namespace NUMINAMATH_CALUDE_five_vents_per_zone_l2948_294827

/-- Represents an HVAC system -/
structure HVACSystem where
  totalCost : ℕ
  numZones : ℕ
  costPerVent : ℕ

/-- Calculates the number of vents in each zone of an HVAC system -/
def ventsPerZone (system : HVACSystem) : ℕ :=
  (system.totalCost / system.costPerVent) / system.numZones

/-- Theorem: For the given HVAC system, there are 5 vents in each zone -/
theorem five_vents_per_zone (system : HVACSystem)
    (h1 : system.totalCost = 20000)
    (h2 : system.numZones = 2)
    (h3 : system.costPerVent = 2000) :
    ventsPerZone system = 5 := by
  sorry

#eval ventsPerZone { totalCost := 20000, numZones := 2, costPerVent := 2000 }

end NUMINAMATH_CALUDE_five_vents_per_zone_l2948_294827


namespace NUMINAMATH_CALUDE_ducks_killed_per_year_is_correct_l2948_294866

/-- The number of ducks killed every year -/
def ducks_killed_per_year : ℕ := 20

/-- The original flock size -/
def original_flock_size : ℕ := 100

/-- The number of ducks born every year -/
def ducks_born_per_year : ℕ := 30

/-- The number of years before joining with another flock -/
def years_before_joining : ℕ := 5

/-- The size of the other flock -/
def other_flock_size : ℕ := 150

/-- The combined flock size after joining -/
def combined_flock_size : ℕ := 300

theorem ducks_killed_per_year_is_correct :
  original_flock_size + years_before_joining * (ducks_born_per_year - ducks_killed_per_year) + other_flock_size = combined_flock_size :=
by sorry

end NUMINAMATH_CALUDE_ducks_killed_per_year_is_correct_l2948_294866


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2948_294871

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_s

/-- Proof that a train's length is approximately 129.96 meters -/
theorem train_length_proof (speed_kmh : ℝ) (time_s : ℝ)
  (h1 : speed_kmh = 52)
  (h2 : time_s = 9) :
  ∃ ε > 0, |train_length speed_kmh time_s - 129.96| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2948_294871


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2948_294889

/-- The line (m-1)x+(2m-1)y=m-5 always passes through the point (9, -4) for all real m -/
theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2948_294889


namespace NUMINAMATH_CALUDE_cos_330_degrees_l2948_294816

theorem cos_330_degrees : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l2948_294816


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2948_294845

theorem least_positive_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 2 = 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 4 = 3) ∧ 
  (n % 5 = 4) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 → m ≥ n) ∧
  n = 59 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2948_294845


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l2948_294896

/-- Represents the number of students in each grade --/
structure GradeDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Represents the sample sizes for each grade --/
structure SampleSizes where
  grade10 : ℕ
  grade11 : ℕ

/-- Checks if the sampling is proportional across grades --/
def isProportionalSampling (dist : GradeDistribution) (sample : SampleSizes) : Prop :=
  (dist.grade10 : ℚ) / sample.grade10 = (dist.grade11 : ℚ) / sample.grade11

theorem stratified_sampling_proportion 
  (dist : GradeDistribution)
  (sample : SampleSizes)
  (h1 : dist.grade10 = 50)
  (h2 : dist.grade11 = 40)
  (h3 : dist.grade12 = 40)
  (h4 : sample.grade11 = 8)
  (h5 : isProportionalSampling dist sample) :
  sample.grade10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l2948_294896


namespace NUMINAMATH_CALUDE_f_unique_zero_and_inequality_l2948_294813

noncomputable section

variable (a : ℝ)

def f (x : ℝ) := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

def g (x : ℝ) := a * Real.exp x + x

theorem f_unique_zero_and_inequality (h : a ≥ 0) :
  (∃! x, f a x = 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ > -1 → x₂ > -1 → f a x₁ = g a x₁ - g a x₂ → x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2) :=
sorry

end

end NUMINAMATH_CALUDE_f_unique_zero_and_inequality_l2948_294813


namespace NUMINAMATH_CALUDE_group_size_l2948_294838

theorem group_size (total : ℕ) (over_30 : ℕ) (under_20 : ℕ) 
  (h1 : over_30 = 90)
  (h2 : total = over_30 + under_20)
  (h3 : (under_20 : ℚ) / total = 1 / 10) : 
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_group_size_l2948_294838


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_inequality_l2948_294804

theorem unique_integer_satisfying_inequality : 
  ∃! (n : ℕ), n > 0 ∧ (105 * n : ℝ)^30 > (n : ℝ)^90 ∧ (n : ℝ)^90 > 3^180 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_inequality_l2948_294804


namespace NUMINAMATH_CALUDE_tangent_line_max_difference_l2948_294802

theorem tangent_line_max_difference (m n : ℝ) :
  ((m + 1)^2 + (n + 1)^2 = 4) →  -- Condition for tangent line
  (∀ x y : ℝ, (m + 1) * x + (n + 1) * y = 2 → x^2 + y^2 ≤ 1) →  -- Line touches or is outside the circle
  (∃ x y : ℝ, (m + 1) * x + (n + 1) * y = 2 ∧ x^2 + y^2 = 1) →  -- Line touches the circle at least at one point
  (m - n ≤ 2 * Real.sqrt 2) ∧ (∃ m₀ n₀ : ℝ, m₀ - n₀ = 2 * Real.sqrt 2 ∧ 
    ((m₀ + 1)^2 + (n₀ + 1)^2 = 4) ∧
    (∀ x y : ℝ, (m₀ + 1) * x + (n₀ + 1) * y = 2 → x^2 + y^2 ≤ 1) ∧
    (∃ x y : ℝ, (m₀ + 1) * x + (n₀ + 1) * y = 2 ∧ x^2 + y^2 = 1)) :=
by sorry


end NUMINAMATH_CALUDE_tangent_line_max_difference_l2948_294802


namespace NUMINAMATH_CALUDE_divisible_by_five_l2948_294815

theorem divisible_by_five (a b : ℕ) : 
  (5 ∣ a * b) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l2948_294815


namespace NUMINAMATH_CALUDE_garden_area_increase_l2948_294859

/-- Represents a rectangular garden with given length and width -/
structure RectGarden where
  length : ℝ
  width : ℝ

/-- Represents a square garden with given side length -/
structure SquareGarden where
  side : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def RectGarden.perimeter (g : RectGarden) : ℝ :=
  2 * (g.length + g.width)

/-- Calculates the area of a rectangular garden -/
def RectGarden.area (g : RectGarden) : ℝ :=
  g.length * g.width

/-- Calculates the perimeter of a square garden -/
def SquareGarden.perimeter (g : SquareGarden) : ℝ :=
  4 * g.side

/-- Calculates the area of a square garden -/
def SquareGarden.area (g : SquareGarden) : ℝ :=
  g.side * g.side

/-- Theorem: Changing a 60 ft by 20 ft rectangular garden to a square garden 
    with the same perimeter increases the area by 400 square feet -/
theorem garden_area_increase :
  let rect := RectGarden.mk 60 20
  let square := SquareGarden.mk (rect.perimeter / 4)
  square.area - rect.area = 400 := by
  sorry


end NUMINAMATH_CALUDE_garden_area_increase_l2948_294859


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l2948_294877

/-- The percentage of motorists who exceed the speed limit -/
def exceed_speed_limit : ℝ := 12.5

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_percentage : ℝ := 20

/-- The percentage of motorists who receive speeding tickets -/
def receive_ticket_percentage : ℝ := 10

/-- Theorem stating that the percentage of motorists receiving speeding tickets is 10% -/
theorem speeding_ticket_percentage :
  receive_ticket_percentage = exceed_speed_limit * (100 - no_ticket_percentage) / 100 := by
  sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l2948_294877


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2948_294867

def vec_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-4, 2)  -- Derived from a - (1/2)b = (3,1)
  let c : ℝ × ℝ := (x, 3)
  vec_parallel (2 * a + b) c → x = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2948_294867


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l2948_294857

theorem root_equation_implies_expression_value (a : ℝ) : 
  a^2 + a - 1 = 0 → 2021 - 2*a^2 - 2*a = 2019 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l2948_294857


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_1105_l2948_294899

theorem modular_inverse_11_mod_1105 :
  let m : ℕ := 1105
  let a : ℕ := 11
  let b : ℕ := 201
  m = 5 * 13 * 17 →
  (a * b) % m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_1105_l2948_294899


namespace NUMINAMATH_CALUDE_polynomial_equality_l2948_294864

theorem polynomial_equality (a b : ℝ) :
  (∀ x : ℝ, (x - 2) * (x + 3) = x^2 + a*x + b) →
  (a = 1 ∧ b = -6) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2948_294864


namespace NUMINAMATH_CALUDE_game_c_higher_probability_l2948_294851

-- Define the probability of getting heads
def p_heads : ℚ := 2/3

-- Define the probability of getting tails
def p_tails : ℚ := 1/3

-- Define the probability of winning Game C
def p_game_c : ℚ :=
  let p_first_three := p_heads^3 + p_tails^3
  let p_last_three := p_heads^3 + p_tails^3
  let p_overlap := p_heads^5 + p_tails^5
  p_first_three + p_last_three - p_overlap

-- Define the probability of winning Game D
def p_game_d : ℚ :=
  let p_first_last_two := (p_heads^2 + p_tails^2)^2
  let p_middle_three := p_heads^3 + p_tails^3
  let p_overlap := 2 * (p_heads^4 + p_tails^4)
  p_first_last_two + p_middle_three - p_overlap

-- Theorem statement
theorem game_c_higher_probability :
  p_game_c - p_game_d = 29/81 :=
sorry

end NUMINAMATH_CALUDE_game_c_higher_probability_l2948_294851


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l2948_294890

theorem smallest_undefined_value (y : ℝ) :
  let f := fun y : ℝ => (y - 3) / (9 * y^2 - 56 * y + 7)
  let roots := {y : ℝ | 9 * y^2 - 56 * y + 7 = 0}
  ∃ (smallest : ℝ), smallest ∈ roots ∧ 
    (∀ y ∈ roots, y ≥ smallest) ∧
    (∀ z < smallest, f z ≠ 0⁻¹) :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l2948_294890


namespace NUMINAMATH_CALUDE_new_stereo_price_l2948_294847

theorem new_stereo_price 
  (old_cost : ℝ) 
  (trade_in_percentage : ℝ) 
  (new_discount_percentage : ℝ) 
  (out_of_pocket : ℝ) 
  (h1 : old_cost = 250)
  (h2 : trade_in_percentage = 0.8)
  (h3 : new_discount_percentage = 0.25)
  (h4 : out_of_pocket = 250) :
  let trade_in_value := old_cost * trade_in_percentage
  let total_spent := trade_in_value + out_of_pocket
  let original_price := total_spent / (1 - new_discount_percentage)
  original_price = 600 := by sorry

end NUMINAMATH_CALUDE_new_stereo_price_l2948_294847


namespace NUMINAMATH_CALUDE_t_shirt_cost_l2948_294823

/-- The cost of one T-shirt -/
def T : ℝ := sorry

/-- The cost of one pair of pants -/
def pants_cost : ℝ := 80

/-- The cost of one pair of shoes -/
def shoes_cost : ℝ := 150

/-- The discount rate applied to all items -/
def discount_rate : ℝ := 0.9

/-- The total cost Eugene pays after discount -/
def total_cost : ℝ := 558

theorem t_shirt_cost : T = 20 := by
  have h1 : total_cost = discount_rate * (4 * T + 3 * pants_cost + 2 * shoes_cost) := by sorry
  sorry

end NUMINAMATH_CALUDE_t_shirt_cost_l2948_294823


namespace NUMINAMATH_CALUDE_decagon_painting_count_l2948_294800

/-- The number of ways to choose 4 colors from 8 available colors -/
def choose_colors : ℕ := Nat.choose 8 4

/-- The number of circular permutations of 4 colors -/
def circular_permutations : ℕ := Nat.factorial 3

/-- The number of distinct colorings of a decagon -/
def decagon_colorings : ℕ := choose_colors * circular_permutations / 2

/-- Theorem stating the number of distinct ways to paint the decagon -/
theorem decagon_painting_count : decagon_colorings = 210 := by
  sorry

#eval decagon_colorings

end NUMINAMATH_CALUDE_decagon_painting_count_l2948_294800


namespace NUMINAMATH_CALUDE_inequality_for_positive_numbers_l2948_294861

theorem inequality_for_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (a + 1)⁻¹ + (b + 1)⁻¹ ≥ 4/3 := by sorry

end NUMINAMATH_CALUDE_inequality_for_positive_numbers_l2948_294861


namespace NUMINAMATH_CALUDE_power_of_two_equation_l2948_294870

theorem power_of_two_equation (m : ℤ) : 
  2^2000 - 3 * 2^1999 + 2^1998 - 2^1997 + 2^1996 = m * 2^1996 → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l2948_294870


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2948_294884

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (((t.leg : ℝ) ^ 2 - ((t.base : ℝ) / 2) ^ 2).sqrt) / 2

/-- Theorem stating the minimum perimeter of two noncongruent isosceles triangles
    with the same area and bases in the ratio 3:2 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    area t1 = area t2 ∧
    t1.base * 2 = t2.base * 3 ∧
    perimeter t1 = perimeter t2 ∧
    perimeter t1 = 508 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      area s1 = area s2 →
      s1.base * 2 = s2.base * 3 →
      perimeter s1 = perimeter s2 →
      perimeter s1 ≥ 508) :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2948_294884


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l2948_294876

theorem angle_sum_around_point (y : ℝ) (h : y > 0) : 
  6 * y + 3 * y + 4 * y + 2 * y + y + 5 * y = 360 → y = 120 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l2948_294876


namespace NUMINAMATH_CALUDE_garden_fence_posts_l2948_294809

/-- Calculates the number of posts needed for a rectangular garden fence --/
def fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side_posts := length / post_spacing + 1
  let short_side_posts := width / post_spacing + 1
  long_side_posts + 2 * (short_side_posts - 1)

/-- Theorem stating the minimum number of posts required for the specified garden --/
theorem garden_fence_posts :
  fence_posts 100 50 10 = 21 :=
sorry

end NUMINAMATH_CALUDE_garden_fence_posts_l2948_294809


namespace NUMINAMATH_CALUDE_binomial_8_3_l2948_294856

theorem binomial_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_3_l2948_294856


namespace NUMINAMATH_CALUDE_unique_solution_f_two_equals_four_l2948_294894

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * (f x) * y + y^2

/-- The theorem stating that x^2 is the only function satisfying the equation -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  ∀ x : ℝ, f x = x^2 :=
sorry

/-- The value of f(2) is 4 -/
theorem f_two_equals_four (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  f 2 = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_f_two_equals_four_l2948_294894


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2948_294820

theorem sum_of_cubes :
  (∀ n : ℤ, ∃ a b c d : ℤ, 6 * n = a^3 + b^3 + c^3 + d^3) ∧
  (∀ k : ℤ, ∃ a b c d e : ℤ, k = a^3 + b^3 + c^3 + d^3 + e^3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2948_294820


namespace NUMINAMATH_CALUDE_exists_carmichael_number_l2948_294863

theorem exists_carmichael_number : 
  ∃ n : ℕ, 
    n > 1 ∧ 
    ¬(Nat.Prime n) ∧ 
    ∀ a : ℤ, (a^n) % n = a % n :=
by sorry

end NUMINAMATH_CALUDE_exists_carmichael_number_l2948_294863


namespace NUMINAMATH_CALUDE_min_sum_of_fractions_l2948_294810

def Digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_sum_of_fractions (A B C D : Nat) :
  A ∈ Digits → B ∈ Digits → C ∈ Digits → D ∈ Digits →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  B ≠ 0 → D ≠ 0 →
  (∀ A' B' C' D' : Nat,
    A' ∈ Digits → B' ∈ Digits → C' ∈ Digits → D' ∈ Digits →
    A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
    B' ≠ 0 → D' ≠ 0 →
    (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) ≤ (A' : ℚ) / (B' : ℚ) + (C' : ℚ) / (D' : ℚ)) →
  (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_fractions_l2948_294810


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2948_294839

def f (x : ℝ) : ℝ := 3 - 2*x

theorem solution_set_of_inequality (x : ℝ) :
  (|f (x + 1) + 2| ≤ 3) ↔ (0 ≤ x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2948_294839


namespace NUMINAMATH_CALUDE_senate_arrangement_l2948_294801

/-- The number of ways to arrange senators around a circular table. -/
def arrange_senators (num_democrats num_republicans : ℕ) : ℕ :=
  (num_republicans - 1).factorial * (num_republicans.choose num_democrats) * num_democrats.factorial

/-- Theorem: The number of ways to arrange 4 Democrats and 6 Republicans around a circular table
    such that no two Democrats sit next to each other is 43,200. -/
theorem senate_arrangement :
  arrange_senators 4 6 = 43200 :=
sorry

end NUMINAMATH_CALUDE_senate_arrangement_l2948_294801


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l2948_294883

/-- Proves that the total number of coins in each chest is 1000 given the pirate's distribution rules --/
theorem pirate_treasure_distribution (total_gold : ℕ) (total_silver : ℕ) (num_chests : ℕ) :
  total_gold = 3500 →
  total_silver = 500 →
  num_chests = 5 →
  let gold_per_chest : ℕ := total_gold / num_chests
  let silver_per_chest : ℕ := total_silver / num_chests
  let bronze_per_chest : ℕ := 2 * silver_per_chest
  gold_per_chest + silver_per_chest + bronze_per_chest = 1000 :=
by
  sorry

#eval (3500 / 5) + (500 / 5) + 2 * (500 / 5)

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l2948_294883


namespace NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l2948_294834

theorem bob_pennies : ℕ → ℕ → Prop :=
  fun a b =>
    (b + 1 = 4 * (a - 1)) ∧
    (b - 1 = 3 * (a + 1)) →
    b = 31

-- The proof goes here
theorem bob_pennies_proof : bob_pennies 9 31 := by
  sorry

end NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l2948_294834


namespace NUMINAMATH_CALUDE_inequality_proof_l2948_294886

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2948_294886


namespace NUMINAMATH_CALUDE_travel_distance_proof_l2948_294880

theorem travel_distance_proof (total_distance : ℝ) (plane_fraction : ℝ) (train_to_bus_ratio : ℝ) 
  (h1 : total_distance = 1800)
  (h2 : plane_fraction = 1/3)
  (h3 : train_to_bus_ratio = 2/3) : 
  ∃ (bus_distance : ℝ), 
    bus_distance = 720 ∧ 
    plane_fraction * total_distance + train_to_bus_ratio * bus_distance + bus_distance = total_distance :=
by sorry

end NUMINAMATH_CALUDE_travel_distance_proof_l2948_294880


namespace NUMINAMATH_CALUDE_exactly_one_and_two_white_mutually_exclusive_not_contradictory_l2948_294814

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The set of all possible outcomes when drawing two balls from the bag -/
def allOutcomes : Finset DrawOutcome := sorry

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Red) ∨
  (outcome.first = BallColor.Red ∧ outcome.second = BallColor.White)

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (event1 event2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, ¬(event1 outcome ∧ event2 outcome)

/-- Two events are contradictory if one of them must occur -/
def contradictory (event1 event2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, event1 outcome ∨ event2 outcome

theorem exactly_one_and_two_white_mutually_exclusive_not_contradictory :
  mutuallyExclusive exactlyOneWhite exactlyTwoWhite ∧
  ¬contradictory exactlyOneWhite exactlyTwoWhite :=
sorry

end NUMINAMATH_CALUDE_exactly_one_and_two_white_mutually_exclusive_not_contradictory_l2948_294814


namespace NUMINAMATH_CALUDE_integer_property_l2948_294854

theorem integer_property (k : ℕ) : k ≥ 3 → (
  (∃ m n : ℕ, 1 < m ∧ m < k ∧
              1 < n ∧ n < k ∧
              Nat.gcd m k = 1 ∧
              Nat.gcd n k = 1 ∧
              m + n > k ∧
              k ∣ (m - 1) * (n - 1))
  ↔ (k = 15 ∨ k = 30)
) := by
  sorry

end NUMINAMATH_CALUDE_integer_property_l2948_294854


namespace NUMINAMATH_CALUDE_overlapping_squares_theorem_l2948_294844

/-- Represents a rectangle with numbers placed inside it -/
structure NumberedRectangle where
  width : ℕ
  height : ℕ
  numbers : List ℕ

/-- Represents the result of rotating a NumberedRectangle by 180° -/
def rotate180 (nr : NumberedRectangle) : NumberedRectangle :=
  { width := nr.width,
    height := nr.height,
    numbers := [6, 1, 2, 1] }

/-- Calculates the number of overlapping shaded squares when a NumberedRectangle is overlaid with its 180° rotation -/
def overlappingSquares (nr : NumberedRectangle) : ℕ :=
  nr.width * nr.height - 10

/-- The main theorem to be proved -/
theorem overlapping_squares_theorem (nr : NumberedRectangle) :
  nr.width = 8 ∧ nr.height = 5 ∧ nr.numbers = [1, 2, 1, 9] →
  rotate180 nr = { width := 8, height := 5, numbers := [6, 1, 2, 1] } →
  overlappingSquares nr = 30 := by
  sorry

#check overlapping_squares_theorem

end NUMINAMATH_CALUDE_overlapping_squares_theorem_l2948_294844


namespace NUMINAMATH_CALUDE_trevors_future_age_l2948_294807

/-- Proves Trevor's age when his older brother is three times Trevor's current age -/
theorem trevors_future_age (t b : ℕ) (h1 : t = 11) (h2 : b = 20) :
  ∃ x : ℕ, b + (x - t) = 3 * t ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_trevors_future_age_l2948_294807


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l2948_294893

theorem imaginary_part_of_complex_product : Complex.im ((1 - Complex.I) * (3 + Complex.I)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l2948_294893


namespace NUMINAMATH_CALUDE_quadratic_equation_k_l2948_294897

/-- Given a quadratic equation x^2 - 3x + k = 0 with two real roots a and b,
    if ab + 2a + 2b = 1, then k = -5 -/
theorem quadratic_equation_k (a b k : ℝ) :
  (∀ x, x^2 - 3*x + k = 0 ↔ x = a ∨ x = b) →
  (a*b + 2*a + 2*b = 1) →
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_l2948_294897


namespace NUMINAMATH_CALUDE_jogging_problem_l2948_294891

/-- Jogging problem -/
theorem jogging_problem (total_distance : ℝ) (total_time : ℝ) (halfway_point : ℝ) :
  total_distance = 3 →
  total_time = 24 →
  halfway_point = total_distance / 2 →
  (halfway_point / total_distance) * total_time = 12 :=
by sorry

end NUMINAMATH_CALUDE_jogging_problem_l2948_294891


namespace NUMINAMATH_CALUDE_beaus_sons_age_l2948_294875

theorem beaus_sons_age (beau_age : ℕ) (sons_age : ℕ) : 
  beau_age = 42 →
  3 * (sons_age - 3) = beau_age - 3 →
  sons_age = 16 := by
sorry

end NUMINAMATH_CALUDE_beaus_sons_age_l2948_294875


namespace NUMINAMATH_CALUDE_minimize_distance_to_point_l2948_294825

/-- Given points P(-2, -2) and R(2, m), prove that the value of m that minimizes 
    the distance PR is -2. -/
theorem minimize_distance_to_point (m : ℝ) : 
  let P : ℝ × ℝ := (-2, -2)
  let R : ℝ × ℝ := (2, m)
  let distance := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  (∀ k : ℝ, distance ≤ Real.sqrt ((P.1 - 2)^2 + (P.2 - k)^2)) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_minimize_distance_to_point_l2948_294825


namespace NUMINAMATH_CALUDE_cosine_amplitude_l2948_294840

/-- Given a cosine function y = a cos(bx) where a > 0 and b > 0,
    prove that a equals the maximum y-value of the graph. -/
theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, ∃ y, y = a * Real.cos (b * x)) →
  (∃ M, M > 0 ∧ ∀ x, a * Real.cos (b * x) ≤ M) →
  (∀ ε > 0, ∃ x, a * Real.cos (b * x) > M - ε) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l2948_294840


namespace NUMINAMATH_CALUDE_positive_sum_l2948_294835

theorem positive_sum (x y z : ℝ) 
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hz : 0.5 < z ∧ z < 1) : 
  y + z > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_l2948_294835


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l2948_294828

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle time -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationWindow (cycle : TrafficLightCycle) (observationDuration : ℕ) : ℕ :=
  3 * observationDuration  -- 3 color changes per cycle

/-- The probability of observing a color change -/
def probabilityOfChange (cycle : TrafficLightCycle) (observationDuration : ℕ) : ℚ :=
  changeObservationWindow cycle observationDuration / cycleDuration cycle

theorem traffic_light_change_probability :
  let cycle := TrafficLightCycle.mk 45 5 40
  let observationDuration := 4
  probabilityOfChange cycle observationDuration = 2 / 15 := by
  sorry


end NUMINAMATH_CALUDE_traffic_light_change_probability_l2948_294828


namespace NUMINAMATH_CALUDE_problem_solution_l2948_294873

/-- f(n) denotes the nth positive integer which is not a perfect square -/
def f (n : ℕ) : ℕ := sorry

/-- Applies the function f n times -/
def iterateF (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | m + 1 => f (iterateF m x)

theorem problem_solution :
  ∃ (n : ℕ), n > 0 ∧ iterateF 2013 n = 2014^2 + 1 ∧ n = 6077248 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2948_294873


namespace NUMINAMATH_CALUDE_max_surface_area_increase_l2948_294892

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cuboid -/
def surfaceArea (d : CuboidDimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- The dimensions of the original small cuboid -/
def originalCuboid : CuboidDimensions :=
  { length := 3, width := 2, height := 1 }

/-- Theorem stating the maximum increase in surface area -/
theorem max_surface_area_increase :
  ∃ (finalCuboid : CuboidDimensions),
    surfaceArea finalCuboid - surfaceArea originalCuboid ≤ 10 ∧
    ∀ (otherCuboid : CuboidDimensions),
      surfaceArea otherCuboid - surfaceArea originalCuboid ≤
        surfaceArea finalCuboid - surfaceArea originalCuboid :=
by sorry

end NUMINAMATH_CALUDE_max_surface_area_increase_l2948_294892


namespace NUMINAMATH_CALUDE_video_game_map_area_l2948_294853

-- Define the map dimensions
def map_width : ℝ := 10
def map_length : ℝ := 2

-- Define the area of a rectangle
def rectangle_area (width length : ℝ) : ℝ := width * length

-- Theorem statement
theorem video_game_map_area : rectangle_area map_width map_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_video_game_map_area_l2948_294853


namespace NUMINAMATH_CALUDE_combined_salaries_l2948_294808

theorem combined_salaries (salary_A : ℕ) (num_people : ℕ) (avg_salary : ℕ) : 
  salary_A = 8000 → 
  num_people = 5 → 
  avg_salary = 9000 → 
  (avg_salary * num_people - salary_A = 37000) := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l2948_294808


namespace NUMINAMATH_CALUDE_special_polynomial_form_l2948_294885

/-- A polynomial satisfying the given conditions -/
class SpecialPolynomial (P : ℝ → ℝ) where
  zero_condition : P 0 = 0
  functional_equation : ∀ x : ℝ, P x = (1/2) * (P (x + 1) + P (x - 1))

/-- Theorem stating that any polynomial satisfying the given conditions is of the form P(x) = ax -/
theorem special_polynomial_form {P : ℝ → ℝ} [SpecialPolynomial P] : 
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_form_l2948_294885


namespace NUMINAMATH_CALUDE_front_page_stickers_l2948_294887

/-- Given:
  * initial_stickers: The initial number of stickers Mary had
  * pages: The number of pages (excluding the front page) where Mary used stickers
  * stickers_per_page: The number of stickers Mary used on each page (excluding the front page)
  * remaining_stickers: The number of stickers Mary has left
  
  Prove that the number of large stickers used on the front page is 3
-/
theorem front_page_stickers 
  (initial_stickers : ℕ) 
  (pages : ℕ) 
  (stickers_per_page : ℕ) 
  (remaining_stickers : ℕ) 
  (h1 : initial_stickers = 89)
  (h2 : pages = 6)
  (h3 : stickers_per_page = 7)
  (h4 : remaining_stickers = 44) :
  initial_stickers - (pages * stickers_per_page) - remaining_stickers = 3 :=
by sorry

end NUMINAMATH_CALUDE_front_page_stickers_l2948_294887


namespace NUMINAMATH_CALUDE_four_roots_iff_a_in_range_l2948_294830

-- Define the function f(x) = |x^2 + 3x|
def f (x : ℝ) : ℝ := |x^2 + 3*x|

-- Define the equation f(x) - a|x-1| = 0
def equation (a : ℝ) (x : ℝ) : Prop := f x - a * |x - 1| = 0

-- Define the property of having exactly 4 distinct real roots
def has_four_distinct_roots (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    equation a x₁ ∧ equation a x₂ ∧ equation a x₃ ∧ equation a x₄ ∧
    ∀ (x : ℝ), equation a x → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄

-- Theorem statement
theorem four_roots_iff_a_in_range :
  ∀ a : ℝ, has_four_distinct_roots a ↔ (a ∈ Set.Ioo 0 1 ∪ Set.Ioi 9) :=
sorry

end NUMINAMATH_CALUDE_four_roots_iff_a_in_range_l2948_294830


namespace NUMINAMATH_CALUDE_expression_evaluation_l2948_294874

theorem expression_evaluation (x y z k : ℤ) 
  (hx : x = 25) (hy : y = 12) (hz : z = 3) (hk : k = 4) :
  (x - (y - z)) - ((x - y) - (z + k)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2948_294874


namespace NUMINAMATH_CALUDE_five_percent_difference_l2948_294841

theorem five_percent_difference (x y : ℝ) 
  (hx : 5 = 0.25 * x) 
  (hy : 5 = 0.50 * y) : 
  x - y = 10 := by
sorry

end NUMINAMATH_CALUDE_five_percent_difference_l2948_294841


namespace NUMINAMATH_CALUDE_white_go_stones_l2948_294869

theorem white_go_stones (total : ℕ) (difference : ℕ) (white : ℕ) (black : ℕ) : 
  total = 120 →
  difference = 36 →
  white = black + difference →
  total = white + black →
  white = 78 := by
sorry

end NUMINAMATH_CALUDE_white_go_stones_l2948_294869


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l2948_294837

theorem power_fraction_simplification :
  (2^2020 - 2^2018) / (2^2020 + 2^2018) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l2948_294837


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_specific_perpendicular_line_l2948_294833

/-- A line passing through a point and perpendicular to another line -/
theorem perpendicular_line_through_point 
  (x₀ y₀ : ℝ) 
  (a b c : ℝ) 
  (h₁ : b ≠ 0) 
  (h₂ : a ≠ 0) :
  ∃ m k : ℝ, 
    (y₀ = m * x₀ + k) ∧ 
    (m = -a / b) ∧
    (k = y₀ - m * x₀) :=
sorry

/-- The specific line passing through (3, -5) and perpendicular to 2x - 6y + 15 = 0 -/
theorem specific_perpendicular_line : 
  ∃ m k : ℝ, 
    (-5 = m * 3 + k) ∧ 
    (m = -(2 : ℝ) / (-6 : ℝ)) ∧ 
    (k = -5 - m * 3) ∧
    (k = -4) ∧ 
    (m = 3) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_specific_perpendicular_line_l2948_294833


namespace NUMINAMATH_CALUDE_square_49_using_50_l2948_294812

theorem square_49_using_50 : ∃ x : ℕ, 49^2 = 50^2 - x + 1 ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_49_using_50_l2948_294812


namespace NUMINAMATH_CALUDE_five_heads_before_two_tails_l2948_294895

/-- The probability of getting 5 heads before 2 consecutive tails when repeatedly flipping a fair coin -/
def probability_5H_before_2T : ℚ :=
  3 / 34

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℚ → Prop) : Prop :=
  p (1/2) ∧ p (1/2)

theorem five_heads_before_two_tails (p : ℚ → Prop) (h : fair_coin p) :
  probability_5H_before_2T = 3 / 34 :=
sorry

end NUMINAMATH_CALUDE_five_heads_before_two_tails_l2948_294895


namespace NUMINAMATH_CALUDE_road_trip_total_hours_l2948_294849

/-- Calculates the total hours spent on a road trip -/
def total_road_trip_hours (jade_hours : Fin 3 → ℕ) (krista_hours : Fin 3 → ℕ) (break_hours : ℕ) : ℕ :=
  (Finset.sum Finset.univ (λ i => jade_hours i + krista_hours i)) + 3 * break_hours

theorem road_trip_total_hours : 
  let jade_hours : Fin 3 → ℕ := ![8, 7, 6]
  let krista_hours : Fin 3 → ℕ := ![6, 5, 4]
  let break_hours : ℕ := 2
  total_road_trip_hours jade_hours krista_hours break_hours = 42 := by
  sorry

#eval total_road_trip_hours ![8, 7, 6] ![6, 5, 4] 2

end NUMINAMATH_CALUDE_road_trip_total_hours_l2948_294849


namespace NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_seven_satisfies_inequality_seven_is_smallest_l2948_294843

theorem smallest_b_for_quadratic_inequality :
  ∀ b : ℝ, b^2 - 16*b + 63 ≤ 0 → b ≥ 7 :=
by
  sorry

theorem seven_satisfies_inequality : 
  7^2 - 16*7 + 63 ≤ 0 :=
by
  sorry

theorem seven_is_smallest :
  ∀ b : ℝ, b^2 - 16*b + 63 ≤ 0 → b ≥ 7 ∧ 
  (∃ ε > 0, (7 - ε)^2 - 16*(7 - ε) + 63 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_seven_satisfies_inequality_seven_is_smallest_l2948_294843


namespace NUMINAMATH_CALUDE_work_speed_l2948_294868

/-- Proves that given a round trip of 2 hours, 72 minutes to work, and 90 km/h return speed, the speed to work is 60 km/h -/
theorem work_speed (total_time : Real) (time_to_work : Real) (return_speed : Real) :
  total_time = 2 ∧ 
  time_to_work = 72 / 60 ∧ 
  return_speed = 90 →
  (2 * return_speed * time_to_work) / (total_time + time_to_work) = 60 := by
  sorry

end NUMINAMATH_CALUDE_work_speed_l2948_294868


namespace NUMINAMATH_CALUDE_product_102_104_divisible_by_8_l2948_294803

theorem product_102_104_divisible_by_8 : (102 * 104) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_102_104_divisible_by_8_l2948_294803


namespace NUMINAMATH_CALUDE_sum_abs_roots_quadratic_l2948_294879

theorem sum_abs_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * r₁^2 + b * r₁ + c = 0 ∧ 
  a * r₂^2 + b * r₂ + c = 0 →
  |r₁| + |r₂| = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_abs_roots_quadratic_l2948_294879


namespace NUMINAMATH_CALUDE_equation_solution_l2948_294824

theorem equation_solution : ∃ (x : ℝ), 
  x > 0 ∧ 
  (1/4) * (5*x^2 - 4) = (x^2 - 40*x - 5) * (x^2 + 20*x + 2) ∧
  x = 20 + 10 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2948_294824


namespace NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l2948_294826

/-- A regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon) : ℕ := p.n

/-- The smallest positive angle of rotational symmetry for a regular polygon (in degrees) -/
def rotationalSymmetryAngle (p : RegularPolygon) : ℚ := 360 / p.n

/-- The theorem to be proved -/
theorem regular_18gon_symmetry_sum :
  let p : RegularPolygon := ⟨18, by norm_num⟩
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by
  sorry

end NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l2948_294826


namespace NUMINAMATH_CALUDE_decimal_addition_l2948_294865

theorem decimal_addition : (0.0935 : ℚ) + (0.007 : ℚ) + (0.2 : ℚ) = (0.3005 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l2948_294865


namespace NUMINAMATH_CALUDE_right_triangle_rotation_l2948_294882

theorem right_triangle_rotation (x y : ℝ) : 
  x > 0 → y > 0 →
  (1 / 3) * π * y^2 * x = 1080 * π →
  (1 / 3) * π * x^2 * y = 2430 * π →
  Real.sqrt (x^2 + y^2) = 6 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_rotation_l2948_294882


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2948_294822

/-- A hyperbola with asymptotes forming an acute angle of 60° and passing through (√2, √3) -/
structure Hyperbola where
  /-- The acute angle formed by the asymptotes -/
  angle : ℝ
  /-- The point through which the hyperbola passes -/
  point : ℝ × ℝ
  /-- The angle is 60° -/
  angle_is_60 : angle = 60 * π / 180
  /-- The point is (√2, √3) -/
  point_is_sqrt : point = (Real.sqrt 2, Real.sqrt 3)

/-- The standard equation of the hyperbola -/
def standard_equation (h : Hyperbola) : (ℝ → ℝ → Prop) → Prop :=
  λ eq ↦ (eq = λ x y ↦ x^2/1 - y^2/3 = 1) ∨ (eq = λ x y ↦ x^2/7 - y^2/(7/3) = 1)

/-- Theorem stating that the given hyperbola has one of the two standard equations -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  ∃ eq : ℝ → ℝ → Prop, standard_equation h eq :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2948_294822
