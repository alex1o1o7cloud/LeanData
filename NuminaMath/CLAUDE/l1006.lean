import Mathlib

namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1006_100670

/-- Given that a² and √b vary inversely, a = 3 when b = 36, and ab = 108, prove that b = 36 -/
theorem inverse_variation_problem (a b : ℝ) (h1 : ∃ k : ℝ, a^2 * Real.sqrt b = k)
  (h2 : a = 3 ∧ b = 36) (h3 : a * b = 108) : b = 36 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1006_100670


namespace NUMINAMATH_CALUDE_min_value_quadratic_function_l1006_100641

theorem min_value_quadratic_function :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x
  ∃ m : ℝ, m = -3 ∧ ∀ x : ℝ, x ∈ Set.Icc 0 1 → f x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_function_l1006_100641


namespace NUMINAMATH_CALUDE_divisibility_and_ratio_theorem_l1006_100607

theorem divisibility_and_ratio_theorem (k : ℕ) (h : k > 1) :
  ∃ a b : ℕ, 1 < a ∧ a < b ∧ (a^2 + b^2 - 1) / (a * b) = k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_and_ratio_theorem_l1006_100607


namespace NUMINAMATH_CALUDE_function_bound_l1006_100636

-- Define the properties of functions f and g
def satisfies_functional_equation (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y

def not_identically_zero (f : ℝ → ℝ) : Prop :=
  ∃ x, f x ≠ 0

def bounded_by_one (f : ℝ → ℝ) : Prop :=
  ∀ x, |f x| ≤ 1

-- Theorem statement
theorem function_bound (f g : ℝ → ℝ) 
  (h1 : satisfies_functional_equation f g)
  (h2 : not_identically_zero f)
  (h3 : bounded_by_one f) :
  bounded_by_one g :=
sorry

end NUMINAMATH_CALUDE_function_bound_l1006_100636


namespace NUMINAMATH_CALUDE_probability_of_mathematics_letter_l1006_100673

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of unique letters in "MATHEMATICS" -/
def unique_letters : ℕ := 8

/-- The probability of selecting a letter from "MATHEMATICS" -/
def probability : ℚ := unique_letters / alphabet_size

theorem probability_of_mathematics_letter :
  probability = 4 / 13 := by sorry

end NUMINAMATH_CALUDE_probability_of_mathematics_letter_l1006_100673


namespace NUMINAMATH_CALUDE_parabola_vertex_l1006_100628

def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 4*y + 2*x + 8 = 0

def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x' y' : ℝ), eq x' y' → (x' - x)^2 + (y' - y)^2 ≥ 0

theorem parabola_vertex :
  is_vertex (-2) 2 parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1006_100628


namespace NUMINAMATH_CALUDE_book_cost_problem_l1006_100687

theorem book_cost_problem (cost_of_three : ℝ) (h : cost_of_three = 45) :
  let cost_of_one : ℝ := cost_of_three / 3
  let cost_of_seven : ℝ := 7 * cost_of_one
  cost_of_seven = 105 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l1006_100687


namespace NUMINAMATH_CALUDE_min_value_of_function_l1006_100639

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  let y := x + 4 / (x - 1)
  (∀ z, z > 1 → y ≤ z + 4 / (z - 1)) ∧ y = 5 ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1006_100639


namespace NUMINAMATH_CALUDE_original_station_count_l1006_100697

/-- The number of combinations of 2 items from a set of k items -/
def combinations (k : ℕ) : ℕ := k * (k - 1) / 2

/-- 
Given:
- m is the original number of stations
- n is the number of new stations added (n > 1)
- The increase in types of passenger tickets is 58

Prove that m = 14
-/
theorem original_station_count (m n : ℕ) 
  (h1 : n > 1) 
  (h2 : combinations (m + n) - combinations m = 58) : 
  m = 14 := by sorry

end NUMINAMATH_CALUDE_original_station_count_l1006_100697


namespace NUMINAMATH_CALUDE_inequality_proof_l1006_100642

theorem inequality_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^4 + y^4 + 2 / (x^2 * y^2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1006_100642


namespace NUMINAMATH_CALUDE_abc_sum_l1006_100695

theorem abc_sum (a b c : ℕ) : 
  (10 ≤ a ∧ a < 100) → 
  (10 ≤ b ∧ b < 100) → 
  (10 ≤ c ∧ c < 100) → 
  a < b → 
  b < c → 
  a * b * c = 3960 → 
  Even (a + b + c) → 
  a + b + c = 50 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l1006_100695


namespace NUMINAMATH_CALUDE_probability_adjacent_points_hexagon_l1006_100643

/-- The number of points on the regular hexagon -/
def num_points : ℕ := 6

/-- The number of adjacent pairs on the regular hexagon -/
def num_adjacent_pairs : ℕ := 6

/-- The probability of selecting two adjacent points on a regular hexagon -/
theorem probability_adjacent_points_hexagon : 
  (num_adjacent_pairs : ℚ) / (num_points.choose 2) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_adjacent_points_hexagon_l1006_100643


namespace NUMINAMATH_CALUDE_vector_opposite_directions_x_value_l1006_100683

/-- Two vectors are in opposite directions if their dot product is negative -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 < 0

theorem vector_opposite_directions_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (4, x)
  opposite_directions a b → x = -2 := by
sorry

end NUMINAMATH_CALUDE_vector_opposite_directions_x_value_l1006_100683


namespace NUMINAMATH_CALUDE_batsman_inning_number_l1006_100644

/-- Represents the batting statistics of a cricket player -/
structure BattingStats where
  totalRuns : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the new average after adding runs to the existing stats -/
def newAverage (stats : BattingStats) (newRuns : ℕ) : ℚ :=
  (stats.totalRuns + newRuns) / (stats.innings + 1)

theorem batsman_inning_number (stats : BattingStats) (h1 : newAverage stats 88 = 40)
    (h2 : stats.average = 37) : stats.innings + 1 = 17 := by
  sorry

#check batsman_inning_number

end NUMINAMATH_CALUDE_batsman_inning_number_l1006_100644


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1006_100633

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, x)
  are_parallel a b → x = -4 := by
    sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1006_100633


namespace NUMINAMATH_CALUDE_basketball_spectators_l1006_100637

theorem basketball_spectators (total : Nat) (men : Nat) (women : Nat) (children : Nat) :
  total = 10000 →
  men = 7000 →
  total = men + women + children →
  children = 5 * women →
  children = 2500 := by
sorry

end NUMINAMATH_CALUDE_basketball_spectators_l1006_100637


namespace NUMINAMATH_CALUDE_doug_lost_marbles_l1006_100672

theorem doug_lost_marbles (d : ℕ) (l : ℕ) : 
  (d + 22 = d - l + 30) → l = 8 := by
  sorry

end NUMINAMATH_CALUDE_doug_lost_marbles_l1006_100672


namespace NUMINAMATH_CALUDE_ab_value_l1006_100648

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1006_100648


namespace NUMINAMATH_CALUDE_tv_selection_theorem_l1006_100684

/-- The number of televisions of type A -/
def typeA : ℕ := 3

/-- The number of televisions of type B -/
def typeB : ℕ := 3

/-- The number of televisions of type C -/
def typeC : ℕ := 4

/-- The total number of televisions -/
def totalTVs : ℕ := typeA + typeB + typeC

/-- The number of televisions to be selected -/
def selectCount : ℕ := 3

/-- Calculates the number of ways to select r items from n items -/
def combination (n r : ℕ) : ℕ :=
  Nat.choose n r

/-- The theorem to be proved -/
theorem tv_selection_theorem : 
  combination totalTVs selectCount - 
  (combination typeA selectCount + combination typeB selectCount + combination typeC selectCount) = 114 := by
  sorry

end NUMINAMATH_CALUDE_tv_selection_theorem_l1006_100684


namespace NUMINAMATH_CALUDE_ferry_travel_time_difference_l1006_100614

/-- Represents the properties of a ferry --/
structure Ferry where
  baseSpeed : ℝ  -- Speed without current in km/h
  currentEffect : ℝ  -- Speed reduction due to current in km/h
  travelTime : ℝ  -- Travel time in hours
  routeLength : ℝ  -- Route length in km

/-- The problem setup --/
def ferryProblem : Prop := ∃ (p q : Ferry),
  -- Ferry p properties
  p.baseSpeed = 6 ∧
  p.currentEffect = 1 ∧
  p.travelTime = 3 ∧
  
  -- Ferry q properties
  q.baseSpeed = p.baseSpeed + 3 ∧
  q.currentEffect = p.currentEffect / 2 ∧
  q.routeLength = 2 * p.routeLength ∧
  
  -- Calculate effective speeds
  let pEffectiveSpeed := p.baseSpeed - p.currentEffect
  let qEffectiveSpeed := q.baseSpeed - q.currentEffect
  
  -- Calculate route lengths
  p.routeLength = pEffectiveSpeed * p.travelTime ∧
  
  -- Calculate q's travel time
  q.travelTime = q.routeLength / qEffectiveSpeed ∧
  
  -- The difference in travel time is approximately 0.5294 hours
  abs (q.travelTime - p.travelTime - 0.5294) < 0.0001

/-- The theorem to be proved --/
theorem ferry_travel_time_difference : ferryProblem := by
  sorry

end NUMINAMATH_CALUDE_ferry_travel_time_difference_l1006_100614


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_plane_perp_l1006_100671

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_plane_perp 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → planePerp α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_plane_perp_l1006_100671


namespace NUMINAMATH_CALUDE_b_hire_charges_l1006_100613

/-- Calculates the hire charges for a specific person given the total cost,
    and the hours used by each person. -/
def hireCharges (totalCost : ℚ) (hoursA hoursB hoursC : ℚ) : ℚ :=
  let totalHours := hoursA + hoursB + hoursC
  let costPerHour := totalCost / totalHours
  costPerHour * hoursB

theorem b_hire_charges :
  hireCharges 720 9 10 13 = 225 := by
  sorry

end NUMINAMATH_CALUDE_b_hire_charges_l1006_100613


namespace NUMINAMATH_CALUDE_marble_problem_l1006_100623

theorem marble_problem (a : ℚ) : 
  (∃ (brian caden daryl : ℚ),
    brian = 2 * a ∧
    caden = 3 * brian ∧
    daryl = 6 * caden ∧
    a + brian + caden + daryl = 150) →
  a = 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l1006_100623


namespace NUMINAMATH_CALUDE_words_with_consonant_l1006_100610

def letter_set : Finset Char := {'A', 'B', 'C', 'D', 'E'}
def vowel_set : Finset Char := {'A', 'E'}
def word_length : Nat := 5

def total_words : Nat := letter_set.card ^ word_length
def all_vowel_words : Nat := vowel_set.card ^ word_length

theorem words_with_consonant :
  total_words - all_vowel_words = 3093 :=
sorry

end NUMINAMATH_CALUDE_words_with_consonant_l1006_100610


namespace NUMINAMATH_CALUDE_largest_percentage_increase_l1006_100632

def students : Fin 7 → ℕ
  | 0 => 80  -- 2010
  | 1 => 85  -- 2011
  | 2 => 88  -- 2012
  | 3 => 90  -- 2013
  | 4 => 95  -- 2014
  | 5 => 100 -- 2015
  | 6 => 120 -- 2016

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseYears : Fin 6 := 5 -- Represents 2015 to 2016

theorem largest_percentage_increase :
  ∀ i : Fin 6, percentageIncrease (students i) (students (i.succ)) ≤ 
    percentageIncrease (students largestIncreaseYears) (students (largestIncreaseYears.succ)) := by
  sorry

end NUMINAMATH_CALUDE_largest_percentage_increase_l1006_100632


namespace NUMINAMATH_CALUDE_x_varies_as_z_to_four_fifths_l1006_100677

/-- Given that x varies as the square of y, y varies as the square of w, 
    and w varies as the fifth root of z, prove that x varies as z^(4/5) -/
theorem x_varies_as_z_to_four_fifths 
  (hxy : ∃ k : ℝ, ∀ x y : ℝ, x = k * y^2)
  (hyw : ∃ j : ℝ, ∀ y w : ℝ, y = j * w^2)
  (hwz : ∃ c : ℝ, ∀ w z : ℝ, w = c * z^(1/5)) :
  ∃ m : ℝ, ∀ x z : ℝ, x = m * z^(4/5) := by
  sorry

end NUMINAMATH_CALUDE_x_varies_as_z_to_four_fifths_l1006_100677


namespace NUMINAMATH_CALUDE_pizza_burger_cost_ratio_l1006_100627

/-- The cost ratio of pizza to burger given certain conditions -/
theorem pizza_burger_cost_ratio :
  let burger_cost : ℚ := 9
  let pizza_cost : ℚ → ℚ := λ k => k * burger_cost
  ∀ k : ℚ, pizza_cost k + 3 * burger_cost = 45 →
  pizza_cost k / burger_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_pizza_burger_cost_ratio_l1006_100627


namespace NUMINAMATH_CALUDE_regression_lines_common_point_l1006_100617

-- Define the type for a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the type for a line in 2D space
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Define the theorem
theorem regression_lines_common_point
  (s t : ℝ)
  (l₁ l₂ : Line)
  (h₁ : pointOnLine ⟨s, t⟩ l₁)
  (h₂ : pointOnLine ⟨s, t⟩ l₂) :
  ∃ (p : Point), pointOnLine p l₁ ∧ pointOnLine p l₂ :=
by sorry

end NUMINAMATH_CALUDE_regression_lines_common_point_l1006_100617


namespace NUMINAMATH_CALUDE_heather_total_distance_l1006_100681

/-- The distance Heather bicycled per day in kilometers -/
def distance_per_day : ℝ := 40.0

/-- The number of days Heather bicycled -/
def number_of_days : ℝ := 8.0

/-- The total distance Heather bicycled -/
def total_distance : ℝ := distance_per_day * number_of_days

theorem heather_total_distance : total_distance = 320.0 := by
  sorry

end NUMINAMATH_CALUDE_heather_total_distance_l1006_100681


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l1006_100600

theorem three_digit_number_divisible_by_seven (a b : ℕ) 
  (h1 : a ≥ 1 ∧ a ≤ 9) 
  (h2 : b ≥ 0 ∧ b ≤ 9) 
  (h3 : (a + b + b) % 7 = 0) : 
  ∃ k : ℕ, (100 * a + 10 * b + b) = 7 * k :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l1006_100600


namespace NUMINAMATH_CALUDE_base_eight_1563_to_ten_l1006_100665

def base_eight_to_ten (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem base_eight_1563_to_ten :
  base_eight_to_ten 1563 = 883 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_1563_to_ten_l1006_100665


namespace NUMINAMATH_CALUDE_fraction_equality_unique_solution_l1006_100685

theorem fraction_equality_unique_solution :
  ∃! (C D : ℝ), ∀ x : ℝ, x ≠ 4 ∧ x ≠ 5 →
    (D * x - 23) / (x^2 - 9*x + 20) = C / (x - 4) + 7 / (x - 5) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_fraction_equality_unique_solution_l1006_100685


namespace NUMINAMATH_CALUDE_dots_per_blouse_is_twenty_l1006_100630

/-- The number of dots on each blouse -/
def dots_per_blouse (total_dye : ℕ) (num_blouses : ℕ) (dye_per_dot : ℕ) : ℕ :=
  (total_dye / num_blouses) / dye_per_dot

/-- Theorem stating that the number of dots per blouse is 20 -/
theorem dots_per_blouse_is_twenty :
  dots_per_blouse (50 * 400) 100 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dots_per_blouse_is_twenty_l1006_100630


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1006_100690

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0) :
  let e := Real.sqrt 5 / 2 - 1 / 2
  ∃ (F₁ F₂ P : ℝ × ℝ),
    -- F₁ and F₂ are the foci of the ellipse
    F₁.1 = -Real.sqrt (a^2 - b^2) ∧ F₁.2 = 0 ∧
    F₂.1 = Real.sqrt (a^2 - b^2) ∧ F₂.2 = 0 ∧
    -- P is on the ellipse
    P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧
    -- PF₂ is perpendicular to the x-axis
    P.1 = F₂.1 ∧
    -- |F₁F₂| = 2|PF₂|
    (F₁.1 - F₂.1)^2 = 4 * P.2^2 ∧
    -- The eccentricity is e
    e = Real.sqrt (a^2 - b^2) / a := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1006_100690


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1006_100667

/-- An isosceles triangle PQR with given side lengths -/
structure IsoscelesTriangle where
  -- Side lengths
  pq : ℝ
  qr : ℝ
  pr : ℝ
  -- Isosceles condition
  isIsosceles : pq = pr
  -- Given side lengths
  qr_eq : qr = 8
  pr_eq : pr = 10

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.pq + t.qr + t.pr

/-- Theorem: The perimeter of the given isosceles triangle is 28 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) :
  perimeter t = 28 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1006_100667


namespace NUMINAMATH_CALUDE_min_monkeys_correct_l1006_100603

/-- Represents the problem of transporting weapons with monkeys --/
structure WeaponTransport where
  total_weight : ℕ
  max_weapon_weight : ℕ
  max_monkey_capacity : ℕ

/-- Calculates the minimum number of monkeys needed to transport all weapons --/
def min_monkeys_needed (wt : WeaponTransport) : ℕ :=
  23

/-- Theorem stating that the minimum number of monkeys needed is correct --/
theorem min_monkeys_correct (wt : WeaponTransport) 
  (h1 : wt.total_weight = 600)
  (h2 : wt.max_weapon_weight = 30)
  (h3 : wt.max_monkey_capacity = 50) :
  min_monkeys_needed wt = 23 ∧ 
  ∀ n : ℕ, n < 23 → ¬ (n * wt.max_monkey_capacity ≥ wt.total_weight) :=
sorry

end NUMINAMATH_CALUDE_min_monkeys_correct_l1006_100603


namespace NUMINAMATH_CALUDE_black_number_equals_sum_of_white_numbers_l1006_100688

theorem black_number_equals_sum_of_white_numbers :
  ∃ (a b c d : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  (Real.sqrt (c + d * Real.sqrt 7) = Real.sqrt (a + b * Real.sqrt 2) + Real.sqrt (a - b * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_black_number_equals_sum_of_white_numbers_l1006_100688


namespace NUMINAMATH_CALUDE_frank_remaining_money_l1006_100650

def cheapest_lamp : ℕ := 20
def expensive_multiplier : ℕ := 3
def frank_money : ℕ := 90

theorem frank_remaining_money :
  frank_money - (cheapest_lamp * expensive_multiplier) = 30 := by
  sorry

end NUMINAMATH_CALUDE_frank_remaining_money_l1006_100650


namespace NUMINAMATH_CALUDE_billys_age_l1006_100696

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 52) : 
  billy = 39 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l1006_100696


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l1006_100666

/-- The cost of the sneakers in dollars -/
def sneaker_cost : ℚ := 45.5

/-- The number of $10 bills Chloe has -/
def ten_dollar_bills : ℕ := 4

/-- The number of quarters Chloe has -/
def quarters : ℕ := 5

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The minimum number of nickels needed -/
def min_nickels : ℕ := 85

theorem minimum_nickels_needed :
  ∀ n : ℕ,
  (n : ℚ) * nickel_value + (ten_dollar_bills * 10 : ℚ) + (quarters * 0.25 : ℚ) ≥ sneaker_cost →
  n ≥ min_nickels :=
by sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l1006_100666


namespace NUMINAMATH_CALUDE_demand_decrease_proportion_l1006_100676

theorem demand_decrease_proportion (P Q : ℝ) (P_new Q_new I_new : ℝ) :
  P_new = 1.2 * P →
  I_new = 1.1 * (P * Q) →
  I_new = P_new * Q_new →
  (Q - Q_new) / Q = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_demand_decrease_proportion_l1006_100676


namespace NUMINAMATH_CALUDE_prism_volume_30_l1006_100649

/-- A right rectangular prism with integer edge lengths -/
structure RightRectangularPrism where
  a : ℕ
  b : ℕ
  h : ℕ

/-- The volume of a right rectangular prism -/
def volume (p : RightRectangularPrism) : ℕ := p.a * p.b * p.h

/-- The areas of the faces of a right rectangular prism -/
def face_areas (p : RightRectangularPrism) : Finset ℕ :=
  {p.a * p.b, p.a * p.h, p.b * p.h}

theorem prism_volume_30 (p : RightRectangularPrism) :
  30 ∈ face_areas p → 13 ∈ face_areas p → volume p = 30 := by
  sorry

#check prism_volume_30

end NUMINAMATH_CALUDE_prism_volume_30_l1006_100649


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1006_100611

theorem quadratic_equation_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁^2 + 4*r₁ = 0 ∧ r₂^2 + 4*r₂ = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_roots_l1006_100611


namespace NUMINAMATH_CALUDE_zoe_leo_difference_l1006_100691

-- Define variables
variable (t : ℝ) -- Leo's driving time
variable (s : ℝ) -- Leo's speed

-- Define Leo's distance
def leo_distance (t s : ℝ) : ℝ := t * s

-- Define Maria's distance
def maria_distance (t s : ℝ) : ℝ := (t + 2) * (s + 15)

-- Define Zoe's distance
def zoe_distance (t s : ℝ) : ℝ := (t + 3) * (s + 20)

-- Theorem statement
theorem zoe_leo_difference (t s : ℝ) :
  maria_distance t s = leo_distance t s + 110 →
  zoe_distance t s - leo_distance t s = 180 := by
  sorry


end NUMINAMATH_CALUDE_zoe_leo_difference_l1006_100691


namespace NUMINAMATH_CALUDE_vertical_angles_are_congruent_l1006_100652

-- Define what it means for two angles to be vertical
def are_vertical_angles (α β : Angle) : Prop := sorry

-- Define what it means for two angles to be congruent
def are_congruent (α β : Angle) : Prop := sorry

-- Theorem statement
theorem vertical_angles_are_congruent (α β : Angle) :
  are_vertical_angles α β → are_congruent α β := by
  sorry

end NUMINAMATH_CALUDE_vertical_angles_are_congruent_l1006_100652


namespace NUMINAMATH_CALUDE_product_of_specific_roots_l1006_100604

/-- Given distinct real numbers a, b, c, d satisfying specific equations, their product is 11 -/
theorem product_of_specific_roots (a b c d : ℝ) 
  (ha : a = Real.sqrt (4 + Real.sqrt (5 + a)))
  (hb : b = Real.sqrt (4 - Real.sqrt (5 + b)))
  (hc : c = Real.sqrt (4 + Real.sqrt (5 - c)))
  (hd : d = Real.sqrt (4 - Real.sqrt (5 - d)))
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  a * b * c * d = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_roots_l1006_100604


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l1006_100619

/-- The number of ways to seat 2 students in a row of 5 desks with at least one empty desk between them -/
def seatingArrangements : ℕ := 6

/-- The number of desks in the row -/
def numDesks : ℕ := 5

/-- The number of students to be seated -/
def numStudents : ℕ := 2

/-- The minimum number of empty desks required between the students -/
def minEmptyDesks : ℕ := 1

theorem correct_seating_arrangements :
  seatingArrangements = 
    (numDesks - numStudents - minEmptyDesks + 1) * (numStudents) :=
by sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l1006_100619


namespace NUMINAMATH_CALUDE_min_value_of_f_l1006_100664

def f (x : ℝ) := x^2 + 14*x + 3

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1006_100664


namespace NUMINAMATH_CALUDE_min_distance_between_points_l1006_100622

/-- The minimum distance between points A(x, √2-x) and B(√2/2, 0) is 1/2 -/
theorem min_distance_between_points :
  let A : ℝ → ℝ × ℝ := λ x ↦ (x, Real.sqrt 2 - x)
  let B : ℝ × ℝ := (Real.sqrt 2 / 2, 0)
  ∃ (min_dist : ℝ), min_dist = 1/2 ∧
    ∀ x, Real.sqrt ((A x).1 - B.1)^2 + ((A x).2 - B.2)^2 ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_points_l1006_100622


namespace NUMINAMATH_CALUDE_marathon_debate_duration_in_minutes_l1006_100699

/-- Converts hours, minutes, and seconds to total minutes and rounds to the nearest whole number -/
def totalMinutesRounded (hours minutes seconds : ℕ) : ℕ :=
  let totalMinutes : ℚ := hours * 60 + minutes + seconds / 60
  (totalMinutes + 1/2).floor.toNat

/-- The marathon debate duration -/
def marathonDebateDuration : ℕ × ℕ × ℕ := (12, 15, 30)

theorem marathon_debate_duration_in_minutes :
  totalMinutesRounded marathonDebateDuration.1 marathonDebateDuration.2.1 marathonDebateDuration.2.2 = 736 := by
  sorry

end NUMINAMATH_CALUDE_marathon_debate_duration_in_minutes_l1006_100699


namespace NUMINAMATH_CALUDE_negative_three_a_cubed_div_a_fourth_l1006_100601

theorem negative_three_a_cubed_div_a_fourth (a : ℝ) (h : a ≠ 0) :
  -3 * a^3 / a^4 = -3 / a := by sorry

end NUMINAMATH_CALUDE_negative_three_a_cubed_div_a_fourth_l1006_100601


namespace NUMINAMATH_CALUDE_ella_seventh_test_score_l1006_100653

def is_valid_score_set (scores : List ℤ) : Prop :=
  scores.length = 8 ∧
  scores.all (λ s => 88 ≤ s ∧ s ≤ 97) ∧
  scores.Nodup ∧
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 8 → (scores.take k).sum % k = 0) ∧
  scores.get! 7 = 90

theorem ella_seventh_test_score (scores : List ℤ) :
  is_valid_score_set scores → scores.get! 6 = 95 := by
  sorry

#check ella_seventh_test_score

end NUMINAMATH_CALUDE_ella_seventh_test_score_l1006_100653


namespace NUMINAMATH_CALUDE_linear_function_increasing_l1006_100656

/-- A linear function y = (2k-6)x + (2k+1) is increasing if and only if k > 3 -/
theorem linear_function_increasing (k : ℝ) :
  (∀ x y : ℝ, y = (2*k - 6)*x + (2*k + 1)) →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((2*k - 6)*x₁ + (2*k + 1) < (2*k - 6)*x₂ + (2*k + 1))) ↔
  k > 3 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l1006_100656


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1006_100659

theorem no_solution_for_equation (x y : ℝ) : xy = 1 → ¬(Real.sqrt (x^2 + y^2) = x + y) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1006_100659


namespace NUMINAMATH_CALUDE_distinct_domino_arrangements_l1006_100668

/-- Represents a grid with width and height -/
structure Grid :=
  (width : Nat)
  (height : Nat)

/-- Represents a domino with width and height -/
structure Domino :=
  (width : Nat)
  (height : Nat)

/-- Calculates the number of distinct paths on a grid using a given number of dominoes -/
def countDistinctPaths (g : Grid) (d : Domino) (numDominoes : Nat) : Nat :=
  Nat.choose (g.width + g.height - 2) (g.width - 1)

/-- Theorem: The number of distinct domino arrangements on a 6x5 grid with 5 dominoes is 126 -/
theorem distinct_domino_arrangements :
  let g : Grid := ⟨6, 5⟩
  let d : Domino := ⟨2, 1⟩
  countDistinctPaths g d 5 = 126 := by
  sorry

#eval countDistinctPaths ⟨6, 5⟩ ⟨2, 1⟩ 5

end NUMINAMATH_CALUDE_distinct_domino_arrangements_l1006_100668


namespace NUMINAMATH_CALUDE_function_property_l1006_100674

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ p q : ℝ, f (p + q) = f p * f q) 
  (h2 : f 1 = 3) : 
  (f 1 * f 1 + f 2) / f 1 + (f 2 * f 2 + f 4) / f 3 + 
  (f 3 * f 3 + f 6) / f 5 + (f 4 * f 4 + f 8) / f 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1006_100674


namespace NUMINAMATH_CALUDE_square_difference_l1006_100631

theorem square_difference (n : ℕ) (h : (n + 1)^2 = n^2 + 2*n + 1) :
  (n - 1)^2 = n^2 - (2*n - 1) :=
by sorry

end NUMINAMATH_CALUDE_square_difference_l1006_100631


namespace NUMINAMATH_CALUDE_fraction_power_equality_l1006_100645

theorem fraction_power_equality (a b : ℝ) (m : ℤ) (ha : a > 0) (hb : b ≠ 0) :
  (b / a) ^ m = a ^ (-m) * b ^ m := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l1006_100645


namespace NUMINAMATH_CALUDE_train_speed_l1006_100680

/-- The speed of a train given specific conditions -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_length = 500 →
  man_speed = 12 →
  passing_time = 10 →
  ∃ (train_speed : ℝ), train_speed = 168 ∧ 
    (train_speed + man_speed) * passing_time / 3.6 = train_length + man_speed * passing_time / 3.6 :=
by sorry


end NUMINAMATH_CALUDE_train_speed_l1006_100680


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_98_l1006_100616

theorem largest_four_digit_divisible_by_98 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 98 = 0 → n ≤ 9998 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_98_l1006_100616


namespace NUMINAMATH_CALUDE_password_count_l1006_100669

def password_length : ℕ := 4
def available_digits : ℕ := 9  -- 10 digits minus 1 (7 is excluded)

def total_passwords : ℕ := available_digits ^ password_length

def all_different_passwords : ℕ := Nat.choose available_digits password_length * Nat.factorial password_length

theorem password_count : 
  total_passwords - all_different_passwords = 3537 :=
by sorry

end NUMINAMATH_CALUDE_password_count_l1006_100669


namespace NUMINAMATH_CALUDE_mean_temperature_l1006_100678

def temperatures : List ℝ := [78, 76, 80, 83, 85]

theorem mean_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 80.4 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1006_100678


namespace NUMINAMATH_CALUDE_cubic_equation_root_l1006_100686

theorem cubic_equation_root (a b : ℚ) : 
  (2 + Real.sqrt 3 : ℝ) ^ 3 + a * (2 + Real.sqrt 3 : ℝ) ^ 2 + b * (2 + Real.sqrt 3 : ℝ) - 15 = 0 → 
  b = -44 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l1006_100686


namespace NUMINAMATH_CALUDE_percent_profit_calculation_l1006_100608

/-- If the cost price of 60 articles is equal to the selling price of 40 articles,
    then the percent profit is 50%. -/
theorem percent_profit_calculation (C S : ℝ) 
  (h : C > 0) 
  (eq : 60 * C = 40 * S) : 
  (S - C) / C * 100 = 50 :=
by sorry

end NUMINAMATH_CALUDE_percent_profit_calculation_l1006_100608


namespace NUMINAMATH_CALUDE_combined_average_age_l1006_100621

theorem combined_average_age (people_a : ℕ) (people_b : ℕ) (avg_age_a : ℝ) (avg_age_b : ℝ)
  (h1 : people_a = 8)
  (h2 : people_b = 2)
  (h3 : avg_age_a = 38)
  (h4 : avg_age_b = 30) :
  (people_a * avg_age_a + people_b * avg_age_b) / (people_a + people_b) = 36.4 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l1006_100621


namespace NUMINAMATH_CALUDE_carltons_shirts_l1006_100657

theorem carltons_shirts (shirts : ℕ) (vests : ℕ) (outfits : ℕ) : 
  vests = 2 * shirts → 
  outfits = vests * shirts → 
  outfits = 18 → 
  shirts = 3 := by
sorry

end NUMINAMATH_CALUDE_carltons_shirts_l1006_100657


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1006_100624

/-- Proves that for a line with slope 4 passing through the point (2, -1),
    where the equation of the line is y = mx + b, the value of m + b is equal to -5. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 4 → (2 : ℝ) * m + b = -1 → m + b = -5 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1006_100624


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1006_100620

theorem geometric_sequence_sum (a b c q : ℝ) (h_seq : (a + b + c) * q = b + c - a ∧
                                                    (b + c - a) * q = c + a - b ∧
                                                    (c + a - b) * q = a + b - c) :
  q^3 + q^2 + q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1006_100620


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1006_100629

theorem smallest_multiple_of_6_and_15 (c : ℕ) :
  (c > 0 ∧ 6 ∣ c ∧ 15 ∣ c) → c ≥ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1006_100629


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l1006_100612

theorem monotonic_f_implies_a_range (a : ℝ) :
  (∀ x : ℝ, StrictMono (fun x => x - (1/3) * Real.sin (2*x) + a * Real.sin x)) →
  -1/3 ≤ a ∧ a ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l1006_100612


namespace NUMINAMATH_CALUDE_volleyball_count_l1006_100698

theorem volleyball_count : ∃ (x y z : ℕ),
  x + y + z = 20 ∧
  60 * x + 30 * y + 10 * z = 330 ∧
  z = 15 := by
sorry

end NUMINAMATH_CALUDE_volleyball_count_l1006_100698


namespace NUMINAMATH_CALUDE_triangle_segment_length_l1006_100602

/-- Given a triangle ADE with point C on AD and point B on AC, prove that FC = 14.6 -/
theorem triangle_segment_length 
  (DC CB : ℝ) 
  (h1 : DC = 9) 
  (h2 : CB = 10) 
  (AD : ℝ) 
  (h3 : (1 : ℝ) / 3 * AD = AD - DC - CB) 
  (ED : ℝ) 
  (h4 : ED = 3 / 4 * AD) 
  (FC : ℝ) 
  (h5 : FC * AD = ED * (DC + CB + (1 / 3 * AD))) : 
  FC = 14.6 := by
sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l1006_100602


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l1006_100635

theorem matrix_N_satisfies_conditions :
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![2, -1, 6; 3, 4, 0; -1, 1, -3]
  let i : Matrix (Fin 3) (Fin 1) ℝ := !![1; 0; 0]
  let j : Matrix (Fin 3) (Fin 1) ℝ := !![0; 1; 0]
  let k : Matrix (Fin 3) (Fin 1) ℝ := !![0; 0; 1]
  N * i = !![1; 2; -5] + !![1; 1; 4] ∧
  N * j = !![-1; 4; 1] ∧
  N * k = !![6; 0; -3] :=
by sorry

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l1006_100635


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1006_100655

-- Define the circle C
def circle_C (a x y : ℝ) : Prop := (x - a)^2 + (y - a + 2)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the condition for point M
def condition_M (a x y : ℝ) : Prop :=
  circle_C a x y ∧ (x^2 + (y - 2)^2) + (x^2 + y^2) = 10

-- Main theorem
theorem circle_intersection_range (a : ℝ) :
  (∃ x y : ℝ, condition_M a x y) → a ∈ Set.Icc 0 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1006_100655


namespace NUMINAMATH_CALUDE_difference_in_sums_l1006_100689

def sum_of_integers (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def sum_of_rounded_integers (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_five |>.sum

theorem difference_in_sums :
  sum_of_rounded_integers 200 - sum_of_integers 200 = 120 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_sums_l1006_100689


namespace NUMINAMATH_CALUDE_height_weight_only_correlation_l1006_100615

-- Define the types of relationships
inductive Relationship
  | HeightWeight
  | DistanceTime
  | HeightVision
  | VolumeEdge

-- Define a property for correlation
def is_correlated (r : Relationship) : Prop :=
  match r with
  | Relationship.HeightWeight => True
  | _ => False

-- Define a property for functional relationships
def is_functional (r : Relationship) : Prop :=
  match r with
  | Relationship.DistanceTime => True
  | Relationship.VolumeEdge => True
  | _ => False

-- Theorem statement
theorem height_weight_only_correlation :
  ∀ r : Relationship, is_correlated r ↔ r = Relationship.HeightWeight ∧ ¬is_functional r :=
sorry

end NUMINAMATH_CALUDE_height_weight_only_correlation_l1006_100615


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l1006_100682

/-- The count of three-digit numbers where either all digits are the same or the first and last digits are different -/
def validThreeDigitNumbers : ℕ :=
  -- Total three-digit numbers
  let totalThreeDigitNumbers := 999 - 100 + 1
  -- Numbers to exclude (ABA form where A ≠ B and B ≠ 0)
  let excludedNumbers := 10 * 9
  -- Calculation
  totalThreeDigitNumbers - excludedNumbers

/-- Theorem stating that the count of valid three-digit numbers is 810 -/
theorem valid_three_digit_numbers_count : validThreeDigitNumbers = 810 := by
  sorry


end NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l1006_100682


namespace NUMINAMATH_CALUDE_min_value_problem_l1006_100605

theorem min_value_problem (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 16) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l1006_100605


namespace NUMINAMATH_CALUDE_first_four_terms_l1006_100675

def a (n : ℕ) : ℚ := (1 + (-1)^(n+1)) / 2

theorem first_four_terms :
  (a 1 = 1) ∧ (a 2 = 0) ∧ (a 3 = 1) ∧ (a 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_first_four_terms_l1006_100675


namespace NUMINAMATH_CALUDE_circle_area_equal_perimeter_l1006_100609

theorem circle_area_equal_perimeter (s : ℝ) (r : ℝ) : 
  s > 0 → 
  r > 0 → 
  s^2 = 16 → 
  4 * s = 2 * Real.pi * r → 
  Real.pi * r^2 = 64 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equal_perimeter_l1006_100609


namespace NUMINAMATH_CALUDE_sin_inequality_equivalence_l1006_100640

theorem sin_inequality_equivalence (a b : ℝ) :
  (∀ x : ℝ, Real.sin x + Real.sin a ≥ b * Real.cos x) ↔
  (∃ n : ℤ, a = (4 * n + 1) * Real.pi / 2) ∧ (b = 0) := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_equivalence_l1006_100640


namespace NUMINAMATH_CALUDE_division_theorem_problem_1999_division_l1006_100626

theorem division_theorem (n d q r : ℕ) (h : n = d * q + r) (h_r : r < d) :
  (n / d = q ∧ n % d = r) :=
sorry

theorem problem_1999_division :
  1999 / 40 = 49 ∧ 1999 % 40 = 39 :=
sorry

end NUMINAMATH_CALUDE_division_theorem_problem_1999_division_l1006_100626


namespace NUMINAMATH_CALUDE_curved_octagon_area_l1006_100634

/-- A closed curve composed of circular arcs centered on an octagon's vertices -/
structure CurvedOctagon where
  /-- Number of circular arcs -/
  n_arcs : ℕ
  /-- Length of each circular arc -/
  arc_length : ℝ
  /-- Side length of the regular octagon -/
  octagon_side : ℝ

/-- The area enclosed by the curved octagon -/
noncomputable def enclosed_area (co : CurvedOctagon) : ℝ :=
  sorry

/-- Theorem stating the enclosed area of a specific curved octagon -/
theorem curved_octagon_area :
  let co : CurvedOctagon := {
    n_arcs := 12,
    arc_length := 3 * Real.pi / 4,
    octagon_side := 3
  }
  enclosed_area co = 18 * (1 + Real.sqrt 2) + 81 * Real.pi / 8 :=
sorry

end NUMINAMATH_CALUDE_curved_octagon_area_l1006_100634


namespace NUMINAMATH_CALUDE_circle_to_octagon_area_ratio_l1006_100662

/-- The ratio of the area of a circle inscribed in a regular octagon
    (where the circle's radius equals the octagon's apothem)
    to the area of the octagon itself. -/
theorem circle_to_octagon_area_ratio : ∃ (a b : ℕ), (a : ℝ).sqrt / b * π = (π / (4 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_circle_to_octagon_area_ratio_l1006_100662


namespace NUMINAMATH_CALUDE_ellipse_fraction_bounds_l1006_100651

theorem ellipse_fraction_bounds (x y : ℝ) (h : (x - 3)^2 + 4*(y - 1)^2 = 4) :
  ∃ (t : ℝ), (x + y - 3) / (x - y + 1) = t ∧ -1 ≤ t ∧ t ≤ 1 ∧
  (∃ (x₁ y₁ : ℝ), (x₁ - 3)^2 + 4*(y₁ - 1)^2 = 4 ∧ (x₁ + y₁ - 3) / (x₁ - y₁ + 1) = -1) ∧
  (∃ (x₂ y₂ : ℝ), (x₂ - 3)^2 + 4*(y₂ - 1)^2 = 4 ∧ (x₂ + y₂ - 3) / (x₂ - y₂ + 1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_fraction_bounds_l1006_100651


namespace NUMINAMATH_CALUDE_power_function_through_one_l1006_100638

theorem power_function_through_one (a : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^a
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_through_one_l1006_100638


namespace NUMINAMATH_CALUDE_age_difference_l1006_100646

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 12) : A - C = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1006_100646


namespace NUMINAMATH_CALUDE_total_weight_of_hay_bales_l1006_100618

/-- Calculates the total weight of hay bales in a barn after adding new bales -/
theorem total_weight_of_hay_bales
  (initial_bales : ℕ)
  (initial_weight : ℕ)
  (total_bales : ℕ)
  (new_weight : ℕ)
  (h1 : initial_bales = 73)
  (h2 : initial_weight = 45)
  (h3 : total_bales = 96)
  (h4 : new_weight = 50)
  (h5 : total_bales > initial_bales) :
  initial_bales * initial_weight + (total_bales - initial_bales) * new_weight = 4435 :=
by sorry

#check total_weight_of_hay_bales

end NUMINAMATH_CALUDE_total_weight_of_hay_bales_l1006_100618


namespace NUMINAMATH_CALUDE_alice_basketball_record_l1006_100693

/-- Alice's basketball record problem -/
theorem alice_basketball_record (total_score : ℝ) (other_players : ℕ) (avg_score : ℝ) 
  (h1 : total_score = 72)
  (h2 : other_players = 7)
  (h3 : avg_score = 4.7) :
  total_score - (other_players : ℝ) * avg_score = 39.1 :=
by sorry

end NUMINAMATH_CALUDE_alice_basketball_record_l1006_100693


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1006_100660

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  b = 3 * a →
  a^2 + b^2 = c^2 →
  a^2 + b^2 + c^2 = 500 →
  c = 5 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1006_100660


namespace NUMINAMATH_CALUDE_max_roses_for_680_l1006_100694

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : ℝ  -- Price of an individual rose
  dozen : ℝ       -- Price of a dozen roses
  twoDozen : ℝ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased with a given budget -/
def maxRoses (pricing : RosePricing) (budget : ℝ) : ℕ :=
  sorry

/-- The theorem stating the maximum number of roses that can be purchased for $680 -/
theorem max_roses_for_680 (pricing : RosePricing) 
  (h1 : pricing.individual = 4.5)
  (h2 : pricing.dozen = 36)
  (h3 : pricing.twoDozen = 50) : 
  maxRoses pricing 680 = 318 :=
sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l1006_100694


namespace NUMINAMATH_CALUDE_expression_equality_l1006_100661

theorem expression_equality : 
  Real.sqrt 8 + Real.sqrt (1/2) + (Real.sqrt 3 - 1)^2 + Real.sqrt 6 / (1/2 * Real.sqrt 2) = 
  5/2 * Real.sqrt 2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1006_100661


namespace NUMINAMATH_CALUDE_book_reading_time_l1006_100606

theorem book_reading_time (pages : ℕ) (extra_pages_per_day : ℕ) (days_difference : ℕ) :
  pages = 480 →
  extra_pages_per_day = 16 →
  days_difference = 5 →
  ∃ (x : ℕ), x > 0 ∧ 
    (pages : ℚ) / x - days_difference = pages / (x + extra_pages_per_day) ∧
    x = 15 := by
  sorry

#check book_reading_time

end NUMINAMATH_CALUDE_book_reading_time_l1006_100606


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1006_100654

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, n * 17 = 85 ∧ 
  (∀ m : ℕ, m * 17 ≤ 99 → m * 17 ≤ 85) := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1006_100654


namespace NUMINAMATH_CALUDE_book_purchase_l1006_100663

theorem book_purchase (total_volumes : ℕ) (paperback_cost hardcover_cost total_cost : ℕ) 
  (h : total_volumes = 10)
  (h1 : paperback_cost = 18)
  (h2 : hardcover_cost = 28)
  (h3 : total_cost = 240) :
  ∃ (hardcover_count : ℕ), 
    hardcover_count * hardcover_cost + (total_volumes - hardcover_count) * paperback_cost = total_cost ∧ 
    hardcover_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_l1006_100663


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1006_100625

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 4*x - 12 = 0 ↔ x = 6 ∨ x = -2) ∧
  (∀ x : ℝ, (2*x - 1)^2 = 3*(2*x - 1) ↔ x = 1/2 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1006_100625


namespace NUMINAMATH_CALUDE_chord_sum_squares_l1006_100692

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 100}

def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry
def E : ℝ × ℝ := sorry

-- State the theorem
theorem chord_sum_squares (h1 : A ∈ Circle) (h2 : B ∈ Circle) (h3 : C ∈ Circle) (h4 : D ∈ Circle) (h5 : E ∈ Circle)
  (h6 : A.1 = -B.1 ∧ A.2 = -B.2) -- AB is a diameter
  (h7 : (E.1 - C.1) * (B.1 - A.1) + (E.2 - C.2) * (B.2 - A.2) = 0) -- CD intersects AB at E
  (h8 : (B.1 - E.1)^2 + (B.2 - E.2)^2 = 40) -- BE = 2√10
  (h9 : (A.1 - E.1) * (C.1 - E.1) + (A.2 - E.2) * (C.2 - E.2) = 
        Real.sqrt 3 * Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) * Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2) / 2) -- Angle AEC = 30°
  : (C.1 - E.1)^2 + (C.2 - E.2)^2 + (D.1 - E.1)^2 + (D.2 - E.2)^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_chord_sum_squares_l1006_100692


namespace NUMINAMATH_CALUDE_largest_angle_at_C_l1006_100647

/-- The line c is defined by the equation y = x + 1 -/
def line_c (x y : ℝ) : Prop := y = x + 1

/-- Point A has coordinates (1, 0) -/
def point_A : ℝ × ℝ := (1, 0)

/-- Point B has coordinates (3, 0) -/
def point_B : ℝ × ℝ := (3, 0)

/-- Point C has coordinates (1, 2) -/
def point_C : ℝ × ℝ := (1, 2)

/-- Function to calculate the angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating that point C is the point on line c from which segment AB is seen at the largest angle -/
theorem largest_angle_at_C :
  line_c point_C.1 point_C.2 ∧
  ∀ (p : ℝ × ℝ), line_c p.1 p.2 → angle p point_A point_B ≤ angle point_C point_A point_B :=
sorry

end NUMINAMATH_CALUDE_largest_angle_at_C_l1006_100647


namespace NUMINAMATH_CALUDE_race_time_differences_l1006_100658

/-- Race competition with three competitors --/
structure RaceCompetition where
  distance : ℝ
  time_A : ℝ
  time_B : ℝ
  time_C : ℝ

/-- Calculate time difference between two competitors --/
def timeDifference (t1 t2 : ℝ) : ℝ := t2 - t1

/-- Theorem stating the time differences between competitors --/
theorem race_time_differences (race : RaceCompetition) 
  (h_distance : race.distance = 250)
  (h_time_A : race.time_A = 40)
  (h_time_B : race.time_B = 50)
  (h_time_C : race.time_C = 55) : 
  (timeDifference race.time_A race.time_B = 10) ∧ 
  (timeDifference race.time_A race.time_C = 15) ∧ 
  (timeDifference race.time_B race.time_C = 5) := by
  sorry

end NUMINAMATH_CALUDE_race_time_differences_l1006_100658


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1984_l1006_100679

theorem smallest_divisible_by_1984 :
  ∃ (a : ℕ), (a > 0) ∧
  (∀ (n : ℕ), Odd n → (47^n + a * 15^n) % 1984 = 0) ∧
  (∀ (b : ℕ), 0 < b ∧ b < a → ∃ (m : ℕ), Odd m ∧ (47^m + b * 15^m) % 1984 ≠ 0) ∧
  (a = 1055) := by
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1984_l1006_100679
