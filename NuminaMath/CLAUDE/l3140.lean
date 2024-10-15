import Mathlib

namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l3140_314044

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount (sugar flour baking_soda : ℝ) 
  (h1 : sugar / flour = 3 / 8)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 900 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sugar_amount_l3140_314044


namespace NUMINAMATH_CALUDE_trains_meeting_time_l3140_314051

/-- Two trains meeting problem -/
theorem trains_meeting_time
  (distance : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (start_time_diff : ℝ)
  (h_distance : distance = 200)
  (h_speed_A : speed_A = 20)
  (h_speed_B : speed_B = 25)
  (h_start_time_diff : start_time_diff = 1) :
  let initial_distance_A := speed_A * start_time_diff
  let remaining_distance := distance - initial_distance_A
  let relative_speed := speed_A + speed_B
  let meeting_time := remaining_distance / relative_speed
  meeting_time + start_time_diff = 5 := by sorry

end NUMINAMATH_CALUDE_trains_meeting_time_l3140_314051


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3140_314013

-- Problem 1
theorem factorization_problem_1 (p q : ℝ) :
  6 * p^3 * q - 10 * p^2 = 2 * p^2 * (3 * p * q - 5) := by sorry

-- Problem 2
theorem factorization_problem_2 (a : ℝ) :
  a^4 - 8 * a^2 + 16 = (a + 2)^2 * (a - 2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3140_314013


namespace NUMINAMATH_CALUDE_range_of_a_l3140_314076

def A : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def B (a : ℝ) : Set ℝ := {x | |x - a| < 5}

theorem range_of_a (a : ℝ) : (A ∪ B a = Set.univ) → a ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3140_314076


namespace NUMINAMATH_CALUDE_sally_buttons_l3140_314093

/-- The number of buttons Sally needs for all shirts -/
def total_buttons (monday tuesday wednesday buttons_per_shirt : ℕ) : ℕ :=
  (monday + tuesday + wednesday) * buttons_per_shirt

/-- Theorem: Sally needs 45 buttons for all shirts -/
theorem sally_buttons : total_buttons 4 3 2 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sally_buttons_l3140_314093


namespace NUMINAMATH_CALUDE_dogs_and_video_games_percentage_l3140_314042

theorem dogs_and_video_games_percentage 
  (total_students : ℕ) 
  (dogs_preference : ℕ) 
  (dogs_and_movies_percent : ℚ) : 
  total_students = 30 →
  dogs_preference = 18 →
  dogs_and_movies_percent = 10 / 100 →
  (dogs_preference - (dogs_and_movies_percent * total_students).num) / total_students = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_dogs_and_video_games_percentage_l3140_314042


namespace NUMINAMATH_CALUDE_value_of_fraction_l3140_314084

theorem value_of_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x/y + y/x = 4) :
  (x + 2*y) / (x - 2*y) = Real.sqrt 33 / 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_fraction_l3140_314084


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l3140_314067

theorem grape_rate_calculation (grape_quantity : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : 
  grape_quantity = 8 →
  mango_quantity = 9 →
  mango_rate = 45 →
  total_paid = 965 →
  ∃ (grape_rate : ℕ), grape_rate * grape_quantity + mango_rate * mango_quantity = total_paid ∧ grape_rate = 70 := by
  sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l3140_314067


namespace NUMINAMATH_CALUDE_train_length_calculation_l3140_314052

/-- Calculates the length of a train given its speed, time to cross a platform, and the platform length. -/
def train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : ℝ :=
  speed * time - platform_length

/-- Proves that a train with speed 35 m/s crossing a 250.056 m platform in 20 seconds has a length of 449.944 m. -/
theorem train_length_calculation :
  train_length 35 20 250.056 = 449.944 := by
  sorry

#eval train_length 35 20 250.056

end NUMINAMATH_CALUDE_train_length_calculation_l3140_314052


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l3140_314096

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  area : ℝ

/-- Theorem: In a rhombus with one diagonal of 20 cm and an area of 250 cm², the other diagonal is 25 cm -/
theorem rhombus_other_diagonal
  (r : Rhombus)
  (h1 : r.diagonal1 = 20)
  (h2 : r.area = 250)
  (h3 : r.area = r.diagonal1 * r.diagonal2 / 2) :
  r.diagonal2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l3140_314096


namespace NUMINAMATH_CALUDE_inequality_proof_l3140_314026

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b ≥ 4 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3140_314026


namespace NUMINAMATH_CALUDE_fault_line_exists_l3140_314059

/-- Represents a 6x6 grid covered by 18 1x2 dominoes -/
structure DominoCoveredGrid :=
  (grid : Fin 6 → Fin 6 → Bool)
  (dominoes : Fin 18 → (Fin 6 × Fin 6) × (Fin 6 × Fin 6))
  (cover_complete : ∀ i j, ∃ k, (dominoes k).1 = (i, j) ∨ (dominoes k).2 = (i, j))
  (domino_size : ∀ k, 
    ((dominoes k).1.1 = (dominoes k).2.1 ∧ (dominoes k).2.2 = (dominoes k).1.2.succ) ∨
    ((dominoes k).1.2 = (dominoes k).2.2 ∧ (dominoes k).2.1 = (dominoes k).1.1.succ))

/-- A fault line is a row or column that doesn't intersect any domino -/
def has_fault_line (g : DominoCoveredGrid) : Prop :=
  (∃ i : Fin 6, ∀ k, (g.dominoes k).1.1 ≠ i ∧ (g.dominoes k).2.1 ≠ i) ∨
  (∃ j : Fin 6, ∀ k, (g.dominoes k).1.2 ≠ j ∧ (g.dominoes k).2.2 ≠ j)

/-- Theorem: Every 6x6 grid covered by 18 1x2 dominoes has a fault line -/
theorem fault_line_exists (g : DominoCoveredGrid) : has_fault_line g :=
sorry

end NUMINAMATH_CALUDE_fault_line_exists_l3140_314059


namespace NUMINAMATH_CALUDE_perpendicular_intersects_side_l3140_314014

/-- A regular polygon with n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry
  is_inscribed : sorry

/-- The opposite side of a vertex in a regular polygon -/
def opposite_side (p : RegularPolygon 101) (i : Fin 101) : Set (ℝ × ℝ) :=
  sorry

/-- The perpendicular from a vertex to the line containing the opposite side -/
def perpendicular (p : RegularPolygon 101) (i : Fin 101) : Set (ℝ × ℝ) :=
  sorry

/-- The intersection point of the perpendicular and the line containing the opposite side -/
def intersection_point (p : RegularPolygon 101) (i : Fin 101) : ℝ × ℝ :=
  sorry

/-- Theorem: In a regular 101-gon inscribed in a circle, there exists at least one vertex 
    such that the perpendicular from this vertex to the line containing the opposite side 
    intersects the opposite side itself, not its extension -/
theorem perpendicular_intersects_side (p : RegularPolygon 101) : 
  ∃ i : Fin 101, intersection_point p i ∈ opposite_side p i :=
sorry

end NUMINAMATH_CALUDE_perpendicular_intersects_side_l3140_314014


namespace NUMINAMATH_CALUDE_fraction_inequality_l3140_314064

theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / ((a - c)^2) > e / ((b - d)^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3140_314064


namespace NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_8000_l3140_314078

theorem count_primes_with_squares_between_5000_and_8000 :
  (Finset.filter (fun p => 5000 < p^2 ∧ p^2 < 8000) (Finset.filter Nat.Prime (Finset.range 90))).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_8000_l3140_314078


namespace NUMINAMATH_CALUDE_cookies_prepared_l3140_314066

theorem cookies_prepared (cookies_per_guest : ℕ) (num_guests : ℕ) (total_cookies : ℕ) :
  cookies_per_guest = 19 →
  num_guests = 2 →
  total_cookies = cookies_per_guest * num_guests →
  total_cookies = 38 := by
sorry

end NUMINAMATH_CALUDE_cookies_prepared_l3140_314066


namespace NUMINAMATH_CALUDE_largest_n_for_product_l3140_314095

/-- An arithmetic sequence with integer terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product (a b : ℕ → ℤ) :
  ArithmeticSequence a →
  ArithmeticSequence b →
  a 1 = 1 →
  b 1 = 1 →
  a 2 ≤ b 2 →
  (∃ n : ℕ, a n * b n = 1764) →
  (∀ m : ℕ, (∃ k : ℕ, a k * b k = 1764) → m ≤ 44) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_product_l3140_314095


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3140_314058

-- Problem 1
theorem problem_1 : (-1)^4 + (1 - 1/2) / 3 * (2 - 2^3) = 2 := by
  sorry

-- Problem 2
theorem problem_2 : (-3/4 - 5/9 + 7/12) / (1/36) = -26 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3140_314058


namespace NUMINAMATH_CALUDE_cone_volume_and_surface_area_l3140_314007

/-- Represents a cone with given slant height and height --/
structure Cone where
  slant_height : ℝ
  height : ℝ

/-- Calculate the volume of a cone --/
def volume (c : Cone) : ℝ := sorry

/-- Calculate the surface area of a cone --/
def surface_area (c : Cone) : ℝ := sorry

/-- Theorem stating the volume and surface area of a specific cone --/
theorem cone_volume_and_surface_area :
  let c : Cone := { slant_height := 15, height := 9 }
  (volume c = 432 * Real.pi) ∧ (surface_area c = 324 * Real.pi) := by sorry

end NUMINAMATH_CALUDE_cone_volume_and_surface_area_l3140_314007


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3140_314054

theorem sum_of_reciprocals (x y : ℕ+) 
  (sum_eq : x + y = 45)
  (hcf_eq : Nat.gcd x y = 3)
  (lcm_eq : Nat.lcm x y = 100) :
  (1 : ℚ) / x + (1 : ℚ) / y = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3140_314054


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l3140_314016

-- (1)
theorem simplify_expression_1 : 3 * Real.sqrt 20 - Real.sqrt 45 - Real.sqrt (1/5) = (14 * Real.sqrt 5) / 5 := by sorry

-- (2)
theorem simplify_expression_2 : (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 - 1 = 2 := by sorry

-- (3)
theorem simplify_expression_3 : Real.sqrt 16 + 327 - 2 * Real.sqrt (1/4) = 330 := by sorry

-- (4)
theorem simplify_expression_4 : (Real.sqrt 3 - Real.sqrt 5) * (Real.sqrt 5 + Real.sqrt 3) - (Real.sqrt 5 - Real.sqrt 3)^2 = 2 * Real.sqrt 15 - 6 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l3140_314016


namespace NUMINAMATH_CALUDE_inequality_solution_l3140_314040

theorem inequality_solution (x : ℝ) (h1 : x > 0) 
  (h2 : x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3140_314040


namespace NUMINAMATH_CALUDE_fraction_simplification_l3140_314048

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) : (x^2 - y^2) / (x - y) = x + y := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3140_314048


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3140_314023

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4 / 7
  let a₂ : ℚ := 16 / 21
  let r : ℚ := a₂ / a₁
  r = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3140_314023


namespace NUMINAMATH_CALUDE_increase_in_average_age_l3140_314074

/-- Calculates the increase in average age when two men in a group are replaced -/
theorem increase_in_average_age
  (n : ℕ) -- Total number of men
  (age1 age2 : ℕ) -- Ages of the two men being replaced
  (new_avg : ℚ) -- Average age of the two new men
  (h1 : n = 15)
  (h2 : age1 = 21)
  (h3 : age2 = 23)
  (h4 : new_avg = 37) :
  (2 * new_avg - (age1 + age2 : ℚ)) / n = 2 := by sorry

end NUMINAMATH_CALUDE_increase_in_average_age_l3140_314074


namespace NUMINAMATH_CALUDE_set_intersection_equality_l3140_314065

-- Define sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define the intersection set
def intersection : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_intersection_equality : M ∩ N = intersection := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l3140_314065


namespace NUMINAMATH_CALUDE_simple_random_for_small_population_systematic_for_large_uniform_population_stratified_for_population_with_strata_l3140_314002

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Define a structure for a sampling scenario
structure SamplingScenario where
  populationSize : ℕ
  sampleSize : ℕ
  hasStrata : Bool
  uniformDistribution : Bool

-- Define the function to determine the most appropriate sampling method
def mostAppropriateSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  sorry

-- Theorem for the first scenario
theorem simple_random_for_small_population :
  mostAppropriateSamplingMethod { populationSize := 20, sampleSize := 4, hasStrata := false, uniformDistribution := true } = SamplingMethod.SimpleRandom :=
  sorry

-- Theorem for the second scenario
theorem systematic_for_large_uniform_population :
  mostAppropriateSamplingMethod { populationSize := 1280, sampleSize := 32, hasStrata := false, uniformDistribution := true } = SamplingMethod.Systematic :=
  sorry

-- Theorem for the third scenario
theorem stratified_for_population_with_strata :
  mostAppropriateSamplingMethod { populationSize := 180, sampleSize := 15, hasStrata := true, uniformDistribution := false } = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_simple_random_for_small_population_systematic_for_large_uniform_population_stratified_for_population_with_strata_l3140_314002


namespace NUMINAMATH_CALUDE_fibonacci_inequality_l3140_314031

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) :
  (fibonacci (n + 1) : ℝ) ^ (1 / n : ℝ) ≥ 1 + (fibonacci n : ℝ) ^ (-1 / n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_inequality_l3140_314031


namespace NUMINAMATH_CALUDE_percentage_increase_l3140_314011

theorem percentage_increase (t : ℝ) (P : ℝ) : 
  t = 80 →
  (t + (P / 100) * t) - (t - (25 / 100) * t) = 30 →
  P = 12.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l3140_314011


namespace NUMINAMATH_CALUDE_tissues_cost_is_two_l3140_314039

def cost_of_tissues (toilet_paper_rolls : ℕ) (paper_towel_rolls : ℕ) (tissue_boxes : ℕ)
                    (toilet_paper_cost : ℚ) (paper_towel_cost : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - (toilet_paper_rolls * toilet_paper_cost + paper_towel_rolls * paper_towel_cost)) / tissue_boxes

theorem tissues_cost_is_two :
  cost_of_tissues 10 7 3 (3/2) 2 35 = 2 :=
by sorry

end NUMINAMATH_CALUDE_tissues_cost_is_two_l3140_314039


namespace NUMINAMATH_CALUDE_factorization_equality_l3140_314072

theorem factorization_equality (m n : ℝ) : -8*m^2 + 2*m*n = -2*m*(4*m - n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3140_314072


namespace NUMINAMATH_CALUDE_magician_trick_possible_magician_trick_smallest_l3140_314055

/-- Represents a sequence of digits -/
def DigitSequence (n : ℕ) := Fin n → Fin 10

/-- Represents a pair of adjacent positions in a sequence -/
structure AdjacentPair (n : ℕ) where
  first : Fin n
  second : Fin n
  adjacent : second = first.succ

/-- 
Given a sequence of digits and a pair of adjacent positions,
returns the sequence with those positions covered
-/
def coverDigits (seq : DigitSequence n) (pair : AdjacentPair n) : 
  Fin (n - 2) → Fin 10 := sorry

/-- 
States that for any sequence of 101 digits, covering any two adjacent digits
still allows for unique determination of the original sequence
-/
theorem magician_trick_possible : 
  ∀ (seq : DigitSequence 101) (pair : AdjacentPair 101),
  ∃! (original : DigitSequence 101), coverDigits original pair = coverDigits seq pair :=
sorry

/-- 
States that 101 is the smallest number for which the magician's trick is always possible
-/
theorem magician_trick_smallest : 
  (∀ n < 101, ¬(∀ (seq : DigitSequence n) (pair : AdjacentPair n),
    ∃! (original : DigitSequence n), coverDigits original pair = coverDigits seq pair)) ∧
  (∀ (seq : DigitSequence 101) (pair : AdjacentPair 101),
    ∃! (original : DigitSequence 101), coverDigits original pair = coverDigits seq pair) :=
sorry

end NUMINAMATH_CALUDE_magician_trick_possible_magician_trick_smallest_l3140_314055


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l3140_314006

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 2

/-- Represents the number of students selected -/
def num_selected : ℕ := 2

/-- Represents the event of selecting exactly one boy -/
def event_one_boy : Set (Fin num_boys × Fin num_girls) := sorry

/-- Represents the event of selecting exactly two boys -/
def event_two_boys : Set (Fin num_boys × Fin num_girls) := sorry

/-- The main theorem stating that the two events are mutually exclusive but not complementary -/
theorem events_mutually_exclusive_not_complementary :
  (event_one_boy ∩ event_two_boys = ∅) ∧ 
  (event_one_boy ∪ event_two_boys ≠ Set.univ) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l3140_314006


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l3140_314010

def total_players : ℕ := 16
def triplets : ℕ := 3
def captain : ℕ := 1
def starters : ℕ := 6

def remaining_players : ℕ := total_players - triplets - captain
def players_to_choose : ℕ := starters - triplets - captain

theorem volleyball_team_selection :
  Nat.choose remaining_players players_to_choose = 66 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l3140_314010


namespace NUMINAMATH_CALUDE_constant_expression_l3140_314062

theorem constant_expression (x y z : ℝ) 
  (h1 : x * y + y * z + z * x = 4) 
  (h2 : x * y * z = 6) : 
  (x*y - 3/2*(x+y)) * (y*z - 3/2*(y+z)) * (z*x - 3/2*(z+x)) = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_l3140_314062


namespace NUMINAMATH_CALUDE_goose_eggs_count_l3140_314049

/-- The number of goose eggs laid at the pond -/
def total_eggs : ℕ := 1125

/-- The fraction of eggs that hatched -/
def hatched_fraction : ℚ := 1/3

/-- The fraction of hatched geese that survived the first month -/
def survived_month_fraction : ℚ := 4/5

/-- The fraction of geese that survived the first month but did not survive the first year -/
def not_survived_year_fraction : ℚ := 3/5

/-- The number of geese that survived the first year -/
def survived_year : ℕ := 120

theorem goose_eggs_count :
  (↑survived_year : ℚ) = (↑total_eggs * hatched_fraction * survived_month_fraction * (1 - not_survived_year_fraction)) ∧
  ∀ n : ℕ, n ≠ total_eggs → 
    (↑survived_year : ℚ) ≠ (↑n * hatched_fraction * survived_month_fraction * (1 - not_survived_year_fraction)) :=
by sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l3140_314049


namespace NUMINAMATH_CALUDE_find_m_value_l3140_314041

theorem find_m_value (m : ℝ) (h1 : m ≠ 0) :
  (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 7)) →
  m = 7 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l3140_314041


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3140_314071

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 --/
def CircleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The given equation of the circle --/
def GivenEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 4*y = 16

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ x y : ℝ, GivenEquation x y ↔ CircleEquation c x y) ∧
                  c.center = (-4, 2) ∧
                  c.radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3140_314071


namespace NUMINAMATH_CALUDE_extremum_at_two_min_value_of_sum_l3140_314079

/-- The function f(x) = -x³ + ax² - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem extremum_at_two (a : ℝ) : f_deriv a 2 = 0 ↔ a = 3 := by sorry

theorem min_value_of_sum (m n : ℝ) (hm : m ∈ Set.Icc (-1 : ℝ) 1) (hn : n ∈ Set.Icc (-1 : ℝ) 1) :
  ∃ (a : ℝ), f_deriv a 2 = 0 ∧ f a m + f_deriv a n ≥ -13 ∧
  ∃ (m' n' : ℝ), m' ∈ Set.Icc (-1 : ℝ) 1 ∧ n' ∈ Set.Icc (-1 : ℝ) 1 ∧ f a m' + f_deriv a n' = -13 := by sorry

end NUMINAMATH_CALUDE_extremum_at_two_min_value_of_sum_l3140_314079


namespace NUMINAMATH_CALUDE_m_range_l3140_314000

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, (|x - m| < 1 ↔ 1/3 < x ∧ x < 1/2)) ↔ 
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3140_314000


namespace NUMINAMATH_CALUDE_house_painting_cost_l3140_314033

/-- Calculates the total cost of painting a house given the areas and costs per square foot for different rooms. -/
def total_painting_cost (living_room_area : ℕ) (living_room_cost : ℕ)
                        (bedroom_area : ℕ) (bedroom_cost : ℕ)
                        (kitchen_area : ℕ) (kitchen_cost : ℕ)
                        (bathroom_area : ℕ) (bathroom_cost : ℕ) : ℕ :=
  living_room_area * living_room_cost +
  2 * bedroom_area * bedroom_cost +
  kitchen_area * kitchen_cost +
  2 * bathroom_area * bathroom_cost

/-- Theorem stating that the total cost of painting the house is 49500 Rs. -/
theorem house_painting_cost :
  total_painting_cost 600 30 450 25 300 20 100 15 = 49500 := by
  sorry

#eval total_painting_cost 600 30 450 25 300 20 100 15

end NUMINAMATH_CALUDE_house_painting_cost_l3140_314033


namespace NUMINAMATH_CALUDE_bob_gave_terry_24_bushels_l3140_314037

/-- Represents the number of bushels Bob grew -/
def total_bushels : ℕ := 50

/-- Represents the number of ears per bushel -/
def ears_per_bushel : ℕ := 14

/-- Represents the number of ears Bob has left -/
def ears_left : ℕ := 357

/-- Calculates the number of bushels Bob gave to Terry -/
def bushels_given_to_terry : ℕ :=
  ((total_bushels * ears_per_bushel) - ears_left) / ears_per_bushel

theorem bob_gave_terry_24_bushels :
  bushels_given_to_terry = 24 := by
  sorry

end NUMINAMATH_CALUDE_bob_gave_terry_24_bushels_l3140_314037


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l3140_314004

theorem geometric_progression_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 2^(1/4 : ℝ)) 
  (h₂ : a₂ = 2^(1/8 : ℝ)) 
  (h₃ : a₃ = 2^(1/16 : ℝ)) 
  (h_geom : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) :
  ∃ a₄ : ℝ, a₄ = a₃ * (a₃ / a₂) ∧ a₄ = 2^(-1/16 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l3140_314004


namespace NUMINAMATH_CALUDE_triangle_formation_l3140_314075

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  ¬(can_form_triangle 2 2 4) ∧
  can_form_triangle 5 6 10 ∧
  ¬(can_form_triangle 3 4 8) ∧
  ¬(can_form_triangle 4 5 10) :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3140_314075


namespace NUMINAMATH_CALUDE_jeff_purchases_total_l3140_314097

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - ↑(⌊x⌋) < 1/2 then ⌊x⌋ else ⌈x⌉

theorem jeff_purchases_total :
  let purchase1 : ℚ := 245/100
  let purchase2 : ℚ := 375/100
  let purchase3 : ℚ := 856/100
  let discount : ℚ := 50/100
  round_to_nearest_dollar purchase1 +
  round_to_nearest_dollar purchase2 +
  round_to_nearest_dollar (purchase3 - discount) = 14 := by
  sorry

end NUMINAMATH_CALUDE_jeff_purchases_total_l3140_314097


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3140_314088

theorem system_of_equations_solution :
  ∃ (x y : ℝ), 2*x - 3*y = -5 ∧ 5*x - 2*y = 4 :=
by
  use 2, 3
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3140_314088


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_iff_rational_l3140_314017

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a d : ℚ) : ℕ → ℚ := fun n ↦ a + n * d

/-- A geometric progression is a sequence where each term after the first
    is found by multiplying the previous term by a fixed, non-zero number. -/
def GeometricProgression (a r : ℚ) : ℕ → ℚ := fun n ↦ a * r^n

/-- A subsequence of a sequence is a sequence that can be derived from the original
    sequence by deleting some or no elements without changing the order of the
    remaining elements. -/
def Subsequence (f g : ℕ → ℚ) : Prop :=
  ∃ h : ℕ → ℕ, Monotone h ∧ ∀ n, f (h n) = g n

theorem arithmetic_to_geometric_iff_rational (a d : ℚ) (hd : d ≠ 0) :
  (∃ (b r : ℚ) (hr : r ≠ 1), Subsequence (ArithmeticProgression a d) (GeometricProgression b r)) ↔
  ∃ q : ℚ, a = q * d := by sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_iff_rational_l3140_314017


namespace NUMINAMATH_CALUDE_goals_scored_over_two_days_l3140_314030

/-- The total number of goals scored by Gina and Tom over two days -/
def total_goals (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ) : ℕ :=
  gina_day1 + gina_day2 + tom_day1 + tom_day2

/-- Theorem stating the total number of goals scored by Gina and Tom -/
theorem goals_scored_over_two_days :
  ∃ (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ),
    gina_day1 = 2 ∧
    tom_day1 = gina_day1 + 3 ∧
    tom_day2 = 6 ∧
    gina_day2 = tom_day2 - 2 ∧
    total_goals gina_day1 gina_day2 tom_day1 tom_day2 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_goals_scored_over_two_days_l3140_314030


namespace NUMINAMATH_CALUDE_unique_root_of_R_l3140_314092

/-- Represents a quadratic trinomial ax^2 + bx + c -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a quadratic trinomial P, construct R by adding P to the trinomial formed by swapping P's a and c -/
def constructR (P : QuadraticTrinomial) : QuadraticTrinomial :=
  { a := P.a + P.c
  , b := 2 * P.b
  , c := P.a + P.c }

theorem unique_root_of_R (P : QuadraticTrinomial) :
  let R := constructR P
  (∃! x : ℝ, R.a * x^2 + R.b * x + R.c = 0) →
  (∃ x : ℝ, x = -2 ∨ x = 2 ∧ R.a * x^2 + R.b * x + R.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_of_R_l3140_314092


namespace NUMINAMATH_CALUDE_marble_collection_proof_l3140_314047

/-- The number of blue marbles collected by the three friends --/
def total_blue_marbles (mary_blue : ℕ) (jenny_blue : ℕ) (anie_blue : ℕ) : ℕ :=
  mary_blue + jenny_blue + anie_blue

theorem marble_collection_proof 
  (jenny_red : ℕ) 
  (jenny_blue : ℕ) 
  (mary_red : ℕ) 
  (mary_blue : ℕ) 
  (anie_red : ℕ) 
  (anie_blue : ℕ) : 
  jenny_red = 30 →
  jenny_blue = 25 →
  mary_red = 2 * jenny_red →
  anie_red = mary_red + 20 →
  anie_blue = 2 * jenny_blue →
  mary_blue = anie_blue / 2 →
  total_blue_marbles mary_blue jenny_blue anie_blue = 100 := by
  sorry

#check marble_collection_proof

end NUMINAMATH_CALUDE_marble_collection_proof_l3140_314047


namespace NUMINAMATH_CALUDE_jolene_total_earnings_l3140_314020

/-- The amount of money Jolene raised through babysitting and car washing -/
def jolene_earnings (num_families : ℕ) (babysitting_rate : ℕ) (num_cars : ℕ) (car_wash_rate : ℕ) : ℕ :=
  num_families * babysitting_rate + num_cars * car_wash_rate

/-- Theorem stating that Jolene raised $180 given the specified conditions -/
theorem jolene_total_earnings :
  jolene_earnings 4 30 5 12 = 180 := by
  sorry

end NUMINAMATH_CALUDE_jolene_total_earnings_l3140_314020


namespace NUMINAMATH_CALUDE_janes_daily_vase_arrangement_l3140_314082

def total_vases : ℕ := 248
def last_day_vases : ℕ := 8

theorem janes_daily_vase_arrangement :
  ∃ (daily_vases : ℕ),
    daily_vases > 0 ∧
    daily_vases = last_day_vases ∧
    (total_vases - last_day_vases) % daily_vases = 0 :=
by sorry

end NUMINAMATH_CALUDE_janes_daily_vase_arrangement_l3140_314082


namespace NUMINAMATH_CALUDE_incenter_inside_BOH_l3140_314094

/-- Triangle type with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Angle measure of a triangle -/
def angle (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point is inside a triangle -/
def is_inside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Main theorem: Incenter lies inside the triangle formed by circumcenter, vertex B, and orthocenter -/
theorem incenter_inside_BOH (t : Triangle) 
  (h1 : angle t t.C > angle t t.B)
  (h2 : angle t t.B > angle t t.A) : 
  is_inside (incenter t) (Triangle.mk (circumcenter t) t.B (orthocenter t)) := by
  sorry

end NUMINAMATH_CALUDE_incenter_inside_BOH_l3140_314094


namespace NUMINAMATH_CALUDE_same_color_probability_l3140_314043

/-- The number of color options for neckties -/
def necktie_colors : ℕ := 6

/-- The number of color options for shirts -/
def shirt_colors : ℕ := 5

/-- The number of color options for hats -/
def hat_colors : ℕ := 4

/-- The number of color options for socks -/
def sock_colors : ℕ := 3

/-- The number of colors available for all item types -/
def common_colors : ℕ := 3

/-- The probability of selecting items of the same color for a box -/
theorem same_color_probability : 
  (common_colors : ℚ) / (necktie_colors * shirt_colors * hat_colors * sock_colors) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3140_314043


namespace NUMINAMATH_CALUDE_probability_green_then_blue_l3140_314045

def total_marbles : ℕ := 10
def blue_marbles : ℕ := 4
def green_marbles : ℕ := 6

theorem probability_green_then_blue :
  (green_marbles : ℚ) / total_marbles * (blue_marbles : ℚ) / (total_marbles - 1) = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_probability_green_then_blue_l3140_314045


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l3140_314089

/-- The quadratic function f(x) = 15x^2 + 75x + 225 -/
def f (x : ℝ) : ℝ := 15 * x^2 + 75 * x + 225

/-- The constants a, b, and c in the form a(x+b)^2+c -/
def a : ℝ := 15
def b : ℝ := 2.5
def c : ℝ := 131.25

/-- The quadratic function g(x) in the form a(x+b)^2+c -/
def g (x : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum_of_constants :
  (∀ x, f x = g x) → a + b + c = 148.75 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l3140_314089


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3140_314005

theorem diophantine_equation_solutions (n : ℕ) : n ∈ ({1, 2, 3} : Set ℕ) ↔ 
  ∃ (a b c : ℤ), a^n + b^n = c^n + n ∧ n ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3140_314005


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l3140_314077

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l3140_314077


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3140_314068

theorem inequality_system_solution :
  ∀ x : ℝ, (2 * x + 1 < 5 ∧ 2 - x ≤ 1) ↔ (1 ≤ x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3140_314068


namespace NUMINAMATH_CALUDE_line_relations_l3140_314086

-- Define the concept of a line in 3D space
variable (Line : Type)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem line_relations (a b c : Line) :
  parallel a b → perpendicular a c → perpendicular b c := by
  sorry

end NUMINAMATH_CALUDE_line_relations_l3140_314086


namespace NUMINAMATH_CALUDE_fencing_cost_is_2210_l3140_314080

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (width : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) : ℝ :=
  perimeter * cost_per_meter

/-- Theorem: The total cost of fencing the rectangular plot is 2210 -/
theorem fencing_cost_is_2210 :
  ∃ (width : ℝ),
    let length := width + 10
    let perimeter := 2 * (length + width)
    perimeter = 340 ∧
    total_fencing_cost width perimeter 6.5 = 2210 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_2210_l3140_314080


namespace NUMINAMATH_CALUDE_bike_ride_speed_l3140_314053

theorem bike_ride_speed (x : ℝ) : 
  (210 / x = 210 / (x - 5) - 1) → x = 35 := by
sorry

end NUMINAMATH_CALUDE_bike_ride_speed_l3140_314053


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3140_314012

theorem inequality_solution_set (x : ℝ) : 
  5 * x^2 + 7 * x > 3 ↔ x < -1 ∨ x > 3/5 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3140_314012


namespace NUMINAMATH_CALUDE_max_value_when_m_1_solution_when_m_neg_2_l3140_314024

-- Define the function f(x, m)
def f (x m : ℝ) : ℝ := |m * x + 1| - |x - 1|

-- Theorem 1: Maximum value of f(x) when m = 1
theorem max_value_when_m_1 :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x : ℝ), f x 1 ≤ max :=
sorry

-- Theorem 2: Solution to f(x) ≥ 1 when m = -2
theorem solution_when_m_neg_2 :
  ∀ (x : ℝ), f x (-2) ≥ 1 ↔ x ≤ -1 ∨ x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_when_m_1_solution_when_m_neg_2_l3140_314024


namespace NUMINAMATH_CALUDE_intersection_M_N_l3140_314015

def M : Set ℝ := {0, 1, 3}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3140_314015


namespace NUMINAMATH_CALUDE_b_41_mod_49_l3140_314021

/-- The sequence b_n defined as 6^n + 8^n -/
def b (n : ℕ) : ℕ := 6^n + 8^n

/-- The theorem stating that b_41 is congruent to 35 modulo 49 -/
theorem b_41_mod_49 : b 41 ≡ 35 [ZMOD 49] := by sorry

end NUMINAMATH_CALUDE_b_41_mod_49_l3140_314021


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3140_314087

theorem pizza_toppings_combinations (n : ℕ) (k : ℕ) : n = 5 ∧ k = 2 → Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3140_314087


namespace NUMINAMATH_CALUDE_inequality_theorem_l3140_314085

theorem inequality_theorem (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h1 : x + a * b * y ≤ a * (y + z))
  (h2 : y + b * c * z ≤ b * (z + x))
  (h3 : z + c * a * x ≤ c * (x + y)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3140_314085


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3140_314056

theorem inequality_solution_set (x : ℝ) : 3 * x - 2 > x ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3140_314056


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3140_314008

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- Given point P -/
def P : Point := ⟨-1, -2⟩

/-- Symmetry about the origin -/
def symmetricAboutOrigin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

/-- Theorem: The point symmetrical to P(-1, -2) about the origin has coordinates (1, 2) -/
theorem symmetric_point_coordinates :
  symmetricAboutOrigin P = Point.mk 1 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3140_314008


namespace NUMINAMATH_CALUDE_min_value_a_l3140_314046

theorem min_value_a (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  ∃ (a : ℝ), ∀ (x y : ℝ), x > 1 → y > 1 → 
    Real.log (x * y) ≤ Real.log a * Real.sqrt (Real.log x ^ 2 + Real.log y ^ 2) ∧
    ∀ (b : ℝ), (∀ (x y : ℝ), x > 1 → y > 1 → 
      Real.log (x * y) ≤ Real.log b * Real.sqrt (Real.log x ^ 2 + Real.log y ^ 2)) → 
    a ≤ b :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l3140_314046


namespace NUMINAMATH_CALUDE_simplify_inverse_sum_l3140_314073

theorem simplify_inverse_sum (k x y : ℝ) (hk : k ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) :
  (k * x⁻¹ + k * y⁻¹)⁻¹ = (x * y) / (k * (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_inverse_sum_l3140_314073


namespace NUMINAMATH_CALUDE_jacks_estimate_is_larger_l3140_314069

theorem jacks_estimate_is_larger (x y a b : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (ha : a > 0) (hb : b > 0) : 
  (x + a) - (y - b) > x - y := by
  sorry

end NUMINAMATH_CALUDE_jacks_estimate_is_larger_l3140_314069


namespace NUMINAMATH_CALUDE_rainy_days_count_l3140_314083

theorem rainy_days_count (n : ℤ) (R : ℕ) (NR : ℕ) : 
  n * R + 3 * NR = 26 →  -- Total cups equation
  3 * NR - n * R = 10 →  -- Difference in cups equation
  R + NR = 7 →           -- Total days equation
  R = 1 :=                -- Conclusion: 1 rainy day
by sorry

end NUMINAMATH_CALUDE_rainy_days_count_l3140_314083


namespace NUMINAMATH_CALUDE_pythagorean_triple_check_l3140_314036

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_check : 
  ¬ is_pythagorean_triple 12 15 18 ∧
  is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 6 9 15 :=
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_check_l3140_314036


namespace NUMINAMATH_CALUDE_simplify_and_express_negative_exponents_l3140_314022

theorem simplify_and_express_negative_exponents 
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = 2 * x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_express_negative_exponents_l3140_314022


namespace NUMINAMATH_CALUDE_categorical_variables_correct_l3140_314060

-- Define the type for variables
inductive Variable
  | Smoking
  | Gender
  | ReligiousBelief
  | Nationality

-- Define a function to check if a variable is categorical
def isCategorical (v : Variable) : Prop :=
  match v with
  | Variable.Smoking => False
  | _ => True

-- Define the set of all variables
def allVariables : Set Variable :=
  {Variable.Smoking, Variable.Gender, Variable.ReligiousBelief, Variable.Nationality}

-- Define the set of categorical variables
def categoricalVariables : Set Variable :=
  {v ∈ allVariables | isCategorical v}

-- Theorem statement
theorem categorical_variables_correct :
  categoricalVariables = {Variable.Gender, Variable.ReligiousBelief, Variable.Nationality} :=
by sorry

end NUMINAMATH_CALUDE_categorical_variables_correct_l3140_314060


namespace NUMINAMATH_CALUDE_odd_numbers_sum_greater_than_20000_l3140_314063

/-- The count of odd numbers between 200 and 405 whose sum is greater than 20000 -/
def count_odd_numbers_with_large_sum : ℕ :=
  let first_odd := 201
  let last_odd := 403
  let count := (last_odd - first_odd) / 2 + 1
  count

theorem odd_numbers_sum_greater_than_20000 :
  count_odd_numbers_with_large_sum = 102 :=
sorry


end NUMINAMATH_CALUDE_odd_numbers_sum_greater_than_20000_l3140_314063


namespace NUMINAMATH_CALUDE_union_of_sets_l3140_314029

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l3140_314029


namespace NUMINAMATH_CALUDE_solve_for_n_l3140_314009

def first_seven_multiples_of_six : List ℕ := [6, 12, 18, 24, 30, 36, 42]

def a : ℚ := (List.sum first_seven_multiples_of_six) / 7

def b (n : ℕ) : ℕ := 2 * n

theorem solve_for_n (n : ℕ) (h : n > 0) : a ^ 2 - (b n) ^ 2 = 0 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_n_l3140_314009


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_quadratic_form_minimum_attainable_l3140_314057

theorem quadratic_form_minimum (x y : ℝ) :
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5 ≥ 8 :=
by sorry

theorem quadratic_form_minimum_attainable :
  ∃ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5 = 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_quadratic_form_minimum_attainable_l3140_314057


namespace NUMINAMATH_CALUDE_team_allocation_proof_l3140_314038

/-- Proves that given the initial team sizes and total transfer, 
    the number of people allocated to Team A that makes its size 
    twice Team B's size is 23 -/
theorem team_allocation_proof 
  (initial_a initial_b transfer : ℕ) 
  (h_initial_a : initial_a = 31)
  (h_initial_b : initial_b = 26)
  (h_transfer : transfer = 24) :
  ∃ (x : ℕ), 
    x ≤ transfer ∧ 
    initial_a + x = 2 * (initial_b + (transfer - x)) ∧ 
    x = 23 := by
  sorry

end NUMINAMATH_CALUDE_team_allocation_proof_l3140_314038


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l3140_314081

theorem geometric_progression_first_term (S a r : ℝ) : 
  S = 10 → 
  a + a * r = 6 → 
  a = 2 * r → 
  (a = -1 + Real.sqrt 13 ∨ a = -1 - Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l3140_314081


namespace NUMINAMATH_CALUDE_complex_order_multiplication_property_l3140_314019

-- Define the order relation on complex numbers
def complex_order (z1 z2 : ℂ) : Prop :=
  z1.re > z2.re ∨ (z1.re = z2.re ∧ z1.im > z2.im)

-- Define the statement to be proven false
theorem complex_order_multiplication_property (z z1 z2 : ℂ) :
  ¬(complex_order z 0 → complex_order z1 z2 → complex_order (z * z1) (z * z2)) :=
sorry

end NUMINAMATH_CALUDE_complex_order_multiplication_property_l3140_314019


namespace NUMINAMATH_CALUDE_max_valid_sequence_length_l3140_314027

/-- A sequence of integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  (∀ i : ℕ, i + 2 < n → a i + a (i + 1) + a (i + 2) > 0) ∧
  (∀ i : ℕ, i + 4 < n → a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) < 0)

/-- The maximum length of a valid sequence is 6 -/
theorem max_valid_sequence_length :
  (∃ (a : ℕ → ℤ), ValidSequence a 6) ∧
  (∀ n : ℕ, n > 6 → ¬∃ (a : ℕ → ℤ), ValidSequence a n) :=
sorry

end NUMINAMATH_CALUDE_max_valid_sequence_length_l3140_314027


namespace NUMINAMATH_CALUDE_factors_of_1320_l3140_314098

theorem factors_of_1320 : Finset.card (Nat.divisors 1320) = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l3140_314098


namespace NUMINAMATH_CALUDE_triangle_regions_l3140_314018

theorem triangle_regions (p : ℕ) (h_prime : Nat.Prime p) (h_ge_3 : p ≥ 3) :
  let num_lines := 3 * p
  (num_lines * (num_lines + 1)) / 2 + 1 = 3 * p^2 - 3 * p + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_regions_l3140_314018


namespace NUMINAMATH_CALUDE_base_4_representation_of_253_base_4_to_decimal_3331_l3140_314028

/-- Converts a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n < 4 then [n]
  else (n % 4) :: toBase4 (n / 4)

/-- Converts a list of base 4 digits to its decimal representation -/
def fromBase4 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 4 * acc) 0

theorem base_4_representation_of_253 :
  toBase4 253 = [1, 3, 3, 3] :=
by sorry

theorem base_4_to_decimal_3331 :
  fromBase4 [1, 3, 3, 3] = 253 :=
by sorry

end NUMINAMATH_CALUDE_base_4_representation_of_253_base_4_to_decimal_3331_l3140_314028


namespace NUMINAMATH_CALUDE_complement_union_A_B_range_of_m_when_B_subset_A_l3140_314061

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

-- Theorem 1
theorem complement_union_A_B : 
  (A ∪ B (-2))ᶜ = {x | x < -2 ∨ x > 2} := by sorry

-- Theorem 2
theorem range_of_m_when_B_subset_A : 
  ∀ m : ℝ, B m ⊆ A ↔ -1 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_range_of_m_when_B_subset_A_l3140_314061


namespace NUMINAMATH_CALUDE_neither_alive_probability_l3140_314035

/-- The probability that a man will be alive for 10 more years -/
def prob_man_alive : ℚ := 1/4

/-- The probability that a woman will be alive for 10 more years -/
def prob_woman_alive : ℚ := 1/3

/-- The probability that neither the man nor the woman will be alive for 10 more years -/
def prob_neither_alive : ℚ := (1 - prob_man_alive) * (1 - prob_woman_alive)

theorem neither_alive_probability : prob_neither_alive = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_neither_alive_probability_l3140_314035


namespace NUMINAMATH_CALUDE_tournament_max_points_l3140_314050

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : Nat)
  (games_per_pair : Nat)
  (points_for_win : Nat)
  (points_for_draw : Nat)
  (points_for_loss : Nat)

/-- Calculates the total number of games in the tournament -/
def total_games (t : Tournament) : Nat :=
  (t.num_teams.choose 2) * t.games_per_pair

/-- Represents the maximum points achievable by top teams -/
def max_points_for_top_teams (t : Tournament) : Nat :=
  let games_against_lower := (t.num_teams - 3) * t.games_per_pair
  let points_from_lower := games_against_lower * t.points_for_win
  let games_among_top := 2 * t.games_per_pair
  let points_from_top := games_among_top * t.points_for_win / 2
  points_from_lower + points_from_top

/-- The main theorem to be proved -/
theorem tournament_max_points :
  ∀ t : Tournament,
    t.num_teams = 8 ∧
    t.games_per_pair = 2 ∧
    t.points_for_win = 3 ∧
    t.points_for_draw = 1 ∧
    t.points_for_loss = 0 →
    max_points_for_top_teams t = 36 := by
  sorry

end NUMINAMATH_CALUDE_tournament_max_points_l3140_314050


namespace NUMINAMATH_CALUDE_dave_first_six_l3140_314070

/-- The probability of tossing a six on a single throw -/
def prob_six : ℚ := 1 / 6

/-- The probability of not tossing a six on a single throw -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of players before Dave in each round -/
def players_before_dave : ℕ := 3

/-- The total number of players -/
def total_players : ℕ := 4

/-- The probability that Dave is the first to toss a six -/
theorem dave_first_six : 
  (prob_six * prob_not_six ^ players_before_dave) / 
  (1 - prob_not_six ^ total_players) = 125 / 671 := by
  sorry

end NUMINAMATH_CALUDE_dave_first_six_l3140_314070


namespace NUMINAMATH_CALUDE_range_of_a_l3140_314003

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀ + 2*a*x₀ + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3140_314003


namespace NUMINAMATH_CALUDE_vacuum_tube_alignment_l3140_314034

theorem vacuum_tube_alignment :
  ∃ (f g : Fin 7 → Fin 7), 
    ∀ (r : Fin 7), ∃ (k : Fin 7), f k = g ((r + k) % 7) := by
  sorry

end NUMINAMATH_CALUDE_vacuum_tube_alignment_l3140_314034


namespace NUMINAMATH_CALUDE_nimathur_prime_l3140_314090

/-- Definition of a-nimathur -/
def is_a_nimathur (a b : ℕ) : Prop :=
  a ≥ 1 ∧ b ≥ 1 ∧ ∀ n : ℕ, n ≥ b / a →
    (a * n + 1) ∣ (Nat.choose (a * n) b - 1)

/-- Main theorem -/
theorem nimathur_prime (a b : ℕ) :
  is_a_nimathur a b ∧ ¬is_a_nimathur a (b + 2) → Nat.Prime (b + 1) :=
by sorry

end NUMINAMATH_CALUDE_nimathur_prime_l3140_314090


namespace NUMINAMATH_CALUDE_sorcerer_elixir_combinations_l3140_314099

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of enchanted crystals available. -/
def num_crystals : ℕ := 6

/-- The number of crystals that are incompatible with some herbs. -/
def num_incompatible_crystals : ℕ := 2

/-- The number of herbs that each incompatible crystal cannot be used with. -/
def num_incompatible_herbs_per_crystal : ℕ := 3

/-- The total number of valid combinations for the sorcerer's elixir. -/
def valid_combinations : ℕ := 18

theorem sorcerer_elixir_combinations :
  (num_herbs * num_crystals) - (num_incompatible_crystals * num_incompatible_herbs_per_crystal) = valid_combinations :=
by sorry

end NUMINAMATH_CALUDE_sorcerer_elixir_combinations_l3140_314099


namespace NUMINAMATH_CALUDE_largest_non_representable_l3140_314091

def is_representable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

theorem largest_non_representable : 
  (∀ m : ℕ, m > 43 → is_representable m) ∧ 
  ¬(is_representable 43) := by sorry

end NUMINAMATH_CALUDE_largest_non_representable_l3140_314091


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l3140_314032

/-- The area of a square with adjacent vertices at (0,5) and (5,0) is 50 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 5)
  let p2 : ℝ × ℝ := (5, 0)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := side_length^2
  area = 50 := by sorry


end NUMINAMATH_CALUDE_square_area_from_vertices_l3140_314032


namespace NUMINAMATH_CALUDE_students_with_A_or_B_l3140_314001

theorem students_with_A_or_B (fraction_A fraction_B : ℝ) 
  (h1 : fraction_A = 0.7)
  (h2 : fraction_B = 0.2) : 
  fraction_A + fraction_B = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_students_with_A_or_B_l3140_314001


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3140_314025

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 7 * b = 3 * c)  -- Seven bowling balls weigh the same as three canoes
  (h2 : 2 * c = 56)     -- Two canoes weigh 56 pounds
  : b = 12 :=           -- One bowling ball weighs 12 pounds
by
  sorry

#check bowling_ball_weight

end NUMINAMATH_CALUDE_bowling_ball_weight_l3140_314025
