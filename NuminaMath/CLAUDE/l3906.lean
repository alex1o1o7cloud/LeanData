import Mathlib

namespace NUMINAMATH_CALUDE_test_scores_mode_l3906_390689

/-- Represents a stem-and-leaf plot entry --/
structure StemLeafEntry where
  stem : ℕ
  leaves : List ℕ

/-- Finds the mode of a list of numbers --/
def mode (l : List ℕ) : ℕ := sorry

/-- Converts a stem-and-leaf plot to a list of numbers --/
def stemLeafToList (plot : List StemLeafEntry) : List ℕ := sorry

theorem test_scores_mode (plot : List StemLeafEntry) 
  (h1 : plot = [
    ⟨5, [1, 1]⟩,
    ⟨6, [5]⟩,
    ⟨7, [2, 4]⟩,
    ⟨8, [0, 3, 6, 6]⟩,
    ⟨9, [1, 5, 5, 5, 8, 8, 8]⟩,
    ⟨10, [2, 2, 2, 2, 4]⟩,
    ⟨11, [0, 0, 0]⟩
  ]) : 
  mode (stemLeafToList plot) = 102 := by sorry

end NUMINAMATH_CALUDE_test_scores_mode_l3906_390689


namespace NUMINAMATH_CALUDE_sqrt_neg_two_squared_l3906_390671

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_two_squared_l3906_390671


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3906_390697

/-- The number of ways to arrange books on a shelf --/
def arrange_books (num_math_books : ℕ) (num_history_books : ℕ) : ℕ :=
  let remaining_books := num_math_books + (num_history_books - 2)
  num_history_books * (num_history_books - 1) * Nat.factorial remaining_books

/-- Theorem stating the correct number of arrangements for the given problem --/
theorem book_arrangement_count :
  arrange_books 5 4 = 60480 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3906_390697


namespace NUMINAMATH_CALUDE_uniform_production_theorem_l3906_390670

def device_A_rate : ℚ := 1 / 90
def device_B_rate : ℚ := 1 / 60
def simultaneous_work_days : ℕ := 30
def remaining_days : ℕ := 13

theorem uniform_production_theorem :
  (∃ x : ℚ, x * (device_A_rate + device_B_rate) = 1 ∧ x = 36) ∧
  (∃ y : ℚ, (simultaneous_work_days + y) * device_A_rate + simultaneous_work_days * device_B_rate = 1 ∧ y > remaining_days) :=
by sorry

end NUMINAMATH_CALUDE_uniform_production_theorem_l3906_390670


namespace NUMINAMATH_CALUDE_newer_train_distance_calculation_l3906_390630

/-- The distance traveled by the older train in miles -/
def older_train_distance : ℝ := 300

/-- The percentage increase in distance for the newer train -/
def percentage_increase : ℝ := 0.30

/-- The distance traveled by the newer train in miles -/
def newer_train_distance : ℝ := older_train_distance * (1 + percentage_increase)

theorem newer_train_distance_calculation : newer_train_distance = 390 := by
  sorry

end NUMINAMATH_CALUDE_newer_train_distance_calculation_l3906_390630


namespace NUMINAMATH_CALUDE_gcd_lcm_45_75_l3906_390645

theorem gcd_lcm_45_75 :
  (Nat.gcd 45 75 = 15) ∧ (Nat.lcm 45 75 = 1125) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_45_75_l3906_390645


namespace NUMINAMATH_CALUDE_jacket_cost_l3906_390662

theorem jacket_cost (total_spent : ℚ) (shorts_cost : ℚ) (shirt_cost : ℚ) 
  (h1 : total_spent = 33.56)
  (h2 : shorts_cost = 13.99)
  (h3 : shirt_cost = 12.14) :
  total_spent - (shorts_cost + shirt_cost) = 7.43 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_l3906_390662


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3906_390609

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_power_sum : i^23 + i^45 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3906_390609


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3906_390681

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 4*x - 3 > 0} = Set.Ioo 1 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3906_390681


namespace NUMINAMATH_CALUDE_fourth_buoy_distance_l3906_390612

/-- Given buoys placed at even intervals in the ocean, with the third buoy 72 meters from the beach,
    this theorem proves that the fourth buoy is 108 meters from the beach. -/
theorem fourth_buoy_distance (interval : ℝ) (h1 : interval > 0) (h2 : 3 * interval = 72) :
  4 * interval = 108 := by
  sorry

end NUMINAMATH_CALUDE_fourth_buoy_distance_l3906_390612


namespace NUMINAMATH_CALUDE_more_students_than_pets_l3906_390660

theorem more_students_than_pets : 
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 20
  let rabbits_per_classroom : ℕ := 2
  let goldfish_per_classroom : ℕ := 3
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_pets : ℕ := num_classrooms * (rabbits_per_classroom + goldfish_per_classroom)
  total_students - total_pets = 75 := by
sorry

end NUMINAMATH_CALUDE_more_students_than_pets_l3906_390660


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l3906_390624

theorem students_taking_one_subject (both : ℕ) (geometry_total : ℕ) (painting_only : ℕ)
  (h1 : both = 15)
  (h2 : geometry_total = 30)
  (h3 : painting_only = 18) :
  (geometry_total - both) + painting_only = 33 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l3906_390624


namespace NUMINAMATH_CALUDE_inequality_proof_l3906_390628

theorem inequality_proof (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) :
  Real.exp x₂ * Real.log x₁ < Real.exp x₁ * Real.log x₂ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3906_390628


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3906_390622

theorem complex_equation_solution (a : ℝ) (i : ℂ) : 
  i * i = -1 → (a - i)^2 = 2*i → a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3906_390622


namespace NUMINAMATH_CALUDE_range_of_m_l3906_390694

-- Define the sets A and B
def A : Set ℝ := {x | (2 - x) / (2 * x - 1) > 1}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*x + 1 - m ≤ 0}

-- State the theorem
theorem range_of_m (h : ∀ m > 0, A ⊆ B m ∧ ∃ x, x ∈ B m ∧ x ∉ A) :
  {m : ℝ | m ≥ 4} = {m : ℝ | m > 0 ∧ A ⊆ B m ∧ ∃ x, x ∈ B m ∧ x ∉ A} :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3906_390694


namespace NUMINAMATH_CALUDE_combined_mass_of_individuals_l3906_390650

/-- The density of water in kg/m³ -/
def water_density : ℝ := 1000

/-- The length of the boat in meters -/
def boat_length : ℝ := 4

/-- The breadth of the boat in meters -/
def boat_breadth : ℝ := 3

/-- The depth the boat sinks when the first person gets on, in meters -/
def first_person_depth : ℝ := 0.01

/-- The additional depth the boat sinks when the second person gets on, in meters -/
def second_person_depth : ℝ := 0.02

/-- Calculates the mass of water displaced by the boat sinking to a given depth -/
def water_mass (depth : ℝ) : ℝ :=
  boat_length * boat_breadth * depth * water_density

theorem combined_mass_of_individuals :
  water_mass (first_person_depth + second_person_depth) = 360 := by
  sorry

end NUMINAMATH_CALUDE_combined_mass_of_individuals_l3906_390650


namespace NUMINAMATH_CALUDE_reservoir_capacity_after_storm_l3906_390667

/-- Proves that a reservoir with given initial conditions will be 60% full after adding water from a storm. -/
theorem reservoir_capacity_after_storm 
  (original_content : ℝ) 
  (original_percentage : ℝ) 
  (added_water : ℝ) 
  (h1 : original_content = 220)
  (h2 : original_percentage = 0.4)
  (h3 : added_water = 110) :
  (original_content + added_water) / (original_content / original_percentage) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_after_storm_l3906_390667


namespace NUMINAMATH_CALUDE_plus_signs_count_l3906_390648

theorem plus_signs_count (total : ℕ) (plus_count : ℕ) (minus_count : ℕ) :
  total = 23 →
  plus_count + minus_count = total →
  (∀ (subset : Finset ℕ), subset.card = 10 → (∃ (i : ℕ), i ∈ subset ∧ i < plus_count)) →
  (∀ (subset : Finset ℕ), subset.card = 15 → (∃ (i : ℕ), i ∈ subset ∧ plus_count ≤ i ∧ i < total)) →
  plus_count = 14 :=
by sorry

end NUMINAMATH_CALUDE_plus_signs_count_l3906_390648


namespace NUMINAMATH_CALUDE_altitude_sum_of_triangle_l3906_390639

/-- The line equation --/
def line_equation (x y : ℝ) : Prop := 15 * x + 6 * y = 90

/-- A point is on the x-axis if its y-coordinate is 0 --/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A point is on the y-axis if its x-coordinate is 0 --/
def on_y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

/-- The triangle vertices --/
def triangle_vertices : Set (ℝ × ℝ) := {(0, 0), (6, 0), (0, 15)}

/-- The sum of altitudes of the triangle --/
noncomputable def altitude_sum : ℝ := 21 + 10 * Real.sqrt (1 / 29)

/-- The main theorem --/
theorem altitude_sum_of_triangle :
  ∀ (p : ℝ × ℝ), p ∈ triangle_vertices →
  (on_x_axis p ∨ on_y_axis p ∨ line_equation p.1 p.2) →
  altitude_sum = 21 + 10 * Real.sqrt (1 / 29) :=
sorry

end NUMINAMATH_CALUDE_altitude_sum_of_triangle_l3906_390639


namespace NUMINAMATH_CALUDE_holly_fence_length_l3906_390602

/-- The length of Holly's fence in yards -/
def fence_length_yards : ℚ := 25

/-- The cost of trees to cover the fence -/
def total_cost : ℚ := 400

/-- The cost of each tree -/
def tree_cost : ℚ := 8

/-- The width of each tree in feet -/
def tree_width_feet : ℚ := 1.5

/-- The number of feet in a yard -/
def feet_per_yard : ℚ := 3

theorem holly_fence_length :
  fence_length_yards * feet_per_yard = (total_cost / tree_cost) * tree_width_feet :=
by sorry

end NUMINAMATH_CALUDE_holly_fence_length_l3906_390602


namespace NUMINAMATH_CALUDE_cube_equality_iff_three_l3906_390626

theorem cube_equality_iff_three (x : ℝ) (hx : x ≠ 0) :
  (3 * x)^3 = (9 * x)^2 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_equality_iff_three_l3906_390626


namespace NUMINAMATH_CALUDE_position_2018_in_spiral_100_l3906_390654

/-- Represents a position in the matrix -/
structure Position where
  i : Nat
  j : Nat

/-- Constructs a spiral matrix of size n x n -/
def spiralMatrix (n : Nat) : Matrix (Fin n) (Fin n) Nat :=
  sorry

/-- Returns the position of a given number in the spiral matrix -/
def findPosition (n : Nat) (num : Nat) : Position :=
  sorry

/-- Theorem stating that 2018 is at position (34, 95) in a 100x100 spiral matrix -/
theorem position_2018_in_spiral_100 :
  findPosition 100 2018 = Position.mk 34 95 := by
  sorry

end NUMINAMATH_CALUDE_position_2018_in_spiral_100_l3906_390654


namespace NUMINAMATH_CALUDE_rhombus_square_equal_area_l3906_390623

/-- The side length of a square with area equal to a rhombus with diagonals 16 and 8 -/
theorem rhombus_square_equal_area (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 8) :
  ∃ (s : ℝ), s > 0 ∧ (d1 * d2) / 2 = s^2 ∧ s = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_square_equal_area_l3906_390623


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l3906_390610

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ,
    is_prime p ∧ is_prime q ∧
    p > 30 ∧ q > 30 ∧
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ r s : ℕ, is_prime r → is_prime s → r > 30 → s > 30 → r ≠ s → p * q ≤ r * s :=
sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l3906_390610


namespace NUMINAMATH_CALUDE_customers_left_l3906_390605

/-- A problem about customers leaving a waiter's section. -/
theorem customers_left (initial_customers : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : 
  initial_customers = 22 → remaining_tables = 2 → people_per_table = 4 →
  initial_customers - (remaining_tables * people_per_table) = 14 := by
sorry

end NUMINAMATH_CALUDE_customers_left_l3906_390605


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3906_390682

theorem absolute_value_inequality (x : ℝ) : 
  (1 ≤ |x - 2| ∧ |x - 2| ≤ 7) ↔ ((-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3906_390682


namespace NUMINAMATH_CALUDE_point_outside_circle_l3906_390604

/-- Given a circle with center O and radius 3, and a point P outside the circle,
    prove that the distance between O and P is greater than 3. -/
theorem point_outside_circle (O P : ℝ × ℝ) (r : ℝ) : 
  r = 3 →  -- The radius of the circle is 3
  (∀ Q : ℝ × ℝ, dist O Q = r → dist O P > dist O Q) →  -- P is outside the circle
  dist O P > 3  -- The distance between O and P is greater than 3
:= by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3906_390604


namespace NUMINAMATH_CALUDE_equation_2x_squared_eq_1_is_quadratic_l3906_390680

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 2x^2 = 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- Theorem: The equation 2x^2 = 1 is a quadratic equation -/
theorem equation_2x_squared_eq_1_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_2x_squared_eq_1_is_quadratic_l3906_390680


namespace NUMINAMATH_CALUDE_product_remainder_mod_seven_l3906_390675

theorem product_remainder_mod_seven : ((-1234 * 1984 * -1460 * 2008) % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_seven_l3906_390675


namespace NUMINAMATH_CALUDE_february_savings_l3906_390651

/-- Represents the savings pattern over 6 months -/
structure SavingsPattern :=
  (january : ℕ)
  (february : ℕ)
  (march : ℕ)
  (increase : ℕ)
  (total : ℕ)

/-- The savings pattern satisfies the given conditions -/
def ValidPattern (p : SavingsPattern) : Prop :=
  p.january = 2 ∧
  p.march = 8 ∧
  p.total = 126 ∧
  p.march - p.january = p.february - p.january ∧
  p.total = p.january + p.february + p.march + 
            (p.march + p.increase) + 
            (p.march + 2 * p.increase) + 
            (p.march + 3 * p.increase)

/-- The theorem to be proved -/
theorem february_savings (p : SavingsPattern) :
  ValidPattern p → p.february = 50 := by
  sorry

end NUMINAMATH_CALUDE_february_savings_l3906_390651


namespace NUMINAMATH_CALUDE_difference_of_squares_division_problem_solution_l3906_390686

theorem difference_of_squares_division (a b : ℕ) (h : a > b) : 
  (a ^ 2 - b ^ 2) / (a - b) = a + b := by sorry

theorem problem_solution : (125 ^ 2 - 117 ^ 2) / 8 = 242 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_problem_solution_l3906_390686


namespace NUMINAMATH_CALUDE_lemonade_solution_water_parts_l3906_390657

theorem lemonade_solution_water_parts (water_parts : ℝ) : 
  (7 : ℝ) / (water_parts + 7) > (1 : ℝ) / 10 ∧ 
  (7 : ℝ) / (water_parts + 7 - 2.1428571428571423 + 2.1428571428571423) = (1 : ℝ) / 10 → 
  water_parts = 63 := by
sorry

end NUMINAMATH_CALUDE_lemonade_solution_water_parts_l3906_390657


namespace NUMINAMATH_CALUDE_folding_problem_l3906_390641

/-- Represents the folding rate for each type of clothing --/
structure FoldingRate where
  shirts : ℕ
  pants : ℕ
  shorts : ℕ

/-- Represents the number of items for each type of clothing --/
structure ClothingItems where
  shirts : ℕ
  pants : ℕ
  shorts : ℕ

/-- Calculates the remaining items to be folded given the initial conditions --/
def remainingItems (initialItems : ClothingItems) (rate : FoldingRate) (totalTime : ℕ) 
    (shirtFoldTime : ℕ) (pantFoldTime : ℕ) (shirtBreakTime : ℕ) (pantBreakTime : ℕ) : ClothingItems :=
  sorry

/-- The main theorem to be proved --/
theorem folding_problem (initialItems : ClothingItems) (rate : FoldingRate) (totalTime : ℕ) 
    (shirtFoldTime : ℕ) (pantFoldTime : ℕ) (shirtBreakTime : ℕ) (pantBreakTime : ℕ) :
    initialItems = ClothingItems.mk 30 15 20 ∧ 
    rate = FoldingRate.mk 12 8 10 ∧
    totalTime = 120 ∧
    shirtFoldTime = 45 ∧
    pantFoldTime = 30 ∧
    shirtBreakTime = 15 ∧
    pantBreakTime = 10 →
    remainingItems initialItems rate totalTime shirtFoldTime pantFoldTime shirtBreakTime pantBreakTime = 
    ClothingItems.mk 21 11 17 :=
  sorry

end NUMINAMATH_CALUDE_folding_problem_l3906_390641


namespace NUMINAMATH_CALUDE_inequality_proof_l3906_390606

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3906_390606


namespace NUMINAMATH_CALUDE_comparison_theorem_l3906_390619

theorem comparison_theorem :
  (∀ m n : ℝ, m > n → -2*m + 1 < -2*n + 1) ∧
  (∀ m n a : ℝ, 
    (m < n ∧ a = 0 → m*a = n*a) ∧
    (m < n ∧ a > 0 → m*a < n*a) ∧
    (m < n ∧ a < 0 → m*a > n*a)) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l3906_390619


namespace NUMINAMATH_CALUDE_unfair_coin_flip_probability_l3906_390620

/-- The probability of flipping exactly 3 heads in 8 flips of an unfair coin -/
theorem unfair_coin_flip_probability (p : ℚ) (h : p = 1/3) :
  let n : ℕ := 8
  let k : ℕ := 3
  let q : ℚ := 1 - p
  Nat.choose n k * p^k * q^(n-k) = 1792/6561 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_flip_probability_l3906_390620


namespace NUMINAMATH_CALUDE_odd_function_value_l3906_390642

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem odd_function_value (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x < 0, f x = x^2 + 3*x) :
  f 2 = 2 := by sorry

end NUMINAMATH_CALUDE_odd_function_value_l3906_390642


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3906_390618

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 2}

-- Define set B
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3906_390618


namespace NUMINAMATH_CALUDE_handshakes_for_seven_people_l3906_390601

/-- The number of handshakes in a group of n people, where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Theorem stating that the number of handshakes for 7 people is 21. -/
theorem handshakes_for_seven_people : handshakes 7 = 21 := by sorry

end NUMINAMATH_CALUDE_handshakes_for_seven_people_l3906_390601


namespace NUMINAMATH_CALUDE_inequality_proof_l3906_390696

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3906_390696


namespace NUMINAMATH_CALUDE_inequality_solution_l3906_390636

theorem inequality_solution (x : ℝ) :
  x ≠ 5 →
  (x * (x + 2)) / ((x - 5)^2) ≥ 15 ↔ x ∈ Set.Iic 3.71 ∪ Set.Icc 7.14 5 ∪ Set.Ioi 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3906_390636


namespace NUMINAMATH_CALUDE_engagement_treats_value_l3906_390629

def hotel_nights : ℕ := 2
def hotel_cost_per_night : ℕ := 4000
def car_value : ℕ := 30000

def total_treat_value : ℕ :=
  hotel_nights * hotel_cost_per_night + car_value + 4 * car_value

theorem engagement_treats_value :
  total_treat_value = 158000 := by
  sorry

end NUMINAMATH_CALUDE_engagement_treats_value_l3906_390629


namespace NUMINAMATH_CALUDE_circle_on_line_tangent_to_axes_l3906_390617

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - y = 3

-- Define the tangency condition to both axes
def tangent_to_axes (center_x center_y radius : ℝ) : Prop :=
  (abs center_x = radius ∧ abs center_y = radius)

-- Define the circle equation
def circle_equation (center_x center_y radius x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2

-- The main theorem
theorem circle_on_line_tangent_to_axes :
  ∀ (center_x center_y radius : ℝ),
    line_equation center_x center_y →
    tangent_to_axes center_x center_y radius →
    (∀ (x y : ℝ), circle_equation center_x center_y radius x y ↔
      ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_on_line_tangent_to_axes_l3906_390617


namespace NUMINAMATH_CALUDE_coordinate_plane_points_theorem_l3906_390674

theorem coordinate_plane_points_theorem (x y : ℝ) :
  (x^2 * y + y^3 = 2 * x^2 + 2 * y^2 → ((x = 0 ∧ y = 0) ∨ y = 2)) ∧
  (x * y + 1 = x + y → (x = 1 ∨ y = 1)) := by
  sorry

end NUMINAMATH_CALUDE_coordinate_plane_points_theorem_l3906_390674


namespace NUMINAMATH_CALUDE_cheryl_craft_project_cheryl_material_ratio_l3906_390655

/-- The total amount of material used in Cheryl's craft project --/
def total_used (bought_A bought_B bought_C leftover_A leftover_B leftover_C : ℚ) : ℚ :=
  (bought_A - leftover_A) + (bought_B - leftover_B) + (bought_C - leftover_C)

/-- Theorem stating the total amount of material used in Cheryl's craft project --/
theorem cheryl_craft_project :
  let bought_A : ℚ := 5/8
  let bought_B : ℚ := 2/9
  let bought_C : ℚ := 2/5
  let leftover_A : ℚ := 1/12
  let leftover_B : ℚ := 5/36
  let leftover_C : ℚ := 1/10
  total_used bought_A bought_B bought_C leftover_A leftover_B leftover_C = 37/40 :=
by
  sorry

/-- The ratio of materials used in Cheryl's craft project --/
def material_ratio (used_A used_B used_C : ℚ) : Prop :=
  2 * used_B = used_A ∧ 3 * used_B = used_C

/-- Theorem stating the ratio of materials used in Cheryl's craft project --/
theorem cheryl_material_ratio :
  let bought_A : ℚ := 5/8
  let bought_B : ℚ := 2/9
  let bought_C : ℚ := 2/5
  let leftover_A : ℚ := 1/12
  let leftover_B : ℚ := 5/36
  let leftover_C : ℚ := 1/10
  let used_A : ℚ := bought_A - leftover_A
  let used_B : ℚ := bought_B - leftover_B
  let used_C : ℚ := bought_C - leftover_C
  material_ratio used_A used_B used_C :=
by
  sorry

end NUMINAMATH_CALUDE_cheryl_craft_project_cheryl_material_ratio_l3906_390655


namespace NUMINAMATH_CALUDE_spider_plant_production_l3906_390669

/-- Represents the number of baby plants produced by a spider plant over time -/
def babyPlants (plantsPerProduction : ℕ) (productionsPerYear : ℕ) (years : ℕ) : ℕ :=
  plantsPerProduction * productionsPerYear * years

/-- Theorem: A spider plant producing 2 baby plants 2 times a year will produce 16 baby plants after 4 years -/
theorem spider_plant_production :
  babyPlants 2 2 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_spider_plant_production_l3906_390669


namespace NUMINAMATH_CALUDE_mark_age_is_18_l3906_390659

/-- Represents the ages of family members --/
structure FamilyAges where
  mark : ℕ
  john : ℕ
  parents : ℕ

/-- Defines the relationships between family members' ages --/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.john = ages.mark - 10 ∧
  ages.parents = 5 * ages.john ∧
  ages.parents - 22 = ages.mark

/-- Theorem stating that Mark's age is 18 given the family age relationships --/
theorem mark_age_is_18 :
  ∀ (ages : FamilyAges), validFamilyAges ages → ages.mark = 18 := by
  sorry

end NUMINAMATH_CALUDE_mark_age_is_18_l3906_390659


namespace NUMINAMATH_CALUDE_inequality_proof_l3906_390647

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x / (y + z) + y / (x + z) + z / (x + y) ≤ x * Real.sqrt x / 2 + y * Real.sqrt y / 2 + z * Real.sqrt z / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3906_390647


namespace NUMINAMATH_CALUDE_dog_paws_on_ground_l3906_390663

theorem dog_paws_on_ground (total_dogs : ℕ) (h1 : total_dogs = 12) : 
  (total_dogs / 2) * 2 + (total_dogs / 2) * 4 = 36 :=
by sorry

#check dog_paws_on_ground

end NUMINAMATH_CALUDE_dog_paws_on_ground_l3906_390663


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3906_390633

/-- Represents the shaded area calculation problem on a grid with circles --/
theorem shaded_area_calculation (grid_size : ℕ) (small_circle_radius : ℝ) (large_circle_radius : ℝ) 
  (small_circle_count : ℕ) (large_circle_count : ℕ) :
  grid_size = 6 ∧ 
  small_circle_radius = 0.5 ∧ 
  large_circle_radius = 1 ∧
  small_circle_count = 4 ∧
  large_circle_count = 2 →
  ∃ (A C : ℝ), 
    (A - C * Real.pi = grid_size^2 - (small_circle_count * small_circle_radius^2 + large_circle_count * large_circle_radius^2) * Real.pi) ∧
    A + C = 39 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3906_390633


namespace NUMINAMATH_CALUDE_correct_division_result_l3906_390684

theorem correct_division_result (dividend : ℕ) 
  (h1 : dividend / 87 = 24) 
  (h2 : dividend % 87 = 0) : 
  dividend / 36 = 58 := by
sorry

end NUMINAMATH_CALUDE_correct_division_result_l3906_390684


namespace NUMINAMATH_CALUDE_factorization_equality_l3906_390649

theorem factorization_equality (x : ℝ) : (x^2 + 9)^2 - 36*x^2 = (x + 3)^2 * (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3906_390649


namespace NUMINAMATH_CALUDE_a_minus_b_equals_15_l3906_390634

/-- Represents the division of money among A, B, and C -/
structure MoneyDivision where
  a : ℝ  -- Amount received by A
  b : ℝ  -- Amount received by B
  c : ℝ  -- Amount received by C

/-- Conditions for the money division problem -/
def validDivision (d : MoneyDivision) : Prop :=
  d.a = (1/3) * (d.b + d.c) ∧
  d.b = (2/7) * (d.a + d.c) ∧
  d.a > d.b ∧
  d.a + d.b + d.c = 540

/-- Theorem stating that A receives $15 more than B -/
theorem a_minus_b_equals_15 (d : MoneyDivision) (h : validDivision d) :
  d.a - d.b = 15 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_15_l3906_390634


namespace NUMINAMATH_CALUDE_janet_number_problem_l3906_390673

theorem janet_number_problem (x : ℤ) : 
  2 * (x + 7) - 4 = 28 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_janet_number_problem_l3906_390673


namespace NUMINAMATH_CALUDE_f_odd_iff_l3906_390665

/-- A function f is odd if f(-x) = -f(x) for all x in the domain of f. -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x|x + a| + b -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  x * |x + a| + b

/-- The necessary and sufficient condition for f to be an odd function -/
theorem f_odd_iff (a b : ℝ) :
  IsOdd (f a b) ↔ a^2 + b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_odd_iff_l3906_390665


namespace NUMINAMATH_CALUDE_average_pens_sold_per_day_l3906_390631

theorem average_pens_sold_per_day 
  (bundles_sold : ℕ) 
  (days : ℕ) 
  (pens_per_bundle : ℕ) 
  (h1 : bundles_sold = 15) 
  (h2 : days = 5) 
  (h3 : pens_per_bundle = 40) : 
  (bundles_sold * pens_per_bundle) / days = 120 := by
  sorry

end NUMINAMATH_CALUDE_average_pens_sold_per_day_l3906_390631


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_example_l3906_390666

/-- A triangle with sides a, b, and c is an isosceles right triangle -/
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧  -- Two sides are equal
  (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ a^2 + c^2 = b^2)  -- Pythagorean theorem holds

/-- The set {5, 5, 5√2} represents the sides of an isosceles right triangle -/
theorem isosceles_right_triangle_example : is_isosceles_right_triangle 5 5 (5 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_example_l3906_390666


namespace NUMINAMATH_CALUDE_environmental_protection_contest_l3906_390611

theorem environmental_protection_contest (A B C : ℝ) 
  (hA : A = 3/4)
  (hAC : (1 - A) * (1 - C) = 1/12)
  (hBC : B * C = 1/4)
  (hIndep : ∀ X Y : ℝ, X * Y = X * Y) : 
  A * B * C + (1 - A) * B * C + A * (1 - B) * C + A * B * (1 - C) = 21/32 := by
  sorry

end NUMINAMATH_CALUDE_environmental_protection_contest_l3906_390611


namespace NUMINAMATH_CALUDE_remainder_of_m_l3906_390695

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_m_l3906_390695


namespace NUMINAMATH_CALUDE_correct_grass_bundle_equations_l3906_390678

/-- Represents the number of roots in grass bundles -/
structure GrassBundles where
  high_quality : ℕ  -- number of roots in one bundle of high-quality grass
  low_quality : ℕ   -- number of roots in one bundle of low-quality grass

/-- Represents the relationships between high-quality and low-quality grass bundles -/
def grass_bundle_relations (g : GrassBundles) : Prop :=
  (5 * g.high_quality - 11 = 7 * g.low_quality) ∧
  (7 * g.high_quality - 25 = 5 * g.low_quality)

/-- Theorem stating that the given system of equations correctly represents the problem -/
theorem correct_grass_bundle_equations (g : GrassBundles) :
  grass_bundle_relations g ↔
  (5 * g.high_quality - 11 = 7 * g.low_quality) ∧
  (7 * g.high_quality - 25 = 5 * g.low_quality) :=
by sorry

end NUMINAMATH_CALUDE_correct_grass_bundle_equations_l3906_390678


namespace NUMINAMATH_CALUDE_probability_white_or_red_l3906_390677

def total_balls : ℕ := 8 + 9 + 3
def white_balls : ℕ := 8
def black_balls : ℕ := 9
def red_balls : ℕ := 3

theorem probability_white_or_red :
  (white_balls + red_balls : ℚ) / total_balls = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_or_red_l3906_390677


namespace NUMINAMATH_CALUDE_sum_and_product_problem_l3906_390637

theorem sum_and_product_problem (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) :
  1 / x + 1 / y = 5 / 12 ∧ x^2 + y^2 = 153 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_problem_l3906_390637


namespace NUMINAMATH_CALUDE_blue_tiles_45th_row_l3906_390685

/-- Calculates the total number of tiles in a row given the row number -/
def totalTiles (n : ℕ) : ℕ := 2 * n - 1

/-- Calculates the number of blue tiles in a row given the total number of tiles -/
def blueTiles (total : ℕ) : ℕ := (total - 1) / 2

theorem blue_tiles_45th_row :
  blueTiles (totalTiles 45) = 44 := by
  sorry

end NUMINAMATH_CALUDE_blue_tiles_45th_row_l3906_390685


namespace NUMINAMATH_CALUDE_valid_parameterization_l3906_390603

/-- A vector parameterization of a line --/
structure VectorParam where
  x0 : ℝ
  y0 : ℝ
  dx : ℝ
  dy : ℝ

/-- The line y = 2x + 4 --/
def line (x : ℝ) : ℝ := 2 * x + 4

/-- Checks if a vector is a scalar multiple of (2, 1) --/
def isValidDirection (dx dy : ℝ) : Prop := ∃ (k : ℝ), dx = 2 * k ∧ dy = k

/-- Checks if a point (x0, y0) lies on the line y = 2x + 4 --/
def isOnLine (x0 y0 : ℝ) : Prop := y0 = line x0

/-- Theorem: A vector parameterization is valid iff its direction is a scalar multiple of (2, 1) and its initial point lies on the line --/
theorem valid_parameterization (p : VectorParam) : 
  (isValidDirection p.dx p.dy ∧ isOnLine p.x0 p.y0) ↔ 
  (∀ t : ℝ, line (p.x0 + t * p.dx) = p.y0 + t * p.dy) :=
sorry

end NUMINAMATH_CALUDE_valid_parameterization_l3906_390603


namespace NUMINAMATH_CALUDE_tractor_financing_term_l3906_390644

/-- Calculates the financing term in years given the monthly payment and total financed amount. -/
def financing_term_years (monthly_payment : ℚ) (total_amount : ℚ) : ℚ :=
  (total_amount / monthly_payment) / 12

/-- Theorem stating that the financing term for the given conditions is 5 years. -/
theorem tractor_financing_term :
  let monthly_payment : ℚ := 150
  let total_amount : ℚ := 9000
  financing_term_years monthly_payment total_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_tractor_financing_term_l3906_390644


namespace NUMINAMATH_CALUDE_ratio_solution_l3906_390690

theorem ratio_solution (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (3 * a + 4 * b) / (b + 12 * a) = 2) : 
  a / b = (5 - Real.sqrt 19) / 6 ∨ a / b = (5 + Real.sqrt 19) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_solution_l3906_390690


namespace NUMINAMATH_CALUDE_lunch_price_with_gratuity_l3906_390640

theorem lunch_price_with_gratuity 
  (num_people : ℕ) 
  (avg_price : ℝ) 
  (gratuity_rate : ℝ) : 
  num_people = 15 →
  avg_price = 12 →
  gratuity_rate = 0.15 →
  num_people * avg_price * (1 + gratuity_rate) = 207 := by
  sorry

end NUMINAMATH_CALUDE_lunch_price_with_gratuity_l3906_390640


namespace NUMINAMATH_CALUDE_implicit_function_derivatives_l3906_390676

/-- Given an implicit function defined by x^y - y^x = 0, this theorem proves
    the expressions for its first and second derivatives. -/
theorem implicit_function_derivatives
  (x y : ℝ) (h : x^y = y^x) (hx : x > 0) (hy : y > 0) :
  let y' := (y^2 * (Real.log x - 1)) / (x^2 * (Real.log y - 1))
  let y'' := (x * (3 - 2 * Real.log x) * (Real.log y - 1)^2 +
              (Real.log x - 1)^2 * (2 * Real.log y - 3) * y) *
             y^2 / (x^4 * (Real.log y - 1)^3)
  ∃ f : ℝ → ℝ, (∀ t, t^(f t) = (f t)^t) ∧
               (deriv f x = y') ∧
               (deriv (deriv f) x = y'') := by
  sorry

end NUMINAMATH_CALUDE_implicit_function_derivatives_l3906_390676


namespace NUMINAMATH_CALUDE_dvd_cost_l3906_390698

/-- The cost of each DVD given the total number of movies, trade-in value per VHS, and total replacement cost. -/
theorem dvd_cost (total_movies : ℕ) (vhs_trade_value : ℚ) (total_replacement_cost : ℚ) :
  total_movies = 100 →
  vhs_trade_value = 2 →
  total_replacement_cost = 800 →
  (total_replacement_cost - (total_movies : ℚ) * vhs_trade_value) / total_movies = 6 := by
  sorry

end NUMINAMATH_CALUDE_dvd_cost_l3906_390698


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3906_390614

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → c = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3906_390614


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3906_390692

theorem area_between_concentric_circles 
  (r_outer : ℝ) 
  (r_inner : ℝ) 
  (chord_length : ℝ) 
  (h_r_outer : r_outer = 60) 
  (h_r_inner : r_inner = 36) 
  (h_chord : chord_length = 96) 
  (h_tangent : chord_length / 2 = Real.sqrt (r_outer^2 - r_inner^2)) : 
  π * (r_outer^2 - r_inner^2) = 2304 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3906_390692


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3906_390656

/-- Represents a parabola y² = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Represents a point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

theorem parabola_focus_distance (C : Parabola) (A : PointOnParabola C)
  (h_focus_dist : Real.sqrt ((A.x - C.p / 2)^2 + A.y^2) = 12)
  (h_y_axis_dist : A.x = 9) :
  C.p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3906_390656


namespace NUMINAMATH_CALUDE_bucket_weight_l3906_390646

/-- Given a bucket where:
    - The weight when half full (including the bucket) is c
    - The weight when completely full (including the bucket) is d
    This theorem proves that the weight when three-quarters full is (1/2)c + (1/2)d -/
theorem bucket_weight (c d : ℝ) : ℝ :=
  let half_full := c
  let full := d
  let three_quarters_full := (1/2 : ℝ) * c + (1/2 : ℝ) * d
  three_quarters_full

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l3906_390646


namespace NUMINAMATH_CALUDE_student_count_l3906_390688

/-- The number of students in the group -/
def num_students : ℕ := 6

/-- The weight decrease when replacing the heavier student with the lighter one -/
def weight_difference : ℕ := 80 - 62

/-- The average weight decrease per student -/
def avg_weight_decrease : ℕ := 3

theorem student_count :
  num_students * avg_weight_decrease = weight_difference :=
sorry

end NUMINAMATH_CALUDE_student_count_l3906_390688


namespace NUMINAMATH_CALUDE_max_exchanges_theorem_l3906_390683

/-- Represents a student with a height -/
structure Student where
  height : ℕ

/-- Represents a circle of students -/
def StudentCircle := List Student

/-- Condition for a student to switch places -/
def canSwitch (s₁ s₂ : Student) (prevHeight : ℕ) : Prop :=
  s₁.height > s₂.height ∧ s₂.height ≤ prevHeight

/-- The maximum number of exchanges possible -/
def maxExchanges (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 6

/-- Theorem stating the maximum number of exchanges -/
theorem max_exchanges_theorem (n : ℕ) (circle : StudentCircle) :
  (circle.length = n) →
  (∀ i j, i < j → (circle.get i).height < (circle.get j).height) →
  (∀ exchanges, exchanges ≤ maxExchanges n) := by
  sorry

end NUMINAMATH_CALUDE_max_exchanges_theorem_l3906_390683


namespace NUMINAMATH_CALUDE_diagonal_increase_l3906_390699

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := sorry

/-- Theorem stating the relationship between the number of diagonals
    in a convex polygon with n sides and n+1 sides -/
theorem diagonal_increase (n : ℕ) :
  num_diagonals (n + 1) = num_diagonals n + n - 1 := by sorry

end NUMINAMATH_CALUDE_diagonal_increase_l3906_390699


namespace NUMINAMATH_CALUDE_area_conversions_l3906_390658

-- Define conversion rates
def sq_dm_to_sq_cm : ℝ := 100
def hectare_to_sq_m : ℝ := 10000
def sq_km_to_hectare : ℝ := 100
def sq_m_to_sq_dm : ℝ := 100

-- Theorem to prove the conversions
theorem area_conversions :
  (7 * sq_dm_to_sq_cm = 700) ∧
  (5 * hectare_to_sq_m = 50000) ∧
  (600 / sq_km_to_hectare = 6) ∧
  (200 / sq_m_to_sq_dm = 2) :=
by sorry

end NUMINAMATH_CALUDE_area_conversions_l3906_390658


namespace NUMINAMATH_CALUDE_intersection_property_l3906_390652

/-- Given a function f(x) = |sin x| and a line y = kx (k > 0) that intersect at exactly three points,
    with the maximum x-coordinate of the intersections being α, prove that:
    cos α / (sin α + sin 3α) = (1 + α²) / (4α) -/
theorem intersection_property (k α : ℝ) (hk : k > 0) 
    (h_intersections : ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ α ∧
      (∀ x, k * x = |Real.sin x| ↔ x = x₁ ∨ x = x₂ ∨ x = x₃))
    (h_max : ∀ x, k * x = |Real.sin x| → x ≤ α) :
  Real.cos α / (Real.sin α + Real.sin (3 * α)) = (1 + α^2) / (4 * α) := by
  sorry

end NUMINAMATH_CALUDE_intersection_property_l3906_390652


namespace NUMINAMATH_CALUDE_vacation_cost_per_person_l3906_390615

theorem vacation_cost_per_person 
  (num_people : ℕ) 
  (airbnb_cost car_cost : ℚ) 
  (h1 : num_people = 8)
  (h2 : airbnb_cost = 3200)
  (h3 : car_cost = 800) :
  (airbnb_cost + car_cost) / num_people = 500 :=
by sorry

end NUMINAMATH_CALUDE_vacation_cost_per_person_l3906_390615


namespace NUMINAMATH_CALUDE_problem_statement_l3906_390693

theorem problem_statement (x y : ℝ) : 
  16 * (4 : ℝ)^x = 3^(y + 2) → y = -2 → x = -2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3906_390693


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3906_390632

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {2, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3906_390632


namespace NUMINAMATH_CALUDE_sheila_picnic_probability_l3906_390661

/-- The probability of Sheila attending the picnic given the weather conditions and transport strike possibility. -/
theorem sheila_picnic_probability :
  let p_rain : ℝ := 0.5
  let p_sunny : ℝ := 1 - p_rain
  let p_attend_if_rain : ℝ := 0.25
  let p_attend_if_sunny : ℝ := 0.8
  let p_transport_strike : ℝ := 0.1
  let p_attend : ℝ := (p_rain * p_attend_if_rain + p_sunny * p_attend_if_sunny) * (1 - p_transport_strike)
  p_attend = 0.4725 := by
  sorry

end NUMINAMATH_CALUDE_sheila_picnic_probability_l3906_390661


namespace NUMINAMATH_CALUDE_sum_first_105_remainder_l3906_390600

theorem sum_first_105_remainder (n : Nat) (sum : Nat → Nat) : 
  n = 105 → 
  (∀ k, sum k = k * (k + 1) / 2) → 
  sum n % 1000 = 565 := by
sorry

end NUMINAMATH_CALUDE_sum_first_105_remainder_l3906_390600


namespace NUMINAMATH_CALUDE_escalator_speed_increase_l3906_390687

theorem escalator_speed_increase (total_steps : ℕ) (first_climb : ℕ) (second_climb : ℕ)
  (h_total : total_steps = 125)
  (h_first : first_climb = 45)
  (h_second : second_climb = 55)
  (h_first_valid : first_climb < total_steps)
  (h_second_valid : second_climb < total_steps) :
  (second_climb : ℚ) / first_climb * (total_steps - first_climb : ℚ) / (total_steps - second_climb) = 88 / 63 :=
by sorry

end NUMINAMATH_CALUDE_escalator_speed_increase_l3906_390687


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3906_390625

-- Define variables
variable (a b : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : 3 * a - (4 * b - 2 * a + 1) = 5 * a - 4 * b - 1 := by
  sorry

-- Theorem for the second expression
theorem simplify_expression_2 : 2 * (5 * a - 3 * b) - 3 * (a^2 - 2 * b) = 10 * a - 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3906_390625


namespace NUMINAMATH_CALUDE_least_beads_beads_solution_l3906_390607

theorem least_beads (b : ℕ) : 
  (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) → b ≥ 179 :=
by
  sorry

theorem beads_solution : 
  ∃ (b : ℕ), (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) ∧ b = 179 :=
by
  sorry

end NUMINAMATH_CALUDE_least_beads_beads_solution_l3906_390607


namespace NUMINAMATH_CALUDE_kendra_minivan_count_l3906_390668

theorem kendra_minivan_count : 
  ∀ (afternoon_count evening_count total_count : ℕ),
  afternoon_count = 4 →
  evening_count = 1 →
  total_count = afternoon_count + evening_count →
  total_count = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_kendra_minivan_count_l3906_390668


namespace NUMINAMATH_CALUDE_solution_set_when_m_eq_5_range_of_m_for_f_geq_7_l3906_390679

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| + |x + 6|

-- Theorem for part I
theorem solution_set_when_m_eq_5 :
  {x : ℝ | f x 5 ≤ 12} = {x : ℝ | -13/2 ≤ x ∧ x ≤ 11/2} := by sorry

-- Theorem for part II
theorem range_of_m_for_f_geq_7 :
  {m : ℝ | ∀ x, f x m ≥ 7} = {m : ℝ | m ≤ -13 ∨ 1 ≤ m} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_eq_5_range_of_m_for_f_geq_7_l3906_390679


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l3906_390672

/-- An odd function defined on ℝ -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_positive : ∀ x ≥ 0, f x = x^2 - 2*x) :
  ∀ x < 0, f x = -x^2 - 2*x :=
by sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l3906_390672


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3906_390638

-- Define the function f(x) = (x-1)^2 - 2
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f x ≤ f y :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3906_390638


namespace NUMINAMATH_CALUDE_prop_truth_values_l3906_390664

-- Define a structure for a line
structure Line where
  slope : ℝ
  x_intercept : ℝ
  y_intercept : ℝ

-- Define the original proposition
def original_prop (l : Line) : Prop :=
  l.slope = -1 → l.x_intercept = l.y_intercept

-- Define the converse
def converse_prop (l : Line) : Prop :=
  l.x_intercept = l.y_intercept → l.slope = -1

-- Define the inverse
def inverse_prop (l : Line) : Prop :=
  l.slope ≠ -1 → l.x_intercept ≠ l.y_intercept

-- Define the contrapositive
def contrapositive_prop (l : Line) : Prop :=
  l.x_intercept ≠ l.y_intercept → l.slope ≠ -1

-- Theorem stating the truth values of the propositions
theorem prop_truth_values :
  ∃ l : Line, original_prop l ∧
  ¬(∀ l : Line, converse_prop l) ∧
  ¬(∀ l : Line, inverse_prop l) ∧
  (∀ l : Line, contrapositive_prop l) :=
sorry

end NUMINAMATH_CALUDE_prop_truth_values_l3906_390664


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_A_intersect_C_nonempty_l3906_390643

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 < x ∧ x ≤ 8}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 8} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | 1 < x ∧ x < 9} := by sorry

-- Theorem for the condition when A ∩ C is non-empty
theorem A_intersect_C_nonempty (a : ℝ) : (A ∩ C a).Nonempty ↔ a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_A_intersect_C_nonempty_l3906_390643


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3906_390621

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ+ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n : ℕ+, a n > 0)
  (h_prod : a 2 * a 4 = 9) :
  a 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3906_390621


namespace NUMINAMATH_CALUDE_man_business_ownership_l3906_390627

/-- 
Given a business valued at 10000 rs, if a man sells 3/5 of his shares for 2000 rs,
then he originally owned 1/3 of the business.
-/
theorem man_business_ownership (man_share : ℚ) : 
  (3 / 5 : ℚ) * man_share * 10000 = 2000 → man_share = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_man_business_ownership_l3906_390627


namespace NUMINAMATH_CALUDE_perforation_information_count_l3906_390613

theorem perforation_information_count : 
  ∀ (n : ℕ) (states : ℕ), 
    n = 8 → 
    states = 2 → 
    states ^ n = 256 := by
  sorry

end NUMINAMATH_CALUDE_perforation_information_count_l3906_390613


namespace NUMINAMATH_CALUDE_good_quadruple_inequality_l3906_390691

/-- A good quadruple is a set of positive integers (p, a, b, c) satisfying certain conditions. -/
structure GoodQuadruple where
  p : Nat
  a : Nat
  b : Nat
  c : Nat
  p_prime : Nat.Prime p
  p_odd : Odd p
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c
  div_ab : p ∣ (a * b + 1)
  div_bc : p ∣ (b * c + 1)
  div_ca : p ∣ (c * a + 1)

/-- The main theorem about good quadruples. -/
theorem good_quadruple_inequality (q : GoodQuadruple) :
  q.p + 2 ≤ (q.a + q.b + q.c) / 3 ∧
  (q.p + 2 = (q.a + q.b + q.c) / 3 ↔ q.a = 2 ∧ q.b = 2 + q.p ∧ q.c = 2 + 2 * q.p) :=
by sorry

end NUMINAMATH_CALUDE_good_quadruple_inequality_l3906_390691


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3906_390653

-- Define the square root function
def square_root (x : ℝ) : Set ℝ :=
  {y : ℝ | y * y = x}

-- Theorem statement
theorem square_root_of_nine :
  square_root 9 = {3, -3} := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3906_390653


namespace NUMINAMATH_CALUDE_unique_mapping_l3906_390608

-- Define the property for the mapping
def SatisfiesProperty (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f (f n) ≤ (n + f n) / 2

-- Define the identity function on ℕ
def IdentityFunc : ℕ → ℕ := λ n => n

-- Theorem statement
theorem unique_mapping :
  ∀ f : ℕ → ℕ, Function.Injective f → SatisfiesProperty f → f = IdentityFunc :=
sorry

end NUMINAMATH_CALUDE_unique_mapping_l3906_390608


namespace NUMINAMATH_CALUDE_bianca_birthday_money_l3906_390616

theorem bianca_birthday_money (total_amount : ℕ) (num_friends : ℕ) (amount_per_friend : ℕ) 
  (h1 : total_amount = 30) 
  (h2 : num_friends = 5) 
  (h3 : total_amount = num_friends * amount_per_friend) : 
  amount_per_friend = 6 := by
  sorry

end NUMINAMATH_CALUDE_bianca_birthday_money_l3906_390616


namespace NUMINAMATH_CALUDE_reconstruct_quadrilateral_l3906_390635

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A convex quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The intersection point of the diagonals -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- The projection of a point onto a line segment -/
def projectPointOntoSegment (p : Point) (a : Point) (b : Point) : Point :=
  sorry

/-- Theorem: Given four points that are projections of the diagonal intersection
    onto the sides of a convex quadrilateral, we can reconstruct the quadrilateral -/
theorem reconstruct_quadrilateral 
  (M N K L : Point) 
  (h : ∃ (q : Quadrilateral), 
    M = projectPointOntoSegment (diagonalIntersection q) q.A q.B ∧
    N = projectPointOntoSegment (diagonalIntersection q) q.B q.C ∧
    K = projectPointOntoSegment (diagonalIntersection q) q.C q.D ∧
    L = projectPointOntoSegment (diagonalIntersection q) q.D q.A) :
  ∃! (q : Quadrilateral), 
    M = projectPointOntoSegment (diagonalIntersection q) q.A q.B ∧
    N = projectPointOntoSegment (diagonalIntersection q) q.B q.C ∧
    K = projectPointOntoSegment (diagonalIntersection q) q.C q.D ∧
    L = projectPointOntoSegment (diagonalIntersection q) q.D q.A :=
  sorry

end NUMINAMATH_CALUDE_reconstruct_quadrilateral_l3906_390635
