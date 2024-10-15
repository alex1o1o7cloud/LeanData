import Mathlib

namespace NUMINAMATH_CALUDE_restaurant_time_is_ten_l1463_146372

-- Define the times as natural numbers (in minutes)
def time_to_hidden_lake : ℕ := 15
def time_from_hidden_lake : ℕ := 7
def total_journey_time : ℕ := 32

-- Define the time to Lake Park restaurant as a function
def time_to_restaurant : ℕ := total_journey_time - (time_to_hidden_lake + time_from_hidden_lake)

-- Theorem statement
theorem restaurant_time_is_ten : time_to_restaurant = 10 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_time_is_ten_l1463_146372


namespace NUMINAMATH_CALUDE_solve_for_a_l1463_146364

theorem solve_for_a : ∃ a : ℝ, 
  (∃ x y : ℝ, x = 1 ∧ y = 2 ∧ a * x - y = 3) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1463_146364


namespace NUMINAMATH_CALUDE_point_2_3_in_first_quadrant_l1463_146317

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Theorem: The point (2, 3) is in the first quadrant -/
theorem point_2_3_in_first_quadrant :
  let p : Point := ⟨2, 3⟩
  isInFirstQuadrant p := by
  sorry

end NUMINAMATH_CALUDE_point_2_3_in_first_quadrant_l1463_146317


namespace NUMINAMATH_CALUDE_taco_cost_is_90_cents_l1463_146331

-- Define the cost of a taco and an enchilada
variable (taco_cost enchilada_cost : ℚ)

-- Define the two orders
def order1_cost := 2 * taco_cost + 3 * enchilada_cost
def order2_cost := 3 * taco_cost + 5 * enchilada_cost

-- State the theorem
theorem taco_cost_is_90_cents 
  (h1 : order1_cost = 780/100)
  (h2 : order2_cost = 1270/100) :
  taco_cost = 90/100 := by
  sorry

end NUMINAMATH_CALUDE_taco_cost_is_90_cents_l1463_146331


namespace NUMINAMATH_CALUDE_part_one_part_two_l1463_146347

-- Define set A
def A : Set ℝ := {x | (x + 1) / (x - 3) ≤ 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - (m - 1) * x + m - 2 ≤ 0}

-- Statement for part (1)
theorem part_one (a b : ℝ) : A ∪ Set.Icc a b = Set.Icc (-1) 4 → b = 4 ∧ -1 ≤ a ∧ a < 3 := by
  sorry

-- Statement for part (2)
theorem part_two (m : ℝ) : A ∪ B m = A → 1 ≤ m ∧ m < 5 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1463_146347


namespace NUMINAMATH_CALUDE_vector_at_t_5_l1463_146334

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  /-- The vector on the line at parameter t -/
  vector_at : ℝ → ℝ × ℝ

/-- Theorem: Given a parameterized line with specific points, the vector at t=5 is (10, -11) -/
theorem vector_at_t_5 
  (line : ParameterizedLine) 
  (h1 : line.vector_at 1 = (2, 5)) 
  (h4 : line.vector_at 4 = (8, -7)) : 
  line.vector_at 5 = (10, -11) := by
sorry


end NUMINAMATH_CALUDE_vector_at_t_5_l1463_146334


namespace NUMINAMATH_CALUDE_common_roots_imply_c_d_l1463_146301

/-- Two cubic polynomials with two distinct common roots -/
def has_two_common_roots (c d : ℝ) : Prop :=
  ∃ r s : ℝ, r ≠ s ∧
    (r^3 + c*r^2 + 12*r + 7 = 0) ∧ 
    (r^3 + d*r^2 + 15*r + 9 = 0) ∧
    (s^3 + c*s^2 + 12*s + 7 = 0) ∧ 
    (s^3 + d*s^2 + 15*s + 9 = 0)

/-- The theorem stating that if the polynomials have two distinct common roots, then c = 5 and d = 4 -/
theorem common_roots_imply_c_d (c d : ℝ) :
  has_two_common_roots c d → c = 5 ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_common_roots_imply_c_d_l1463_146301


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1463_146398

theorem age_ratio_problem (cindy_age jan_age marcia_age greg_age : ℕ) : 
  cindy_age = 5 →
  jan_age = cindy_age + 2 →
  ∃ k : ℕ, marcia_age = k * jan_age →
  greg_age = marcia_age + 2 →
  greg_age = 16 →
  marcia_age / jan_age = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1463_146398


namespace NUMINAMATH_CALUDE_cow_count_is_83_l1463_146325

/-- Calculates the final number of cows given the initial count and changes in the herd -/
def final_cow_count (initial : ℕ) (died : ℕ) (sold : ℕ) (increased : ℕ) (bought : ℕ) (gifted : ℕ) : ℕ :=
  initial - died - sold + increased + bought + gifted

/-- Theorem stating that given the specific changes in the herd, the final count is 83 -/
theorem cow_count_is_83 :
  final_cow_count 39 25 6 24 43 8 = 83 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_is_83_l1463_146325


namespace NUMINAMATH_CALUDE_f_is_even_l1463_146382

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem stating that f is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1463_146382


namespace NUMINAMATH_CALUDE_hiker_total_distance_l1463_146337

-- Define the hiker's walking parameters
def day1_distance : ℕ := 18
def day1_speed : ℕ := 3
def day2_speed_increase : ℕ := 1
def day3_speed : ℕ := 5
def day3_hours : ℕ := 6

-- Theorem to prove
theorem hiker_total_distance :
  let day1_hours : ℕ := day1_distance / day1_speed
  let day2_hours : ℕ := day1_hours - 1
  let day2_speed : ℕ := day1_speed + day2_speed_increase
  let day2_distance : ℕ := day2_speed * day2_hours
  let day3_distance : ℕ := day3_speed * day3_hours
  day1_distance + day2_distance + day3_distance = 68 := by
  sorry


end NUMINAMATH_CALUDE_hiker_total_distance_l1463_146337


namespace NUMINAMATH_CALUDE_locus_of_tangent_circles_l1463_146353

/-- The equation of circle C1 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The equation of circle C3 -/
def C3 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

/-- A circle is externally tangent to C1 and internally tangent to C3 -/
def is_tangent_to_C1_C3 (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (5 - r)^2)

/-- The locus equation -/
def locus_equation (a b : ℝ) : Prop :=
  40 * a^2 + 49 * b^2 - 48 * a - 64 = 0

theorem locus_of_tangent_circles :
  ∀ a b : ℝ, (∃ r : ℝ, is_tangent_to_C1_C3 a b r) ↔ locus_equation a b :=
sorry

end NUMINAMATH_CALUDE_locus_of_tangent_circles_l1463_146353


namespace NUMINAMATH_CALUDE_unique_D_value_l1463_146393

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Definition of our addition problem -/
def AdditionProblem (A B C D : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  1000 * A.val + 100 * A.val + 10 * C.val + B.val +
  1000 * B.val + 100 * C.val + 10 * B.val + D.val =
  1000 * B.val + 100 * D.val + 10 * A.val + B.val

theorem unique_D_value (A B C D : Digit) :
  AdditionProblem A B C D → D.val = 0 ∧ ∀ E : Digit, AdditionProblem A B C E → E = D :=
by sorry

end NUMINAMATH_CALUDE_unique_D_value_l1463_146393


namespace NUMINAMATH_CALUDE_money_difference_proof_l1463_146311

/-- The number of nickels in a quarter -/
def nickels_per_quarter : ℕ := 5

/-- Charles' quarters as a function of q -/
def charles_quarters (q : ℤ) : ℤ := 7 * q + 2

/-- Richard's quarters as a function of q -/
def richard_quarters (q : ℤ) : ℤ := 3 * q + 8

/-- The difference in money between Charles and Richard, expressed in nickels -/
def money_difference_in_nickels (q : ℤ) : ℤ :=
  nickels_per_quarter * (charles_quarters q - richard_quarters q)

theorem money_difference_proof (q : ℤ) :
  money_difference_in_nickels q = 20 * q - 30 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_proof_l1463_146311


namespace NUMINAMATH_CALUDE_trig_identity_l1463_146338

theorem trig_identity : 
  (2 * Real.sin (46 * π / 180) - Real.sqrt 3 * Real.cos (74 * π / 180)) / Real.cos (16 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1463_146338


namespace NUMINAMATH_CALUDE_chosen_number_proof_l1463_146346

theorem chosen_number_proof (x : ℝ) : (x / 6) - 189 = 3 → x = 1152 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l1463_146346


namespace NUMINAMATH_CALUDE_find_m_l1463_146386

theorem find_m : ∃ m : ℤ, (|m| = 2 ∧ m - 2 ≠ 0) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1463_146386


namespace NUMINAMATH_CALUDE_melanie_dimes_l1463_146385

theorem melanie_dimes (x : ℕ) : x + 8 + 4 = 19 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l1463_146385


namespace NUMINAMATH_CALUDE_polygon_sides_l1463_146328

theorem polygon_sides (sum_interior_angles : ℝ) :
  sum_interior_angles = 1620 →
  ∃ n : ℕ, n = 11 ∧ sum_interior_angles = 180 * (n - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1463_146328


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1463_146305

theorem smallest_sum_of_sequence (X Y Z W : ℕ) : 
  X > 0 → Y > 0 → Z > 0 → W > 0 →
  (∃ d : ℤ, Z - Y = Y - X ∧ Z - Y = d) →
  (∃ r : ℚ, Z = r * Y ∧ W = r * Z) →
  Z = (9 : ℚ) / 5 * Y →
  (∀ a b c d : ℕ, a > 0 → b > 0 → c > 0 → d > 0 →
    (∃ d' : ℤ, c - b = b - a ∧ c - b = d') →
    (∃ r' : ℚ, c = r' * b ∧ d = r' * c) →
    c = (9 : ℚ) / 5 * b →
    X + Y + Z + W ≤ a + b + c + d) →
  X + Y + Z + W = 156 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1463_146305


namespace NUMINAMATH_CALUDE_log_25_between_consecutive_integers_l1463_146396

theorem log_25_between_consecutive_integers :
  ∃ c d : ℤ, c + 1 = d ∧ (c : ℝ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < d ∧ c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_25_between_consecutive_integers_l1463_146396


namespace NUMINAMATH_CALUDE_erdos_szekeres_l1463_146306

theorem erdos_szekeres (n : ℕ) (seq : Fin (n^2 + 1) → ℝ) :
  (∃ (subseq : Fin (n + 1) → Fin (n^2 + 1)), Monotone (seq ∘ subseq)) ∨
  (∃ (subseq : Fin (n + 1) → Fin (n^2 + 1)), StrictAnti (seq ∘ subseq)) :=
sorry

end NUMINAMATH_CALUDE_erdos_szekeres_l1463_146306


namespace NUMINAMATH_CALUDE_square_region_area_l1463_146312

/-- A region consisting of equal squares inscribed in a rectangle -/
structure SquareRegion where
  num_squares : ℕ
  rect_width : ℝ
  rect_height : ℝ

/-- Calculate the area of a SquareRegion -/
def area (r : SquareRegion) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem square_region_area (r : SquareRegion) 
  (h1 : r.num_squares = 13)
  (h2 : r.rect_width = 28)
  (h3 : r.rect_height = 26) : 
  area r = 338 := by
  sorry

end NUMINAMATH_CALUDE_square_region_area_l1463_146312


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_nine_l1463_146318

theorem seven_digit_divisible_by_nine (m : ℕ) : 
  m < 10 →
  (746 * 1000000 + m * 10000 + 813) % 9 = 0 →
  m = 7 := by
sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_nine_l1463_146318


namespace NUMINAMATH_CALUDE_plane_equation_theorem_l1463_146369

/-- The equation of a plane given its normal vector and a point on the plane -/
def plane_equation (normal : ℝ × ℝ × ℝ) (point : ℝ × ℝ × ℝ) : ℤ × ℤ × ℤ × ℤ :=
  sorry

/-- Check if the first coefficient is positive -/
def first_coeff_positive (coeffs : ℤ × ℤ × ℤ × ℤ) : Prop :=
  sorry

/-- Calculate the GCD of the absolute values of all coefficients -/
def gcd_of_coeffs (coeffs : ℤ × ℤ × ℤ × ℤ) : ℕ :=
  sorry

theorem plane_equation_theorem :
  let normal : ℝ × ℝ × ℝ := (10, -5, 6)
  let point : ℝ × ℝ × ℝ := (10, -5, 6)
  let coeffs := plane_equation normal point
  first_coeff_positive coeffs ∧ gcd_of_coeffs coeffs = 1 ∧ coeffs = (10, -5, 6, -161) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_theorem_l1463_146369


namespace NUMINAMATH_CALUDE_area_of_triangle_abc_l1463_146332

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * b * c * Real.sin A

theorem area_of_triangle_abc (a b c : ℝ) (A B C : ℝ) 
  (h1 : A = π/4)
  (h2 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) :
  triangle_area a b c A B C = 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_abc_l1463_146332


namespace NUMINAMATH_CALUDE_perfect_square_trinomials_l1463_146377

-- Perfect square trinomial properties
theorem perfect_square_trinomials 
  (x a b : ℝ) : 
  (x^2 + 6*x + 9 = (x + 3)^2) ∧ 
  (x^2 + 8*x + 16 = (x + 4)^2) ∧ 
  (x^2 - 12*x + 36 = (x - 6)^2) ∧ 
  (a^2 + 2*a*b + b^2 = (a + b)^2) ∧ 
  (a^2 - 2*a*b + b^2 = (a - b)^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomials_l1463_146377


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1463_146355

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1463_146355


namespace NUMINAMATH_CALUDE_range_of_f_l1463_146391

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 - 2)^2

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ 4} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1463_146391


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l1463_146359

theorem smallest_factorization_coefficient : ∃ (r s : ℕ+), 
  (r : ℤ) * s = 1620 ∧ 
  r + s = 84 ∧ 
  (∀ (r' s' : ℕ+), (r' : ℤ) * s' = 1620 → r' + s' ≥ 84) := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l1463_146359


namespace NUMINAMATH_CALUDE_hotel_rooms_theorem_l1463_146330

/-- The minimum number of rooms needed for 100 tourists with k rooms under renovation -/
def min_rooms (k : ℕ) : ℕ :=
  let m := k / 2
  if k % 2 = 0 then
    100 * (m + 1)
  else
    100 * (m + 1) + 1

/-- Theorem stating the minimum number of rooms needed for 100 tourists -/
theorem hotel_rooms_theorem (k : ℕ) :
  ∀ n : ℕ, n ≥ min_rooms k →
  ∃ strategy : (Fin 100 → Fin n → Option (Fin n)),
  (∀ i : Fin 100, ∃ room : Fin n, strategy i room = some room) ∧
  (∀ i j : Fin 100, i ≠ j →
    ∀ room : Fin n, strategy i room ≠ none → strategy j room = none) :=
by
  sorry

#check hotel_rooms_theorem

end NUMINAMATH_CALUDE_hotel_rooms_theorem_l1463_146330


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1463_146344

/-- Proves that a train with given length, crossing a bridge of given length in a given time, has a specific speed in km/hr -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 140 ∧ 
  bridge_length = 132 ∧ 
  crossing_time = 13.598912087033037 →
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1463_146344


namespace NUMINAMATH_CALUDE_cube_sum_and_product_theorem_l1463_146352

theorem cube_sum_and_product_theorem :
  ∃! (n : ℕ), ∃ (a b : ℕ+),
    a ^ 3 + b ^ 3 = 189 ∧
    a * b = 20 ∧
    n = 2 :=
sorry

end NUMINAMATH_CALUDE_cube_sum_and_product_theorem_l1463_146352


namespace NUMINAMATH_CALUDE_bacteria_habitat_limits_l1463_146383

/-- Represents a bacterial colony with its growth characteristics -/
structure BacterialColony where
  growthFactor : ℕ       -- How much the colony multiplies in size
  growthPeriod : ℕ       -- Number of days between each growth
  totalDays : ℕ          -- Total number of days the colony grows

/-- Calculates the number of days it takes for a colony to reach its habitat limit -/
def daysToHabitatLimit (colony : BacterialColony) : ℕ :=
  colony.totalDays

/-- Colony A doubles every day for 22 days -/
def colonyA : BacterialColony :=
  { growthFactor := 2
  , growthPeriod := 1
  , totalDays := 22 }

/-- Colony B triples every 2 days for 30 days -/
def colonyB : BacterialColony :=
  { growthFactor := 3
  , growthPeriod := 2
  , totalDays := 30 }

theorem bacteria_habitat_limits :
  daysToHabitatLimit colonyA = 22 ∧ daysToHabitatLimit colonyB = 30 := by
  sorry

#eval daysToHabitatLimit colonyA
#eval daysToHabitatLimit colonyB

end NUMINAMATH_CALUDE_bacteria_habitat_limits_l1463_146383


namespace NUMINAMATH_CALUDE_unique_divisor_square_sum_l1463_146324

theorem unique_divisor_square_sum (p n : ℕ) (hp : Prime p) (hn : n > 0) (hodd : Odd p) :
  ∃! d : ℕ, d > 0 ∧ d ∣ (p * n^2) ∧ ∃ m : ℕ, d + n^2 = m^2 ↔ ∃ k : ℕ, n = k * ((p - 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_unique_divisor_square_sum_l1463_146324


namespace NUMINAMATH_CALUDE_jacob_age_l1463_146308

/-- Given Rehana's current age, her age relative to Phoebe's in 5 years, and Jacob's age relative to Phoebe's, prove Jacob's current age. -/
theorem jacob_age (rehana_age : ℕ) (phoebe_age : ℕ) (jacob_age : ℕ) : 
  rehana_age = 25 →
  rehana_age + 5 = 3 * (phoebe_age + 5) →
  jacob_age = 3 * phoebe_age / 5 →
  jacob_age = 3 :=
by sorry

end NUMINAMATH_CALUDE_jacob_age_l1463_146308


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1463_146392

theorem complex_fraction_simplification :
  let z : ℂ := (10 : ℂ) - 8 * Complex.I
  let w : ℂ := (3 : ℂ) + 4 * Complex.I
  z / w = -(2 : ℂ) / 25 - (64 : ℂ) / 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1463_146392


namespace NUMINAMATH_CALUDE_lu_daokui_scholarship_winners_l1463_146316

theorem lu_daokui_scholarship_winners 
  (total_winners : ℕ) 
  (first_prize_amount : ℕ) 
  (second_prize_amount : ℕ) 
  (total_prize_money : ℕ) 
  (h1 : total_winners = 28)
  (h2 : first_prize_amount = 10000)
  (h3 : second_prize_amount = 2000)
  (h4 : total_prize_money = 80000) :
  ∃ (first_prize_winners second_prize_winners : ℕ),
    first_prize_winners + second_prize_winners = total_winners ∧
    first_prize_winners * first_prize_amount + second_prize_winners * second_prize_amount = total_prize_money ∧
    first_prize_winners = 3 ∧
    second_prize_winners = 25 := by
  sorry

end NUMINAMATH_CALUDE_lu_daokui_scholarship_winners_l1463_146316


namespace NUMINAMATH_CALUDE_standard_deviation_of_random_variable_l1463_146358

def random_variable (ξ : ℝ → ℝ) : Prop :=
  (ξ 1 = 0.4) ∧ (ξ 3 = 0.1) ∧ (∃ x, ξ 5 = x) ∧ (ξ 1 + ξ 3 + ξ 5 = 1)

def expected_value (ξ : ℝ → ℝ) : ℝ :=
  1 * ξ 1 + 3 * ξ 3 + 5 * ξ 5

def variance (ξ : ℝ → ℝ) : ℝ :=
  (1 - expected_value ξ)^2 * ξ 1 + 
  (3 - expected_value ξ)^2 * ξ 3 + 
  (5 - expected_value ξ)^2 * ξ 5

theorem standard_deviation_of_random_variable (ξ : ℝ → ℝ) :
  random_variable ξ → Real.sqrt (variance ξ) = Real.sqrt 3.56 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_of_random_variable_l1463_146358


namespace NUMINAMATH_CALUDE_x_range_given_quadratic_inequality_l1463_146310

theorem x_range_given_quadratic_inequality (x : ℝ) :
  4 - x^2 ≤ 0 → x ≤ -2 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_given_quadratic_inequality_l1463_146310


namespace NUMINAMATH_CALUDE_a_10_has_1000_nines_l1463_146303

def sequence_a : ℕ → ℕ
  | 0 => 9
  | (k + 1) => 3 * (sequence_a k)^4 + 4 * (sequence_a k)^3

def has_consecutive_nines (n : ℕ) (count : ℕ) : Prop :=
  ∃ m : ℕ, n = m * 10^count + (10^count - 1)

theorem a_10_has_1000_nines : 
  has_consecutive_nines (sequence_a 10) 1000 := by sorry

end NUMINAMATH_CALUDE_a_10_has_1000_nines_l1463_146303


namespace NUMINAMATH_CALUDE_fraction_value_l1463_146307

theorem fraction_value : (2200 - 2096)^2 / 121 = 89 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l1463_146307


namespace NUMINAMATH_CALUDE_cats_meowing_time_l1463_146368

/-- The number of minutes the cats were meowing -/
def minutes : ℚ := 5

/-- The number of meows per minute for the first cat -/
def first_cat_meows : ℚ := 3

/-- The number of meows per minute for the second cat -/
def second_cat_meows : ℚ := 2 * first_cat_meows

/-- The number of meows per minute for the third cat -/
def third_cat_meows : ℚ := (1/3) * second_cat_meows

/-- The total number of meows -/
def total_meows : ℚ := 55

theorem cats_meowing_time :
  minutes * (first_cat_meows + second_cat_meows + third_cat_meows) = total_meows :=
by sorry

end NUMINAMATH_CALUDE_cats_meowing_time_l1463_146368


namespace NUMINAMATH_CALUDE_shortest_player_height_l1463_146361

theorem shortest_player_height (tallest_height : Float) (height_difference : Float) :
  tallest_height = 77.75 →
  height_difference = 9.5 →
  tallest_height - height_difference = 68.25 := by
sorry

end NUMINAMATH_CALUDE_shortest_player_height_l1463_146361


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1463_146333

/-- The perimeter of a regular hexagon given the distance between opposite sides -/
theorem regular_hexagon_perimeter (d : ℝ) (h : d = 15) : 
  let s := 2 * d / Real.sqrt 3
  6 * s = 60 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1463_146333


namespace NUMINAMATH_CALUDE_crate_missing_dimension_l1463_146327

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  width : Real
  length : Real
  height : Real

/-- Represents a cylindrical tank -/
structure CylindricalTank where
  radius : Real
  height : Real

def fits_in_crate (tank : CylindricalTank) (crate : CrateDimensions) : Prop :=
  2 * tank.radius ≤ min crate.width crate.length ∧
  tank.height ≤ crate.height

def is_max_volume (tank : CylindricalTank) (crate : CrateDimensions) : Prop :=
  fits_in_crate tank crate ∧
  ∀ other_tank : CylindricalTank,
    fits_in_crate other_tank crate →
    tank.radius * tank.radius * tank.height ≥ other_tank.radius * other_tank.radius * other_tank.height

theorem crate_missing_dimension
  (crate : CrateDimensions)
  (h_width : crate.width = 8)
  (h_length : crate.length = 12)
  (tank : CylindricalTank)
  (h_radius : tank.radius = 6)
  (h_max_volume : is_max_volume tank crate) :
  crate.height = 12 :=
sorry

end NUMINAMATH_CALUDE_crate_missing_dimension_l1463_146327


namespace NUMINAMATH_CALUDE_probability_two_cards_sum_19_l1463_146388

/-- Represents a standard 52-card deck --/
def StandardDeck : ℕ := 52

/-- Number of cards that can be part of the pair (9 or 10) --/
def ValidFirstCards : ℕ := 8

/-- Number of complementary cards after drawing the first card --/
def ComplementaryCards : ℕ := 4

/-- Probability of drawing two number cards totaling 19 from a standard deck --/
theorem probability_two_cards_sum_19 :
  (ValidFirstCards : ℚ) / StandardDeck * ComplementaryCards / (StandardDeck - 1) = 8 / 663 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_cards_sum_19_l1463_146388


namespace NUMINAMATH_CALUDE_cubic_equation_with_double_root_l1463_146304

theorem cubic_equation_with_double_root (k : ℝ) :
  (∃ a b : ℝ, (3 * a^3 + 9 * a^2 - 150 * a + k = 0) ∧
              (3 * b^3 + 9 * b^2 - 150 * b + k = 0) ∧
              (a ≠ b)) ∧
  (∃ x : ℝ, (3 * x^3 + 9 * x^2 - 150 * x + k = 0) ∧
            (∃ y : ℝ, y ≠ x ∧ 3 * y^3 + 9 * y^2 - 150 * y + k = 0)) ∧
  (k > 0) →
  k = 84 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_with_double_root_l1463_146304


namespace NUMINAMATH_CALUDE_equation_solution_l1463_146300

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Ceiling function: smallest integer greater than or equal to x -/
noncomputable def ceil (x : ℝ) : ℤ :=
  -Int.floor (-x)

/-- Nearest integer function: integer closest to x (x ≠ n + 0.5 for any integer n) -/
noncomputable def nearest (x : ℝ) : ℤ :=
  if x - Int.floor x < 0.5 then Int.floor x else Int.floor x + 1

/-- Theorem: The equation 3⌊x⌋ + 2⌈x⌉ + ⟨x⟩ = 8 is satisfied if and only if 1 < x < 1.5 -/
theorem equation_solution (x : ℝ) :
  3 * (floor x) + 2 * (ceil x) + (nearest x) = 8 ↔ 1 < x ∧ x < 1.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1463_146300


namespace NUMINAMATH_CALUDE_bipartite_graph_completion_l1463_146320

/-- A bipartite graph with n vertices in each partition -/
structure BipartiteGraph (n : ℕ) :=
  (A B : Finset (Fin n))
  (edges : Finset (Fin n × Fin n))
  (bipartite : ∀ (e : Fin n × Fin n), e ∈ edges → (e.1 ∈ A ∧ e.2 ∈ B) ∨ (e.1 ∈ B ∧ e.2 ∈ A))

/-- The degree of a vertex in a bipartite graph -/
def degree (G : BipartiteGraph n) (v : Fin n) : ℕ :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- The theorem statement -/
theorem bipartite_graph_completion
  (n d : ℕ) (h_pos : 0 < n ∧ 0 < d) (h_bound : d < n / 2)
  (G : BipartiteGraph n)
  (h_degree : ∀ v, degree G v ≤ d) :
  ∃ G' : BipartiteGraph n,
    (∀ e ∈ G.edges, e ∈ G'.edges) ∧
    (∀ v, degree G' v = 2 * d) :=
sorry

end NUMINAMATH_CALUDE_bipartite_graph_completion_l1463_146320


namespace NUMINAMATH_CALUDE_function_equation_solver_l1463_146397

theorem function_equation_solver (f : ℝ → ℝ) :
  (∀ x, f (x + 1) = x^2 + 4*x + 1) →
  (∀ x, f x = x^2 + 2*x - 2) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solver_l1463_146397


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l1463_146374

theorem largest_multiple_of_8_under_100 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 100 → n ≤ 96 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l1463_146374


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1463_146343

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) :
  perimeter = 40 →
  area = (perimeter / 4)^2 →
  area = 100 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1463_146343


namespace NUMINAMATH_CALUDE_distance_between_centers_l1463_146322

/-- An isosceles triangle with its circumcircle and inscribed circle -/
structure IsoscelesTriangleWithCircles where
  /-- The radius of the circumcircle -/
  R : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- R is positive -/
  R_pos : R > 0
  /-- r is positive -/
  r_pos : r > 0
  /-- r is less than R (as the inscribed circle must fit inside the circumcircle) -/
  r_lt_R : r < R

/-- The distance between the centers of the circumcircle and inscribed circle
    of an isosceles triangle is √(R(R - 2r)) -/
theorem distance_between_centers (t : IsoscelesTriangleWithCircles) :
  ∃ d : ℝ, d = Real.sqrt (t.R * (t.R - 2 * t.r)) ∧ d > 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_centers_l1463_146322


namespace NUMINAMATH_CALUDE_chocolate_division_l1463_146323

theorem chocolate_division (total_chocolate : ℚ) (piles : ℕ) (friends : ℕ) : 
  total_chocolate = 60 / 7 →
  piles = 5 →
  friends = 3 →
  (total_chocolate / piles * (piles - 1)) / friends = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l1463_146323


namespace NUMINAMATH_CALUDE_recurring_decimal_difference_l1463_146367

theorem recurring_decimal_difference : 
  let x : ℚ := 8/11  -- 0.overline{72}
  let y : ℚ := 18/25 -- 0.72
  x - y = 2/275 := by
sorry

end NUMINAMATH_CALUDE_recurring_decimal_difference_l1463_146367


namespace NUMINAMATH_CALUDE_sum_of_decimals_l1463_146348

theorem sum_of_decimals : 2.75 + 0.003 + 0.158 = 2.911 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l1463_146348


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1463_146378

/-- The solution set of the inequality (x-2)(ax-2) > 0 -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then Set.Iio 2
  else if a < 0 then Set.Ioo (2/a) 2
  else if 0 < a ∧ a < 1 then Set.Iio 2 ∪ Set.Ioi (2/a)
  else if a > 1 then Set.Iio (2/a) ∪ Set.Ioi 2
  else Set.Iio 2 ∪ Set.Ioi 2

theorem inequality_solution_set (a : ℝ) (x : ℝ) :
  (x - 2) * (a * x - 2) > 0 ↔ x ∈ solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1463_146378


namespace NUMINAMATH_CALUDE_exponent_property_l1463_146315

theorem exponent_property (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_property_l1463_146315


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1463_146351

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the original inequality
def S := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x, f a b c x > 0 ↔ x ∈ S) :
  ∀ x, f c b a x > 0 ↔ x < -1 ∨ x > (1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1463_146351


namespace NUMINAMATH_CALUDE_tulips_to_remaining_ratio_l1463_146389

def total_flowers : ℕ := 12
def daisies : ℕ := 2
def sunflowers : ℕ := 4

def tulips : ℕ := total_flowers - (daisies + sunflowers)
def remaining_flowers : ℕ := tulips + sunflowers

theorem tulips_to_remaining_ratio :
  (tulips : ℚ) / (remaining_flowers : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tulips_to_remaining_ratio_l1463_146389


namespace NUMINAMATH_CALUDE_domino_probability_and_attempts_l1463_146349

/-- The total number of domino tiles -/
def total_tiles : ℕ := 45

/-- The number of tiles drawn -/
def drawn_tiles : ℕ := 3

/-- The probability of the event occurring in a single attempt -/
def event_probability : ℚ := 54 / 473

/-- The minimum probability we want to achieve -/
def target_probability : ℝ := 0.9

/-- The minimum number of attempts needed -/
def min_attempts : ℕ := 19

/-- Theorem stating the probability of the event and the minimum number of attempts needed -/
theorem domino_probability_and_attempts :
  (event_probability : ℝ) = 54 / 473 ∧
  (1 - (1 - event_probability) ^ min_attempts : ℝ) ≥ target_probability ∧
  ∀ n : ℕ, n < min_attempts → (1 - (1 - event_probability) ^ n : ℝ) < target_probability :=
sorry


end NUMINAMATH_CALUDE_domino_probability_and_attempts_l1463_146349


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1463_146375

/-- Given that 8y varies inversely as the cube of x, and y = 25 when x = 2,
    prove that y = 25/8 when x = 4. -/
theorem inverse_variation_problem (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, 8 * y x = k / x^3) →  -- 8y varies inversely as the cube of x
  y 2 = 25 →                 -- y = 25 when x = 2
  y 4 = 25 / 8 :=             -- y = 25/8 when x = 4
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1463_146375


namespace NUMINAMATH_CALUDE_erased_line_length_l1463_146390

/-- Proves that erasing 33 cm from a 1 m line results in a 67 cm line -/
theorem erased_line_length : 
  let initial_length_m : ℝ := 1
  let initial_length_cm : ℝ := initial_length_m * 100
  let erased_length_cm : ℝ := 33
  let final_length_cm : ℝ := initial_length_cm - erased_length_cm
  final_length_cm = 67 := by sorry

end NUMINAMATH_CALUDE_erased_line_length_l1463_146390


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1463_146329

-- Define the universal set U
def U : Set ℝ := {x | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 < x ∧ x < 3)} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1463_146329


namespace NUMINAMATH_CALUDE_new_people_total_weight_l1463_146363

/-- Proves that the total weight of five new people joining a group is 270kg -/
theorem new_people_total_weight (initial_count : ℕ) (first_replacement_count : ℕ) (second_replacement_count : ℕ)
  (initial_average_increase : ℝ) (second_average_decrease : ℝ) 
  (first_outgoing_weights : Fin 3 → ℝ) (second_outgoing_total : ℝ) :
  initial_count = 20 ∧ 
  first_replacement_count = 3 ∧
  second_replacement_count = 2 ∧
  initial_average_increase = 2.5 ∧
  second_average_decrease = 1.8 ∧
  first_outgoing_weights 0 = 36 ∧
  first_outgoing_weights 1 = 48 ∧
  first_outgoing_weights 2 = 62 ∧
  second_outgoing_total = 110 →
  (initial_count : ℝ) * initial_average_increase + (first_outgoing_weights 0 + first_outgoing_weights 1 + first_outgoing_weights 2) +
  (second_outgoing_total - (initial_count : ℝ) * second_average_decrease) = 270 := by
  sorry

end NUMINAMATH_CALUDE_new_people_total_weight_l1463_146363


namespace NUMINAMATH_CALUDE_function_inequality_l1463_146366

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (1/3)^x - x^2

-- State the theorem
theorem function_inequality (x₀ x₁ x₂ m : ℝ) 
  (h1 : f x₀ = m) 
  (h2 : x₁ ∈ Set.Ioo 0 x₀) 
  (h3 : x₂ ∈ Set.Ioi x₀) : 
  f x₁ > m ∧ f x₂ < m := by
  sorry

end

end NUMINAMATH_CALUDE_function_inequality_l1463_146366


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l1463_146373

/-- A complete graph with 6 vertices where each edge is colored either black or red -/
def ColoredGraph6 := Fin 6 → Fin 6 → Bool

/-- A triangle in the graph is represented by three distinct vertices -/
def Triangle (G : ColoredGraph6) (a b c : Fin 6) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- A monochromatic triangle has all edges of the same color -/
def MonochromaticTriangle (G : ColoredGraph6) (a b c : Fin 6) : Prop :=
  Triangle G a b c ∧
  ((G a b = G b c ∧ G b c = G a c) ∨
   (G a b ≠ G b c ∧ G b c ≠ G a c ∧ G a c ≠ G a b))

/-- The main theorem: every 2-coloring of K6 contains a monochromatic triangle -/
theorem monochromatic_triangle_exists (G : ColoredGraph6) :
  ∃ (a b c : Fin 6), MonochromaticTriangle G a b c := by
  sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l1463_146373


namespace NUMINAMATH_CALUDE_contest_end_time_l1463_146336

def contest_start : Nat := 12 * 60  -- noon in minutes since midnight
def contest_duration : Nat := 1000  -- duration in minutes

theorem contest_end_time :
  (contest_start + contest_duration) % (24 * 60) = 4 * 60 + 40 :=
sorry

end NUMINAMATH_CALUDE_contest_end_time_l1463_146336


namespace NUMINAMATH_CALUDE_max_d_is_two_l1463_146339

def a (n : ℕ) : ℕ := 101 + (n + 1)^2 + 3*n

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_is_two : 
  (∃ (n : ℕ), d n = 2) ∧ (∀ (n : ℕ), d n ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_max_d_is_two_l1463_146339


namespace NUMINAMATH_CALUDE_typing_speed_equation_l1463_146365

theorem typing_speed_equation (x : ℝ) : x > 0 → x + 6 > 0 →
  (Xiao_Ming_speed : ℝ) →
  (Xiao_Zhang_speed : ℝ) →
  (Xiao_Ming_speed = x) →
  (Xiao_Zhang_speed = x + 6) →
  (120 / Xiao_Ming_speed = 180 / Xiao_Zhang_speed) →
  120 / x = 180 / (x + 6) := by
sorry

end NUMINAMATH_CALUDE_typing_speed_equation_l1463_146365


namespace NUMINAMATH_CALUDE_range_of_t_l1463_146387

def M : Set ℝ := {x | -2 < x ∧ x < 5}

def N (t : ℝ) : Set ℝ := {x | 2 - t < x ∧ x < 2*t + 1}

theorem range_of_t : 
  (∀ t : ℝ, M ∩ N t = N t) ↔ (∀ t : ℝ, t ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_t_l1463_146387


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1463_146319

def i : ℂ := Complex.I

theorem complex_modulus_problem (a : ℝ) (z : ℂ) 
  (h1 : z = (2 - a * i) / i) 
  (h2 : z.re = 0) : 
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1463_146319


namespace NUMINAMATH_CALUDE_min_colors_condition_1_min_colors_condition_2_l1463_146341

variable (n : ℕ)

/-- The set of all lattice points in n-dimensional space -/
def X : Set (Fin n → ℤ) := Set.univ

/-- Distance between two lattice points -/
def distance (A B : Fin n → ℤ) : ℕ :=
  (Finset.univ.sum fun i => (A i - B i).natAbs)

/-- A coloring of X is valid for Condition 1 if any two points of the same color have distance ≥ 2 -/
def valid_coloring_1 (c : (Fin n → ℤ) → Fin r) : Prop :=
  ∀ A B : Fin n → ℤ, c A = c B → distance n A B ≥ 2

/-- A coloring of X is valid for Condition 2 if any two points of the same color have distance ≥ 3 -/
def valid_coloring_2 (c : (Fin n → ℤ) → Fin r) : Prop :=
  ∀ A B : Fin n → ℤ, c A = c B → distance n A B ≥ 3

/-- The minimum number of colors needed to satisfy Condition 1 is 2 -/
theorem min_colors_condition_1 :
  (∃ c : (Fin n → ℤ) → Fin 2, valid_coloring_1 n c) ∧
  (∀ r < 2, ¬∃ c : (Fin n → ℤ) → Fin r, valid_coloring_1 n c) :=
sorry

/-- The minimum number of colors needed to satisfy Condition 2 is 2n + 1 -/
theorem min_colors_condition_2 :
  (∃ c : (Fin n → ℤ) → Fin (2 * n + 1), valid_coloring_2 n c) ∧
  (∀ r < 2 * n + 1, ¬∃ c : (Fin n → ℤ) → Fin r, valid_coloring_2 n c) :=
sorry

end NUMINAMATH_CALUDE_min_colors_condition_1_min_colors_condition_2_l1463_146341


namespace NUMINAMATH_CALUDE_cubic_poly_b_value_l1463_146362

/-- Represents a cubic polynomial of the form x^3 - ax^2 + bx - b --/
def cubic_poly (a b : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + b*x - b

/-- Predicate to check if all roots of the polynomial are real and positive --/
def all_roots_real_positive (a b : ℝ) : Prop :=
  ∀ x : ℝ, cubic_poly a b x = 0 → x > 0

/-- The main theorem stating the value of b --/
theorem cubic_poly_b_value :
  ∃ (a : ℝ), a > 0 ∧
  (∀ a' : ℝ, a' > 0 → all_roots_real_positive a' (a'^2/3) → a ≤ a') ∧
  all_roots_real_positive a (a^2/3) ∧
  a^2/3 = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_poly_b_value_l1463_146362


namespace NUMINAMATH_CALUDE_shoes_sold_shoes_sold_is_six_l1463_146371

theorem shoes_sold (shoe_price : ℕ) (shirt_price : ℕ) (num_shirts : ℕ) (individual_earnings : ℕ) : ℕ :=
  let total_earnings := 2 * individual_earnings
  let shirt_earnings := shirt_price * num_shirts
  let shoe_earnings := total_earnings - shirt_earnings
  shoe_earnings / shoe_price

theorem shoes_sold_is_six : shoes_sold 3 2 18 27 = 6 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_shoes_sold_is_six_l1463_146371


namespace NUMINAMATH_CALUDE_largest_non_expressible_l1463_146335

/-- A function that checks if a number is composite -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a number can be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
def CanBeExpressed (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ IsComposite m ∧ n = 30 * k + m

/-- Theorem stating that 211 is the largest positive integer that cannot be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
theorem largest_non_expressible : ∀ n : ℕ, n > 211 → CanBeExpressed n ∧ ¬CanBeExpressed 211 :=
sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l1463_146335


namespace NUMINAMATH_CALUDE_f_six_equals_one_half_l1463_146395

-- Define the function f
noncomputable def f : ℝ → ℝ := λ u => (u^2 - 8*u + 20) / 16

-- State the theorem
theorem f_six_equals_one_half :
  (∀ x : ℝ, f (4*x + 2) = x^2 - x + 1) → f 6 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_six_equals_one_half_l1463_146395


namespace NUMINAMATH_CALUDE_employee_pay_l1463_146314

theorem employee_pay (total : ℝ) (x y : ℝ) (h1 : total = 616) (h2 : x = 1.2 * y) (h3 : total = x + y) : y = 280 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l1463_146314


namespace NUMINAMATH_CALUDE_annie_hamburgers_l1463_146384

theorem annie_hamburgers (initial_amount : ℕ) (hamburger_cost : ℕ) (milkshake_cost : ℕ)
  (milkshakes_bought : ℕ) (amount_left : ℕ) :
  initial_amount = 120 →
  hamburger_cost = 4 →
  milkshake_cost = 3 →
  milkshakes_bought = 6 →
  amount_left = 70 →
  ∃ (hamburgers_bought : ℕ),
    hamburgers_bought = 8 ∧
    initial_amount = amount_left + hamburger_cost * hamburgers_bought + milkshake_cost * milkshakes_bought :=
by
  sorry

end NUMINAMATH_CALUDE_annie_hamburgers_l1463_146384


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l1463_146340

theorem max_sum_on_circle (x y : ℝ) (h : (x - 3)^2 + (y - 3)^2 = 8) :
  ∃ (max : ℝ), ∀ (a b : ℝ), (a - 3)^2 + (b - 3)^2 = 8 → a + b ≤ max ∧ ∃ (u v : ℝ), (u - 3)^2 + (v - 3)^2 = 8 ∧ u + v = max :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l1463_146340


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1463_146370

theorem quadratic_root_problem (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x - 7 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 + m * y - 7 = 0 ∧ y = -7/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1463_146370


namespace NUMINAMATH_CALUDE_distance_circle_center_to_point_l1463_146356

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y - 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 3)

-- Define the given point
def given_point : ℝ × ℝ := (10, 8)

-- Theorem statement
theorem distance_circle_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  (cx - px)^2 + (cy - py)^2 = 89 :=
sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_point_l1463_146356


namespace NUMINAMATH_CALUDE_thousand_to_100_equals_googol_cubed_l1463_146342

-- Define googol
def googol : ℕ := 10^100

-- Theorem statement
theorem thousand_to_100_equals_googol_cubed :
  1000^100 = googol^3 := by
  sorry

end NUMINAMATH_CALUDE_thousand_to_100_equals_googol_cubed_l1463_146342


namespace NUMINAMATH_CALUDE_reciprocal_multiplier_l1463_146399

theorem reciprocal_multiplier (x m : ℝ) : 
  x > 0 → x = 7 → x - 4 = m * (1/x) → m = 21 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_multiplier_l1463_146399


namespace NUMINAMATH_CALUDE_product_of_base_nine_digits_7654_l1463_146380

/-- Represents a number in base 9 as a list of digits -/
def BaseNineRepresentation := List Nat

/-- Converts a base 10 number to its base 9 representation -/
def toBaseNine (n : Nat) : BaseNineRepresentation :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List Nat) : Nat :=
  sorry

theorem product_of_base_nine_digits_7654 :
  productOfList (toBaseNine 7654) = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base_nine_digits_7654_l1463_146380


namespace NUMINAMATH_CALUDE_equations_truth_l1463_146350

-- Define the theorem
theorem equations_truth :
  -- Equation 1
  (∀ a : ℝ, Real.sqrt ((a^2 + 1)^2) = a^2 + 1) ∧
  -- Equation 2
  (∀ a : ℝ, Real.sqrt (a^2) = abs a) ∧
  -- Equation 4
  (∀ x : ℝ, x ≥ 1 → Real.sqrt ((x + 1) * (x - 1)) = Real.sqrt (x + 1) * Real.sqrt (x - 1)) ∧
  -- Equation 3 (counterexample)
  (∃ a b : ℝ, Real.sqrt (a * b) ≠ Real.sqrt a * Real.sqrt b) :=
by
  sorry

end NUMINAMATH_CALUDE_equations_truth_l1463_146350


namespace NUMINAMATH_CALUDE_geometric_sequence_product_range_l1463_146326

theorem geometric_sequence_product_range (a₁ a₂ a₃ m q : ℝ) (hm : m > 0) (hq : q > 0) :
  (a₁ + a₂ + a₃ = 3 * m) →
  (a₂ = a₁ * q) →
  (a₃ = a₂ * q) →
  let t := a₁ * a₂ * a₃
  0 < t ∧ t ≤ m^3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_range_l1463_146326


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1463_146394

/-- Calculates the total cost for a group at a restaurant given specific pricing and group composition. -/
theorem restaurant_bill_calculation 
  (adult_meal_cost : ℚ)
  (kid_meal_cost : ℚ)
  (adult_drink_cost : ℚ)
  (kid_drink_cost : ℚ)
  (dessert_cost : ℚ)
  (total_people : ℕ)
  (num_adults : ℕ)
  (num_children : ℕ)
  (h1 : adult_meal_cost = 12)
  (h2 : kid_meal_cost = 0)
  (h3 : adult_drink_cost = 5/2)
  (h4 : kid_drink_cost = 3/2)
  (h5 : dessert_cost = 4)
  (h6 : total_people = 11)
  (h7 : num_adults = 7)
  (h8 : num_children = 4)
  (h9 : total_people = num_adults + num_children) :
  (num_adults * adult_meal_cost) +
  (num_adults * adult_drink_cost) +
  (num_children * kid_drink_cost) +
  (total_people * dessert_cost) = 151.5 := by
    sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1463_146394


namespace NUMINAMATH_CALUDE_smaller_cuboid_length_l1463_146360

/-- Proves that the length of smaller cuboids is 5 meters, given the specified conditions --/
theorem smaller_cuboid_length : 
  ∀ (large_length large_width large_height : ℝ) 
    (small_width small_height : ℝ) 
    (num_small_cuboids : ℕ),
  large_length = 18 →
  large_width = 15 →
  large_height = 2 →
  small_width = 2 →
  small_height = 3 →
  num_small_cuboids = 18 →
  ∃ (small_length : ℝ),
    small_length = 5 ∧
    large_length * large_width * large_height = 
      num_small_cuboids * small_length * small_width * small_height :=
by sorry

end NUMINAMATH_CALUDE_smaller_cuboid_length_l1463_146360


namespace NUMINAMATH_CALUDE_optimal_pole_minimizes_time_l1463_146302

/-- The optimal pole number for tying Bolivar -/
def optimal_pole : ℕ := 21

/-- The total number of poles -/
def total_poles : ℕ := 27

/-- The total number of segments -/
def total_segments : ℕ := 28

/-- Dotson's walking speed in units per minute -/
def dotson_speed : ℚ := 1 / 9

/-- Williams' walking speed in units per minute -/
def williams_speed : ℚ := 1 / 11

/-- Bolivar's riding speed in units per minute -/
def bolivar_speed : ℚ := 1 / 3

/-- The time taken by Dotson to complete the journey -/
def dotson_time (k : ℕ) : ℚ := 9 - (6 * k : ℚ) / 28

/-- The time taken by Williams to complete the journey -/
def williams_time (k : ℕ) : ℚ := 3 + (2 * k : ℚ) / 7

theorem optimal_pole_minimizes_time :
  ∀ k : ℕ, k ≤ total_poles →
    max (dotson_time k) (williams_time k) ≥ dotson_time optimal_pole :=
by sorry


end NUMINAMATH_CALUDE_optimal_pole_minimizes_time_l1463_146302


namespace NUMINAMATH_CALUDE_division_theorem_l1463_146376

theorem division_theorem (dividend divisor remainder quotient : ℕ) :
  dividend = 176 →
  divisor = 14 →
  remainder = 8 →
  quotient = 12 →
  dividend = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_division_theorem_l1463_146376


namespace NUMINAMATH_CALUDE_johns_bonus_is_twenty_l1463_146309

/-- Calculate the performance bonus for John's job --/
def performance_bonus (normal_wage : ℝ) (normal_hours : ℝ) (extra_hours : ℝ) (bonus_rate : ℝ) : ℝ :=
  (normal_hours + extra_hours) * bonus_rate - normal_wage

/-- Theorem stating that John's performance bonus is $20 per day --/
theorem johns_bonus_is_twenty :
  performance_bonus 80 8 2 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_johns_bonus_is_twenty_l1463_146309


namespace NUMINAMATH_CALUDE_gmat_test_problem_l1463_146357

theorem gmat_test_problem (first_correct : Real) (second_correct : Real) (neither_correct : Real) :
  first_correct = 85 / 100 →
  second_correct = 65 / 100 →
  neither_correct = 5 / 100 →
  first_correct + second_correct - (1 - neither_correct) = 55 / 100 := by
  sorry

end NUMINAMATH_CALUDE_gmat_test_problem_l1463_146357


namespace NUMINAMATH_CALUDE_total_markers_count_l1463_146354

/-- The number of red markers Connie has -/
def red_markers : ℕ := 2315

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 1028

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers

/-- Theorem stating that the total number of markers is 3343 -/
theorem total_markers_count : total_markers = 3343 := by
  sorry

end NUMINAMATH_CALUDE_total_markers_count_l1463_146354


namespace NUMINAMATH_CALUDE_largest_n_for_divisibility_l1463_146313

theorem largest_n_for_divisibility (n p q : ℕ+) : 
  (n.val ^ 3 + p.val) % (n.val + q.val) = 0 → 
  n.val ≤ 3060 ∧ 
  (n.val = 3060 → p.val = 300 ∧ q.val = 15) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_divisibility_l1463_146313


namespace NUMINAMATH_CALUDE_all_triangles_isosceles_l1463_146345

-- Define a point on the grid
structure GridPoint where
  x : Int
  y : Int

-- Define a triangle on the grid
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : GridPoint) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : GridTriangle) : Prop :=
  let d12 := squaredDistance t.p1 t.p2
  let d13 := squaredDistance t.p1 t.p3
  let d23 := squaredDistance t.p2 t.p3
  d12 = d13 ∨ d12 = d23 ∨ d13 = d23

-- Define the four triangles
def triangle1 : GridTriangle := ⟨⟨2, 2⟩, ⟨5, 2⟩, ⟨2, 5⟩⟩
def triangle2 : GridTriangle := ⟨⟨1, 1⟩, ⟨4, 1⟩, ⟨1, 4⟩⟩
def triangle3 : GridTriangle := ⟨⟨3, 3⟩, ⟨6, 3⟩, ⟨6, 6⟩⟩
def triangle4 : GridTriangle := ⟨⟨0, 0⟩, ⟨3, 0⟩, ⟨3, 3⟩⟩

-- Theorem: All four triangles are isosceles
theorem all_triangles_isosceles :
  isIsosceles triangle1 ∧
  isIsosceles triangle2 ∧
  isIsosceles triangle3 ∧
  isIsosceles triangle4 := by
  sorry

end NUMINAMATH_CALUDE_all_triangles_isosceles_l1463_146345


namespace NUMINAMATH_CALUDE_linear_function_through_origin_l1463_146321

/-- A linear function y = nx + (n^2 - 7) passing through (0, 2) with negative slope has n = -3 -/
theorem linear_function_through_origin (n : ℝ) : 
  (2 = n^2 - 7) →  -- The graph passes through (0, 2)
  (n < 0) →        -- y decreases as x increases (negative slope)
  n = -3 := by
sorry

end NUMINAMATH_CALUDE_linear_function_through_origin_l1463_146321


namespace NUMINAMATH_CALUDE_village_population_equality_l1463_146381

theorem village_population_equality (t : ℝ) (G : ℝ) : ¬(t > 0 ∧ 
  78000 - 1200 * t = 42000 + 800 * t ∧
  78000 - 1200 * t = 65000 + G * t ∧
  42000 + 800 * t = 65000 + G * t) :=
sorry

end NUMINAMATH_CALUDE_village_population_equality_l1463_146381


namespace NUMINAMATH_CALUDE_chris_fishing_trips_l1463_146379

theorem chris_fishing_trips (brian_trips : ℕ) (chris_trips : ℕ) (brian_fish_per_trip : ℕ) (total_fish : ℕ) :
  brian_trips = 2 * chris_trips →
  brian_fish_per_trip = 400 →
  total_fish = 13600 →
  brian_fish_per_trip * brian_trips + (chris_trips * (brian_fish_per_trip * 7 / 5)) = total_fish →
  chris_trips = 10 := by
sorry

end NUMINAMATH_CALUDE_chris_fishing_trips_l1463_146379
