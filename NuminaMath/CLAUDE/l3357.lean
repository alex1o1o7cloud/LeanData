import Mathlib

namespace NUMINAMATH_CALUDE_largest_n_for_quadratic_equation_l3357_335704

theorem largest_n_for_quadratic_equation : 
  (∃ (n : ℕ), ∀ (m : ℕ), 
    (∃ (x y z : ℤ), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) ∧
    (m > n → ¬∃ (x y z : ℤ), m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12)) ∧
  (∃ (x y z : ℤ), 10^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_quadratic_equation_l3357_335704


namespace NUMINAMATH_CALUDE_complete_square_sum_l3357_335709

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 + 6*x - 9 = 0 ↔ (x + b)^2 = c) → 
  b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l3357_335709


namespace NUMINAMATH_CALUDE_waist_size_conversion_l3357_335784

/-- Converts inches to centimeters given the conversion rates and waist size --/
def inches_to_cm (inches_per_foot : ℚ) (cm_per_foot : ℚ) (waist_inches : ℚ) : ℚ :=
  (waist_inches / inches_per_foot) * cm_per_foot

/-- Theorem: Given the conversion rates and waist size, proves that 40 inches equals 100 cm --/
theorem waist_size_conversion :
  let inches_per_foot : ℚ := 10
  let cm_per_foot : ℚ := 25
  let waist_inches : ℚ := 40
  inches_to_cm inches_per_foot cm_per_foot waist_inches = 100 := by
  sorry

end NUMINAMATH_CALUDE_waist_size_conversion_l3357_335784


namespace NUMINAMATH_CALUDE_intersection_range_l3357_335726

def set_A (a : ℝ) : Set ℝ := {x | |x - a| < 1}
def set_B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem intersection_range (a : ℝ) : (set_A a ∩ set_B).Nonempty → 0 < a ∧ a < 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_range_l3357_335726


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3357_335785

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℝ), x ≠ 1 → x ≠ 4 → x ≠ -2 →
      (x^3 - x - 4) / ((x - 1) * (x - 4) * (x + 2)) =
      A / (x - 1) + B / (x - 4) + C / (x + 2) ∧
      A = 4/9 ∧ B = 28/9 ∧ C = -1/3 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3357_335785


namespace NUMINAMATH_CALUDE_star_pattern_identifiable_and_separable_l3357_335710

/-- Represents a patch in the tablecloth -/
structure Patch :=
  (shape : Type)
  (material : Type)

/-- Represents the tablecloth -/
structure Tablecloth :=
  (patches : Set Patch)
  (isTriangular : ∀ p ∈ patches, p.shape = Triangle)
  (isSilk : ∀ p ∈ patches, p.material = Silk)

/-- Represents a star pattern -/
structure StarPattern :=
  (patches : Set Patch)
  (isSymmetrical : Bool)
  (fitsWithRest : Tablecloth → Bool)

/-- Theorem: If a symmetrical star pattern exists in the tablecloth, it can be identified and separated -/
theorem star_pattern_identifiable_and_separable 
  (tc : Tablecloth) 
  (sp : StarPattern) 
  (h1 : sp.patches ⊆ tc.patches) 
  (h2 : sp.isSymmetrical = true) 
  (h3 : sp.fitsWithRest tc = true) : 
  ∃ (identified_sp : StarPattern), identified_sp = sp ∧ 
  ∃ (separated_tc : Tablecloth), separated_tc.patches = tc.patches \ sp.patches :=
sorry


end NUMINAMATH_CALUDE_star_pattern_identifiable_and_separable_l3357_335710


namespace NUMINAMATH_CALUDE_operation_b_correct_operation_c_correct_l3357_335741

-- Operation B
theorem operation_b_correct (t : ℝ) : (-2 * t) * (3 * t + t^2 - 1) = -6 * t^2 - 2 * t^3 + 2 * t := by
  sorry

-- Operation C
theorem operation_c_correct (x y : ℝ) : (-2 * x * y^3)^2 = 4 * x^2 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_operation_b_correct_operation_c_correct_l3357_335741


namespace NUMINAMATH_CALUDE_octopus_leg_counts_l3357_335758

/-- Represents the possible number of legs an octopus can have -/
inductive LegCount
  | six
  | seven
  | eight

/-- Represents an octopus with a name and a number of legs -/
structure Octopus :=
  (name : String)
  (legs : LegCount)

/-- Determines if an octopus is telling the truth based on its leg count -/
def isTruthful (o : Octopus) : Bool :=
  match o.legs with
  | LegCount.seven => false
  | _ => true

/-- Converts LegCount to a natural number -/
def legCountToNat (lc : LegCount) : Nat :=
  match lc with
  | LegCount.six => 6
  | LegCount.seven => 7
  | LegCount.eight => 8

/-- The main theorem about the octopuses' leg counts -/
theorem octopus_leg_counts (blue green red yellow : Octopus)
  (h1 : blue.name = "Blue" ∧ green.name = "Green" ∧ red.name = "Red" ∧ yellow.name = "Yellow")
  (h2 : (isTruthful blue) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 25))
  (h3 : (isTruthful green) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 26))
  (h4 : (isTruthful red) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 27))
  (h5 : (isTruthful yellow) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 28)) :
  blue.legs = LegCount.seven ∧ green.legs = LegCount.seven ∧ red.legs = LegCount.six ∧ yellow.legs = LegCount.seven :=
sorry

end NUMINAMATH_CALUDE_octopus_leg_counts_l3357_335758


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_one_l3357_335778

theorem smallest_number_with_remainder_one : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 7 = 1 ∧ 
  n % 11 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 7 = 1 ∧ m % 11 = 1 → m ≥ n) ∧ 
  n = 78 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_one_l3357_335778


namespace NUMINAMATH_CALUDE_ball_count_theorem_l3357_335748

theorem ball_count_theorem (total : ℕ) (red_freq black_freq : ℚ) :
  total = 120 →
  red_freq = 15 / 100 →
  black_freq = 45 / 100 →
  ∃ (red black white : ℕ),
    red = (total : ℚ) * red_freq ∧
    black = (total : ℚ) * black_freq ∧
    white = total - red - black ∧
    white = 48 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l3357_335748


namespace NUMINAMATH_CALUDE_people_per_tent_l3357_335793

theorem people_per_tent 
  (total_people : ℕ) 
  (house_capacity : ℕ) 
  (num_tents : ℕ) 
  (h1 : total_people = 14) 
  (h2 : house_capacity = 4) 
  (h3 : num_tents = 5) :
  (total_people - house_capacity) / num_tents = 2 :=
by sorry

end NUMINAMATH_CALUDE_people_per_tent_l3357_335793


namespace NUMINAMATH_CALUDE_parking_lot_wheel_count_l3357_335725

def parking_lot_wheels (num_cars : ℕ) (num_bikes : ℕ) (wheels_per_car : ℕ) (wheels_per_bike : ℕ) : ℕ :=
  num_cars * wheels_per_car + num_bikes * wheels_per_bike

theorem parking_lot_wheel_count : parking_lot_wheels 14 10 4 2 = 76 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheel_count_l3357_335725


namespace NUMINAMATH_CALUDE_count_twos_in_hotel_l3357_335727

/-- Represents a hotel room number -/
structure RoomNumber where
  floor : Nat
  room : Nat
  h1 : 1 ≤ floor ∧ floor ≤ 5
  h2 : 1 ≤ room ∧ room ≤ 35

/-- Counts occurrences of a digit in a natural number -/
def countDigit (digit : Nat) (n : Nat) : Nat :=
  sorry

/-- All room numbers in the hotel -/
def allRoomNumbers : List RoomNumber :=
  sorry

/-- Counts occurrences of digit 2 in all room numbers -/
def countTwos : Nat :=
  sorry

theorem count_twos_in_hotel : countTwos = 105 := by
  sorry

end NUMINAMATH_CALUDE_count_twos_in_hotel_l3357_335727


namespace NUMINAMATH_CALUDE_light_glow_start_time_l3357_335736

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Calculates the difference between two times in seconds -/
def timeDiffInSeconds (t1 t2 : Time) : Nat :=
  (t1.hours * 3600 + t1.minutes * 60 + t1.seconds) -
  (t2.hours * 3600 + t2.minutes * 60 + t2.seconds)

theorem light_glow_start_time 
  (glow_interval : Nat) 
  (glow_count : Nat) 
  (end_time : Time) 
  (start_time : Time) : 
  glow_interval = 17 →
  glow_count = 292 →
  end_time = { hours := 3, minutes := 20, seconds := 47 } →
  start_time = { hours := 1, minutes := 58, seconds := 3 } →
  timeDiffInSeconds end_time start_time = glow_interval * glow_count :=
by sorry

end NUMINAMATH_CALUDE_light_glow_start_time_l3357_335736


namespace NUMINAMATH_CALUDE_peters_pizza_fraction_l3357_335787

-- Define the number of slices in the pizza
def total_slices : ℕ := 16

-- Define the number of whole slices Peter ate
def whole_slices_eaten : ℕ := 1

-- Define the number of slices shared
def shared_slices : ℕ := 2

-- Theorem statement
theorem peters_pizza_fraction :
  (whole_slices_eaten : ℚ) / total_slices + 
  (shared_slices : ℚ) / total_slices / 2 = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_peters_pizza_fraction_l3357_335787


namespace NUMINAMATH_CALUDE_remainder_mod_48_l3357_335753

theorem remainder_mod_48 (x : ℤ) 
  (h1 : (2 + x) % (2^3) = 2^3 % (2^3))
  (h2 : (4 + x) % (4^3) = 4^2 % (4^3))
  (h3 : (6 + x) % (6^3) = 6^2 % (6^3)) :
  x % 48 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_mod_48_l3357_335753


namespace NUMINAMATH_CALUDE_train_speed_l3357_335732

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 1500) (h2 : time = 50) :
  length / time = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3357_335732


namespace NUMINAMATH_CALUDE_inequality_proof_l3357_335764

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / (1 + b + c) + b / (1 + c + a) + c / (1 + a + b) ≥ 
       a * b / (1 + a + b) + b * c / (1 + b + c) + c * a / (1 + c + a)) :
  (a^2 + b^2 + c^2) / (a * b + b * c + c * a) + a + b + c + 2 ≥ 
  2 * (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3357_335764


namespace NUMINAMATH_CALUDE_min_value_polynomial_l3357_335783

theorem min_value_polynomial (x : ℝ) : 
  (∃ (m : ℝ), ∀ (x : ℝ), (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 ≥ m) ∧ 
  (∃ (x : ℝ), (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 = 2022) := by
  sorry

end NUMINAMATH_CALUDE_min_value_polynomial_l3357_335783


namespace NUMINAMATH_CALUDE_bananas_per_truck_l3357_335743

-- Define the given quantities
def total_apples : ℝ := 132.6
def apples_per_truck : ℝ := 13.26
def total_bananas : ℝ := 6.4

-- Define the theorem
theorem bananas_per_truck :
  (total_bananas / (total_apples / apples_per_truck)) = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_truck_l3357_335743


namespace NUMINAMATH_CALUDE_area_bounds_l3357_335771

/-- An acute triangle with sides a, b, c and area t, satisfying abc = a + b + c -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  t : ℝ
  acute : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  area_condition : t > 0
  side_condition : a * b * c = a + b + c

/-- The area of an acute triangle satisfying the given conditions is bounded -/
theorem area_bounds (triangle : AcuteTriangle) : 1 < triangle.t ∧ triangle.t ≤ (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_bounds_l3357_335771


namespace NUMINAMATH_CALUDE_fifth_friend_payment_l3357_335767

/-- Represents the payment made by each friend -/
structure Payment where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ
  fifth : ℝ

/-- Conditions for the gift payment problem -/
def GiftPaymentConditions (p : Payment) : Prop :=
  p.first + p.second + p.third + p.fourth + p.fifth = 120 ∧
  p.first = (1/3) * (p.second + p.third + p.fourth + p.fifth) ∧
  p.second = (1/4) * (p.first + p.third + p.fourth + p.fifth) ∧
  p.third = (1/5) * (p.first + p.second + p.fourth + p.fifth)

/-- Theorem stating that under the given conditions, the fifth friend paid $40 -/
theorem fifth_friend_payment (p : Payment) : 
  GiftPaymentConditions p → p.fifth = 40 := by
  sorry

end NUMINAMATH_CALUDE_fifth_friend_payment_l3357_335767


namespace NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l3357_335750

/-- Represents the number of crates of eggs sold on each day of the week --/
structure EggSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Theorem stating the ratio of Tuesday's sales to Monday's sales --/
theorem tuesday_to_monday_ratio (sales : EggSales) : 
  sales.monday = 5 ∧ 
  sales.wednesday = sales.tuesday - 2 ∧ 
  sales.thursday = sales.tuesday / 2 ∧
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 28 →
  sales.tuesday = 2 * sales.monday := by
  sorry

#check tuesday_to_monday_ratio

end NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l3357_335750


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l3357_335708

-- Define the given points and circles
def M : ℝ × ℝ := (2, -2)
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 3
def circle_2 (x y : ℝ) : Prop := x^2 + y^2 + 3*x = 0

-- Define the resulting circle and line
def result_circle (x y : ℝ) : Prop := 3*x^2 + 3*y^2 - 5*x - 14 = 0
def line_AB (x y : ℝ) : Prop := 2*x - 2*y = 3

theorem circle_and_line_theorem :
  -- (1) The result_circle passes through M and intersects with circle_O and circle_2
  (result_circle M.1 M.2) ∧
  (∃ x y : ℝ, result_circle x y ∧ circle_O x y) ∧
  (∃ x y : ℝ, result_circle x y ∧ circle_2 x y) ∧
  -- (2) line_AB is tangent to circle_O at two points
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧
    (∀ x y : ℝ, line_AB x y → circle_O x y → (x, y) = A ∨ (x, y) = B)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_theorem_l3357_335708


namespace NUMINAMATH_CALUDE_classrooms_needed_l3357_335777

def total_students : ℕ := 1675
def students_per_classroom : ℕ := 37

theorem classrooms_needed : 
  ∃ (n : ℕ), n * students_per_classroom ≥ total_students ∧ 
  ∀ (m : ℕ), m * students_per_classroom ≥ total_students → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_classrooms_needed_l3357_335777


namespace NUMINAMATH_CALUDE_triangle_inequality_l3357_335781

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3357_335781


namespace NUMINAMATH_CALUDE_parallelogram_is_rhombus_l3357_335734

/-- A parallelogram ABCD in a 2D Euclidean space. -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Vector addition -/
def vecAdd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

/-- Vector subtraction -/
def vecSub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

/-- Dot product of two vectors -/
def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The zero vector -/
def zeroVec : ℝ × ℝ := (0, 0)

/-- Theorem: A parallelogram is a rhombus if it satisfies certain vector conditions -/
theorem parallelogram_is_rhombus (ABCD : Parallelogram)
  (h1 : vecAdd (vecSub ABCD.B ABCD.A) (vecSub ABCD.D ABCD.C) = zeroVec)
  (h2 : dotProduct (vecSub (vecSub ABCD.B ABCD.A) (vecSub ABCD.D ABCD.A)) (vecSub ABCD.C ABCD.A) = 0) :
  ABCD.A = ABCD.B ∧ ABCD.B = ABCD.C ∧ ABCD.C = ABCD.D ∧ ABCD.D = ABCD.A := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_is_rhombus_l3357_335734


namespace NUMINAMATH_CALUDE_four_drivers_sufficient_l3357_335738

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_minutes : minutes < 60

/-- Represents a driver -/
inductive Driver
| A | B | C | D

/-- Represents a trip -/
structure Trip where
  driver : Driver
  departure : Time
  arrival : Time

def one_way_duration : Time := ⟨2, 40, sorry⟩
def round_trip_duration : Time := ⟨5, 20, sorry⟩
def min_rest_duration : Time := ⟨1, 0, sorry⟩

def driver_A_return : Time := ⟨12, 40, sorry⟩
def driver_D_departure : Time := ⟨13, 5, sorry⟩
def driver_B_return : Time := ⟨16, 0, sorry⟩
def driver_A_fifth_departure : Time := ⟨16, 10, sorry⟩
def driver_B_sixth_departure : Time := ⟨17, 30, sorry⟩
def alexey_return : Time := ⟨21, 30, sorry⟩

def is_valid_schedule (trips : List Trip) : Prop :=
  sorry

theorem four_drivers_sufficient :
  ∃ (trips : List Trip),
    trips.length ≥ 6 ∧
    is_valid_schedule trips ∧
    (∃ (last_trip : Trip),
      last_trip ∈ trips ∧
      last_trip.driver = Driver.A ∧
      last_trip.departure = driver_A_fifth_departure ∧
      last_trip.arrival = alexey_return) ∧
    (∀ (trip : Trip), trip ∈ trips → trip.driver ∈ [Driver.A, Driver.B, Driver.C, Driver.D]) :=
  sorry

end NUMINAMATH_CALUDE_four_drivers_sufficient_l3357_335738


namespace NUMINAMATH_CALUDE_problem_statement_l3357_335766

theorem problem_statement (a b : ℝ) (ha : a > 0) (heq : Real.exp a + Real.log b = 1) :
  (a + Real.log b < 0) ∧ (Real.exp a + b > 2) ∧ (a + b > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3357_335766


namespace NUMINAMATH_CALUDE_infinitely_many_disconnected_pLandia_l3357_335740

/-- A function that determines if two islands are connected in p-Landia -/
def isConnected (p n m : ℕ) : Prop :=
  p ∣ (n^2 - m + 1) * (m^2 - n + 1)

/-- The graph representation of p-Landia -/
def pLandiaGraph (p : ℕ) : SimpleGraph ℕ :=
  SimpleGraph.fromRel (λ n m ↦ n ≠ m ∧ isConnected p n m)

/-- The theorem stating that infinitely many p-Landia graphs are disconnected -/
theorem infinitely_many_disconnected_pLandia :
  ∃ (S : Set ℕ), (∀ p ∈ S, Nat.Prime p) ∧ Set.Infinite S ∧
    ∀ p ∈ S, ¬(pLandiaGraph p).Connected :=
  sorry

end NUMINAMATH_CALUDE_infinitely_many_disconnected_pLandia_l3357_335740


namespace NUMINAMATH_CALUDE_multiply_by_9999_l3357_335703

theorem multiply_by_9999 : ∃! x : ℤ, x * 9999 = 806006795 :=
  by sorry

end NUMINAMATH_CALUDE_multiply_by_9999_l3357_335703


namespace NUMINAMATH_CALUDE_triangle_side_length_l3357_335752

/-- Given a triangle ABC where angle A is 6 degrees, angle C is 75 degrees, 
    and side BC has length √3, prove that the length of side AC 
    is equal to (√3 * sin 6°) / sin 45° -/
theorem triangle_side_length (A B C : ℝ) (AC BC : ℝ) : 
  A = 6 * π / 180 →  -- Convert 6° to radians
  C = 75 * π / 180 →  -- Convert 75° to radians
  BC = Real.sqrt 3 →
  AC = (Real.sqrt 3 * Real.sin (6 * π / 180)) / Real.sin (45 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3357_335752


namespace NUMINAMATH_CALUDE_least_common_denominator_l3357_335723

theorem least_common_denominator : 
  Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 8))))) = 840 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l3357_335723


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_l3357_335763

/-- The range of k for which the intersection of two lines lies in the first quadrant -/
theorem intersection_in_first_quadrant (k : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x - 1 ∧ x + y - 1 = 0) ↔ k > 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_first_quadrant_l3357_335763


namespace NUMINAMATH_CALUDE_trig_expression_equals_three_l3357_335730

theorem trig_expression_equals_three :
  let sin_60 : ℝ := Real.sqrt 3 / 2
  let tan_45 : ℝ := 1
  let tan_60 : ℝ := Real.sqrt 3
  ∀ (sin_25 cos_25 : ℝ), 
    sin_25^2 + cos_25^2 = 1 →
    sin_25^2 + 2 * sin_60 + tan_45 - tan_60 + cos_25^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_three_l3357_335730


namespace NUMINAMATH_CALUDE_sally_cards_bought_l3357_335769

def cards_bought (initial : ℕ) (received : ℕ) (final : ℕ) : ℕ :=
  final - (initial + received)

theorem sally_cards_bought :
  cards_bought 27 41 88 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sally_cards_bought_l3357_335769


namespace NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l3357_335720

/-- Proves that mixing solutions A and B results in a solution with approximately 1.44% liquid X -/
theorem liquid_x_percentage_in_mixed_solution :
  let solution_a_weight : ℝ := 400
  let solution_b_weight : ℝ := 700
  let liquid_x_percent_a : ℝ := 0.8
  let liquid_x_percent_b : ℝ := 1.8
  let total_weight := solution_a_weight + solution_b_weight
  let liquid_x_weight_a := solution_a_weight * (liquid_x_percent_a / 100)
  let liquid_x_weight_b := solution_b_weight * (liquid_x_percent_b / 100)
  let total_liquid_x_weight := liquid_x_weight_a + liquid_x_weight_b
  let result_percent := (total_liquid_x_weight / total_weight) * 100
  ∃ ε > 0, |result_percent - 1.44| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l3357_335720


namespace NUMINAMATH_CALUDE_union_equals_A_l3357_335746

def A : Set ℝ := {x | x^2 + x - 2 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem union_equals_A (m : ℝ) : A ∪ B m = A → m = 0 ∨ m = -1 ∨ m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l3357_335746


namespace NUMINAMATH_CALUDE_hyperbola_equivalence_l3357_335706

-- Define the equation
def hyperbola_eq (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + y^2) - Real.sqrt ((x + 3)^2 + y^2) = 4

-- Define the standard form of the hyperbola
def hyperbola_standard_form (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1 ∧ x ≤ -2

-- Theorem stating the equivalence
theorem hyperbola_equivalence :
  ∀ x y : ℝ, hyperbola_eq x y ↔ hyperbola_standard_form x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equivalence_l3357_335706


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3357_335775

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ↔ C ≤ Real.sqrt (4/3) := by
  sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3357_335775


namespace NUMINAMATH_CALUDE_ellipse_condition_ellipse_condition_converse_l3357_335742

/-- Represents a point in a 2D rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation m(x^2 + y^2 + 2y + 1) = (x - 2y + 3)^2 -/
def ellipseEquation (m : ℝ) (p : Point) : Prop :=
  m * (p.x^2 + p.y^2 + 2*p.y + 1) = (p.x - 2*p.y + 3)^2

/-- Defines what it means for the equation to represent an ellipse -/
def isEllipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    ∀ (p : Point), ellipseEquation m p ↔ 
      (p.x^2 / a^2) + (p.y^2 / b^2) = 1

/-- The main theorem: if the equation represents an ellipse, then m > 5 -/
theorem ellipse_condition (m : ℝ) :
  isEllipse m → m > 5 := by
  sorry

/-- The converse: if m > 5, then the equation represents an ellipse -/
theorem ellipse_condition_converse (m : ℝ) :
  m > 5 → isEllipse m := by
  sorry

end NUMINAMATH_CALUDE_ellipse_condition_ellipse_condition_converse_l3357_335742


namespace NUMINAMATH_CALUDE_min_cos_B_angle_A_values_l3357_335776

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a + t.c = 3 * Real.sqrt 3 ∧ t.b = 3

/-- The minimum value of cos B -/
theorem min_cos_B (t : Triangle) (h : TriangleConditions t) :
    (∀ t' : Triangle, TriangleConditions t' → Real.cos t'.B ≥ Real.cos t.B) →
    Real.cos t.B = 1/3 := by sorry

/-- The possible values of angle A when BA · BC = 3 -/
theorem angle_A_values (t : Triangle) (h : TriangleConditions t) :
    t.a * t.c * Real.cos t.B = 3 →
    t.A = Real.pi / 2 ∨ t.A = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_min_cos_B_angle_A_values_l3357_335776


namespace NUMINAMATH_CALUDE_interior_alternate_angles_equal_implies_parallel_l3357_335713

/-- Two lines in a plane -/
structure Line

/-- A transversal line cutting two other lines -/
structure Transversal

/-- An angle formed by the intersection of lines -/
structure Angle

/-- Defines the concept of interior alternate angles -/
def interior_alternate_angles (l1 l2 : Line) (t : Transversal) (a1 a2 : Angle) : Prop :=
  sorry

/-- Defines parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  sorry

/-- The main theorem: if interior alternate angles are equal, then the lines are parallel -/
theorem interior_alternate_angles_equal_implies_parallel 
  (l1 l2 : Line) (t : Transversal) (a1 a2 : Angle) :
  interior_alternate_angles l1 l2 t a1 a2 → a1 = a2 → parallel l1 l2 :=
sorry

end NUMINAMATH_CALUDE_interior_alternate_angles_equal_implies_parallel_l3357_335713


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3357_335735

/-- Two quantities vary inversely if their product is constant -/
def vary_inversely (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a x * b x = k

theorem inverse_variation_problem (a b : ℝ → ℝ) 
  (h_inverse : vary_inversely a b)
  (h_initial : b 800 = 0.5) :
  b 3200 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3357_335735


namespace NUMINAMATH_CALUDE_scientific_notation_of_2590000_l3357_335757

theorem scientific_notation_of_2590000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2590000 = a * (10 : ℝ) ^ n ∧ a = 2.59 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2590000_l3357_335757


namespace NUMINAMATH_CALUDE_cos_alpha_for_given_point_l3357_335792

/-- If the terminal side of angle α passes through the point (√3/2, 1/2), then cos α = √3/2 -/
theorem cos_alpha_for_given_point (α : Real) :
  (∃ (r : Real), r * (Real.sqrt 3 / 2) = Real.cos α ∧ r * (1 / 2) = Real.sin α) →
  Real.cos α = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_given_point_l3357_335792


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l3357_335702

/-- Defines an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Defines a line -/
structure Line where
  k : ℝ
  m : ℝ

/-- Theorem statement -/
theorem ellipse_fixed_point_intersection 
  (E : Ellipse) 
  (h_point : E.a^2 + (3/2)^2 / E.b^2 = 1) 
  (h_ecc : (E.a^2 - E.b^2) / E.a^2 = 1/4) 
  (l : Line) 
  (h_intersect : ∃ (M N : ℝ × ℝ), M ≠ N ∧ 
    M.1^2 / E.a^2 + M.2^2 / E.b^2 = 1 ∧
    N.1^2 / E.a^2 + N.2^2 / E.b^2 = 1 ∧
    M.2 = l.k * M.1 + l.m ∧
    N.2 = l.k * N.1 + l.m)
  (h_perp : ∀ (M N : ℝ × ℝ), 
    M.1^2 / E.a^2 + M.2^2 / E.b^2 = 1 →
    N.1^2 / E.a^2 + N.2^2 / E.b^2 = 1 →
    M.2 = l.k * M.1 + l.m →
    N.2 = l.k * N.1 + l.m →
    (M.1 - E.a) * (N.1 - E.a) + M.2 * N.2 = 0) :
  l.k * (2/7) + l.m = 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l3357_335702


namespace NUMINAMATH_CALUDE_figure_area_l3357_335728

/-- The total area of a figure composed of five rectangles -/
def total_area (a b c d e f g h i j : ℕ) : ℕ :=
  a * b + c * d + e * f + g * h + i * j

theorem figure_area : 
  total_area 7 4 5 4 7 3 5 2 3 1 = 82 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_l3357_335728


namespace NUMINAMATH_CALUDE_prism_volume_l3357_335712

/-- A right rectangular prism with given face areas has the specified volume -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3357_335712


namespace NUMINAMATH_CALUDE_cube_sum_over_product_equals_thirteen_l3357_335774

theorem cube_sum_over_product_equals_thirteen
  (a b c : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (sum_equals_ten : a + b + c = 10)
  (squared_diff_sum : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 13 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_equals_thirteen_l3357_335774


namespace NUMINAMATH_CALUDE_f_bijective_iff_power_of_two_l3357_335722

/-- The set of all possible lamp configurations for n lamps -/
def Ψ (n : ℕ) := Fin (2^n)

/-- The cool procedure function -/
def f (n : ℕ) : Ψ n → Ψ n := sorry

/-- Theorem stating that f is bijective if and only if n is a power of 2 -/
theorem f_bijective_iff_power_of_two (n : ℕ) :
  Function.Bijective (f n) ↔ ∃ k : ℕ, n = 2^k := by sorry

end NUMINAMATH_CALUDE_f_bijective_iff_power_of_two_l3357_335722


namespace NUMINAMATH_CALUDE_f_properties_l3357_335779

-- Define the function f(x) = sin(1/x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (1 / x)

-- State the theorem
theorem f_properties :
  -- The range of f(x) is [-1, 1]
  (∀ y ∈ Set.range f, -1 ≤ y ∧ y ≤ 1) ∧
  (∀ y : ℝ, -1 ≤ y ∧ y ≤ 1 → ∃ x ≠ 0, f x = y) ∧
  -- f(x) is monotonically decreasing on [2/π, +∞)
  (∀ x₁ x₂ : ℝ, x₁ ≥ 2/Real.pi ∧ x₂ ≥ 2/Real.pi ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  -- For any m ∈ [-1, 1], f(x) = m has infinitely many solutions in (0, 1)
  (∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → ∃ (S : Set ℝ), S.Infinite ∧ S ⊆ Set.Ioo 0 1 ∧ ∀ x ∈ S, f x = m) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3357_335779


namespace NUMINAMATH_CALUDE_unique_integral_solution_l3357_335719

theorem unique_integral_solution :
  ∃! (x y z : ℕ), 
    (z^x = y^(3*x)) ∧ 
    (2^z = 4 * 8^x) ∧ 
    (x + y + z = 20) ∧
    x = 2 ∧ y = 2 ∧ z = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_integral_solution_l3357_335719


namespace NUMINAMATH_CALUDE_circle_properties_l3357_335749

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Theorem statement
theorem circle_properties :
  -- The center is on the y-axis
  ∃ b : ℝ, circle_equation 0 b ∧
  -- The radius is 1
  (∀ x y : ℝ, circle_equation x y → (x^2 + (y - 2)^2 = 1)) ∧
  -- The circle passes through (1, 2)
  circle_equation 1 2 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3357_335749


namespace NUMINAMATH_CALUDE_sqrt_81_equals_9_l3357_335739

theorem sqrt_81_equals_9 : Real.sqrt 81 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_equals_9_l3357_335739


namespace NUMINAMATH_CALUDE_stratified_sampling_boys_l3357_335754

theorem stratified_sampling_boys (total_boys : ℕ) (total_girls : ℕ) (sample_size : ℕ) :
  total_boys = 48 →
  total_girls = 36 →
  sample_size = 21 →
  (total_boys * sample_size) / (total_boys + total_girls) = 12 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_boys_l3357_335754


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l3357_335797

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 12 → (1 / x) = 3 * (1 / y) → x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l3357_335797


namespace NUMINAMATH_CALUDE_roof_collapse_days_l3357_335770

theorem roof_collapse_days (roof_weight_limit : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) : 
  roof_weight_limit = 500 → 
  leaves_per_day = 100 → 
  leaves_per_pound = 1000 → 
  (roof_weight_limit * leaves_per_pound) / leaves_per_day = 5000 := by
  sorry

#check roof_collapse_days

end NUMINAMATH_CALUDE_roof_collapse_days_l3357_335770


namespace NUMINAMATH_CALUDE_total_investment_amount_l3357_335755

/-- Prove that the total investment is $8000 given the specified conditions --/
theorem total_investment_amount (total_income : ℝ) (rate1 rate2 : ℝ) (investment1 : ℝ) :
  total_income = 575 →
  rate1 = 0.085 →
  rate2 = 0.064 →
  investment1 = 3000 →
  ∃ (investment2 : ℝ),
    total_income = investment1 * rate1 + investment2 * rate2 ∧
    investment1 + investment2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_total_investment_amount_l3357_335755


namespace NUMINAMATH_CALUDE_inequality_proof_l3357_335751

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3357_335751


namespace NUMINAMATH_CALUDE_min_value_expression_l3357_335733

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x^2 / (x + 2)) + (y^2 / (y + 1)) ≥ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3357_335733


namespace NUMINAMATH_CALUDE_computer_price_after_15_years_l3357_335788

/-- The price of a computer after a certain number of 5-year periods, given an initial price and a price decrease rate. -/
def computer_price (initial_price : ℝ) (decrease_rate : ℝ) (periods : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ periods

/-- Theorem stating that a computer with an initial price of 8100 yuan and a price decrease of 1/3 every 5 years will cost 2400 yuan after 15 years. -/
theorem computer_price_after_15_years :
  computer_price 8100 (1/3) 3 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_after_15_years_l3357_335788


namespace NUMINAMATH_CALUDE_intersection_A_B_l3357_335717

-- Define sets A and B
def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | (x - 1) / (x - 4) < 0}

-- State the theorem
theorem intersection_A_B :
  ∀ x : ℝ, x ∈ A ∩ B ↔ 3 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3357_335717


namespace NUMINAMATH_CALUDE_least_integer_with_2023_divisors_l3357_335765

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Check if n is divisible by d -/
def is_divisible_by (n d : ℕ) : Prop := sorry

theorem least_integer_with_2023_divisors :
  ∃ (m k : ℕ),
    (num_divisors (m * 6^k) = 2023) ∧
    (¬ is_divisible_by m 6) ∧
    (∀ n : ℕ, num_divisors n = 2023 → n ≥ m * 6^k) ∧
    m = 9216 ∧
    k = 6 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_2023_divisors_l3357_335765


namespace NUMINAMATH_CALUDE_smallest_divisible_k_l3357_335789

/-- The polynomial p(z) = z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1 -/
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

/-- The function f(k) = z^k - 1 -/
def f (k : ℕ) (z : ℂ) : ℂ := z^k - 1

/-- Theorem: The smallest positive integer k such that p(z) divides f(k)(z) is 112 -/
theorem smallest_divisible_k : (∀ z : ℂ, p z ∣ f 112 z) ∧
  (∀ k : ℕ, k < 112 → ∃ z : ℂ, ¬(p z ∣ f k z)) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_k_l3357_335789


namespace NUMINAMATH_CALUDE_rectangle_combinations_l3357_335773

-- Define the number of horizontal and vertical lines
def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4

-- Define the number of lines needed to form a rectangle
def lines_for_rectangle : ℕ := 2

-- Theorem statement
theorem rectangle_combinations :
  (Nat.choose horizontal_lines lines_for_rectangle) *
  (Nat.choose vertical_lines lines_for_rectangle) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_combinations_l3357_335773


namespace NUMINAMATH_CALUDE_equation_solutions_l3357_335791

theorem equation_solutions (m : ℕ+) :
  let f := fun (x y : ℕ) => x^2 + y^2 + 2*x*y - m*x - m*y - m - 1
  (∃! s : Finset (ℕ × ℕ), s.card = m ∧ 
    ∀ (p : ℕ × ℕ), p ∈ s ↔ (f p.1 p.2 = 0 ∧ p.1 > 0 ∧ p.2 > 0)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3357_335791


namespace NUMINAMATH_CALUDE_regular_hexagon_most_symmetry_l3357_335705

/-- Number of lines of symmetry for a given shape -/
def linesOfSymmetry (shape : String) : ℕ :=
  match shape with
  | "regular pentagon" => 5
  | "parallelogram" => 0
  | "oval" => 2
  | "right triangle" => 0
  | "regular hexagon" => 6
  | _ => 0

/-- The set of shapes we're considering -/
def shapes : List String := ["regular pentagon", "parallelogram", "oval", "right triangle", "regular hexagon"]

theorem regular_hexagon_most_symmetry :
  ∀ s ∈ shapes, linesOfSymmetry "regular hexagon" ≥ linesOfSymmetry s :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_most_symmetry_l3357_335705


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3357_335731

theorem cube_sum_theorem (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p * q + q * r + r * p = 11)
  (prod_eq : p * q * r = -6) :
  p^3 + q^3 + r^3 = -90 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3357_335731


namespace NUMINAMATH_CALUDE_abs_eq_neg_iff_nonpositive_l3357_335782

theorem abs_eq_neg_iff_nonpositive (a : ℝ) : |a| = -a ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_abs_eq_neg_iff_nonpositive_l3357_335782


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3357_335715

-- Problem 1
theorem problem_1 (a b : ℝ) (h : a ≠ b) : 
  (a^2 / (a - b)) - (b^2 / (a - b)) = a + b :=
sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) (h3 : x ≠ 1) : 
  ((x^2 - 1) / (x^2 + 2*x + 1)) / ((x^2 - x) / (x + 1)) = 1 / x :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3357_335715


namespace NUMINAMATH_CALUDE_sons_age_is_24_l3357_335795

/-- Proves that the son's age is 24 given the conditions of the problem -/
theorem sons_age_is_24 (son_age father_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_sons_age_is_24_l3357_335795


namespace NUMINAMATH_CALUDE_integral_equals_sqrt3_over_2_minus_ln2_l3357_335721

noncomputable def integral_function (x : ℝ) : ℝ := (Real.cos x)^2 / (1 + Real.cos x - Real.sin x)^2

theorem integral_equals_sqrt3_over_2_minus_ln2 :
  ∫ x in -((2 * Real.pi) / 3)..0, integral_function x = Real.sqrt 3 / 2 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_sqrt3_over_2_minus_ln2_l3357_335721


namespace NUMINAMATH_CALUDE_beta_max_success_ratio_l3357_335724

theorem beta_max_success_ratio 
  (alpha_day1_score alpha_day1_total : ℕ)
  (alpha_day2_score alpha_day2_total : ℕ)
  (beta_day1_score beta_day1_total : ℕ)
  (beta_day2_score beta_day2_total : ℕ)
  (h1 : alpha_day1_score = 160)
  (h2 : alpha_day1_total = 300)
  (h3 : alpha_day2_score = 140)
  (h4 : alpha_day2_total = 200)
  (h5 : beta_day1_total + beta_day2_total = 500)
  (h6 : beta_day1_total ≠ 300)
  (h7 : beta_day1_score > 0)
  (h8 : beta_day2_score > 0)
  (h9 : (beta_day1_score : ℚ) / beta_day1_total < (alpha_day1_score : ℚ) / alpha_day1_total)
  (h10 : (beta_day2_score : ℚ) / beta_day2_total < (alpha_day2_score : ℚ) / alpha_day2_total)
  (h11 : (alpha_day1_score + alpha_day2_score : ℚ) / (alpha_day1_total + alpha_day2_total) = 3/5) :
  (beta_day1_score + beta_day2_score : ℚ) / (beta_day1_total + beta_day2_total) ≤ 349/500 :=
by sorry

end NUMINAMATH_CALUDE_beta_max_success_ratio_l3357_335724


namespace NUMINAMATH_CALUDE_greatest_t_value_l3357_335700

theorem greatest_t_value (t : ℝ) : 
  (t^2 - t - 56) / (t - 8) = 3 / (t + 5) → t ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_greatest_t_value_l3357_335700


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l3357_335768

theorem increasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inequality : f (a^2 - a) > f (2*a^2 - 4*a)) : 
  0 < a ∧ a < 3 := by
sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l3357_335768


namespace NUMINAMATH_CALUDE_function_property_l3357_335718

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ p q, f (p + q) = f p * f q) 
  (h2 : f 1 = 3) : 
  (f 1^2 + f 2) / f 1 + 
  (f 2^2 + f 4) / f 3 + 
  (f 3^2 + f 6) / f 5 + 
  (f 4^2 + f 8) / f 7 + 
  (f 5^2 + f 10) / f 9 = 30 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3357_335718


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3357_335716

/-- The cost of a candy bar given the conditions -/
theorem candy_bar_cost : 
  ∀ (candy_cost chocolate_cost : ℝ),
  candy_cost + chocolate_cost = 3 →
  candy_cost = chocolate_cost + 3 →
  candy_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l3357_335716


namespace NUMINAMATH_CALUDE_line_length_after_erasing_l3357_335794

-- Define the original length in meters
def original_length_m : ℝ := 1.5

-- Define the erased length in centimeters
def erased_length_cm : ℝ := 37.5

-- Define the conversion factor from meters to centimeters
def m_to_cm : ℝ := 100

-- Theorem statement
theorem line_length_after_erasing :
  (original_length_m * m_to_cm - erased_length_cm) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_line_length_after_erasing_l3357_335794


namespace NUMINAMATH_CALUDE_units_digit_of_150_factorial_l3357_335761

theorem units_digit_of_150_factorial (n : ℕ) : n = 150 → n.factorial % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_150_factorial_l3357_335761


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3357_335747

theorem quadratic_equation_roots (m : ℝ) : m > 0 →
  (∃ (x : ℝ), x^2 + x - m = 0) ∧
  (∃ (x : ℝ), x^2 + x - m = 0) → m > 0 ∨
  m ≤ 0 → ¬(∃ (x : ℝ), x^2 + x - m = 0) ∨
  ¬(∃ (x : ℝ), x^2 + x - m = 0) → m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3357_335747


namespace NUMINAMATH_CALUDE_problem_solution_l3357_335701

theorem problem_solution (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49/(x - 3)^2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3357_335701


namespace NUMINAMATH_CALUDE_astronaut_selection_probability_l3357_335745

theorem astronaut_selection_probability : 
  let total_astronauts : ℕ := 4
  let male_astronauts : ℕ := 2
  let female_astronauts : ℕ := 2
  let selected_astronauts : ℕ := 2

  -- Probability of selecting one male and one female
  (Nat.choose male_astronauts 1 * Nat.choose female_astronauts 1 : ℚ) / 
  (Nat.choose total_astronauts selected_astronauts) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_astronaut_selection_probability_l3357_335745


namespace NUMINAMATH_CALUDE_train_passing_bridge_time_l3357_335762

/-- Calculates the time for a train to pass a bridge -/
theorem train_passing_bridge_time (train_length : Real) (bridge_length : Real) (train_speed_kmh : Real) :
  let total_distance : Real := train_length + bridge_length
  let train_speed_ms : Real := train_speed_kmh * 1000 / 3600
  let time : Real := total_distance / train_speed_ms
  train_length = 200 ∧ bridge_length = 180 ∧ train_speed_kmh = 65 →
  ∃ ε > 0, |time - 21.04| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_passing_bridge_time_l3357_335762


namespace NUMINAMATH_CALUDE_fraction_of_repeating_decimals_l3357_335759

def repeating_decimal_142857 : ℚ := 142857 / 999999
def repeating_decimal_857143 : ℚ := 857143 / 999999

theorem fraction_of_repeating_decimals : 
  (repeating_decimal_142857) / (2 + repeating_decimal_857143) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_repeating_decimals_l3357_335759


namespace NUMINAMATH_CALUDE_unripe_oranges_count_l3357_335780

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges : ℕ := 44

/-- The difference between the number of sacks of ripe and unripe oranges harvested per day -/
def difference : ℕ := 19

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges : ℕ := ripe_oranges - difference

theorem unripe_oranges_count : unripe_oranges = 25 := by
  sorry

end NUMINAMATH_CALUDE_unripe_oranges_count_l3357_335780


namespace NUMINAMATH_CALUDE_square_area_possibilities_l3357_335729

/-- Represents a square in a 2D plane -/
structure Square where
  side_length : ℝ
  area : ℝ := side_length ^ 2

/-- Represents a parallelogram in a 2D plane -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  area : ℝ

/-- Represents an oblique projection from a square to a parallelogram -/
def oblique_projection (s : Square) (p : Parallelogram) : Prop :=
  (s.side_length = p.side1 ∨ s.side_length = p.side2) ∧ p.area = s.area

theorem square_area_possibilities (s : Square) (p : Parallelogram) :
  oblique_projection s p → p.side1 = 4 → s.area = 16 ∨ s.area = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_possibilities_l3357_335729


namespace NUMINAMATH_CALUDE_trajectory_and_line_theorem_l3357_335798

-- Define the circle P
def circle_P (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 36

-- Define point B
def point_B : ℝ × ℝ := (-2, 0)

-- Define the condition that P is on line segment AB
def P_on_AB (A P : ℝ × ℝ) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * point_B.1, t * A.2 + (1 - t) * point_B.2)

-- Define the ratio condition
def ratio_condition (A P : ℝ × ℝ) : Prop :=
  (P.1 - point_B.1)^2 + (P.2 - point_B.2)^2 = 1/4 * ((A.1 - P.1)^2 + (A.2 - P.2)^2)

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2 + y^2 = 4 ∧ x ≠ -2

-- Define line l
def line_l (x y : ℝ) : Prop := 4*x + 3*y - 5 = 0 ∨ x = -1

-- Define the intersection condition
def intersection_condition (M N : ℝ × ℝ) : Prop :=
  trajectory_C M.1 M.2 ∧ trajectory_C N.1 N.2 ∧
  line_l M.1 M.2 ∧ line_l N.1 N.2 ∧
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 12

-- Main theorem
theorem trajectory_and_line_theorem 
  (A P : ℝ × ℝ) 
  (h1 : circle_P A.1 A.2)
  (h2 : P_on_AB A P)
  (h3 : ratio_condition A P)
  (h4 : ∃ M N : ℝ × ℝ, line_l (-1) 3 ∧ intersection_condition M N) :
  trajectory_C P.1 P.2 ∧ line_l (-1) 3 :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_line_theorem_l3357_335798


namespace NUMINAMATH_CALUDE_triangle_theorem_l3357_335707

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.a + t.b - t.c) * (t.a + t.b + t.c) = t.a * t.b)
  (h2 : t.c = 2 * t.a * Real.cos t.B)
  (h3 : t.b = 2) : 
  t.C = 2 * Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3357_335707


namespace NUMINAMATH_CALUDE_certain_amount_calculation_l3357_335772

theorem certain_amount_calculation (amount : ℝ) : 
  (5 / 100) * ((25 / 100) * amount) = 20 → amount = 1600 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_calculation_l3357_335772


namespace NUMINAMATH_CALUDE_program_production_cost_l3357_335799

/-- The cost to produce a program for a college football game. -/
def cost_to_produce : ℝ :=
  sorry

/-- Theorem: Given the conditions, the cost to produce a program is 5500 rupees. -/
theorem program_production_cost :
  let advertisement_revenue : ℝ := 15000
  let copies_sold : ℝ := 35000
  let price_per_copy : ℝ := 0.50
  let desired_profit : ℝ := 8000
  cost_to_produce = advertisement_revenue + (copies_sold * price_per_copy) - (advertisement_revenue + desired_profit) :=
by
  sorry

end NUMINAMATH_CALUDE_program_production_cost_l3357_335799


namespace NUMINAMATH_CALUDE_linear_regression_not_guaranteed_point_l3357_335786

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  dataPoints : List Point

/-- Checks if a point is in the list of data points -/
def isDataPoint (p : Point) (lr : LinearRegression) : Prop :=
  p ∈ lr.dataPoints

/-- Theorem: The linear regression line is not guaranteed to pass through (6.5, 8) -/
theorem linear_regression_not_guaranteed_point (lr : LinearRegression) 
  (h1 : isDataPoint ⟨2, 3⟩ lr)
  (h2 : isDataPoint ⟨5, 7⟩ lr)
  (h3 : isDataPoint ⟨8, 9⟩ lr)
  (h4 : isDataPoint ⟨11, 13⟩ lr) :
  ¬ ∀ (regression_line : Point → Prop), 
    (∀ p, isDataPoint p lr → regression_line p) → 
    regression_line ⟨6.5, 8⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_linear_regression_not_guaranteed_point_l3357_335786


namespace NUMINAMATH_CALUDE_trigonometric_calculation_l3357_335744

theorem trigonometric_calculation :
  let a := |1 - Real.tan (60 * π / 180)| - (-1/2)⁻¹ + Real.sin (45 * π / 180) + Real.sqrt (1/2)
  let b := -1^2022 + Real.sqrt 12 - (π - 3)^0 - Real.cos (30 * π / 180)
  (a = Real.sqrt 3 + Real.sqrt 2 + 1) ∧ (b = -2 + (3/2) * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_trigonometric_calculation_l3357_335744


namespace NUMINAMATH_CALUDE_building_shadow_length_l3357_335790

/-- Given a flagpole and a building under similar conditions, 
    calculate the length of the shadow cast by the building -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : flagpole_shadow = 45)
  (h3 : building_height = 26)
  : ∃ building_shadow : ℝ, building_shadow = 65 := by
  sorry

#check building_shadow_length

end NUMINAMATH_CALUDE_building_shadow_length_l3357_335790


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l3357_335796

theorem geometric_arithmetic_sequence_ratio (x y z : ℝ) 
  (h1 : (4*y)^2 = (3*x)*(5*z))  -- Geometric sequence condition
  (h2 : 2/y = 1/x + 1/z)        -- Arithmetic sequence condition
  : x/z + z/x = 34/15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l3357_335796


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3357_335756

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (1 + 1 / (Real.sqrt 5 + 2)) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3357_335756


namespace NUMINAMATH_CALUDE_shooting_test_probability_l3357_335737

/-- The number of shots in the test -/
def num_shots : ℕ := 3

/-- The minimum number of successful shots required to pass -/
def min_success : ℕ := 2

/-- The probability of making a single shot -/
def shot_probability : ℝ := 0.6

/-- The probability of passing the test -/
def pass_probability : ℝ := 0.648

/-- Theorem stating that the calculated probability of passing the test is correct -/
theorem shooting_test_probability : 
  (Finset.sum (Finset.range (num_shots - min_success + 1))
    (λ k => Nat.choose num_shots (num_shots - k) * 
      shot_probability ^ (num_shots - k) * 
      (1 - shot_probability) ^ k)) = pass_probability := by
  sorry

end NUMINAMATH_CALUDE_shooting_test_probability_l3357_335737


namespace NUMINAMATH_CALUDE_supplementary_angles_problem_l3357_335711

theorem supplementary_angles_problem (x y : ℝ) :
  x + y = 180 ∧ y = x + 18 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angles_problem_l3357_335711


namespace NUMINAMATH_CALUDE_dart_target_probability_l3357_335714

theorem dart_target_probability (n : ℕ) : 
  (n : ℝ) * π / (n : ℝ)^2 ≥ (1 : ℝ) / 2 → n ≤ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_dart_target_probability_l3357_335714


namespace NUMINAMATH_CALUDE_sqrt_square_abs_l3357_335760

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_l3357_335760
