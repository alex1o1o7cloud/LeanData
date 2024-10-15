import Mathlib

namespace NUMINAMATH_CALUDE_students_in_front_of_yuna_l3335_333578

/-- Given a line of students with Yuna somewhere in the line, this theorem
    proves the number of students in front of Yuna. -/
theorem students_in_front_of_yuna 
  (total_students : ℕ) 
  (students_behind_yuna : ℕ) 
  (h1 : total_students = 25)
  (h2 : students_behind_yuna = 9) :
  total_students - (students_behind_yuna + 1) = 15 :=
by sorry

end NUMINAMATH_CALUDE_students_in_front_of_yuna_l3335_333578


namespace NUMINAMATH_CALUDE_max_ratio_squared_l3335_333574

theorem max_ratio_squared (a b x z : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b)
  (hx : 0 ≤ x) (hxa : x < a) (hz : 0 ≤ z) (hzb : z < b)
  (heq : a^2 + z^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b - z)^2) :
  (a / b)^2 ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l3335_333574


namespace NUMINAMATH_CALUDE_circle_center_l3335_333551

/-- The center of a circle given by the equation x^2 - 4x + y^2 - 6y - 12 = 0 is (2, 3) -/
theorem circle_center (x y : ℝ) : 
  x^2 - 4*x + y^2 - 6*y - 12 = 0 → (2, 3) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3335_333551


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l3335_333509

/-- The number of emails Jack received in a day -/
def total_emails : ℕ := 10

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 1

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := total_emails - morning_emails - evening_emails

theorem jack_afternoon_emails :
  afternoon_emails = 4 := by sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l3335_333509


namespace NUMINAMATH_CALUDE_jingJing_bought_four_notebooks_l3335_333529

/-- Represents the purchase of stationery items -/
structure StationeryPurchase where
  carbonPens : ℕ
  notebooks : ℕ
  pencilCases : ℕ

/-- Calculates the total cost of a stationery purchase -/
def totalCost (p : StationeryPurchase) : ℚ :=
  1.8 * p.carbonPens + 3.5 * p.notebooks + 4.2 * p.pencilCases

/-- Theorem stating that Jing Jing bought 4 notebooks -/
theorem jingJing_bought_four_notebooks :
  ∃ (p : StationeryPurchase),
    p.carbonPens > 0 ∧
    p.notebooks > 0 ∧
    p.pencilCases > 0 ∧
    totalCost p = 20 ∧
    p.notebooks = 4 :=
by sorry

end NUMINAMATH_CALUDE_jingJing_bought_four_notebooks_l3335_333529


namespace NUMINAMATH_CALUDE_jess_remaining_distance_l3335_333506

/-- The remaining distance Jess must walk to arrive at work -/
def remaining_distance (store_distance gallery_distance work_distance walked_distance : ℕ) : ℕ :=
  store_distance + gallery_distance + work_distance - walked_distance

/-- Proof that Jess must walk 20 more blocks to arrive at work -/
theorem jess_remaining_distance :
  remaining_distance 11 6 8 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jess_remaining_distance_l3335_333506


namespace NUMINAMATH_CALUDE_wrappers_found_at_park_l3335_333556

/-- Represents the number of bottle caps Danny found at the park. -/
def bottle_caps_found : ℕ := 58

/-- Represents the number of wrappers Danny now has in his collection. -/
def wrappers_now : ℕ := 11

/-- Represents the number of bottle caps Danny now has in his collection. -/
def bottle_caps_now : ℕ := 12

/-- Represents the difference between bottle caps and wrappers Danny has now. -/
def cap_wrapper_difference : ℕ := 1

/-- Proves that the number of wrappers Danny found at the park is 11. -/
theorem wrappers_found_at_park : ℕ := by
  sorry

end NUMINAMATH_CALUDE_wrappers_found_at_park_l3335_333556


namespace NUMINAMATH_CALUDE_percentage_relation_l3335_333572

theorem percentage_relation (x y : ℝ) (h : 0.6 * (x - y) = 0.2 * (x + y)) : y = 0.5 * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3335_333572


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3335_333585

/-- The perimeter of a rhombus with diagonals of 12 inches and 16 inches is 40 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 40 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3335_333585


namespace NUMINAMATH_CALUDE_trigonometric_identity_equivalence_l3335_333570

theorem trigonometric_identity_equivalence (x : ℝ) :
  (1 + Real.cos (4 * x)) * Real.sin (2 * x) = (Real.cos (2 * x))^2 ↔
  (∃ k : ℤ, x = (-1)^k * (π / 12) + k * (π / 2)) ∨
  (∃ n : ℤ, x = π / 4 * (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_equivalence_l3335_333570


namespace NUMINAMATH_CALUDE_unique_triplet_satisfying_conditions_l3335_333558

theorem unique_triplet_satisfying_conditions :
  ∃! (a b c : ℝ),
    ({a^2 - 4*c, b^2 - 2*a, c^2 - 2*b} : Set ℝ) = {a - c, b - 4*c, a + b} ∧
    2*a + 2*b + 6 = 5*c ∧
    (a^2 - 4*c ≠ b^2 - 2*a ∧ a^2 - 4*c ≠ c^2 - 2*b ∧ b^2 - 2*a ≠ c^2 - 2*b) ∧
    (a - c ≠ b - 4*c ∧ a - c ≠ a + b ∧ b - 4*c ≠ a + b) ∧
    a = 1 ∧ b = 1 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_satisfying_conditions_l3335_333558


namespace NUMINAMATH_CALUDE_not_prime_n4_2n2_3_l3335_333544

theorem not_prime_n4_2n2_3 (n : ℤ) : ∃ k : ℤ, n^4 + 2*n^2 + 3 = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n4_2n2_3_l3335_333544


namespace NUMINAMATH_CALUDE_sara_marbles_l3335_333517

theorem sara_marbles (initial lost left : ℕ) : 
  lost = 7 → left = 3 → initial = lost + left → initial = 10 := by
sorry

end NUMINAMATH_CALUDE_sara_marbles_l3335_333517


namespace NUMINAMATH_CALUDE_museum_trip_buses_l3335_333542

/-- The number of people on the first bus -/
def first_bus : ℕ := 12

/-- The number of people on the second bus -/
def second_bus : ℕ := 2 * first_bus

/-- The number of people on the third bus -/
def third_bus : ℕ := second_bus - 6

/-- The number of people on the fourth bus -/
def fourth_bus : ℕ := first_bus + 9

/-- The total number of people going to the museum -/
def total_people : ℕ := 75

/-- The number of buses hired -/
def num_buses : ℕ := 4

theorem museum_trip_buses :
  first_bus + second_bus + third_bus + fourth_bus = total_people ∧
  num_buses = 4 := by sorry

end NUMINAMATH_CALUDE_museum_trip_buses_l3335_333542


namespace NUMINAMATH_CALUDE_complex_number_equality_l3335_333522

theorem complex_number_equality (z : ℂ) : z = 2 - (13 / 6) * I →
  Complex.abs (z - 2) = Complex.abs (z + 2) ∧
  Complex.abs (z - 2) = Complex.abs (z - 3 * I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3335_333522


namespace NUMINAMATH_CALUDE_bookstore_location_l3335_333532

/-- The floor number of the academy -/
def academy_floor : ℕ := 7

/-- The number of floors the reading room is above the academy -/
def reading_room_above_academy : ℕ := 4

/-- The number of floors the bookstore is below the reading room -/
def bookstore_below_reading_room : ℕ := 9

/-- The floor number of the bookstore -/
def bookstore_floor : ℕ := academy_floor + reading_room_above_academy - bookstore_below_reading_room

theorem bookstore_location : bookstore_floor = 2 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_location_l3335_333532


namespace NUMINAMATH_CALUDE_problem_1_l3335_333582

theorem problem_1 (m n : ℤ) (h1 : 4*m + n = 90) (h2 : 2*m - 3*n = 10) :
  (m + 2*n)^2 - (3*m - n)^2 = -900 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3335_333582


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l3335_333557

theorem six_digit_divisibility (A B : ℕ) 
  (hA : A ≥ 100 ∧ A < 1000) 
  (hB : B ≥ 100 ∧ B < 1000) 
  (hAnotDiv : ¬ (37 ∣ A)) 
  (hBnotDiv : ¬ (37 ∣ B)) 
  (hSum : 37 ∣ (A + B)) : 
  37 ∣ (1000 * A + B) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l3335_333557


namespace NUMINAMATH_CALUDE_gcd_9009_14014_l3335_333594

theorem gcd_9009_14014 : Nat.gcd 9009 14014 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9009_14014_l3335_333594


namespace NUMINAMATH_CALUDE_sin_double_angle_for_point_on_terminal_side_l3335_333508

theorem sin_double_angle_for_point_on_terminal_side :
  ∀ α : ℝ,
  let P : ℝ × ℝ := (-4, -6 * Real.sin (150 * π / 180))
  (P.1 = -4 ∧ P.2 = -6 * Real.sin (150 * π / 180)) →
  Real.sin (2 * α) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_for_point_on_terminal_side_l3335_333508


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3335_333518

theorem sum_of_three_numbers (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xy : x * y = 60) (h_xz : x * z = 90) (h_yz : y * z = 150) : 
  x + y + z = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3335_333518


namespace NUMINAMATH_CALUDE_vanessa_album_pictures_l3335_333503

/-- The number of albums created by Vanessa -/
def num_albums : ℕ := 10

/-- The number of pictures from the phone in each album -/
def phone_pics_per_album : ℕ := 8

/-- The number of pictures from the camera in each album -/
def camera_pics_per_album : ℕ := 4

/-- The total number of pictures in each album -/
def pics_per_album : ℕ := phone_pics_per_album + camera_pics_per_album

theorem vanessa_album_pictures :
  pics_per_album = 12 :=
sorry

end NUMINAMATH_CALUDE_vanessa_album_pictures_l3335_333503


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l3335_333579

/-- Given real numbers a and b where ab ≠ 0, prove that ax - y + b = 0 represents a line
    and bx² + ay² = ab represents an ellipse -/
theorem line_intersects_ellipse (a b : ℝ) (h : a * b ≠ 0) :
  ∃ (line : ℝ → ℝ) (ellipse : Set (ℝ × ℝ)),
    (∀ x y, ax - y + b = 0 ↔ y = line x) ∧
    (∀ x y, (x, y) ∈ ellipse ↔ b * x^2 + a * y^2 = a * b) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l3335_333579


namespace NUMINAMATH_CALUDE_f_at_two_l3335_333537

def f (x : ℝ) : ℝ := 15 * x^5 - 24 * x^4 + 33 * x^3 - 42 * x^2 + 51 * x

theorem f_at_two : f 2 = 294 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_l3335_333537


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3335_333536

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y = x * y) :
  x + y ≥ 18 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + 8 * y = x * y ∧ x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3335_333536


namespace NUMINAMATH_CALUDE_sum_less_than_addends_implies_negative_l3335_333587

theorem sum_less_than_addends_implies_negative (a b : ℝ) :
  (a + b < a ∧ a + b < b) → (a < 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_addends_implies_negative_l3335_333587


namespace NUMINAMATH_CALUDE_application_methods_count_l3335_333511

def number_of_universities : ℕ := 6
def universities_to_choose : ℕ := 3
def universities_with_conflict : ℕ := 2

theorem application_methods_count :
  (number_of_universities.choose universities_to_choose) -
  (universities_with_conflict * (number_of_universities - universities_with_conflict).choose (universities_to_choose - 1)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_application_methods_count_l3335_333511


namespace NUMINAMATH_CALUDE_price_per_large_bottle_l3335_333560

/-- The price per large bottle, given the number of large and small bottles,
    the price of small bottles, and the average price of all bottles. -/
theorem price_per_large_bottle (large_count small_count : ℕ)
                                (small_price avg_price : ℚ) :
  large_count = 1325 →
  small_count = 750 →
  small_price = 138/100 →
  avg_price = 17057/10000 →
  ∃ (large_price : ℚ), 
    (large_count * large_price + small_count * small_price) / (large_count + small_count) = avg_price ∧
    abs (large_price - 189/100) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_price_per_large_bottle_l3335_333560


namespace NUMINAMATH_CALUDE_soda_comparison_l3335_333565

theorem soda_comparison (J : ℝ) (L A : ℝ) 
  (h1 : L = J * 1.5)  -- Liliane has 50% more soda than Jacqueline
  (h2 : A = J * 1.25) -- Alice has 25% more soda than Jacqueline
  : L = A * 1.2       -- Liliane has 20% more soda than Alice
:= by sorry

end NUMINAMATH_CALUDE_soda_comparison_l3335_333565


namespace NUMINAMATH_CALUDE_graphics_cards_sold_l3335_333533

/-- Represents the number of graphics cards sold. -/
def graphics_cards : ℕ := sorry

/-- Represents the number of hard drives sold. -/
def hard_drives : ℕ := 14

/-- Represents the number of CPUs sold. -/
def cpus : ℕ := 8

/-- Represents the number of RAM pairs sold. -/
def ram_pairs : ℕ := 4

/-- Represents the price of a single graphics card in dollars. -/
def graphics_card_price : ℕ := 600

/-- Represents the price of a single hard drive in dollars. -/
def hard_drive_price : ℕ := 80

/-- Represents the price of a single CPU in dollars. -/
def cpu_price : ℕ := 200

/-- Represents the price of a pair of RAM in dollars. -/
def ram_pair_price : ℕ := 60

/-- Represents the total earnings of the store in dollars. -/
def total_earnings : ℕ := 8960

/-- Theorem stating that the number of graphics cards sold is 10. -/
theorem graphics_cards_sold : graphics_cards = 10 := by
  sorry

end NUMINAMATH_CALUDE_graphics_cards_sold_l3335_333533


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3335_333581

def U : Set ℕ := {x | x ≥ 2}
def A : Set ℕ := {x | x^2 ≥ 5}

theorem complement_of_A_in_U : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3335_333581


namespace NUMINAMATH_CALUDE_pole_length_reduction_l3335_333573

theorem pole_length_reduction (original_length current_length : ℝ) 
  (h1 : original_length = 20)
  (h2 : current_length = 14) :
  (original_length - current_length) / original_length * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_pole_length_reduction_l3335_333573


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3335_333504

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The equation of a plane in 3D space -/
structure PlaneEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Given points A, B, and C, proves that the equation x + 2y + 4z - 5 = 0
    represents the plane passing through point A and perpendicular to vector BC -/
theorem plane_equation_proof 
  (A : Point3D) 
  (B : Point3D) 
  (C : Point3D) 
  (h1 : A.x = -7 ∧ A.y = 0 ∧ A.z = 3)
  (h2 : B.x = 1 ∧ B.y = -5 ∧ B.z = -4)
  (h3 : C.x = 2 ∧ C.y = -3 ∧ C.z = 0) :
  let BC : Vector3D := ⟨C.x - B.x, C.y - B.y, C.z - B.z⟩
  let plane : PlaneEquation := ⟨1, 2, 4, -5⟩
  (plane.a * (A.x - x) + plane.b * (A.y - y) + plane.c * (A.z - z) = 0) ∧
  (plane.a * BC.x + plane.b * BC.y + plane.c * BC.z = 0) :=
by sorry


end NUMINAMATH_CALUDE_plane_equation_proof_l3335_333504


namespace NUMINAMATH_CALUDE_jessica_rearrangement_time_l3335_333563

/-- The time in hours required to write all rearrangements of a name -/
def time_to_write_rearrangements (name_length : ℕ) (repeated_letter_count : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  let total_permutations := (name_length.factorial / repeated_letter_count.factorial : ℚ)
  let time_in_minutes := total_permutations / rearrangements_per_minute
  time_in_minutes / 60

/-- Theorem stating the time required to write all rearrangements of Jessica's name -/
theorem jessica_rearrangement_time :
  time_to_write_rearrangements 7 2 18 = 2333 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_jessica_rearrangement_time_l3335_333563


namespace NUMINAMATH_CALUDE_parking_arrangements_l3335_333553

theorem parking_arrangements (total_spaces : ℕ) (cars : ℕ) (consecutive_empty : ℕ) 
  (h1 : total_spaces = 12) 
  (h2 : cars = 8) 
  (h3 : consecutive_empty = 4) : 
  (Nat.factorial cars) * (total_spaces - cars - consecutive_empty + 1) = 362880 := by
  sorry

end NUMINAMATH_CALUDE_parking_arrangements_l3335_333553


namespace NUMINAMATH_CALUDE_inequality_preservation_l3335_333541

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 5 > b - 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3335_333541


namespace NUMINAMATH_CALUDE_monomial_properties_l3335_333538

/-- A monomial is a product of a coefficient and variables raised to non-negative integer powers. -/
structure Monomial (R : Type*) [CommRing R] where
  coeff : R
  exponents : List ℕ

/-- The degree of a monomial is the sum of its exponents. -/
def Monomial.degree {R : Type*} [CommRing R] (m : Monomial R) : ℕ :=
  m.exponents.sum

/-- Our specific monomial -3x^2y -/
def our_monomial : Monomial ℤ :=
  { coeff := -3
  , exponents := [2, 1] }

theorem monomial_properties :
  our_monomial.coeff = -3 ∧ our_monomial.degree = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l3335_333538


namespace NUMINAMATH_CALUDE_remainder_theorem_l3335_333559

theorem remainder_theorem : (2^210 + 210) % (2^105 + 2^63 + 1) = 210 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3335_333559


namespace NUMINAMATH_CALUDE_divisibility_property_l3335_333568

theorem divisibility_property (n : ℕ) : n ≥ 1 ∧ n ∣ (3^n + 1) ∧ n ∣ (11^n + 1) ↔ n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3335_333568


namespace NUMINAMATH_CALUDE_lcm_problem_l3335_333520

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3335_333520


namespace NUMINAMATH_CALUDE_car_speed_acceleration_l3335_333510

/-- Proves that given an initial speed of 45 m/s, an acceleration of 2.5 m/s² for 10 seconds,
    the final speed will be 70 m/s and 252 km/h. -/
theorem car_speed_acceleration (initial_speed : Real) (acceleration : Real) (time : Real) :
  initial_speed = 45 ∧ acceleration = 2.5 ∧ time = 10 →
  let final_speed := initial_speed + acceleration * time
  final_speed = 70 ∧ final_speed * 3.6 = 252 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_acceleration_l3335_333510


namespace NUMINAMATH_CALUDE_phone_time_proof_l3335_333526

/-- 
Given a person who spends time on the phone for 5 days, 
doubling the time each day after the first, 
and spending a total of 155 minutes,
prove that they spent 5 minutes on the first day.
-/
theorem phone_time_proof (x : ℝ) : 
  x + 2*x + 4*x + 8*x + 16*x = 155 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_phone_time_proof_l3335_333526


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l3335_333540

/-- An arithmetic sequence with common difference -2 and S_3 = 21 reaches its maximum sum at n = 5 -/
theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, a (n + 1) - a n = -2) →  -- Common difference is -2
  S 3 = 21 →                     -- S_3 = 21
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n for arithmetic sequence
  (∃ m, ∀ k, S k ≤ S m) →       -- S_n has a maximum value
  (∀ k, S k ≤ S 5) :=            -- The maximum occurs at n = 5
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l3335_333540


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_no_false_main_theorem_l3335_333547

-- Proposition 1
theorem proposition_1 (k : ℝ) : k > 0 → ∃ x : ℝ, x^2 - 2*x - k = 0 := by sorry

-- Proposition 2
theorem proposition_2 (x y : ℝ) : x + y ≠ 8 → x ≠ 2 ∨ y ≠ 6 := by sorry

-- Proposition 3
theorem proposition_3_no_false : ¬∃ (P : Prop), P ↔ ¬(∀ x y : ℝ, x*y = 0 → x = 0 ∨ y = 0) := by sorry

-- Main theorem combining all propositions
theorem main_theorem : 
  (∀ k : ℝ, k > 0 → ∃ x : ℝ, x^2 - 2*x - k = 0) ∧ 
  (∀ x y : ℝ, x + y ≠ 8 → x ≠ 2 ∨ y ≠ 6) ∧ 
  ¬∃ (P : Prop), P ↔ ¬(∀ x y : ℝ, x*y = 0 → x = 0 ∨ y = 0) := by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_no_false_main_theorem_l3335_333547


namespace NUMINAMATH_CALUDE_probability_theorem_l3335_333599

/-- The set of ball numbers in the bag -/
def BallNumbers : Finset ℕ := {1, 2, 3, 4}

/-- The probability of drawing two balls with sum not exceeding 4 -/
def prob_sum_not_exceeding_4 : ℚ :=
  (Finset.filter (fun pair => pair.1 + pair.2 ≤ 4) (BallNumbers.product BallNumbers)).card /
  (BallNumbers.product BallNumbers).card

/-- The probability of drawing two balls with replacement where n < m + 2 -/
def prob_n_less_than_m_plus_2 : ℚ :=
  (Finset.filter (fun pair => pair.2 < pair.1 + 2) (BallNumbers.product BallNumbers)).card /
  (BallNumbers.product BallNumbers).card

theorem probability_theorem :
  prob_sum_not_exceeding_4 = 1/3 ∧ prob_n_less_than_m_plus_2 = 13/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3335_333599


namespace NUMINAMATH_CALUDE_rounded_number_accuracy_l3335_333545

/-- Represents a number with its value and accuracy -/
structure ApproximateNumber where
  value : ℝ
  accuracy : ℕ

/-- Defines the concept of "accurate to the hundreds place" -/
def accurate_to_hundreds (n : ApproximateNumber) : Prop :=
  ∃ (k : ℤ), n.value = (k * 100 : ℝ) ∧ 
  ∀ (m : ℤ), |n.value - (m * 100 : ℝ)| ≥ 50

/-- The main theorem to prove -/
theorem rounded_number_accuracy :
  let n := ApproximateNumber.mk (8.80 * 10^4) 2
  accurate_to_hundreds n :=
by sorry

end NUMINAMATH_CALUDE_rounded_number_accuracy_l3335_333545


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3335_333523

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  pos_terms : ∀ n, a n > 0
  geom_prop : ∀ n, a (n + 1) = q * a n

/-- Sum of first n terms of a geometric sequence -/
def S (g : GeometricSequence) (n : ℕ) : ℝ := sorry

theorem geometric_sequence_properties (g : GeometricSequence) :
  (-1 : ℝ) < S g 5 ∧ S g 5 < S g 10 ∧  -- S_5 and S_10 are positive
  (S g 5 - (-1) = S g 10 - S g 5) →    -- -1, S_5, S_10 form an arithmetic sequence
  (S g 10 - 2 * S g 5 = 1) ∧           -- First result
  (∀ h : GeometricSequence, 
    ((-1 : ℝ) < S h 5 ∧ S h 5 < S h 10 ∧ 
     S h 5 - (-1) = S h 10 - S h 5) → 
    S g 15 - S g 10 ≤ S h 15 - S h 10) ∧  -- S_15 - S_10 is minimized for g
  (S g 15 - S g 10 = 4) :=              -- Minimum value is 4
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3335_333523


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l3335_333502

theorem function_not_in_first_quadrant (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : b < -1) :
  ∀ x : ℝ, x > 0 → a^x + b < 0 := by sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l3335_333502


namespace NUMINAMATH_CALUDE_log_xy_z_in_terms_of_log_x_z_and_log_y_z_l3335_333516

theorem log_xy_z_in_terms_of_log_x_z_and_log_y_z
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.log z / Real.log (x * y) = (Real.log z / Real.log x * Real.log z / Real.log y) /
                                  (Real.log z / Real.log x + Real.log z / Real.log y) :=
by sorry

end NUMINAMATH_CALUDE_log_xy_z_in_terms_of_log_x_z_and_log_y_z_l3335_333516


namespace NUMINAMATH_CALUDE_min_garden_cost_l3335_333501

/-- Represents a rectangular region in the flower bed -/
structure Region where
  length : ℝ
  width : ℝ

/-- Represents a type of flower -/
structure Flower where
  name : String
  price : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Calculates the cost of filling a region with a specific flower -/
def cost (r : Region) (f : Flower) : ℝ := area r * f.price

/-- The flower bed arrangement -/
def flowerBed : List Region := [
  { length := 5, width := 2 },
  { length := 4, width := 2 },
  { length := 7, width := 4 },
  { length := 3, width := 5 }
]

/-- Available flower types -/
def flowers : List Flower := [
  { name := "Fuchsia", price := 3.5 },
  { name := "Gardenia", price := 4 },
  { name := "Canna", price := 2 },
  { name := "Begonia", price := 1.5 }
]

/-- Theorem stating the minimum cost of the garden -/
theorem min_garden_cost :
  ∃ (arrangement : List (Region × Flower)),
    arrangement.length = flowerBed.length ∧
    (∀ r ∈ flowerBed, ∃ f ∈ flowers, (r, f) ∈ arrangement) ∧
    (arrangement.map (λ (r, f) => cost r f)).sum = 140 ∧
    ∀ (other_arrangement : List (Region × Flower)),
      other_arrangement.length = flowerBed.length →
      (∀ r ∈ flowerBed, ∃ f ∈ flowers, (r, f) ∈ other_arrangement) →
      (other_arrangement.map (λ (r, f) => cost r f)).sum ≥ 140 := by
  sorry

end NUMINAMATH_CALUDE_min_garden_cost_l3335_333501


namespace NUMINAMATH_CALUDE_moon_weight_calculation_l3335_333592

/-- The weight of the moon in tons -/
def moon_weight : ℝ := 250

/-- The weight of Mars in tons -/
def mars_weight : ℝ := 500

/-- The percentage of iron in the composition -/
def iron_percentage : ℝ := 50

/-- The percentage of carbon in the composition -/
def carbon_percentage : ℝ := 20

/-- The percentage of other elements in the composition -/
def other_percentage : ℝ := 100 - iron_percentage - carbon_percentage

/-- The weight of other elements on Mars in tons -/
def mars_other_elements : ℝ := 150

theorem moon_weight_calculation :
  moon_weight = mars_weight / 2 ∧
  mars_weight = mars_other_elements / (other_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_moon_weight_calculation_l3335_333592


namespace NUMINAMATH_CALUDE_greatest_x_value_l3335_333586

theorem greatest_x_value (x : ℤ) (h : 2.134 * (10 : ℝ) ^ (x : ℝ) < 220000) :
  x ≤ 5 ∧ 2.134 * (10 : ℝ) ^ (5 : ℝ) < 220000 := by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3335_333586


namespace NUMINAMATH_CALUDE_odd_function_symmetry_l3335_333562

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y
def hasMinimumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop := 
  ∀ x, a ≤ x ∧ x ≤ b → f x ≥ m

-- State the theorem
theorem odd_function_symmetry (hOdd : isOdd f) 
  (hDec : isDecreasingOn f (-2) (-1)) 
  (hMin : hasMinimumOn f (-2) (-1) 3) :
  isDecreasingOn f 1 2 ∧ hasMinimumOn f 1 2 (-3) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_l3335_333562


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3335_333514

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 2 + a 7 = 12)
  (h_product : a 4 * a 5 = 35) :
  (∀ n : ℕ, a n = 2 * n - 3) ∨ (∀ n : ℕ, a n = 15 - 2 * n) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3335_333514


namespace NUMINAMATH_CALUDE_sequence_solution_l3335_333564

def x (n : ℕ+) : ℚ := n / (n + 2016)

theorem sequence_solution :
  ∃ (m n : ℕ+), x 2016 = x m * x n ∧ m = 4032 ∧ n = 6048 :=
by sorry

end NUMINAMATH_CALUDE_sequence_solution_l3335_333564


namespace NUMINAMATH_CALUDE_equation_is_linear_and_has_solution_l3335_333571

-- Define the equation
def equation (x : ℝ) : Prop := 1 - x = -3

-- State the theorem
theorem equation_is_linear_and_has_solution :
  (∃ a b : ℝ, ∀ x, equation x ↔ a * x + b = 0) ∧ 
  equation 4 := by sorry

end NUMINAMATH_CALUDE_equation_is_linear_and_has_solution_l3335_333571


namespace NUMINAMATH_CALUDE_x_greater_than_half_l3335_333596

theorem x_greater_than_half (x : ℝ) (h : (1/2) * x = 1) : 
  (x - 1/2) / (1/2) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_half_l3335_333596


namespace NUMINAMATH_CALUDE_florist_roses_theorem_l3335_333543

/-- Calculates the final number of roses a florist has after selling and picking more roses. -/
def final_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : ℕ :=
  initial - sold + picked

/-- Proves that the final number of roses is correct given the initial number,
    the number sold, and the number picked. -/
theorem florist_roses_theorem (initial : ℕ) (sold : ℕ) (picked : ℕ) 
    (h1 : initial ≥ sold) : 
  final_roses initial sold picked = initial - sold + picked :=
by
  -- The proof goes here
  sorry

/-- Verifies the specific case from the original problem. -/
example : final_roses 37 16 19 = 40 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_florist_roses_theorem_l3335_333543


namespace NUMINAMATH_CALUDE_intersection_inequality_solution_l3335_333552

/-- Given two linear functions y₁ = ax + b and y₂ = cx + d with a > c > 0,
    intersecting at the point (2, m), prove that the solution set of
    the inequality (a-c)x ≤ d-b is x ≤ 2. -/
theorem intersection_inequality_solution
  (a b c d m : ℝ)
  (h1 : a > c)
  (h2 : c > 0)
  (h3 : a * 2 + b = c * 2 + d)
  (h4 : a * 2 + b = m) :
  ∀ x, (a - c) * x ≤ d - b ↔ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_inequality_solution_l3335_333552


namespace NUMINAMATH_CALUDE_nonnegative_integer_pairs_l3335_333507

theorem nonnegative_integer_pairs (x y : ℕ) : (x * y + 2)^2 = x^2 + y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_integer_pairs_l3335_333507


namespace NUMINAMATH_CALUDE_power_of_four_three_halves_l3335_333549

theorem power_of_four_three_halves : (4 : ℝ) ^ (3/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_three_halves_l3335_333549


namespace NUMINAMATH_CALUDE_hostel_mess_expenditure_l3335_333577

/-- The original daily expenditure of a hostel mess given certain conditions -/
theorem hostel_mess_expenditure 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (expense_increase : ℕ) 
  (avg_expense_decrease : ℕ) 
  (h1 : initial_students = 35)
  (h2 : new_students = 7)
  (h3 : expense_increase = 42)
  (h4 : avg_expense_decrease = 1) : 
  ∃ (original_expenditure : ℕ), original_expenditure = 420 :=
by sorry

end NUMINAMATH_CALUDE_hostel_mess_expenditure_l3335_333577


namespace NUMINAMATH_CALUDE_advertising_agency_clients_l3335_333515

theorem advertising_agency_clients (total : ℕ) (tv radio mag tv_mag tv_radio radio_mag : ℕ) 
  (h_total : total = 180)
  (h_tv : tv = 115)
  (h_radio : radio = 110)
  (h_mag : mag = 130)
  (h_tv_mag : tv_mag = 85)
  (h_tv_radio : tv_radio = 75)
  (h_radio_mag : radio_mag = 95) :
  total = tv + radio + mag - tv_mag - tv_radio - radio_mag + 80 :=
sorry

end NUMINAMATH_CALUDE_advertising_agency_clients_l3335_333515


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l3335_333505

theorem rational_inequality_solution (x : ℝ) : 
  (x + 2) / (x^2 + 3*x + 10) ≥ 0 ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l3335_333505


namespace NUMINAMATH_CALUDE_age_difference_l3335_333527

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A - C = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3335_333527


namespace NUMINAMATH_CALUDE_lunch_spending_difference_l3335_333580

/-- Given a lunch scenario where two people spent a total of $15,
    with one person spending $10, prove that the difference in
    spending between the two people is $5. -/
theorem lunch_spending_difference :
  ∀ (your_spending friend_spending : ℕ),
  your_spending + friend_spending = 15 →
  friend_spending = 10 →
  friend_spending > your_spending →
  friend_spending - your_spending = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_lunch_spending_difference_l3335_333580


namespace NUMINAMATH_CALUDE_modulus_of_z_l3335_333589

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem modulus_of_z : 
  let z : ℂ := (7 - Complex.I) / (1 + Complex.I)
  Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3335_333589


namespace NUMINAMATH_CALUDE_triangle_frame_stability_l3335_333555

/-- A bicycle frame is a structure used in bicycles. -/
structure BicycleFrame where
  shape : Type

/-- A triangle is a geometric shape with three sides and three angles. -/
inductive Triangle : Type where
  | mk : Triangle

/-- Stability is a property of structures that resist deformation under load. -/
def Stability : Prop := sorry

/-- A bicycle frame made in the shape of a triangle provides stability. -/
theorem triangle_frame_stability (frame : BicycleFrame) (h : frame.shape = Triangle) : 
  Stability :=
sorry

end NUMINAMATH_CALUDE_triangle_frame_stability_l3335_333555


namespace NUMINAMATH_CALUDE_smallest_a1_l3335_333575

/-- A sequence of positive real numbers satisfying aₙ = 7aₙ₋₁ - n for n > 1 -/
def ValidSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 7 * a (n - 1) - n)

/-- The smallest possible value of a₁ in a valid sequence is 13/36 -/
theorem smallest_a1 :
    ∀ a : ℕ → ℝ, ValidSequence a → a 1 ≥ 13/36 ∧ ∃ a', ValidSequence a' ∧ a' 1 = 13/36 :=
  sorry

end NUMINAMATH_CALUDE_smallest_a1_l3335_333575


namespace NUMINAMATH_CALUDE_aquafaba_needed_l3335_333597

/-- The number of tablespoons of aquafaba equivalent to one egg white -/
def aquafaba_per_egg : ℕ := 2

/-- The number of cakes Christine is making -/
def num_cakes : ℕ := 2

/-- The number of egg whites required for each cake -/
def egg_whites_per_cake : ℕ := 8

/-- Theorem stating the total number of tablespoons of aquafaba needed -/
theorem aquafaba_needed : 
  aquafaba_per_egg * num_cakes * egg_whites_per_cake = 32 := by
  sorry

end NUMINAMATH_CALUDE_aquafaba_needed_l3335_333597


namespace NUMINAMATH_CALUDE_max_tickets_buyable_l3335_333530

def regular_price : ℝ := 15
def discount_threshold : ℕ := 6
def discount_rate : ℝ := 0.1
def budget : ℝ := 120

def discounted_price : ℝ := regular_price * (1 - discount_rate)

def cost (n : ℕ) : ℝ :=
  if n ≤ discount_threshold then n * regular_price
  else n * discounted_price

theorem max_tickets_buyable :
  ∀ n : ℕ, cost n ≤ budget → n ≤ 8 ∧ cost 8 ≤ budget :=
sorry

end NUMINAMATH_CALUDE_max_tickets_buyable_l3335_333530


namespace NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l3335_333500

theorem min_values_ab_and_a_plus_2b (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 2) : 
  (∀ a b, a > 0 → b > 0 → 1/a + 2/b = 2 → a * b ≥ 2) ∧ 
  (∀ a b, a > 0 → b > 0 → 1/a + 2/b = 2 → a + 2*b ≥ 9/2) ∧
  (a = 3/2 ∧ b = 3/2 → a * b = 2 ∧ a + 2*b = 9/2) :=
sorry

end NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l3335_333500


namespace NUMINAMATH_CALUDE_derivative_of_power_function_l3335_333521

theorem derivative_of_power_function (a k : ℝ) (x : ℝ) :
  deriv (λ x => (3 * a * x - x^2)^k) x = k * (3 * a - 2 * x) * (3 * a * x - x^2)^(k - 1) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_power_function_l3335_333521


namespace NUMINAMATH_CALUDE_max_sum_is_27_l3335_333576

/-- Represents the arrangement of numbers in the grid -/
structure Arrangement where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {2, 5, 8, 11, 14}

/-- Checks if an arrangement is valid according to the problem conditions -/
def isValidArrangement (arr : Arrangement) : Prop :=
  (arr.a ∈ availableNumbers) ∧
  (arr.b ∈ availableNumbers) ∧
  (arr.c ∈ availableNumbers) ∧
  (arr.d ∈ availableNumbers) ∧
  (arr.e ∈ availableNumbers) ∧
  (arr.f ∈ availableNumbers) ∧
  (arr.a + arr.b + arr.e = arr.c + arr.d + arr.f) ∧
  (arr.a + arr.c = arr.b + arr.d) ∧
  (arr.a + arr.c = arr.e + arr.f)

/-- The theorem to be proven -/
theorem max_sum_is_27 :
  ∀ (arr : Arrangement), isValidArrangement arr →
  (arr.a + arr.b + arr.e ≤ 27 ∧ arr.c + arr.d + arr.f ≤ 27) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_is_27_l3335_333576


namespace NUMINAMATH_CALUDE_excavation_time_equality_l3335_333593

/-- Represents the dimensions of an excavation site -/
structure Dimensions where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of an excavation site given its dimensions -/
def volume (d : Dimensions) : ℝ := d.depth * d.length * d.breadth

/-- The number of days required to dig an excavation site is directly proportional to its volume when the number of laborers is constant -/
axiom days_proportional_to_volume {d1 d2 : Dimensions} {days1 : ℝ} (h : volume d1 = volume d2) :
  days1 = days1 * (volume d2 / volume d1)

theorem excavation_time_equality (initial : Dimensions) (new : Dimensions) (initial_days : ℝ) 
    (h_initial : initial = { depth := 100, length := 25, breadth := 30 })
    (h_new : new = { depth := 75, length := 20, breadth := 50 })
    (h_initial_days : initial_days = 12) :
    initial_days = initial_days * (volume new / volume initial) := by
  sorry

end NUMINAMATH_CALUDE_excavation_time_equality_l3335_333593


namespace NUMINAMATH_CALUDE_five_ruble_coins_l3335_333591

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  one : Nat
  two : Nat
  five : Nat
  ten : Nat

/-- The problem setup -/
def coin_problem (c : CoinCounts) : Prop :=
  c.one + c.two + c.five + c.ten = 25 ∧
  c.one + c.five + c.ten = 19 ∧
  c.one + c.two + c.five = 20 ∧
  c.two + c.five + c.ten = 16

/-- The theorem to be proved -/
theorem five_ruble_coins (c : CoinCounts) : 
  coin_problem c → c.five = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_ruble_coins_l3335_333591


namespace NUMINAMATH_CALUDE_empty_graph_l3335_333539

theorem empty_graph (x y : ℝ) : ¬∃ (x y : ℝ), 3*x^2 + y^2 - 9*x - 4*y + 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_graph_l3335_333539


namespace NUMINAMATH_CALUDE_certain_term_is_12th_l3335_333548

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  -- Sum of a certain term and the 12th term is 20
  certain_term_sum : ∃ n : ℕ, a + (n - 1) * d + (a + 11 * d) = 20
  -- Sum of first 12 terms is 120
  sum_12_terms : 6 * (2 * a + 11 * d) = 120

/-- The certain term is the 12th term itself -/
theorem certain_term_is_12th (ap : ArithmeticProgression) : 
  ∃ n : ℕ, n = 12 ∧ a + (n - 1) * d + (a + 11 * d) = 20 := by
  sorry

#check certain_term_is_12th

end NUMINAMATH_CALUDE_certain_term_is_12th_l3335_333548


namespace NUMINAMATH_CALUDE_radical_expression_simplification_l3335_333528

theorem radical_expression_simplification
  (a b x : ℝ) 
  (h1 : a < b) 
  (h2 : -b ≤ x) 
  (h3 : x ≤ -a) :
  Real.sqrt (-(x + a)^3 * (x + b)) = -(x + a) * Real.sqrt (-(x + a) * (x + b)) :=
by sorry

end NUMINAMATH_CALUDE_radical_expression_simplification_l3335_333528


namespace NUMINAMATH_CALUDE_no_nonneg_integer_solution_l3335_333546

theorem no_nonneg_integer_solution :
  ¬ ∃ y : ℕ, Real.sqrt ((y - 2)^2 + 4^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_nonneg_integer_solution_l3335_333546


namespace NUMINAMATH_CALUDE_cafeteria_bags_l3335_333550

theorem cafeteria_bags (total : ℕ) (x : ℕ) : 
  total = 351 → 
  (x + 20) - 3 * ((total - x) - 50) = 1 → 
  x = 221 ∧ (total - x) = 130 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_bags_l3335_333550


namespace NUMINAMATH_CALUDE_f_derivative_l3335_333584

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem f_derivative (x : ℝ) (hx : x ≠ 0) :
  deriv f x = (Real.exp x * (x - 1)) / (x^2) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_l3335_333584


namespace NUMINAMATH_CALUDE_fruit_selection_ways_l3335_333513

/-- The number of ways to choose k items from n distinct items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of fruits in the basket -/
def num_fruits : ℕ := 5

/-- The number of fruits to be selected -/
def num_selected : ℕ := 2

/-- Theorem: There are 10 ways to select 2 fruits from a basket of 5 fruits -/
theorem fruit_selection_ways : choose num_fruits num_selected = 10 := by
  sorry

end NUMINAMATH_CALUDE_fruit_selection_ways_l3335_333513


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3335_333535

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (Complex.I ^ 2016) / (3 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3335_333535


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l3335_333590

theorem sum_of_fourth_powers (a b : ℝ) 
  (h1 : a^2 - b^2 = 10) 
  (h2 : a * b = 5) : 
  a^4 + b^4 = 150 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l3335_333590


namespace NUMINAMATH_CALUDE_maya_max_number_l3335_333524

theorem maya_max_number : ∃ (max : ℕ), max = 600 ∧ 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 3 * (300 - n) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_maya_max_number_l3335_333524


namespace NUMINAMATH_CALUDE_parabola_directrix_l3335_333554

open Real

-- Define the parabola
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

-- Define points
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def problem_setup (C : Parabola) (O F P Q : Point) : Prop :=
  -- O is the coordinate origin
  O.x = 0 ∧ O.y = 0
  -- F is the focus of parabola C
  ∧ F.x = C.p/2 ∧ F.y = 0
  -- P is a point on C
  ∧ C.eq P.x P.y
  -- PF is perpendicular to the x-axis
  ∧ P.x = F.x
  -- Q is a point on the x-axis
  ∧ Q.y = 0
  -- PQ is perpendicular to OP
  ∧ (Q.y - P.y) * (P.x - O.x) + (Q.x - P.x) * (P.y - O.y) = 0
  -- |FQ| = 6
  ∧ |F.x - Q.x| = 6

-- Theorem statement
theorem parabola_directrix (C : Parabola) (O F P Q : Point) 
  (h : problem_setup C O F P Q) : 
  ∃ (x : ℝ), x = -3/2 ∧ ∀ (y : ℝ), C.eq x y ↔ False :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3335_333554


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l3335_333598

/-- Given a total spend, sales tax, and tax rate, calculate the cost of tax-free items -/
def cost_of_tax_free_items (total_spend : ℚ) (sales_tax : ℚ) (tax_rate : ℚ) : ℚ :=
  total_spend - sales_tax / tax_rate

/-- Theorem: Given the specific values from the problem, the cost of tax-free items is 22 rupees -/
theorem tax_free_items_cost :
  let total_spend : ℚ := 25
  let sales_tax : ℚ := 30 / 100 -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 10 / 100 -- 10%
  cost_of_tax_free_items total_spend sales_tax tax_rate = 22 := by
  sorry

#eval cost_of_tax_free_items 25 (30/100) (10/100)

end NUMINAMATH_CALUDE_tax_free_items_cost_l3335_333598


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3335_333583

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 3 * x / (x - 3) + (3 * x^2 - 27) / x
  ∃ (min_sol : ℝ), min_sol = (8 - Real.sqrt 145) / 3 ∧
    f min_sol = 14 ∧
    ∀ (y : ℝ), f y = 14 → y ≥ min_sol :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3335_333583


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l3335_333588

theorem triangle_cosine_inequality (A B C : ℝ) (h_triangle : A + B + C = π) :
  1/3 * (Real.cos A + Real.cos B + Real.cos C) ≤ 1/2 ∧
  1/2 ≤ Real.sqrt (1/3 * (Real.cos A^2 + Real.cos B^2 + Real.cos C^2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l3335_333588


namespace NUMINAMATH_CALUDE_savannah_gift_wrapping_l3335_333534

/-- Given the conditions of Savannah's gift wrapping, prove that the first roll wraps 3 gifts -/
theorem savannah_gift_wrapping (total_rolls : ℕ) (total_gifts : ℕ) (second_roll_gifts : ℕ) (third_roll_gifts : ℕ) 
  (h1 : total_rolls = 3)
  (h2 : total_gifts = 12)
  (h3 : second_roll_gifts = 5)
  (h4 : third_roll_gifts = 4) :
  total_gifts - (second_roll_gifts + third_roll_gifts) = 3 := by
  sorry

end NUMINAMATH_CALUDE_savannah_gift_wrapping_l3335_333534


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3335_333569

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis. -/
def symmetricXAxis (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

/-- The given point N. -/
def N : Point2D :=
  ⟨2, 3⟩

theorem symmetric_point_coordinates :
  symmetricXAxis N = ⟨2, -3⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3335_333569


namespace NUMINAMATH_CALUDE_truncated_tetrahedron_lateral_area_l3335_333519

/-- Given a truncated tetrahedron with base area A₁, top area A₂ (where A₂ ≤ A₁),
    and sum of lateral face areas P, if the solid can be cut by a plane parallel
    to the base such that a sphere can be inscribed in each of the resulting sections,
    then P = (√A₁ + √A₂)(⁴√A₁ + ⁴√A₂)² -/
theorem truncated_tetrahedron_lateral_area
  (A₁ A₂ P : ℝ)
  (h₁ : 0 < A₁)
  (h₂ : 0 < A₂)
  (h₃ : A₂ ≤ A₁)
  (h₄ : ∃ (A : ℝ), 0 < A ∧ A < A₁ ∧ A > A₂ ∧
    ∃ (R₁ R₂ : ℝ), 0 < R₁ ∧ 0 < R₂ ∧
      A = Real.sqrt (A₁ * A₂) ∧
      (A / A₂) = (A₁ / A) ∧ (A / A₂) = (R₁ / R₂)^2) :
  P = (Real.sqrt A₁ + Real.sqrt A₂) * (Real.sqrt (Real.sqrt A₁) + Real.sqrt (Real.sqrt A₂))^2 := by
sorry

end NUMINAMATH_CALUDE_truncated_tetrahedron_lateral_area_l3335_333519


namespace NUMINAMATH_CALUDE_unique_solution_implies_equal_absolute_values_l3335_333525

theorem unique_solution_implies_equal_absolute_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃! x, a * (x - a)^2 + b * (x - b)^2 = 0) → |a| = |b| :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_equal_absolute_values_l3335_333525


namespace NUMINAMATH_CALUDE_slope_range_l3335_333512

theorem slope_range (a : ℝ) : 
  (∃ x y : ℝ, (a^2 + 2*a)*x - y + 1 = 0 ∧ a^2 + 2*a < 0) ↔ 
  -2 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_slope_range_l3335_333512


namespace NUMINAMATH_CALUDE_earliest_retirement_year_l3335_333566

/-- Represents the retirement eligibility rule -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Represents the employee's age in a given year -/
def age_in_year (hire_year : ℕ) (hire_age : ℕ) (current_year : ℕ) : ℕ :=
  hire_age + (current_year - hire_year)

/-- Represents the employee's years of employment in a given year -/
def years_employed (hire_year : ℕ) (current_year : ℕ) : ℕ :=
  current_year - hire_year

/-- Theorem stating the earliest retirement year for the employee -/
theorem earliest_retirement_year 
  (hire_year : ℕ) 
  (hire_age : ℕ) 
  (retirement_year : ℕ) :
  hire_year = 1987 →
  hire_age = 32 →
  retirement_year = 2006 →
  (∀ y : ℕ, y < retirement_year → 
    ¬(rule_of_70 (age_in_year hire_year hire_age y) (years_employed hire_year y))) →
  rule_of_70 (age_in_year hire_year hire_age retirement_year) (years_employed hire_year retirement_year) :=
by
  sorry


end NUMINAMATH_CALUDE_earliest_retirement_year_l3335_333566


namespace NUMINAMATH_CALUDE_solve_equation_l3335_333561

theorem solve_equation (a : ℚ) (h : 2 * a + a / 2 = 9 / 2) : a = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3335_333561


namespace NUMINAMATH_CALUDE_geometric_sequence_iff_c_eq_neg_one_l3335_333567

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) (c : ℝ) : ℝ := 2^n + c

/-- The n-th term of the sequence a_n -/
def a (n : ℕ) (c : ℝ) : ℝ := S n c - S (n-1) c

/-- Predicate to check if a sequence is geometric -/
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, n > 0 → a (n+1) = r * a n

theorem geometric_sequence_iff_c_eq_neg_one (c : ℝ) :
  is_geometric (a · c) ↔ c = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_iff_c_eq_neg_one_l3335_333567


namespace NUMINAMATH_CALUDE_largest_integer_prime_abs_quadratic_l3335_333531

theorem largest_integer_prime_abs_quadratic : 
  ∃ (x : ℤ), (∀ y : ℤ, y > x → ¬ Nat.Prime (Int.natAbs (4*y^2 - 39*y + 35))) ∧ 
  Nat.Prime (Int.natAbs (4*x^2 - 39*x + 35)) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_prime_abs_quadratic_l3335_333531


namespace NUMINAMATH_CALUDE_remove_seven_maintain_coverage_l3335_333595

/-- Represents a collection of objects covering a surface -/
structure CoveringSet (n : ℕ) :=
  (area : ℝ)
  (total_coverage : ℝ)
  (coverage : Fin n → ℝ)
  (covers_completely : total_coverage = area)
  (non_negative_coverage : ∀ i, coverage i ≥ 0)
  (sum_coverage : (Finset.sum Finset.univ coverage) = total_coverage)

/-- Theorem stating that it's possible to remove 7 objects from a set of 15
    such that the remaining 8 cover at least 8/15 of the total area -/
theorem remove_seven_maintain_coverage 
  (s : CoveringSet 15) : 
  ∃ (removed : Finset (Fin 15)), 
    Finset.card removed = 7 ∧ 
    (Finset.sum (Finset.univ \ removed) s.coverage) ≥ (8/15) * s.area := by
  sorry

end NUMINAMATH_CALUDE_remove_seven_maintain_coverage_l3335_333595
