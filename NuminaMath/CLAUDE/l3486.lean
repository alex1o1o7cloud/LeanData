import Mathlib

namespace NUMINAMATH_CALUDE_flight_departure_time_l3486_348645

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Adds a duration in minutes to a Time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

theorem flight_departure_time :
  let checkInTime : ℕ := 120 -- 2 hours in minutes
  let drivingTime : ℕ := 45
  let parkingTime : ℕ := 15
  let latestDepartureTime : Time := ⟨17, 0, sorry⟩ -- 5:00 pm
  let flightDepartureTime : Time := addMinutes latestDepartureTime (checkInTime + drivingTime + parkingTime)
  flightDepartureTime = ⟨20, 0, sorry⟩ -- 8:00 pm
:= by sorry

end NUMINAMATH_CALUDE_flight_departure_time_l3486_348645


namespace NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l3486_348614

theorem cube_tetrahedron_surface_area_ratio :
  let cube_side_length : ℝ := 2
  let tetrahedron_side_length : ℝ := 2 * Real.sqrt 2
  let cube_surface_area : ℝ := 6 * cube_side_length^2
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length^2
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l3486_348614


namespace NUMINAMATH_CALUDE_power_of_power_l3486_348616

theorem power_of_power (a : ℝ) : (a^5)^3 = a^15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3486_348616


namespace NUMINAMATH_CALUDE_expression_evaluation_l3486_348682

theorem expression_evaluation : 200 * (200 - 7) - (200 * 200 - 7 * 3) = -1379 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3486_348682


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_ABCD_l3486_348605

/-- Given points A, B, C, and D in a 2D coordinate system, prove that ABCD is an isosceles trapezoid with AB parallel to CD -/
theorem isosceles_trapezoid_ABCD :
  let A : ℝ × ℝ := (-6, -1)
  let B : ℝ × ℝ := (2, 3)
  let C : ℝ × ℝ := (-1, 4)
  let D : ℝ × ℝ := (-5, 2)
  
  -- AB is parallel to CD
  (B.2 - A.2) / (B.1 - A.1) = (D.2 - C.2) / (D.1 - C.1) ∧
  
  -- AD = BC (isosceles condition)
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
  
  -- AB ≠ CD (trapezoid condition)
  (B.1 - A.1)^2 + (B.2 - A.2)^2 ≠ (D.1 - C.1)^2 + (D.2 - C.2)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_trapezoid_ABCD_l3486_348605


namespace NUMINAMATH_CALUDE_flu_spread_l3486_348643

theorem flu_spread (initial_infected : ℕ) (total_infected : ℕ) (x : ℝ) : 
  initial_infected = 1 →
  total_infected = 81 →
  (1 + x)^2 = total_infected →
  x ≥ 0 →
  ∃ (y : ℝ), y = x ∧ (initial_infected : ℝ) + y + y^2 = total_infected :=
sorry

end NUMINAMATH_CALUDE_flu_spread_l3486_348643


namespace NUMINAMATH_CALUDE_total_match_sticks_l3486_348697

/-- Given the number of boxes ordered by Farrah -/
def num_boxes : ℕ := 4

/-- The number of matchboxes in each box -/
def matchboxes_per_box : ℕ := 20

/-- The number of sticks in each matchbox -/
def sticks_per_matchbox : ℕ := 300

/-- Theorem stating the total number of match sticks ordered by Farrah -/
theorem total_match_sticks : 
  num_boxes * matchboxes_per_box * sticks_per_matchbox = 24000 := by
  sorry

end NUMINAMATH_CALUDE_total_match_sticks_l3486_348697


namespace NUMINAMATH_CALUDE_complement_of_A_l3486_348626

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x < -1 ∨ 2 ≤ x} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3486_348626


namespace NUMINAMATH_CALUDE_derivative_of_f_l3486_348611

noncomputable def f (x : ℝ) : ℝ := 
  (1 / (2 * Real.sqrt 2)) * (Real.sin (Real.log x) - (Real.sqrt 2 - 1) * Real.cos (Real.log x)) * x^(Real.sqrt 2 + 1)

theorem derivative_of_f (x : ℝ) (h : x > 0) : 
  deriv f x = (x^(Real.sqrt 2) / (2 * Real.sqrt 2)) * 
    (2 * Real.cos (Real.log x) - Real.sqrt 2 * Real.cos (Real.log x) + 2 * Real.sqrt 2 * Real.sin (Real.log x)) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3486_348611


namespace NUMINAMATH_CALUDE_first_shift_participation_is_twenty_percent_l3486_348617

/-- Represents a company with three shifts of employees and a pension program. -/
structure Company where
  first_shift : ℕ
  second_shift : ℕ
  third_shift : ℕ
  second_shift_participation : ℚ
  third_shift_participation : ℚ
  total_participation : ℚ

/-- The percentage of first shift employees participating in the pension program. -/
def first_shift_participation (c : Company) : ℚ :=
  let total_employees := c.first_shift + c.second_shift + c.third_shift
  let total_participants := (c.total_participation * total_employees) / 100
  let second_shift_participants := (c.second_shift_participation * c.second_shift) / 100
  let third_shift_participants := (c.third_shift_participation * c.third_shift) / 100
  let first_shift_participants := total_participants - second_shift_participants - third_shift_participants
  (first_shift_participants * 100) / c.first_shift

theorem first_shift_participation_is_twenty_percent (c : Company) 
  (h1 : c.first_shift = 60)
  (h2 : c.second_shift = 50)
  (h3 : c.third_shift = 40)
  (h4 : c.second_shift_participation = 40)
  (h5 : c.third_shift_participation = 10)
  (h6 : c.total_participation = 24) :
  first_shift_participation c = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_shift_participation_is_twenty_percent_l3486_348617


namespace NUMINAMATH_CALUDE_green_face_probability_l3486_348663

/-- The probability of rolling a green face on a 10-sided die with 3 green faces is 3/10. -/
theorem green_face_probability (total_faces : ℕ) (green_faces : ℕ) 
  (h1 : total_faces = 10) (h2 : green_faces = 3) : 
  (green_faces : ℚ) / total_faces = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_green_face_probability_l3486_348663


namespace NUMINAMATH_CALUDE_cone_volume_l3486_348689

/-- Given a cone with generatrix length 2 and unfolded side sector area 2π, its volume is (√3 * π) / 3 -/
theorem cone_volume (generatrix : ℝ) (sector_area : ℝ) :
  generatrix = 2 →
  sector_area = 2 * Real.pi →
  ∃ (volume : ℝ), volume = (Real.sqrt 3 * Real.pi) / 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l3486_348689


namespace NUMINAMATH_CALUDE_student_handshake_problem_l3486_348620

/-- Given an m x n array of students where m, n ≥ 3, if each student shakes hands
    with adjacent students (horizontally, vertically, or diagonally) and there
    are 1020 handshakes in total, then the total number of students N is 140. -/
theorem student_handshake_problem (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (8 * m * n - 6 * m - 6 * n + 4) / 2 = 1020 →
  m * n = 140 := by
  sorry

#check student_handshake_problem

end NUMINAMATH_CALUDE_student_handshake_problem_l3486_348620


namespace NUMINAMATH_CALUDE_training_hours_calculation_l3486_348677

/-- Calculates the total training hours given daily training hours, initial training days, and additional training days. -/
def total_training_hours (daily_hours : ℕ) (initial_days : ℕ) (additional_days : ℕ) : ℕ :=
  daily_hours * (initial_days + additional_days)

/-- Theorem stating that training 5 hours daily for 30 days and continuing for 12 more days results in 210 total training hours. -/
theorem training_hours_calculation : total_training_hours 5 30 12 = 210 := by
  sorry

end NUMINAMATH_CALUDE_training_hours_calculation_l3486_348677


namespace NUMINAMATH_CALUDE_min_value_of_f_l3486_348641

def f (x : ℝ) : ℝ := x^2 + 14*x + 24

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3486_348641


namespace NUMINAMATH_CALUDE_absolute_value_sum_l3486_348657

theorem absolute_value_sum (x q : ℝ) : 
  |x - 5| = q ∧ x > 5 → x + q = 2*q + 5 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l3486_348657


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3486_348675

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (a 1 + a n) * n / 2

/-- Theorem: For an arithmetic sequence where S₁₅ - S₁₀ = 1, S₂₅ = 5 -/
theorem arithmetic_sequence_sum
  (seq : ArithmeticSequence)
  (h : seq.S 15 - seq.S 10 = 1) :
  seq.S 25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3486_348675


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3486_348668

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*x₁ - 9 = 0) ∧ (x₂^2 - 2*x₂ - 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3486_348668


namespace NUMINAMATH_CALUDE_train_fraction_is_four_fifths_l3486_348658

/-- Proves that the fraction of the journey traveled by train is 0.8 -/
theorem train_fraction_is_four_fifths
  (D : ℝ) -- Total distance
  (h_D_pos : D > 0) -- Assume distance is positive
  (train_speed : ℝ) -- Train speed
  (h_train_speed : train_speed = 80) -- Train speed is 80 mph
  (car_speed : ℝ) -- Car speed
  (h_car_speed : car_speed = 20) -- Car speed is 20 mph
  (avg_speed : ℝ) -- Average speed
  (h_avg_speed : avg_speed = 50) -- Average speed is 50 mph
  (x : ℝ) -- Fraction of journey by train
  (h_x_range : 0 ≤ x ∧ x ≤ 1) -- x is between 0 and 1
  (h_speed_equation : D / ((x * D / train_speed) + ((1 - x) * D / car_speed)) = avg_speed) -- Speed equation
  : x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_train_fraction_is_four_fifths_l3486_348658


namespace NUMINAMATH_CALUDE_managers_salary_l3486_348664

/-- Proves that given 24 employees with an average salary of Rs. 2400, 
    if adding a manager's salary increases the average by Rs. 100, 
    then the manager's salary is Rs. 4900. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) : 
  num_employees = 24 → 
  avg_salary = 2400 → 
  salary_increase = 100 → 
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - 
   num_employees * avg_salary) = 4900 := by
  sorry

#check managers_salary

end NUMINAMATH_CALUDE_managers_salary_l3486_348664


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3486_348618

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x - 3

-- State the theorem
theorem quadratic_function_properties :
  ∀ m : ℝ,
  (m > 0) →
  (∀ x : ℝ, f m x < 0 ↔ -1 < x ∧ x < 3) →
  (m = 1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 3 > 2 * x - 1 ↔ x < 1 ∨ x > 2) ∧
  (∃ a : ℝ, 0 < a ∧ a < 1 ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f m (a^x) - 4 * a^(x+1) ≥ -4) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f m (a^x) - 4 * a^(x+1) = -4) ∧
    a = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3486_348618


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3486_348649

/-- The quadratic equation mx^2 + x - m^2 + 1 = 0 has -1 as a root if and only if m = 1 -/
theorem quadratic_root_implies_m_value (m : ℝ) : 
  (m * (-1)^2 + (-1) - m^2 + 1 = 0) ↔ (m = 1) := by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3486_348649


namespace NUMINAMATH_CALUDE_ship_length_l3486_348621

/-- The length of a ship given its speed and time to pass a fixed point -/
theorem ship_length (speed : ℝ) (time : ℝ) (h1 : speed = 18) (h2 : time = 20) :
  speed * time * (1000 / 3600) = 100 := by
  sorry

#check ship_length

end NUMINAMATH_CALUDE_ship_length_l3486_348621


namespace NUMINAMATH_CALUDE_no_coin_exchange_solution_l3486_348693

theorem no_coin_exchange_solution : ¬∃ (x y z : ℕ), 
  x + y + z = 500 ∧ 
  36 * x + 6 * y + z = 3564 ∧ 
  x ≤ 99 := by
sorry

end NUMINAMATH_CALUDE_no_coin_exchange_solution_l3486_348693


namespace NUMINAMATH_CALUDE_orange_count_l3486_348640

/-- Represents the number of oranges in a basket -/
structure Basket where
  good : ℕ
  bad : ℕ

/-- Defines the ratio between good and bad oranges -/
def hasRatio (b : Basket) (g : ℕ) (d : ℕ) : Prop :=
  g * b.bad = d * b.good

theorem orange_count (b : Basket) (h1 : hasRatio b 3 1) (h2 : b.bad = 8) : b.good = 24 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_l3486_348640


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3486_348680

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum (seq.a ∘ Nat.succ)

/-- Main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  S seq 8 - S seq 3 = 10 → S seq 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3486_348680


namespace NUMINAMATH_CALUDE_q_div_p_equals_450_l3486_348650

def total_slips : ℕ := 50
def num_range : ℕ := 10
def slips_per_num : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := num_range / (total_slips.choose drawn_slips)
def q : ℚ := (num_range.choose 2) * (slips_per_num.choose 3) * (slips_per_num.choose 2) / (total_slips.choose drawn_slips)

theorem q_div_p_equals_450 : q / p = 450 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_450_l3486_348650


namespace NUMINAMATH_CALUDE_divisibility_by_thirty_l3486_348603

theorem divisibility_by_thirty (n : ℕ) (h_prime : Nat.Prime n) (h_geq_7 : n ≥ 7) :
  30 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_thirty_l3486_348603


namespace NUMINAMATH_CALUDE_valid_midpoint_on_hyperbola_l3486_348632

/-- The hyperbola equation --/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

/-- Definition of midpoint --/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

/-- Theorem stating that (-1,-4) is the only valid midpoint --/
theorem valid_midpoint_on_hyperbola :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_hyperbola x₁ y₁ ∧
    is_on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ ∧
    (∀ (x y : ℝ), (x, y) ∈ [(1, 1), (-1, 2), (1, 3)] →
      ¬∃ (x₁' y₁' x₂' y₂' : ℝ),
        is_on_hyperbola x₁' y₁' ∧
        is_on_hyperbola x₂' y₂' ∧
        is_midpoint x y x₁' y₁' x₂' y₂') :=
by sorry

end NUMINAMATH_CALUDE_valid_midpoint_on_hyperbola_l3486_348632


namespace NUMINAMATH_CALUDE_square_side_difference_l3486_348691

/-- Given four squares with side lengths s₁ ≥ s₂ ≥ s₃ ≥ s₄, prove that s₁ - s₄ = 29 -/
theorem square_side_difference (s₁ s₂ s₃ s₄ : ℝ) 
  (h₁ : s₁ ≥ s₂) (h₂ : s₂ ≥ s₃) (h₃ : s₃ ≥ s₄)
  (ab : s₁ - s₂ = 11) (cd : s₂ - s₃ = 5) (fe : s₃ - s₄ = 13) :
  s₁ - s₄ = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_side_difference_l3486_348691


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3486_348646

theorem sum_of_three_numbers (x y z : ℝ) : 
  z / x = 18.48 / 15.4 →
  z = 0.4 * y →
  x + y = 400 →
  x + y + z = 520 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3486_348646


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3486_348683

/-- Represents a triangle with side lengths and an angle -/
structure Triangle :=
  (sideAB : ℝ)
  (sideAC : ℝ)
  (sideBC : ℝ)
  (angleBAC : ℝ)

/-- Represents a circle with a radius -/
structure Circle :=
  (radius : ℝ)

/-- Calculates the area of two shaded regions in a specific geometric configuration -/
def shadedArea (t : Triangle) (c : Circle) : ℝ :=
  sorry

theorem shaded_area_calculation (t : Triangle) (c : Circle) :
  t.sideAB = 16 ∧ t.sideAC = 16 ∧ t.sideBC = c.radius * 2 ∧ t.angleBAC = 120 * π / 180 →
  shadedArea t c = 43 * π - 128 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3486_348683


namespace NUMINAMATH_CALUDE_ab_value_proof_l3486_348619

theorem ab_value_proof (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (2 - i) * (a - b * i) = (-8 - i) * i) : a * b = 42 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_proof_l3486_348619


namespace NUMINAMATH_CALUDE_product_closure_l3486_348638

-- Define the set A
def A : Set ℤ := {z | ∃ (a b k n : ℤ), z = a^2 + k*a*b + n*b^2}

-- State the theorem
theorem product_closure (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := by
  sorry

end NUMINAMATH_CALUDE_product_closure_l3486_348638


namespace NUMINAMATH_CALUDE_smallest_coprime_to_180_seven_coprime_to_180_seven_is_smallest_coprime_to_180_l3486_348688

theorem smallest_coprime_to_180 : ∀ x : ℕ, x > 1 ∧ x < 7 → Nat.gcd x 180 ≠ 1 :=
by
  sorry

theorem seven_coprime_to_180 : Nat.gcd 7 180 = 1 :=
by
  sorry

theorem seven_is_smallest_coprime_to_180 : ∀ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 → x ≥ 7 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_coprime_to_180_seven_coprime_to_180_seven_is_smallest_coprime_to_180_l3486_348688


namespace NUMINAMATH_CALUDE_double_and_square_reverse_digits_l3486_348692

/-- For any base greater than 2, doubling (base - 1) and squaring (base - 1) 
    result in numbers with the same digits in reverse order. -/
theorem double_and_square_reverse_digits (a : ℕ) (h : a > 2) :
  ∃ (d₁ d₂ : ℕ), d₁ < a ∧ d₂ < a ∧ 
  2 * (a - 1) = d₁ * a + d₂ ∧
  (a - 1)^2 = d₂ * a + d₁ :=
sorry

end NUMINAMATH_CALUDE_double_and_square_reverse_digits_l3486_348692


namespace NUMINAMATH_CALUDE_reading_time_comparison_l3486_348608

/-- Given two people A and B, where A reads 5 times faster than B,
    prove that if B takes 3 hours to read a book,
    then A will take 36 minutes to read the same book. -/
theorem reading_time_comparison (reading_speed_ratio : ℝ) (person_b_time : ℝ) :
  reading_speed_ratio = 5 →
  person_b_time = 3 →
  (person_b_time * 60) / reading_speed_ratio = 36 :=
by sorry

end NUMINAMATH_CALUDE_reading_time_comparison_l3486_348608


namespace NUMINAMATH_CALUDE_molly_rode_3285_miles_l3486_348676

/-- The number of miles Molly rode her bike from her 13th to 16th birthday -/
def molly_bike_miles : ℕ :=
  let start_age : ℕ := 13
  let end_age : ℕ := 16
  let years_riding : ℕ := end_age - start_age
  let days_per_year : ℕ := 365
  let miles_per_day : ℕ := 3
  years_riding * days_per_year * miles_per_day

/-- Theorem stating that Molly rode her bike for 3285 miles -/
theorem molly_rode_3285_miles : molly_bike_miles = 3285 := by
  sorry

end NUMINAMATH_CALUDE_molly_rode_3285_miles_l3486_348676


namespace NUMINAMATH_CALUDE_point_M_on_x_axis_l3486_348647

-- Define a point M with coordinates (a+2, a-3)
def M (a : ℝ) : ℝ × ℝ := (a + 2, a - 3)

-- Define what it means for a point to lie on the x-axis
def lies_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Theorem statement
theorem point_M_on_x_axis :
  ∀ a : ℝ, lies_on_x_axis (M a) → M a = (5, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_M_on_x_axis_l3486_348647


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l3486_348609

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored : ℚ) / (b.innings + 1 : ℚ)

/-- Theorem: If a batsman's average increases by 2 after scoring 80 in the 17th innings,
    then the new average is 48 -/
theorem batsman_average_theorem (b : Batsman) :
  b.innings = 16 →
  newAverage b 80 = b.average + 2 →
  newAverage b 80 = 48 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_batsman_average_theorem_l3486_348609


namespace NUMINAMATH_CALUDE_right_triangle_among_options_l3486_348653

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_among_options : 
  is_right_triangle 1 2 3 ∧ 
  ¬is_right_triangle 3 4 5 ∧ 
  ¬is_right_triangle 6 8 10 ∧ 
  ¬is_right_triangle 5 10 12 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_among_options_l3486_348653


namespace NUMINAMATH_CALUDE_apple_cost_theorem_l3486_348686

theorem apple_cost_theorem (cost_two_dozen : ℝ) (h : cost_two_dozen = 15.60) :
  let cost_per_dozen : ℝ := cost_two_dozen / 2
  let cost_four_dozen : ℝ := 4 * cost_per_dozen
  cost_four_dozen = 31.20 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_theorem_l3486_348686


namespace NUMINAMATH_CALUDE_hyperbola_proof_l3486_348651

def polar_equation (ρ φ : ℝ) : Prop := ρ = 36 / (4 - 5 * Real.cos φ)

theorem hyperbola_proof (ρ φ : ℝ) (h : polar_equation ρ φ) :
  ∃ (a b : ℝ), 
    (a = 16 ∧ b = 12) ∧ 
    (∃ (e : ℝ), e > 1 ∧ ρ = (e * (b^2 / a)) / (1 - e * Real.cos φ)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_proof_l3486_348651


namespace NUMINAMATH_CALUDE_math_festival_divisibility_l3486_348685

/-- The year of the first math festival -/
def first_festival_year : ℕ := 1990

/-- The base year for calculating festival years -/
def base_year : ℕ := 1989

/-- Predicate to check if a given ordinal number satisfies the divisibility condition -/
def satisfies_condition (N : ℕ) : Prop :=
  (base_year + N) % N = 0

theorem math_festival_divisibility :
  (∃ (first : ℕ), first > 0 ∧ satisfies_condition first ∧
    ∀ (k : ℕ), 0 < k ∧ k < first → ¬satisfies_condition k) ∧
  (∃ (last : ℕ), satisfies_condition last ∧
    ∀ (k : ℕ), k > last → ¬satisfies_condition k) :=
sorry

end NUMINAMATH_CALUDE_math_festival_divisibility_l3486_348685


namespace NUMINAMATH_CALUDE_sum_of_five_terms_l3486_348654

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_five_terms (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 + a 15 = 6 →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_five_terms_l3486_348654


namespace NUMINAMATH_CALUDE_f_explicit_formula_l3486_348678

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_explicit_formula 
  (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_period : has_period f 2)
  (h_known : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| :=
sorry

end NUMINAMATH_CALUDE_f_explicit_formula_l3486_348678


namespace NUMINAMATH_CALUDE_power_function_odd_l3486_348667

/-- A power function f(x) = (a-1)x^b passing through the point (a, 1/8) is odd. -/
theorem power_function_odd (a b : ℝ) (h : (a - 1) * a^b = 1/8) : 
  ∀ x ≠ 0, (a - 1) * (-x)^b = -((a - 1) * x^b) :=
by sorry

end NUMINAMATH_CALUDE_power_function_odd_l3486_348667


namespace NUMINAMATH_CALUDE_pens_given_to_sharon_problem_l3486_348604

/-- Calculates the number of pens given to Sharon -/
def pens_given_to_sharon (initial_pens : ℕ) (mike_pens : ℕ) (final_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - final_pens

theorem pens_given_to_sharon_problem :
  let initial_pens : ℕ := 5
  let mike_pens : ℕ := 20
  let final_pens : ℕ := 31
  pens_given_to_sharon initial_pens mike_pens final_pens = 19 := by
  sorry

#eval pens_given_to_sharon 5 20 31

end NUMINAMATH_CALUDE_pens_given_to_sharon_problem_l3486_348604


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l3486_348633

theorem sqrt_product_plus_one : 
  Real.sqrt ((25 : ℝ) * 24 * 23 * 22 + 1) = 551 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l3486_348633


namespace NUMINAMATH_CALUDE_line_parameterization_l3486_348679

/-- Given a line y = (3/4)x - 15 parameterized by (x,y) = (f(t), 20t - 10),
    prove that f(t) = (80/3)t + (20/3) -/
theorem line_parameterization (f : ℝ → ℝ) :
  (∀ x y, y = (3/4) * x - 15 ↔ ∃ t, x = f t ∧ y = 20 * t - 10) →
  f = λ t => (80/3) * t + 20/3 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l3486_348679


namespace NUMINAMATH_CALUDE_min_value_theorem_l3486_348636

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 4) :
  (1 / (4 - x) + 2 / x) ≥ (3 + 2 * Real.sqrt 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3486_348636


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3486_348615

/-- Given an arithmetic sequence with common difference 2 where a₁, a₂, a₅ form a geometric sequence, prove a₂ = 3 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 1 * a 5 = a 2 * a 2) →     -- a₁, a₂, a₅ form a geometric sequence
  a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3486_348615


namespace NUMINAMATH_CALUDE_tan_3x_eq_sin_x_solutions_l3486_348669

open Real

theorem tan_3x_eq_sin_x_solutions (x : ℝ) :
  ∃ (s : Finset ℝ), s.card = 12 ∧
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2*π ∧ tan (3*x) = sin x) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 2*π ∧ tan (3*y) = sin y → y ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_tan_3x_eq_sin_x_solutions_l3486_348669


namespace NUMINAMATH_CALUDE_convex_quad_interior_point_inequality_l3486_348601

/-- A convex quadrilateral with an interior point and parallel lines -/
structure ConvexQuadWithInteriorPoint where
  /-- The area of the convex quadrilateral ABCD -/
  T : ℝ
  /-- The area of quadrilateral AEPH -/
  t₁ : ℝ
  /-- The area of quadrilateral PFCG -/
  t₂ : ℝ
  /-- The areas are non-negative -/
  h₁ : 0 ≤ T
  h₂ : 0 ≤ t₁
  h₃ : 0 ≤ t₂

/-- The inequality holds for any convex quadrilateral with an interior point -/
theorem convex_quad_interior_point_inequality (q : ConvexQuadWithInteriorPoint) :
  Real.sqrt q.t₁ + Real.sqrt q.t₂ ≤ Real.sqrt q.T :=
sorry

end NUMINAMATH_CALUDE_convex_quad_interior_point_inequality_l3486_348601


namespace NUMINAMATH_CALUDE_third_number_is_41_l3486_348695

/-- A sequence of six numbers with specific properties -/
def GoldStickerSequence (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₁ = 29 ∧ 
  a₂ = 35 ∧ 
  a₄ = 47 ∧ 
  a₅ = 53 ∧ 
  a₆ = 59 ∧ 
  a₂ - a₁ = 6 ∧ 
  a₄ - a₂ = 12 ∧ 
  a₆ - a₄ = 12

theorem third_number_is_41 {a₁ a₂ a₃ a₄ a₅ a₆ : ℕ} 
  (h : GoldStickerSequence a₁ a₂ a₃ a₄ a₅ a₆) : a₃ = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_third_number_is_41_l3486_348695


namespace NUMINAMATH_CALUDE_n_equals_four_l3486_348600

theorem n_equals_four (n : ℝ) (h : 3 * n = 6 * 2) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_n_equals_four_l3486_348600


namespace NUMINAMATH_CALUDE_steven_seed_collection_l3486_348673

/-- Represents the number of seeds in different fruits -/
structure FruitSeeds where
  apple : ℕ
  pear : ℕ
  grape : ℕ

/-- Represents the number of fruits Steven has -/
structure FruitCount where
  apples : ℕ
  pears : ℕ
  grapes : ℕ

/-- Calculates the total number of seeds Steven needs to collect -/
def totalSeedsNeeded (avg : FruitSeeds) (count : FruitCount) (additional : ℕ) : ℕ :=
  avg.apple * count.apples + avg.pear * count.pears + avg.grape * count.grapes + additional

/-- Theorem: Steven needs to collect 60 seeds in total -/
theorem steven_seed_collection :
  let avg : FruitSeeds := ⟨6, 2, 3⟩
  let count : FruitCount := ⟨4, 3, 9⟩
  let additional : ℕ := 3
  totalSeedsNeeded avg count additional = 60 := by
  sorry


end NUMINAMATH_CALUDE_steven_seed_collection_l3486_348673


namespace NUMINAMATH_CALUDE_triangle_properties_l3486_348674

theorem triangle_properties (A B C : ℝ × ℝ) (S : ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2 = S →
  (C.1 - A.1) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3/5 →
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 2 →
  Real.tan (2 * Real.arctan ((B.2 - A.2) / (B.1 - A.1))) = -4/3 ∧
  S = 8/5 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3486_348674


namespace NUMINAMATH_CALUDE_min_cost_is_128_l3486_348690

/-- Represents the cost of each type of flower -/
structure FlowerCost where
  sunflower : ℕ
  tulip : ℕ
  orchid : ℚ
  rose : ℕ
  hydrangea : ℕ

/-- Represents the areas of different regions in the garden -/
structure GardenRegions where
  small_region1 : ℕ
  small_region2 : ℕ
  medium_region : ℕ
  large_region : ℕ

/-- Calculates the minimum cost of the garden given the flower costs and garden regions -/
def min_garden_cost (costs : FlowerCost) (regions : GardenRegions) : ℚ :=
  costs.sunflower * regions.small_region1 +
  costs.sunflower * regions.small_region2 +
  costs.tulip * regions.medium_region +
  costs.hydrangea * regions.large_region

theorem min_cost_is_128 (costs : FlowerCost) (regions : GardenRegions) :
  costs.sunflower = 1 ∧ 
  costs.tulip = 2 ∧ 
  costs.orchid = 5/2 ∧ 
  costs.rose = 3 ∧ 
  costs.hydrangea = 4 ∧
  regions.small_region1 = 8 ∧
  regions.small_region2 = 8 ∧
  regions.medium_region = 6 ∧
  regions.large_region = 25 →
  min_garden_cost costs regions = 128 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_is_128_l3486_348690


namespace NUMINAMATH_CALUDE_center_equidistant_from_hexagon_vertices_l3486_348696

/-- Represents a nickel coin -/
structure Nickel where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- States that a circle's diameter is equal to a nickel's diameter -/
def circle_diameter_eq_nickel (c : Circle) (n : Nickel) : Prop :=
  c.radius * 2 = n.diameter

/-- States that a hexagon is inscribed in a circle -/
def hexagon_inscribed_in_circle (h : RegularHexagon) (c : Circle) : Prop :=
  ∀ i : Fin 6, dist c.center (h.vertices i) = c.radius

/-- States that a hexagon can be constructed using three nickels -/
def hexagon_constructible_with_nickels (h : RegularHexagon) (n : Nickel) : Prop :=
  ∀ i : Fin 6, ∀ j : Fin 6, i ≠ j → dist (h.vertices i) (h.vertices j) = n.diameter

/-- The main theorem -/
theorem center_equidistant_from_hexagon_vertices
  (c : Circle) (n : Nickel) (h : RegularHexagon)
  (h1 : circle_diameter_eq_nickel c n)
  (h2 : hexagon_inscribed_in_circle h c)
  (h3 : hexagon_constructible_with_nickels h n) :
  ∀ i j : Fin 6, dist c.center (h.vertices i) = dist c.center (h.vertices j) := by
  sorry

end NUMINAMATH_CALUDE_center_equidistant_from_hexagon_vertices_l3486_348696


namespace NUMINAMATH_CALUDE_repeating_decimal_proof_l3486_348635

/-- The repeating decimal 0.76204̄ as a rational number -/
def repeating_decimal : ℚ := 761280 / 999000

theorem repeating_decimal_proof : repeating_decimal = 0.76 + (204 : ℚ) / 999000 := by sorry


end NUMINAMATH_CALUDE_repeating_decimal_proof_l3486_348635


namespace NUMINAMATH_CALUDE_y_value_l3486_348655

theorem y_value (m : ℕ) (y : ℝ) 
  (h1 : ((1 ^ m) / (y ^ m)) * ((1 ^ 16) / (4 ^ 16)) = 1 / (2 * (10 ^ 31)))
  (h2 : m = 31) : 
  y = 5 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3486_348655


namespace NUMINAMATH_CALUDE_hundred_mile_fare_l3486_348687

/-- Represents the cost of a taxi journey based on distance traveled -/
structure TaxiFare where
  /-- The distance traveled in miles -/
  distance : ℝ
  /-- The cost of the journey in dollars -/
  cost : ℝ

/-- Taxi fare is directly proportional to the distance traveled -/
axiom fare_proportional (d₁ d₂ c₁ c₂ : ℝ) :
  d₁ * c₂ = d₂ * c₁

theorem hundred_mile_fare (f : TaxiFare) (h : f.distance = 80 ∧ f.cost = 192) :
  ∃ (g : TaxiFare), g.distance = 100 ∧ g.cost = 240 :=
sorry

end NUMINAMATH_CALUDE_hundred_mile_fare_l3486_348687


namespace NUMINAMATH_CALUDE_ellipse_foci_l3486_348648

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 169 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(0, 12), (0, -12)}

-- Theorem statement
theorem ellipse_foci :
  ∀ (x y : ℝ), ellipse_equation x y →
  ∃ (f₁ f₂ : ℝ × ℝ), f₁ ∈ foci ∧ f₂ ∈ foci ∧
  (x - f₁.1)^2 + (y - f₁.2)^2 + (x - f₂.1)^2 + (y - f₂.2)^2 =
  4 * Real.sqrt (13^2 * 5^2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l3486_348648


namespace NUMINAMATH_CALUDE_even_expression_l3486_348698

theorem even_expression (n : ℕ) (h : n = 101) : Even (2 * n - 2) := by
  sorry

end NUMINAMATH_CALUDE_even_expression_l3486_348698


namespace NUMINAMATH_CALUDE_unique_pair_satisfying_inequality_l3486_348628

theorem unique_pair_satisfying_inequality :
  ∃! (a b : ℝ), ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    |Real.sqrt (1 - x^2) - a*x - b| ≤ (Real.sqrt 2 - 1) / 2 ∧ a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_satisfying_inequality_l3486_348628


namespace NUMINAMATH_CALUDE_square_area_is_two_l3486_348627

/-- A complex number z is a vertex of a square with z^2 and z^4 if it satisfies the equation z^3 - iz + i - 1 = 0 -/
def is_square_vertex (z : ℂ) : Prop :=
  z ≠ 0 ∧ z^3 - Complex.I * z + Complex.I - 1 = 0

/-- The area of a square formed by z, z^2, and z^4 in the complex plane -/
noncomputable def square_area (z : ℂ) : ℝ :=
  (1/2) * Complex.abs (z^4 - z)^2

theorem square_area_is_two (z : ℂ) (h : is_square_vertex z) :
  square_area z = 2 :=
sorry

end NUMINAMATH_CALUDE_square_area_is_two_l3486_348627


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3486_348666

noncomputable def f (x : ℝ) : ℝ := Real.log ((2 - x) / (2 + x))

theorem tangent_line_equation :
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := -4/3
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = -4/3 * x + Real.log 3 - 4/3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3486_348666


namespace NUMINAMATH_CALUDE_orange_juice_remaining_l3486_348642

theorem orange_juice_remaining (initial_amount : ℚ) (given_away : ℚ) : 
  initial_amount = 5 → given_away = 18/7 → initial_amount - given_away = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_remaining_l3486_348642


namespace NUMINAMATH_CALUDE_discount_difference_l3486_348637

/-- Proves that the difference between the claimed discount and the true discount is 9% -/
theorem discount_difference (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) :
  initial_discount = 0.4 →
  additional_discount = 0.1 →
  claimed_discount = 0.55 →
  claimed_discount - (1 - (1 - initial_discount) * (1 - additional_discount)) = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l3486_348637


namespace NUMINAMATH_CALUDE_min_value_quadratic_max_value_quadratic_l3486_348624

-- Question 1
theorem min_value_quadratic (m : ℝ) : m^2 - 6*m + 10 ≥ 1 := by sorry

-- Question 2
theorem max_value_quadratic (x : ℝ) : -2*x^2 - 4*x + 3 ≤ 5 := by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_max_value_quadratic_l3486_348624


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3486_348629

theorem expression_simplification_and_evaluation :
  let x : ℝ := 3 * Real.cos (60 * π / 180)
  let original_expression := (2 * x) / (x + 1) - (2 * x - 4) / (x^2 - 1) / ((x - 2) / (x^2 - 2 * x + 1))
  let simplified_expression := 4 / (x + 1)
  original_expression = simplified_expression ∧ simplified_expression = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3486_348629


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3486_348612

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def switch_outermost_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem unique_three_digit_number : 
  ∃! n : ℕ, is_three_digit n ∧ n + 1 = 2 * (switch_outermost_digits n) :=
by
  use 793
  sorry

#eval switch_outermost_digits 793
#eval 793 + 1 = 2 * (switch_outermost_digits 793)

end NUMINAMATH_CALUDE_unique_three_digit_number_l3486_348612


namespace NUMINAMATH_CALUDE_hotel_charge_comparison_l3486_348660

theorem hotel_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.55 * R) 
  (h2 : P = G - 0.1 * G) : 
  (R - G) / G = 1 := by
sorry

end NUMINAMATH_CALUDE_hotel_charge_comparison_l3486_348660


namespace NUMINAMATH_CALUDE_shape_to_square_transformation_exists_l3486_348671

/-- A shape on a graph paper --/
structure GraphShape where
  -- Add necessary fields to represent the shape

/-- A triangle on a graph paper --/
structure Triangle where
  -- Add necessary fields to represent a triangle

/-- A square on a graph paper --/
structure Square where
  -- Add necessary fields to represent a square

/-- Function to divide a shape into triangles --/
def divideIntoTriangles (shape : GraphShape) : List Triangle :=
  sorry

/-- Function to check if a list of triangles can form a square --/
def canFormSquare (triangles : List Triangle) : Bool :=
  sorry

/-- Theorem stating that there exists a shape that can be divided into 5 triangles
    which can be reassembled to form a square --/
theorem shape_to_square_transformation_exists :
  ∃ (shape : GraphShape),
    let triangles := divideIntoTriangles shape
    triangles.length = 5 ∧ canFormSquare triangles :=
by
  sorry

end NUMINAMATH_CALUDE_shape_to_square_transformation_exists_l3486_348671


namespace NUMINAMATH_CALUDE_install_remaining_windows_time_l3486_348665

/-- Calculates the time needed to install remaining windows -/
def time_to_install_remaining (total : ℕ) (installed : ℕ) (time_per_window : ℕ) : ℕ :=
  (total - installed) * time_per_window

/-- Proves that the time to install the remaining windows is 20 hours -/
theorem install_remaining_windows_time :
  time_to_install_remaining 10 6 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_install_remaining_windows_time_l3486_348665


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l3486_348699

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (x ≠ 1) →  -- ensure fraction is defined
  ((|x| - 1) / (x - 1) = 0) →
  x = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l3486_348699


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l3486_348613

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 4)^2 - 6*(a 4) + 5 = 0 →
  (a 8)^2 - 6*(a 8) + 5 = 0 →
  a 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l3486_348613


namespace NUMINAMATH_CALUDE_triangle_problem_l3486_348610

open Real

noncomputable def f (x : ℝ) := 2 * (cos x)^2 + sin (2*x - π/6)

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  f A = 3/2 →
  b + c = 2 →
  (∀ x, f x ≤ 2) ∧
  A = π/3 ∧
  1 ≤ a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3486_348610


namespace NUMINAMATH_CALUDE_least_multiplier_for_72_l3486_348606

theorem least_multiplier_for_72 (n : ℕ) : n = 62087668 ↔ 
  n > 0 ∧
  (∀ m : ℕ, m > 0 → m < n →
    (¬(112 ∣ (72 * m)) ∨
     ¬(199 ∣ (72 * m)) ∨
     ¬∃ k : ℕ, 72 * m = k * k)) ∧
  (112 ∣ (72 * n)) ∧
  (199 ∣ (72 * n)) ∧
  ∃ k : ℕ, 72 * n = k * k :=
sorry

end NUMINAMATH_CALUDE_least_multiplier_for_72_l3486_348606


namespace NUMINAMATH_CALUDE_output_is_fifteen_l3486_348694

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 2
  if step1 > 18 then step1 - 5 else step1 + 8

theorem output_is_fifteen : function_machine 10 = 15 := by sorry

end NUMINAMATH_CALUDE_output_is_fifteen_l3486_348694


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l3486_348634

theorem min_value_of_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x + 9 / y) ≥ 25 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 4 / x + 9 / y = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l3486_348634


namespace NUMINAMATH_CALUDE_product_of_trig_expressions_l3486_348607

theorem product_of_trig_expressions :
  (1 - Real.sin (π / 8)) * (1 - Real.sin (3 * π / 8)) *
  (1 + Real.sin (π / 8)) * (1 + Real.sin (3 * π / 8)) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_trig_expressions_l3486_348607


namespace NUMINAMATH_CALUDE_problem_statement_l3486_348631

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 + 2*a*b = -2) 
  (h2 : a*b - b^2 = -4) : 
  2*a^2 + (7/2)*a*b + (1/2)*b^2 = -2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3486_348631


namespace NUMINAMATH_CALUDE_expression_evaluation_l3486_348630

theorem expression_evaluation : (28 + 48 / 69) * 69 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3486_348630


namespace NUMINAMATH_CALUDE_continuity_at_one_l3486_348670

def f (x : ℝ) := -4 * x^2 - 7

theorem continuity_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_one_l3486_348670


namespace NUMINAMATH_CALUDE_john_boxes_l3486_348644

/-- The number of boxes each person has -/
structure Boxes where
  stan : ℕ
  joseph : ℕ
  jules : ℕ
  john : ℕ

/-- The conditions of the problem -/
def problem_conditions (b : Boxes) : Prop :=
  b.stan = 100 ∧
  b.joseph = b.stan - (80 * b.stan / 100) ∧
  b.jules = b.joseph + 5 ∧
  b.john > b.jules

/-- The theorem to prove -/
theorem john_boxes (b : Boxes) (h : problem_conditions b) : b.john = 30 := by
  sorry

end NUMINAMATH_CALUDE_john_boxes_l3486_348644


namespace NUMINAMATH_CALUDE_sum_of_digits_c_equals_five_l3486_348662

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def a : ℕ := sum_of_digits (4568^777)
def b : ℕ := sum_of_digits a
def c : ℕ := sum_of_digits b

theorem sum_of_digits_c_equals_five : c = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_c_equals_five_l3486_348662


namespace NUMINAMATH_CALUDE_minimum_room_size_for_table_l3486_348622

theorem minimum_room_size_for_table (table_length : ℝ) (table_width : ℝ) 
  (h1 : table_length = 12) (h2 : table_width = 9) : 
  ∃ (S : ℕ), S = 15 ∧ 
  (∀ (room_size : ℕ), (Real.sqrt (table_length^2 + table_width^2) ≤ room_size) ↔ (S ≤ room_size)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_room_size_for_table_l3486_348622


namespace NUMINAMATH_CALUDE_solution_existence_l3486_348681

theorem solution_existence (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) 
  (h : x + y + 2 * Real.sqrt (x * y) = 2017) :
  (x = 0 ∧ y = 2017) ∨ (x = 2017 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_l3486_348681


namespace NUMINAMATH_CALUDE_new_person_weight_l3486_348659

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 105 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3486_348659


namespace NUMINAMATH_CALUDE_mandy_book_ratio_l3486_348672

/-- Represents Mandy's book reading progression --/
structure BookReading where
  initial_length : ℕ
  initial_age : ℕ
  current_length : ℕ

/-- Calculates the ratio of book length at twice the starting age to initial book length --/
def length_ratio (r : BookReading) : ℚ :=
  let twice_age_length := r.initial_length * (r.current_length / (4 * 3 * r.initial_length))
  twice_age_length / r.initial_length

/-- Theorem stating the ratio of book length at twice Mandy's starting age to her initial book length --/
theorem mandy_book_ratio : 
  ∀ (r : BookReading), 
  r.initial_length = 8 ∧ 
  r.initial_age = 6 ∧ 
  r.current_length = 480 → 
  length_ratio r = 5 := by
  sorry

#eval length_ratio { initial_length := 8, initial_age := 6, current_length := 480 }

end NUMINAMATH_CALUDE_mandy_book_ratio_l3486_348672


namespace NUMINAMATH_CALUDE_john_savings_after_interest_l3486_348684

/-- The percentage of earnings John will have left after one year, given his spending habits and bank interest rate --/
theorem john_savings_after_interest
  (earnings : ℝ)
  (rent_percent : ℝ)
  (dishwasher_percent : ℝ)
  (groceries_percent : ℝ)
  (interest_rate : ℝ)
  (h1 : rent_percent = 0.4)
  (h2 : dishwasher_percent = 0.7 * rent_percent)
  (h3 : groceries_percent = 1.15 * rent_percent)
  (h4 : interest_rate = 0.05)
  : (1 - (rent_percent + dishwasher_percent + groceries_percent)) * (1 + interest_rate) = 0.903 := by
  sorry

end NUMINAMATH_CALUDE_john_savings_after_interest_l3486_348684


namespace NUMINAMATH_CALUDE_variance_2X_plus_1_l3486_348652

/-- A random variable following a Binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Variance of a Binomial distribution -/
def variance (X : BinomialDistribution) : ℝ :=
  X.n * X.p * (1 - X.p)

/-- Variance of a linear transformation of a random variable -/
def varianceLinearTransform (a b : ℝ) (X : BinomialDistribution) : ℝ :=
  a^2 * variance X

/-- Theorem: Variance of 2X+1 for X ~ B(10, 0.8) equals 6.4 -/
theorem variance_2X_plus_1 (X : BinomialDistribution) 
    (h2 : X.n = 10) (h3 : X.p = 0.8) : 
    varianceLinearTransform 2 1 X = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_variance_2X_plus_1_l3486_348652


namespace NUMINAMATH_CALUDE_least_coins_l3486_348661

theorem least_coins (n : ℕ) : 
  (n > 0) → 
  (n % 7 = 3) → 
  (n % 5 = 4) → 
  (∀ m : ℕ, m > 0 → m % 7 = 3 → m % 5 = 4 → n ≤ m) → 
  n = 24 :=
by sorry

end NUMINAMATH_CALUDE_least_coins_l3486_348661


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_equals_3850_l3486_348656

def prime_factorization := (2, 12) :: (3, 18) :: (5, 20) :: (7, 8) :: []

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def count_perfect_square_factors (factorization : List (ℕ × ℕ)) : ℕ :=
  factorization.foldl (fun acc (p, e) => acc * ((e / 2) + 1)) 1

theorem count_perfect_square_factors_equals_3850 :
  count_perfect_square_factors prime_factorization = 3850 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_equals_3850_l3486_348656


namespace NUMINAMATH_CALUDE_equation_simplification_l3486_348639

theorem equation_simplification (y : ℝ) (S : ℝ) :
  5 * (2 * y + 3 * Real.sqrt 3) = S →
  10 * (4 * y + 6 * Real.sqrt 3) = 4 * S := by
sorry

end NUMINAMATH_CALUDE_equation_simplification_l3486_348639


namespace NUMINAMATH_CALUDE_baba_yaga_students_l3486_348625

theorem baba_yaga_students (B G : ℕ) : 
  B + G = 33 →
  (2 * G + 2 * B) / 3 = 22 :=
by
  sorry

#check baba_yaga_students

end NUMINAMATH_CALUDE_baba_yaga_students_l3486_348625


namespace NUMINAMATH_CALUDE_E_equals_F_l3486_348623

def E : Set ℝ := {x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3)}

def F : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}

theorem E_equals_F : E = F := by sorry

end NUMINAMATH_CALUDE_E_equals_F_l3486_348623


namespace NUMINAMATH_CALUDE_cubic_expression_factorization_l3486_348602

theorem cubic_expression_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (-(x*y + x*z + y*z)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_factorization_l3486_348602
