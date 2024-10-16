import Mathlib

namespace NUMINAMATH_CALUDE_number_problem_l388_38830

theorem number_problem (a b c : ℕ) :
  Nat.gcd a b = 15 →
  Nat.gcd b c = 6 →
  b * c = 1800 →
  Nat.lcm a b = 3150 →
  a = 315 ∧ b = 150 ∧ c = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l388_38830


namespace NUMINAMATH_CALUDE_two_numbers_with_110_divisors_and_nine_zeros_sum_l388_38876

/-- A number ends with 9 zeros if it's divisible by 10^9 -/
def ends_with_nine_zeros (n : ℕ) : Prop := n % (10^9) = 0

/-- Count the number of divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The main theorem -/
theorem two_numbers_with_110_divisors_and_nine_zeros_sum :
  ∃ (a b : ℕ), a ≠ b ∧
                ends_with_nine_zeros a ∧
                ends_with_nine_zeros b ∧
                count_divisors a = 110 ∧
                count_divisors b = 110 ∧
                a + b = 7000000000 := by sorry

end NUMINAMATH_CALUDE_two_numbers_with_110_divisors_and_nine_zeros_sum_l388_38876


namespace NUMINAMATH_CALUDE_shooting_statistics_l388_38848

def scores : List ℕ := [7, 5, 8, 9, 6, 6, 7, 7, 8, 7]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

theorem shooting_statistics :
  mode scores = 7 ∧
  median scores = 7 ∧
  mean scores = 7 ∧
  variance scores = 6/5 := by sorry

end NUMINAMATH_CALUDE_shooting_statistics_l388_38848


namespace NUMINAMATH_CALUDE_student_lecture_choices_l388_38837

/-- The number of different choices when n students can each independently
    choose one of m lectures to attend -/
def number_of_choices (n m : ℕ) : ℕ := m^n

/-- Theorem: Given 5 students and 3 lectures, where each student can independently
    choose one lecture to attend, the total number of different possible choices is 3^5 -/
theorem student_lecture_choices :
  number_of_choices 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_student_lecture_choices_l388_38837


namespace NUMINAMATH_CALUDE_AAA_not_sufficient_for_congruence_l388_38856

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_angles : α + β + γ = π

-- Define triangle congruence
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define AAA criterion
def AAA_equal (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem: AAA criterion is not sufficient for triangle congruence
theorem AAA_not_sufficient_for_congruence :
  ¬(∀ (t1 t2 : Triangle), AAA_equal t1 t2 → congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_AAA_not_sufficient_for_congruence_l388_38856


namespace NUMINAMATH_CALUDE_first_half_speed_l388_38814

/-- Proves that given a trip of 8 hours, where the second half is traveled at 85 km/h, 
    and the total distance is 620 km, the speed during the first half of the trip is 70 km/h. -/
theorem first_half_speed 
  (total_time : ℝ) 
  (second_half_speed : ℝ) 
  (total_distance : ℝ) 
  (h1 : total_time = 8) 
  (h2 : second_half_speed = 85) 
  (h3 : total_distance = 620) : 
  (total_distance - (second_half_speed * (total_time / 2))) / (total_time / 2) = 70 := by
sorry

end NUMINAMATH_CALUDE_first_half_speed_l388_38814


namespace NUMINAMATH_CALUDE_total_spent_on_cards_l388_38844

def digimon_pack_price : ℚ := 4.45
def digimon_pack_count : ℕ := 4
def baseball_deck_price : ℚ := 6.06

theorem total_spent_on_cards :
  digimon_pack_price * digimon_pack_count + baseball_deck_price = 23.86 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_cards_l388_38844


namespace NUMINAMATH_CALUDE_grasshoppers_on_plant_count_l388_38874

def total_grasshoppers : ℕ := 31
def baby_grasshoppers_dozens : ℕ := 2

def grasshoppers_on_plant : ℕ := total_grasshoppers - (baby_grasshoppers_dozens * 12)

theorem grasshoppers_on_plant_count : grasshoppers_on_plant = 7 := by
  sorry

end NUMINAMATH_CALUDE_grasshoppers_on_plant_count_l388_38874


namespace NUMINAMATH_CALUDE_fraction_value_l388_38842

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l388_38842


namespace NUMINAMATH_CALUDE_teacher_class_choices_l388_38893

theorem teacher_class_choices (n_teachers : ℕ) (n_classes : ℕ) : 
  n_teachers = 5 → n_classes = 4 → (n_classes : ℕ) ^ n_teachers = 4^5 := by
  sorry

end NUMINAMATH_CALUDE_teacher_class_choices_l388_38893


namespace NUMINAMATH_CALUDE_inequality_system_solution_l388_38826

theorem inequality_system_solution (x : ℝ) :
  x - 2 ≤ 0 ∧ (x - 1) / 2 < x → -1 < x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l388_38826


namespace NUMINAMATH_CALUDE_dividend_calculation_l388_38883

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 8)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 141 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l388_38883


namespace NUMINAMATH_CALUDE_triangle_sine_sum_zero_l388_38853

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  
-- State the theorem
theorem triangle_sine_sum_zero (t : Triangle) : 
  t.a^3 * Real.sin (t.B - t.C) + t.b^3 * Real.sin (t.C - t.A) + t.c^3 * Real.sin (t.A - t.B) = 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_zero_l388_38853


namespace NUMINAMATH_CALUDE_bryans_books_l388_38838

/-- Calculates the total number of books given the number of bookshelves and books per shelf. -/
def total_books (num_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  num_shelves * books_per_shelf

/-- Theorem stating that Bryan's total number of books is 504. -/
theorem bryans_books : 
  total_books 9 56 = 504 := by
  sorry

end NUMINAMATH_CALUDE_bryans_books_l388_38838


namespace NUMINAMATH_CALUDE_sin_45_degrees_l388_38867

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l388_38867


namespace NUMINAMATH_CALUDE_data_transmission_time_l388_38869

/-- Represents the number of blocks of data to be sent -/
def num_blocks : ℕ := 100

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 450

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 200

/-- Represents the number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem stating that the time to send the data is 0.0625 hours -/
theorem data_transmission_time :
  (num_blocks * chunks_per_block : ℚ) / transmission_rate / seconds_per_hour = 0.0625 := by
  sorry

end NUMINAMATH_CALUDE_data_transmission_time_l388_38869


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l388_38810

theorem joshua_bottle_caps (initial bought given_away : ℕ) : 
  initial = 150 → bought = 23 → given_away = 37 → 
  initial + bought - given_away = 136 := by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l388_38810


namespace NUMINAMATH_CALUDE_ricks_ironing_rate_l388_38851

/-- The number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := sorry

/-- The number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := 4

/-- The number of hours Rick spent ironing dress shirts -/
def shirt_hours : ℕ := 3

/-- The number of hours Rick spent ironing dress pants -/
def pant_hours : ℕ := 5

/-- The total number of pieces of clothing Rick ironed -/
def total_pieces : ℕ := 27

theorem ricks_ironing_rate :
  shirts_per_hour * shirt_hours + pants_per_hour * pant_hours = total_pieces ∧
  pants_per_hour = 3 :=
sorry

end NUMINAMATH_CALUDE_ricks_ironing_rate_l388_38851


namespace NUMINAMATH_CALUDE_supermarket_profit_l388_38866

/-- Represents the daily sales quantity as a function of the selling price. -/
def sales_quantity (x : ℤ) : ℤ := -5 * x + 150

/-- Represents the daily profit as a function of the selling price. -/
def daily_profit (x : ℤ) : ℤ := (x - 8) * (sales_quantity x)

theorem supermarket_profit (x : ℤ) (h1 : 8 ≤ x) (h2 : x ≤ 15) :
  (daily_profit 14 = 480) ∧
  (∀ y : ℤ, 8 ≤ y → y ≤ 15 → daily_profit y ≤ daily_profit 15) ∧
  (daily_profit 15 = 525) :=
sorry


end NUMINAMATH_CALUDE_supermarket_profit_l388_38866


namespace NUMINAMATH_CALUDE_A_oxen_count_l388_38818

/-- Represents the number of oxen A put for grazing -/
def X : ℕ := sorry

/-- Total rent of the pasture in Rs -/
def total_rent : ℕ := 175

/-- Number of months A's oxen grazed -/
def A_months : ℕ := 7

/-- Number of oxen B put for grazing -/
def B_oxen : ℕ := 12

/-- Number of months B's oxen grazed -/
def B_months : ℕ := 5

/-- Number of oxen C put for grazing -/
def C_oxen : ℕ := 15

/-- Number of months C's oxen grazed -/
def C_months : ℕ := 3

/-- C's share of rent in Rs -/
def C_share : ℕ := 45

/-- Theorem stating that A put 10 oxen for grazing -/
theorem A_oxen_count : X = 10 := by sorry

end NUMINAMATH_CALUDE_A_oxen_count_l388_38818


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l388_38832

theorem real_part_of_complex_product : 
  let z : ℂ := (1 + Complex.I) * (1 + 2 * Complex.I)
  Complex.re z = -1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l388_38832


namespace NUMINAMATH_CALUDE_profit_percentage_is_twenty_l388_38880

/-- Calculates the percentage profit on wholesale price given wholesale price, retail price, and discount percentage. -/
def percentage_profit (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price / 100
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that given the specific values in the problem, the percentage profit is 20%. -/
theorem profit_percentage_is_twenty :
  percentage_profit 108 144 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_twenty_l388_38880


namespace NUMINAMATH_CALUDE_range_of_R_l388_38884

/-- The polar equation of curve C1 is ρ = R (R > 0) -/
def C1 (R : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = R^2 ∧ R > 0}

/-- The parametric equation of curve C2 is x = 2 + sin²α, y = sin²α -/
def C2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ α : ℝ, p.1 = 2 + Real.sin α ^ 2 ∧ p.2 = Real.sin α ^ 2}

/-- C1 and C2 have common points -/
def have_common_points (R : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ C1 R ∧ p ∈ C2

theorem range_of_R :
  ∀ R : ℝ, have_common_points R ↔ 2 ≤ R ∧ R ≤ Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_range_of_R_l388_38884


namespace NUMINAMATH_CALUDE_messages_in_week_after_removal_l388_38825

/-- Calculates the total number of messages sent in a week by remaining members of a group after some members were removed. -/
def total_messages_in_week (initial_members : ℕ) (removed_members : ℕ) (messages_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  (initial_members - removed_members) * messages_per_day * days_in_week

/-- Proves that the total number of messages sent in a week by remaining members is 45500, given the specified conditions. -/
theorem messages_in_week_after_removal :
  total_messages_in_week 150 20 50 7 = 45500 := by
  sorry

end NUMINAMATH_CALUDE_messages_in_week_after_removal_l388_38825


namespace NUMINAMATH_CALUDE_earliest_saturday_after_second_monday_after_second_thursday_l388_38806

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the next date -/
def nextDate (d : Date) : Date :=
  { day := d.day + 1, dayOfWeek := nextDay d.dayOfWeek }

/-- Finds the nth occurrence of a specific day of the week, starting from a given date -/
def findNthDay (start : Date) (target : DayOfWeek) (n : Nat) : Date :=
  sorry

/-- Finds the first occurrence of a specific day of the week, starting from a given date -/
def findNextDay (start : Date) (target : DayOfWeek) : Date :=
  sorry

/-- Main theorem: The earliest possible date for the first Saturday after the second Monday 
    following the second Thursday of any month is the 17th -/
theorem earliest_saturday_after_second_monday_after_second_thursday (startDate : Date) : 
  (findNextDay 
    (findNthDay 
      (findNthDay startDate DayOfWeek.Thursday 2) 
      DayOfWeek.Monday 
      2) 
    DayOfWeek.Saturday).day ≥ 17 :=
  sorry

end NUMINAMATH_CALUDE_earliest_saturday_after_second_monday_after_second_thursday_l388_38806


namespace NUMINAMATH_CALUDE_lcm_and_prime_factorization_l388_38831

theorem lcm_and_prime_factorization :
  let a := 48
  let b := 180
  let c := 250
  let lcm_result := Nat.lcm (Nat.lcm a b) c
  lcm_result = 18000 ∧ 
  18000 = 2^4 * 3^2 * 5^3 := by
sorry

end NUMINAMATH_CALUDE_lcm_and_prime_factorization_l388_38831


namespace NUMINAMATH_CALUDE_sum_of_extrema_x_l388_38820

theorem sum_of_extrema_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ x', ∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 8/3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_extrema_x_l388_38820


namespace NUMINAMATH_CALUDE_certain_number_proof_l388_38811

theorem certain_number_proof : 
  ∃ x : ℕ, (7899665 : ℕ) - (12 * 3 * x) = 7899593 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l388_38811


namespace NUMINAMATH_CALUDE_sqrt_25_l388_38862

theorem sqrt_25 : {x : ℝ | x^2 = 25} = {5, -5} := by sorry

end NUMINAMATH_CALUDE_sqrt_25_l388_38862


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_schools_l388_38809

theorem stratified_sampling_middle_schools 
  (total_schools : ℕ) 
  (high_schools : ℕ) 
  (middle_schools : ℕ) 
  (elementary_schools : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_schools = high_schools + middle_schools + elementary_schools)
  (h2 : total_schools = 100)
  (h3 : high_schools = 10)
  (h4 : middle_schools = 30)
  (h5 : elementary_schools = 60)
  (h6 : sample_size = 20) :
  (middle_schools : ℚ) * sample_size / total_schools = 6 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_schools_l388_38809


namespace NUMINAMATH_CALUDE_negation_of_proposition_l388_38870

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + 1 < 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l388_38870


namespace NUMINAMATH_CALUDE_one_isosceles_triangle_l388_38895

-- Define a point in 2D space
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a triangle by its three vertices
structure Triangle :=
  (v1 : Point)
  (v2 : Point)
  (v3 : Point)

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := squaredDistance t.v1 t.v2
  let d2 := squaredDistance t.v2 t.v3
  let d3 := squaredDistance t.v3 t.v1
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the four triangles
def triangle1 : Triangle := ⟨⟨0, 7⟩, ⟨3, 7⟩, ⟨1, 5⟩⟩
def triangle2 : Triangle := ⟨⟨4, 5⟩, ⟨4, 7⟩, ⟨6, 5⟩⟩
def triangle3 : Triangle := ⟨⟨0, 2⟩, ⟨3, 3⟩, ⟨7, 2⟩⟩
def triangle4 : Triangle := ⟨⟨11, 5⟩, ⟨10, 7⟩, ⟨12, 5⟩⟩

-- Theorem: Exactly one of the four triangles is isosceles
theorem one_isosceles_triangle :
  (isIsosceles triangle1 ∨ isIsosceles triangle2 ∨ isIsosceles triangle3 ∨ isIsosceles triangle4) ∧
  ¬(isIsosceles triangle1 ∧ isIsosceles triangle2) ∧
  ¬(isIsosceles triangle1 ∧ isIsosceles triangle3) ∧
  ¬(isIsosceles triangle1 ∧ isIsosceles triangle4) ∧
  ¬(isIsosceles triangle2 ∧ isIsosceles triangle3) ∧
  ¬(isIsosceles triangle2 ∧ isIsosceles triangle4) ∧
  ¬(isIsosceles triangle3 ∧ isIsosceles triangle4) :=
sorry

end NUMINAMATH_CALUDE_one_isosceles_triangle_l388_38895


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l388_38873

/-- The function f(x) = x³ - 3ax² + 3bx -/
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*b*x

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 6*a*x + 3*b

theorem tangent_line_and_extrema :
  ∃ (a b : ℝ),
    /- f(x) is tangent to 12x + y - 1 = 0 at (1, -11) -/
    (f a b 1 = -11 ∧ f_derivative a b 1 = -12) ∧
    /- a = 1 and b = -3 -/
    (a = 1 ∧ b = -3) ∧
    /- Maximum value of f(x) in [-2, 4] is 5 -/
    (∀ x, x ∈ Set.Icc (-2) 4 → f a b x ≤ 5) ∧
    (∃ x, x ∈ Set.Icc (-2) 4 ∧ f a b x = 5) ∧
    /- Minimum value of f(x) in [-2, 4] is -27 -/
    (∀ x, x ∈ Set.Icc (-2) 4 → f a b x ≥ -27) ∧
    (∃ x, x ∈ Set.Icc (-2) 4 ∧ f a b x = -27) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l388_38873


namespace NUMINAMATH_CALUDE_cone_base_circumference_l388_38812

/-- 
Given a right circular cone with volume 27π cubic centimeters and height 9 cm,
prove that the circumference of the base is 6π cm.
-/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 27 * Real.pi ∧ h = 9 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l388_38812


namespace NUMINAMATH_CALUDE_range_of_g_bounds_achievable_l388_38872

theorem range_of_g (x : ℝ) : ∃ (y : ℝ), y ∈ Set.Icc (3/4 : ℝ) 1 ∧ y = Real.cos x ^ 4 + Real.sin x ^ 2 :=
sorry

theorem bounds_achievable :
  (∃ (x : ℝ), Real.cos x ^ 4 + Real.sin x ^ 2 = 3/4) ∧
  (∃ (x : ℝ), Real.cos x ^ 4 + Real.sin x ^ 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_g_bounds_achievable_l388_38872


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l388_38854

def anna_lap_time : ℕ := 5
def stephanie_lap_time : ℕ := 8
def james_lap_time : ℕ := 9
def tom_lap_time : ℕ := 10

theorem earliest_meeting_time :
  let lap_times := [anna_lap_time, stephanie_lap_time, james_lap_time, tom_lap_time]
  Nat.lcm (Nat.lcm (Nat.lcm anna_lap_time stephanie_lap_time) james_lap_time) tom_lap_time = 360 :=
by sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l388_38854


namespace NUMINAMATH_CALUDE_count_permutations_2007_l388_38887

/-- The number of permutations of integers 1 to n with exactly one descent -/
def permutations_with_one_descent (n : ℕ) : ℕ :=
  2^n - (n + 1)

/-- The theorem to be proved -/
theorem count_permutations_2007 :
  permutations_with_one_descent 2007 = 2^3 * (2^2004 - 251) := by
  sorry

end NUMINAMATH_CALUDE_count_permutations_2007_l388_38887


namespace NUMINAMATH_CALUDE_percentage_failed_english_l388_38885

theorem percentage_failed_english (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) 
  (h_total : total = 100)
  (h_failed_hindi : failed_hindi = 32)
  (h_failed_both : failed_both = 12)
  (h_passed_both : passed_both = 24) :
  ∃ failed_english : ℝ, failed_english = 56 ∧ 
  total - (failed_hindi + failed_english - failed_both) = passed_both :=
by sorry

end NUMINAMATH_CALUDE_percentage_failed_english_l388_38885


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l388_38898

/-- Represents a rectangular piece of plywood -/
structure Plywood where
  length : ℝ
  width : ℝ

/-- Represents a cut of the plywood into congruent rectangles -/
structure Cut where
  num_pieces : ℕ
  piece_length : ℝ
  piece_width : ℝ

/-- Calculate the perimeter of a rectangular piece -/
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

/-- Check if a cut is valid for a given plywood -/
def is_valid_cut (p : Plywood) (c : Cut) : Prop :=
  c.num_pieces * c.piece_length = p.length ∧ 
  c.num_pieces * c.piece_width = p.width

/-- The main theorem -/
theorem plywood_cut_perimeter_difference 
  (p : Plywood) 
  (h1 : p.length = 10 ∧ p.width = 5) 
  (h2 : ∃ c : Cut, is_valid_cut p c ∧ c.num_pieces = 5) :
  ∃ (max_perim min_perim : ℝ),
    (∀ c : Cut, is_valid_cut p c ∧ c.num_pieces = 5 → 
      perimeter c.piece_length c.piece_width ≤ max_perim) ∧
    (∀ c : Cut, is_valid_cut p c ∧ c.num_pieces = 5 → 
      perimeter c.piece_length c.piece_width ≥ min_perim) ∧
    max_perim - min_perim = 8 := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l388_38898


namespace NUMINAMATH_CALUDE_linear_dependence_iff_k_eq_8_l388_38801

def vector1 : ℝ × ℝ × ℝ := (1, 4, -1)
def vector2 (k : ℝ) : ℝ × ℝ × ℝ := (2, k, 3)

def is_linearly_dependent (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 
    c1 • v1 + c2 • v2 = (0, 0, 0)

theorem linear_dependence_iff_k_eq_8 :
  ∀ k : ℝ, is_linearly_dependent vector1 (vector2 k) ↔ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_linear_dependence_iff_k_eq_8_l388_38801


namespace NUMINAMATH_CALUDE_multiple_of_smaller_integer_l388_38890

theorem multiple_of_smaller_integer (s l : ℤ) (k : ℚ) : 
  s + l = 30 → 
  s = 10 → 
  2 * l = k * s - 10 → 
  k = 5 := by sorry

end NUMINAMATH_CALUDE_multiple_of_smaller_integer_l388_38890


namespace NUMINAMATH_CALUDE_johns_earnings_ratio_l388_38855

def saturday_earnings : ℤ := 18
def previous_weekend_earnings : ℤ := 20
def pogo_stick_cost : ℤ := 60
def additional_needed : ℤ := 13

def total_earnings : ℤ := pogo_stick_cost - additional_needed

theorem johns_earnings_ratio :
  let sunday_earnings := total_earnings - saturday_earnings - previous_weekend_earnings
  saturday_earnings / sunday_earnings = 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_earnings_ratio_l388_38855


namespace NUMINAMATH_CALUDE_x_value_theorem_l388_38891

theorem x_value_theorem (x n : ℕ) : 
  x = 2^n - 32 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 3 ∧ q ≠ 3 ∧
    x = 3 * p * q) →
  x = 480 ∨ x = 2016 := by
sorry

end NUMINAMATH_CALUDE_x_value_theorem_l388_38891


namespace NUMINAMATH_CALUDE_fundraising_excess_l388_38857

/-- Proves that Scott, Mary, and Ken exceeded their fundraising goal by $600 --/
theorem fundraising_excess (ken : ℕ) (mary scott : ℕ) (goal : ℕ) : 
  ken = 600 →
  mary = 5 * ken →
  mary = 3 * scott →
  goal = 4000 →
  mary + scott + ken - goal = 600 := by
sorry

end NUMINAMATH_CALUDE_fundraising_excess_l388_38857


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l388_38835

theorem quadratic_expression_value (x : ℝ) : 2 * x^2 + 3 * x - 1 = 7 → 4 * x^2 + 6 * x + 9 = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l388_38835


namespace NUMINAMATH_CALUDE_f_zero_and_range_l388_38845

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

-- State the theorem
theorem f_zero_and_range :
  -- f(x) has one zero in (-1, 1)
  ∃ (x : ℝ), -1 < x ∧ x < 1 ∧ f a x = 0 →
  -- The range of a
  (12 * (27 - 4 * Real.sqrt 6) / 211 ≤ a ∧ a ≤ 12 * (27 + 4 * Real.sqrt 6) / 211) ∧
  -- When a = 32/17, the solution is 1/2
  (a = 32/17 → f (32/17) (1/2) = 0) :=
sorry


end NUMINAMATH_CALUDE_f_zero_and_range_l388_38845


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l388_38882

/-- The volume of a rectangular solid with given face areas and a dimension relation -/
theorem rectangular_solid_volume (a b c : ℝ) 
  (side_area : a * b = 15)
  (front_area : b * c = 10)
  (top_area : a * c = 6)
  (dimension_relation : b = 2 * a ∨ a = 2 * b ∨ c = 2 * a ∨ a = 2 * c ∨ c = 2 * b ∨ b = 2 * c) :
  a * b * c = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l388_38882


namespace NUMINAMATH_CALUDE_roots_problem_l388_38863

theorem roots_problem :
  (∀ x : ℝ, x ^ 2 = 0 → x = 0) ∧
  (∃ x : ℝ, x ≥ 0 ∧ x ^ 2 = 9 ∧ ∀ y : ℝ, y ≥ 0 ∧ y ^ 2 = 9 → x = y) ∧
  (∃ x : ℝ, x ^ 3 = (64 : ℝ).sqrt ∧ ∀ y : ℝ, y ^ 3 = (64 : ℝ).sqrt → x = y) :=
by sorry

end NUMINAMATH_CALUDE_roots_problem_l388_38863


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_5_l388_38843

theorem smallest_perfect_square_divisible_by_4_and_5 :
  ∀ n : ℕ, n > 0 → n.sqrt ^ 2 = n → n % 4 = 0 → n % 5 = 0 → n ≥ 400 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_5_l388_38843


namespace NUMINAMATH_CALUDE_journey_distance_correct_total_distance_l388_38802

-- Define the journey parameters
def total_time : ℝ := 30
def speed_first_half : ℝ := 20
def speed_second_half : ℝ := 10

-- Define the total distance
def total_distance : ℝ := 400

-- Theorem statement
theorem journey_distance :
  (total_distance / 2 / speed_first_half) + (total_distance / 2 / speed_second_half) = total_time :=
by sorry

-- Proof that the total distance is correct
theorem correct_total_distance : total_distance = 400 :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_correct_total_distance_l388_38802


namespace NUMINAMATH_CALUDE_line_equation_correct_l388_38833

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Check if a point (x, y) is on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y - l.point.2 = l.slope * (x - l.point.1)

/-- The specific line l with slope 2 passing through (2, -1) -/
def l : Line :=
  { slope := 2
  , point := (2, -1) }

/-- Theorem: The equation 2x - y - 5 = 0 represents the line l -/
theorem line_equation_correct :
  ∀ x y : ℝ, l.contains x y ↔ 2 * x - y - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_correct_l388_38833


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_210_l388_38840

theorem greatest_prime_factor_of_210 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 210 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 210 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_210_l388_38840


namespace NUMINAMATH_CALUDE_polynomial_simplification_l388_38886

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + y^10 + 2 * y^9) =
  15 * y^13 - y^12 - 3 * y^11 + 4 * y^10 - 4 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l388_38886


namespace NUMINAMATH_CALUDE_mixture_composition_l388_38815

/-- Represents a seed mixture --/
structure SeedMixture where
  ryegrass : ℝ
  other : ℝ
  sum_to_one : ryegrass + other = 1

/-- The final mixture of X and Y --/
def final_mixture (x y : SeedMixture) (p : ℝ) : SeedMixture :=
  { ryegrass := p * x.ryegrass + (1 - p) * y.ryegrass,
    other := p * x.other + (1 - p) * y.other,
    sum_to_one := by sorry }

theorem mixture_composition 
  (x : SeedMixture)
  (y : SeedMixture)
  (hx : x.ryegrass = 0.4)
  (hy : y.ryegrass = 0.25)
  : ∃ p : ℝ, 
    0 ≤ p ∧ p ≤ 1 ∧ 
    (final_mixture x y p).ryegrass = 0.38 ∧
    abs (p - 0.8667) < 0.0001 := by sorry

end NUMINAMATH_CALUDE_mixture_composition_l388_38815


namespace NUMINAMATH_CALUDE_T_formula_l388_38807

def T : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * T (n + 2) - 4 * (n + 3) * T (n + 1) + (4 * (n + 3) - 8) * T n

theorem T_formula (n : ℕ) : T n = n.factorial + 2^n := by
  sorry

end NUMINAMATH_CALUDE_T_formula_l388_38807


namespace NUMINAMATH_CALUDE_horner_rule_v4_l388_38836

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  v3 * x + 240

theorem horner_rule_v4 :
  horner_v4 2 = 80 :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v4_l388_38836


namespace NUMINAMATH_CALUDE_women_work_hours_l388_38868

/-- Given work completed by men and women under specific conditions, prove that women worked 6 hours per day. -/
theorem women_work_hours (men : ℕ) (women : ℕ) (men_days : ℕ) (women_days : ℕ) (men_hours : ℕ) 
  (h_men : men = 15)
  (h_women : women = 21)
  (h_men_days : men_days = 21)
  (h_women_days : women_days = 30)
  (h_men_hours : men_hours = 8)
  (h_work_rate : (3 : ℚ) / women = (2 : ℚ) / men) :
  ∃ women_hours : ℚ, women_hours = 6 ∧ 
    (men * men_days * men_hours : ℚ) = (women * women_days * women_hours) :=
sorry

end NUMINAMATH_CALUDE_women_work_hours_l388_38868


namespace NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_156_l388_38877

/-- A function that counts the number of positive three-digit integers less than 700 
    with at least two digits that are the same. -/
def count_integers_with_repeated_digits : ℕ :=
  sorry

/-- The theorem stating that the count of integers with the given properties is 156. -/
theorem count_integers_with_repeated_digits_is_156 : 
  count_integers_with_repeated_digits = 156 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_156_l388_38877


namespace NUMINAMATH_CALUDE_min_calls_for_complete_info_sharing_l388_38871

/-- Represents a person in the information sharing network -/
structure Person where
  id : Nat
  initialInfo : Nat

/-- Represents the state of information sharing -/
structure InfoState where
  people : Finset Person
  calls : Nat
  allInfoShared : Bool

/-- The minimum number of calls needed for complete information sharing -/
def minCalls (n : Nat) : Nat := 2 * n - 2

/-- Theorem stating the minimum number of calls needed for complete information sharing -/
theorem min_calls_for_complete_info_sharing (n : Nat) (h : n > 0) :
  ∀ (state : InfoState),
    state.people.card = n →
    (∀ p : Person, p ∈ state.people → ∃! i, p.initialInfo = i) →
    (state.allInfoShared → state.calls ≥ minCalls n) :=
sorry

end NUMINAMATH_CALUDE_min_calls_for_complete_info_sharing_l388_38871


namespace NUMINAMATH_CALUDE_custom_mult_seven_three_l388_38834

/-- Custom multiplication operation -/
def custom_mult (a b : ℤ) : ℤ := 4*a + 5*b - a*b + 1

/-- Theorem stating that 7 * 3 = 23 under the custom multiplication -/
theorem custom_mult_seven_three : custom_mult 7 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_seven_three_l388_38834


namespace NUMINAMATH_CALUDE_score_statistics_l388_38864

def scores : List ℝ := [80, 85, 90, 95]
def frequencies : List ℕ := [4, 6, 8, 2]

def total_students : ℕ := frequencies.sum

def median (s : List ℝ) (f : List ℕ) : ℝ := sorry

def mode (s : List ℝ) (f : List ℕ) : ℝ := sorry

theorem score_statistics :
  median scores frequencies = 87.5 ∧ mode scores frequencies = 90 := by sorry

end NUMINAMATH_CALUDE_score_statistics_l388_38864


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l388_38823

theorem trigonometric_system_solution (x y z : ℝ) : 
  Real.sin x + Real.sin y + Real.sin (x + y + z) = 0 ∧
  Real.sin x + 2 * Real.sin z = 0 ∧
  Real.sin y + 3 * Real.sin z = 0 →
  ∃ (k m n : ℤ), x = π * k ∧ y = π * m ∧ z = π * n := by
sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l388_38823


namespace NUMINAMATH_CALUDE_k_max_is_closest_to_expected_l388_38816

/-- The probability of rolling a one on a fair die -/
def p : ℚ := 1 / 6

/-- The number of times the die is tossed -/
def n : ℕ := 20

/-- The expected number of ones when tossing a fair die n times -/
def expected_ones : ℚ := n * p

/-- The probability of rolling k ones in n tosses of a fair die -/
noncomputable def P (k : ℕ) : ℝ := sorry

/-- The value of k that maximizes P(k) -/
noncomputable def k_max : ℕ := sorry

/-- Theorem stating that k_max is the integer closest to the expected number of ones -/
theorem k_max_is_closest_to_expected : 
  k_max = round expected_ones := by sorry

end NUMINAMATH_CALUDE_k_max_is_closest_to_expected_l388_38816


namespace NUMINAMATH_CALUDE_paint_time_per_room_l388_38827

theorem paint_time_per_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_rooms = 11) 
  (h2 : painted_rooms = 2) 
  (h3 : remaining_time = 63) : 
  remaining_time / (total_rooms - painted_rooms) = 7 := by
  sorry

end NUMINAMATH_CALUDE_paint_time_per_room_l388_38827


namespace NUMINAMATH_CALUDE_binomial_8_5_l388_38805

theorem binomial_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_5_l388_38805


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_specific_prism_volume_l388_38824

/-- Given a right rectangular prism with face areas a₁, a₂, and a₃,
    prove that its volume is the square root of the product of these areas. -/
theorem right_rectangular_prism_volume 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) : 
  ∃ (l w h : ℝ), l * w = a₁ ∧ w * h = a₂ ∧ l * h = a₃ ∧ 
  l * w * h = Real.sqrt (a₁ * a₂ * a₃) := by
  sorry

/-- The volume of a right rectangular prism with face areas 56, 63, and 72 
    square units is 504 cubic units. -/
theorem specific_prism_volume : 
  ∃ (l w h : ℝ), l * w = 56 ∧ w * h = 63 ∧ l * h = 72 ∧ 
  l * w * h = 504 := by
  sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_specific_prism_volume_l388_38824


namespace NUMINAMATH_CALUDE_triangle_side_values_l388_38841

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_values :
  ∀ y : ℕ+,
  (triangle_exists 8 11 (y.val ^ 2)) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l388_38841


namespace NUMINAMATH_CALUDE_odd_sum_15_to_55_l388_38860

theorem odd_sum_15_to_55 : 
  let a₁ : ℕ := 15  -- First term
  let d : ℕ := 4    -- Common difference
  let n : ℕ := (55 - 15) / d + 1  -- Number of terms
  let aₙ : ℕ := a₁ + (n - 1) * d  -- Last term
  (n : ℝ) / 2 * (a₁ + aₙ) = 385 :=
by sorry

end NUMINAMATH_CALUDE_odd_sum_15_to_55_l388_38860


namespace NUMINAMATH_CALUDE_distance_minimized_at_one_third_l388_38804

noncomputable def f (x : ℝ) : ℝ := 9 * x^3

noncomputable def g (x : ℝ) : ℝ := Real.log x

noncomputable def h (m : ℝ) : ℝ := |f m - g m|

theorem distance_minimized_at_one_third (m : ℝ) (hm : m > 0) :
  h m ≥ h (1/3) := by sorry

end NUMINAMATH_CALUDE_distance_minimized_at_one_third_l388_38804


namespace NUMINAMATH_CALUDE_parabola_point_order_l388_38822

/-- Given a parabola y = ax^2 - 2ax + 3 where a > 0, and points A(-1, y₁), B(2, y₂), C(4, y₃) on the parabola,
    prove that y₂ < y₁ < y₃ -/
theorem parabola_point_order (a : ℝ) (y₁ y₂ y₃ : ℝ) 
    (h_a : a > 0)
    (h_y₁ : y₁ = a * (-1)^2 - 2 * a * (-1) + 3)
    (h_y₂ : y₂ = a * 2^2 - 2 * a * 2 + 3)
    (h_y₃ : y₃ = a * 4^2 - 2 * a * 4 + 3) :
  y₂ < y₁ ∧ y₁ < y₃ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_order_l388_38822


namespace NUMINAMATH_CALUDE_projection_result_l388_38808

/-- Given two vectors a and b in ℝ², if both are projected onto the same vector v
    resulting in p, then p is equal to (48/53, 168/53). -/
theorem projection_result (a b v p : ℝ × ℝ) : 
  a = (5, 2) → 
  b = (-2, 4) → 
  (∃ (k₁ k₂ : ℝ), p = k₁ • v ∧ (a - p) • v = 0) → 
  (∃ (k₃ k₄ : ℝ), p = k₃ • v ∧ (b - p) • v = 0) → 
  p = (48/53, 168/53) := by
  sorry

end NUMINAMATH_CALUDE_projection_result_l388_38808


namespace NUMINAMATH_CALUDE_marbles_remaining_l388_38846

theorem marbles_remaining (total : ℕ) (white : ℕ) (removed : ℕ) : 
  total = 50 → 
  white = 20 → 
  removed = 2 * (white - (total - white) / 2) → 
  total - removed = 40 := by
sorry

end NUMINAMATH_CALUDE_marbles_remaining_l388_38846


namespace NUMINAMATH_CALUDE_absolute_difference_26th_terms_l388_38896

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + d * (n - 1)

theorem absolute_difference_26th_terms : 
  let C := arithmetic_sequence 50 15
  let D := arithmetic_sequence 85 (-20)
  |C 26 - D 26| = 840 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_26th_terms_l388_38896


namespace NUMINAMATH_CALUDE_next_next_perfect_square_l388_38894

theorem next_next_perfect_square (x : ℕ) (k : ℕ) (h : x = k^2) :
  (k + 2)^2 = x + 4 * Int.sqrt x + 4 :=
sorry

end NUMINAMATH_CALUDE_next_next_perfect_square_l388_38894


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_functions_l388_38875

-- Define a function f from reals to reals
variable (f : ℝ → ℝ)

-- Define the property of symmetry about x = -1
def symmetric_about_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + 1) = y ↔ f (-x - 1) = y

-- State the theorem
theorem symmetry_of_shifted_functions :
  symmetric_about_neg_one f := by sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_functions_l388_38875


namespace NUMINAMATH_CALUDE_quadratic_sum_value_l388_38852

/-- 
Given two quadratic trinomials that differ by the interchange of the constant term 
and the second coefficient, if their sum has a unique root, then the value of their 
sum at x = 2 is either 8 or 32.
-/
theorem quadratic_sum_value (p q : ℝ) : 
  let f := fun x : ℝ => x^2 + p*x + q
  let g := fun x : ℝ => x^2 + q*x + p
  let sum := fun x : ℝ => f x + g x
  (∃! r : ℝ, sum r = 0) → (sum 2 = 8 ∨ sum 2 = 32) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_value_l388_38852


namespace NUMINAMATH_CALUDE_min_score_theorem_l388_38821

/-- Leon's current test scores -/
def current_scores : List ℕ := [72, 68, 75, 81, 79]

/-- The number of current test scores -/
def num_current_scores : ℕ := current_scores.length

/-- The sum of current test scores -/
def sum_current_scores : ℕ := current_scores.sum

/-- The desired increase in average -/
def desired_increase : ℕ := 5

/-- The minimum score needed on the next test -/
def min_score_needed : ℕ := 105

theorem min_score_theorem :
  ∀ (next_score : ℕ),
    next_score ≥ min_score_needed →
    (sum_current_scores + next_score) / (num_current_scores + 1) ≥
    (sum_current_scores / num_current_scores + desired_increase) :=
by sorry

end NUMINAMATH_CALUDE_min_score_theorem_l388_38821


namespace NUMINAMATH_CALUDE_amount_left_after_pool_l388_38800

-- Define the given conditions
def total_earned : ℝ := 30
def cost_per_person : ℝ := 2.5
def number_of_people : ℕ := 10

-- Define the theorem
theorem amount_left_after_pool : 
  total_earned - (cost_per_person * number_of_people) = 5 := by
  sorry

end NUMINAMATH_CALUDE_amount_left_after_pool_l388_38800


namespace NUMINAMATH_CALUDE_intersection_with_complement_l388_38899

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_with_complement : M ∩ (U \ N) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l388_38899


namespace NUMINAMATH_CALUDE_liquid_film_radius_l388_38813

/-- The radius of a circular film formed by a liquid on water -/
theorem liquid_film_radius 
  (thickness : ℝ) 
  (volume : ℝ) 
  (h1 : thickness = 0.2)
  (h2 : volume = 320) : 
  ∃ (r : ℝ), r = Real.sqrt (1600 / Real.pi) ∧ π * r^2 * thickness = volume :=
sorry

end NUMINAMATH_CALUDE_liquid_film_radius_l388_38813


namespace NUMINAMATH_CALUDE_harry_age_l388_38879

/-- Given the ages of Kiarra, Bea, Job, Figaro, and Harry, prove that Harry is 26 years old. -/
theorem harry_age (kiarra bea job figaro harry : ℕ) 
  (h1 : kiarra = 2 * bea)
  (h2 : job = 3 * bea)
  (h3 : figaro = job + 7)
  (h4 : harry * 2 = figaro)
  (h5 : kiarra = 30) : 
  harry = 26 := by
  sorry

end NUMINAMATH_CALUDE_harry_age_l388_38879


namespace NUMINAMATH_CALUDE_unique_digit_for_divisibility_l388_38861

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def four_digit_number (B : ℕ) : ℕ := 5000 + 100 * B + 10 * B + 3

theorem unique_digit_for_divisibility :
  ∃! B : ℕ, B ≤ 9 ∧ is_divisible_by_9 (four_digit_number B) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_digit_for_divisibility_l388_38861


namespace NUMINAMATH_CALUDE_prob_second_draw_3_eq_11_48_l388_38881

-- Define the boxes and their initial contents
def box1 : Finset ℕ := {1, 1, 2, 3}
def box2 : Finset ℕ := {1, 1, 3}
def box3 : Finset ℕ := {1, 1, 1, 2, 2}

-- Define the probability of drawing a ball from a box
def prob_draw (box : Finset ℕ) (label : ℕ) : ℚ :=
  (box.filter (λ x => x = label)).card / box.card

-- Define the probability of the second draw being 3
def prob_second_draw_3 : ℚ :=
  (prob_draw box1 1 * prob_draw (box1 ∪ {1}) 3) +
  (prob_draw box1 2 * prob_draw (box2 ∪ {2}) 3) +
  (prob_draw box1 3 * prob_draw (box3 ∪ {3}) 3)

-- Theorem statement
theorem prob_second_draw_3_eq_11_48 : prob_second_draw_3 = 11 / 48 := by
  sorry


end NUMINAMATH_CALUDE_prob_second_draw_3_eq_11_48_l388_38881


namespace NUMINAMATH_CALUDE_graduating_class_size_l388_38865

theorem graduating_class_size (boys : ℕ) (girls : ℕ) : 
  boys = 127 → 
  girls = boys + 212 → 
  boys + girls = 466 := by sorry

end NUMINAMATH_CALUDE_graduating_class_size_l388_38865


namespace NUMINAMATH_CALUDE_ones_digit_of_31_power_l388_38850

theorem ones_digit_of_31_power (n : ℕ) : (31^(15 * 7^7) : ℕ) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_31_power_l388_38850


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l388_38888

/-- The radius of the inscribed circle in a triangle with sides 26, 15, and 17 is √6 -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 26) (hb : b = 15) (hc : c = 17) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l388_38888


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l388_38892

/-- The speed of cyclist C in miles per hour -/
def speed_C : ℝ := 28.5

/-- The speed of cyclist D in miles per hour -/
def speed_D : ℝ := speed_C + 6

/-- The distance between City X and City Y in miles -/
def distance_XY : ℝ := 100

/-- The distance C travels before turning back in miles -/
def distance_C_before_turn : ℝ := 80

/-- The distance from City Y where C and D meet after turning back in miles -/
def meeting_distance : ℝ := 15

theorem cyclist_speed_proof :
  speed_C = 28.5 ∧
  speed_D = speed_C + 6 ∧
  (distance_C_before_turn + meeting_distance) / speed_C = 
  (distance_XY + meeting_distance) / speed_D :=
sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l388_38892


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l388_38803

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop := ∃ m : ℝ, 1 < m ∧ m < 2 ∧ x = (1/2)^(m-1)

-- Part 1
theorem range_of_x_when_a_is_quarter :
  ∀ x : ℝ, p x (1/4) ∧ q x → 1/2 < x ∧ x < 3/4 :=
sorry

-- Part 2
theorem range_of_a_when_q_sufficient_not_necessary :
  (∀ x a : ℝ, q x → p x a) ∧ 
  (∃ x a : ℝ, p x a ∧ ¬q x) →
  ∀ a : ℝ, (∃ x : ℝ, q x → p x a) → 1/3 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l388_38803


namespace NUMINAMATH_CALUDE_prism_volume_l388_38819

/-- Given a rectangular prism with dimensions a, b, and c satisfying certain conditions,
    prove that its volume is 30√10 -/
theorem prism_volume (a b c : ℝ) 
    (h1 : a * b = 15)
    (h2 : b * c = 18)
    (h3 : c * a = 20)
    (h4 : c = 2 * a) : 
  a * b * c = 30 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l388_38819


namespace NUMINAMATH_CALUDE_different_color_probability_l388_38847

def blue_chips : ℕ := 4
def yellow_chips : ℕ := 5
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + yellow_chips + green_chips

theorem different_color_probability :
  let p_blue_first := blue_chips / total_chips
  let p_yellow_first := yellow_chips / total_chips
  let p_green_first := green_chips / total_chips
  let p_not_blue_second := (yellow_chips + green_chips) / (total_chips - 1)
  let p_not_yellow_second := (blue_chips + green_chips) / (total_chips - 1)
  let p_not_green_second := (blue_chips + yellow_chips) / (total_chips - 1)
  (p_blue_first * p_not_blue_second + 
   p_yellow_first * p_not_yellow_second + 
   p_green_first * p_not_green_second) = 47 / 66 :=
by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l388_38847


namespace NUMINAMATH_CALUDE_aarti_work_theorem_l388_38839

/-- Given that Aarti can complete a piece of work in a certain number of days,
    this function calculates how many days she needs to complete a multiple of that work. -/
def days_for_multiple_work (base_days : ℕ) (multiple : ℕ) : ℕ :=
  base_days * multiple

/-- Theorem stating that if Aarti can complete a piece of work in 5 days,
    then she will need 15 days to complete three times the work of the same type. -/
theorem aarti_work_theorem :
  days_for_multiple_work 5 3 = 15 := by sorry

end NUMINAMATH_CALUDE_aarti_work_theorem_l388_38839


namespace NUMINAMATH_CALUDE_cos_arctan_equal_x_squared_l388_38849

theorem cos_arctan_equal_x_squared :
  ∃ (x : ℝ), x > 0 ∧ Real.cos (Real.arctan x) = x → x^2 = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_arctan_equal_x_squared_l388_38849


namespace NUMINAMATH_CALUDE_inequality_proof_l388_38829

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b ≤ 1) :
  9 * a^2 * b + 9 * a * b^2 - a^2 - 10 * a * b - b^2 + a + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l388_38829


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l388_38859

-- Define the sample space
def Ω : Type := Unit

-- Define the event of hitting the target on the first shot
def hit1 : Set Ω := sorry

-- Define the event of hitting the target on the second shot
def hit2 : Set Ω := sorry

-- Define the event of hitting the target at least once
def hitAtLeastOnce : Set Ω := hit1 ∪ hit2

-- Define the event of missing the target both times
def missBoth : Set Ω := (hit1 ∪ hit2)ᶜ

-- Theorem stating that hitAtLeastOnce and missBoth are mutually exclusive
theorem mutually_exclusive_events : 
  hitAtLeastOnce ∩ missBoth = ∅ ∧ hitAtLeastOnce ∪ missBoth = Set.univ :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l388_38859


namespace NUMINAMATH_CALUDE_perpendicular_slope_l388_38828

/-- Given two points (3, -4) and (-2, 5) on a line, the slope of a line perpendicular to this line is 5/9. -/
theorem perpendicular_slope : 
  let p1 : ℝ × ℝ := (3, -4)
  let p2 : ℝ × ℝ := (-2, 5)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  (- (1 / m)) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l388_38828


namespace NUMINAMATH_CALUDE_parabola_no_x_intersection_l388_38889

/-- The parabola defined by y = -2x^2 + x - 1 has no intersection with the x-axis -/
theorem parabola_no_x_intersection :
  ∀ x : ℝ, -2 * x^2 + x - 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_no_x_intersection_l388_38889


namespace NUMINAMATH_CALUDE_stating_distinguishable_triangles_count_l388_38897

/-- Represents the number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles needed to construct a large triangle -/
def triangles_per_large : ℕ := 4

/-- 
Calculates the number of distinguishable large equilateral triangles that can be constructed
given the number of available colors and the number of small triangles per large triangle.
-/
def count_distinguishable_triangles (colors : ℕ) (triangles : ℕ) : ℕ :=
  colors * (colors - 1) * (colors - 2) * (colors - 3)

/-- 
Theorem stating that the number of distinguishable large equilateral triangles
that can be constructed under the given conditions is 1680.
-/
theorem distinguishable_triangles_count :
  count_distinguishable_triangles num_colors triangles_per_large = 1680 := by
  sorry


end NUMINAMATH_CALUDE_stating_distinguishable_triangles_count_l388_38897


namespace NUMINAMATH_CALUDE_elevator_probability_l388_38817

/-- The number of floors in the building -/
def num_floors : ℕ := 6

/-- The number of floors where people can exit (excluding ground floor) -/
def exit_floors : ℕ := num_floors - 1

/-- The probability of two people leaving the elevator on different floors -/
def prob_different_floors : ℚ := 4/5

theorem elevator_probability :
  prob_different_floors = 1 - (1 : ℚ) / exit_floors :=
by sorry

end NUMINAMATH_CALUDE_elevator_probability_l388_38817


namespace NUMINAMATH_CALUDE_train_journey_time_l388_38878

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (4 / 5 * usual_speed) * (usual_time + 3 / 4) = usual_speed * usual_time → 
  usual_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l388_38878


namespace NUMINAMATH_CALUDE_inequality_equality_l388_38858

theorem inequality_equality (x : ℝ) : 
  x > 0 → (x * Real.sqrt (16 - x^2) + Real.sqrt (16*x - x^4) ≥ 16 ↔ x = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equality_l388_38858
