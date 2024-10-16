import Mathlib

namespace NUMINAMATH_CALUDE_mobileRadiationNotSuitable_l3834_383428

/-- Represents a statistical activity that can be potentially collected through a questionnaire. -/
inductive StatisticalActivity
  | BlueCars
  | TVsInHomes
  | WakeUpTime
  | MobileRadiation

/-- Predicate to determine if a statistical activity is suitable for questionnaire data collection. -/
def suitableForQuestionnaire (activity : StatisticalActivity) : Prop :=
  match activity with
  | StatisticalActivity.BlueCars => True
  | StatisticalActivity.TVsInHomes => True
  | StatisticalActivity.WakeUpTime => True
  | StatisticalActivity.MobileRadiation => False

/-- Theorem stating that mobile radiation is the only activity not suitable for questionnaire data collection. -/
theorem mobileRadiationNotSuitable :
    ∀ (activity : StatisticalActivity),
      ¬(suitableForQuestionnaire activity) ↔ activity = StatisticalActivity.MobileRadiation := by
  sorry

end NUMINAMATH_CALUDE_mobileRadiationNotSuitable_l3834_383428


namespace NUMINAMATH_CALUDE_sum_of_odd_naturals_900_l3834_383498

theorem sum_of_odd_naturals_900 :
  ∃ n : ℕ, n^2 = 900 ∧ (∀ k : ℕ, k ≤ n → (2*k - 1) ≤ n^2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_odd_naturals_900_l3834_383498


namespace NUMINAMATH_CALUDE_female_students_like_pe_l3834_383466

def total_students : ℕ := 1500
def male_percentage : ℚ := 2/5
def female_dislike_pe_percentage : ℚ := 13/20

theorem female_students_like_pe : 
  (total_students : ℚ) * (1 - male_percentage) * (1 - female_dislike_pe_percentage) = 315 := by
  sorry

end NUMINAMATH_CALUDE_female_students_like_pe_l3834_383466


namespace NUMINAMATH_CALUDE_integer_equation_solution_l3834_383490

theorem integer_equation_solution (x y : ℤ) : 
  12 * x^2 + 6 * x * y + 3 * y^2 = 28 * (x + y) ↔ 
  ∃ m n : ℤ, y = 4 * n ∧ x = 3 * m - 4 * n := by
sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l3834_383490


namespace NUMINAMATH_CALUDE_smallest_x_is_correct_l3834_383444

/-- The smallest positive integer x such that 1980x is a perfect fourth power -/
def smallest_x : ℕ := 6006250

/-- Predicate to check if a number is a perfect fourth power -/
def is_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^4

theorem smallest_x_is_correct :
  (∀ y : ℕ, y < smallest_x → ¬ is_fourth_power (1980 * y)) ∧
  is_fourth_power (1980 * smallest_x) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_is_correct_l3834_383444


namespace NUMINAMATH_CALUDE_units_digit_of_2_power_2010_l3834_383439

-- Define the function for the units digit of 2^n
def unitsDigitOf2Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 0 => 6
  | _ => 0  -- This case should never occur

-- Theorem statement
theorem units_digit_of_2_power_2010 :
  unitsDigitOf2Power 2010 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_power_2010_l3834_383439


namespace NUMINAMATH_CALUDE_two_satisfying_functions_l3834_383451

/-- A function satisfying the given property -/
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x^2 - y * f z) = x * f x - z * f y

/-- The set of functions satisfying the property -/
def SatisfyingFunctions : Set (ℝ → ℝ) :=
  {f | SatisfiesProperty f}

/-- The constant zero function -/
def ZeroFunction : ℝ → ℝ := λ _ ↦ 0

/-- The identity function -/
def IdentityFunction : ℝ → ℝ := λ x ↦ x

theorem two_satisfying_functions :
  SatisfyingFunctions = {ZeroFunction, IdentityFunction} := by
  sorry

#check two_satisfying_functions

end NUMINAMATH_CALUDE_two_satisfying_functions_l3834_383451


namespace NUMINAMATH_CALUDE_correct_option_is_B_l3834_383473

-- Define the statements
def statement1 : Prop := False
def statement2 : Prop := True
def statement3 : Prop := True
def statement4 : Prop := False

-- Define the options
def optionA : Prop := statement1 ∧ statement2 ∧ statement3
def optionB : Prop := statement2 ∧ statement3
def optionC : Prop := statement2 ∧ statement4
def optionD : Prop := statement1 ∧ statement3 ∧ statement4

-- Theorem: The correct option is B
theorem correct_option_is_B : 
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) → 
  (optionB ∧ ¬optionA ∧ ¬optionC ∧ ¬optionD) :=
by sorry

end NUMINAMATH_CALUDE_correct_option_is_B_l3834_383473


namespace NUMINAMATH_CALUDE_roller_plate_acceleration_l3834_383470

noncomputable def g : ℝ := 10
noncomputable def R : ℝ := 1
noncomputable def r : ℝ := 0.4
noncomputable def m : ℝ := 150
noncomputable def α : ℝ := Real.arccos 0.68

theorem roller_plate_acceleration 
  (h_no_slip : True) -- Assumption of no slipping
  (h_weightless : True) -- Assumption of weightless rollers
  : ∃ (plate_acc_mag plate_acc_dir roller_acc : ℝ),
    plate_acc_mag = 4 ∧ 
    plate_acc_dir = Real.arcsin 0.4 ∧
    roller_acc = 4 := by
  sorry

end NUMINAMATH_CALUDE_roller_plate_acceleration_l3834_383470


namespace NUMINAMATH_CALUDE_road_repaving_l3834_383458

theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ) 
  (h1 : total_repaved = 4938)
  (h2 : repaved_today = 805) :
  total_repaved - repaved_today = 4133 :=
by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l3834_383458


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3834_383482

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the angle between asymptotes
def angle_between_asymptotes (h : (x y : ℝ) → Prop) : ℝ := sorry

-- Theorem statement
theorem hyperbola_asymptote_angle :
  angle_between_asymptotes hyperbola = 60 * π / 180 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3834_383482


namespace NUMINAMATH_CALUDE_line_chart_division_l3834_383445

/-- Represents a line chart -/
structure LineChart where
  /-- The line chart uses rise or fall of a line to represent increase or decrease in statistical quantities -/
  represents_statistical_quantities : Bool

/-- Represents a simple line chart -/
structure SimpleLineChart extends LineChart

/-- Represents a compound line chart -/
structure CompoundLineChart extends LineChart

/-- Theorem stating that line charts can be divided into simple and compound line charts -/
theorem line_chart_division (lc : LineChart) : 
  (∃ (slc : SimpleLineChart), slc.toLineChart = lc) ∨ 
  (∃ (clc : CompoundLineChart), clc.toLineChart = lc) :=
sorry

end NUMINAMATH_CALUDE_line_chart_division_l3834_383445


namespace NUMINAMATH_CALUDE_messages_cleared_in_seven_days_l3834_383424

/-- Given the initial number of unread messages, messages read per day,
    and new messages received per day, calculate the number of days
    required to read all unread messages. -/
def days_to_read_messages (initial_messages : ℕ) (messages_read_per_day : ℕ) (new_messages_per_day : ℕ) : ℕ :=
  initial_messages / (messages_read_per_day - new_messages_per_day)

/-- Theorem stating that it takes 7 days to read all unread messages
    under the given conditions. -/
theorem messages_cleared_in_seven_days :
  days_to_read_messages 98 20 6 = 7 := by
  sorry

#eval days_to_read_messages 98 20 6

end NUMINAMATH_CALUDE_messages_cleared_in_seven_days_l3834_383424


namespace NUMINAMATH_CALUDE_handshake_theorem_l3834_383432

theorem handshake_theorem (n : ℕ) (k : ℕ) (h : n = 30 ∧ k = 3) :
  (n * k) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_handshake_theorem_l3834_383432


namespace NUMINAMATH_CALUDE_dave_tickets_l3834_383453

def tickets_problem (initial_tickets spent_tickets later_tickets : ℕ) : Prop :=
  initial_tickets - spent_tickets + later_tickets = 16

theorem dave_tickets : tickets_problem 11 5 10 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l3834_383453


namespace NUMINAMATH_CALUDE_cone_volume_l3834_383403

theorem cone_volume (s : Real) (c : Real) (h : s = 8) (k : c = 6 * Real.pi) :
  let r := c / (2 * Real.pi)
  let height := Real.sqrt (s^2 - r^2)
  (1/3) * Real.pi * r^2 * height = 3 * Real.sqrt 55 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cone_volume_l3834_383403


namespace NUMINAMATH_CALUDE_apple_distribution_l3834_383422

theorem apple_distribution (jim jerry : ℕ) (h1 : jim = 20) (h2 : jerry = 40) : 
  ∃ jane : ℕ, 
    (2 * jim = (jim + jerry + jane) / 3) ∧ 
    (jane = 30) := by
sorry

end NUMINAMATH_CALUDE_apple_distribution_l3834_383422


namespace NUMINAMATH_CALUDE_divisibility_and_finiteness_l3834_383406

theorem divisibility_and_finiteness :
  (∀ x : ℕ+, ∃ y : ℕ+, (x + y + 1) ∣ (x^3 + y^3 + 1)) ∧
  (∀ x : ℕ+, Set.Finite {y : ℕ+ | (x + y + 1) ∣ (x^3 + y^3 + 1)}) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_and_finiteness_l3834_383406


namespace NUMINAMATH_CALUDE_exists_negative_value_implies_a_greater_than_nine_halves_l3834_383407

theorem exists_negative_value_implies_a_greater_than_nine_halves
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = x^3 - a*x^2 + 10)
  (a : ℝ)
  (h_exists : ∃ x ∈ Set.Icc 1 2, f x < 0) :
  a > 9/2 := by
sorry

end NUMINAMATH_CALUDE_exists_negative_value_implies_a_greater_than_nine_halves_l3834_383407


namespace NUMINAMATH_CALUDE_equation_has_three_real_solutions_l3834_383450

-- Define the equation
def equation (x : ℝ) : Prop :=
  (18 * x - 2) ^ (1/3) + (14 * x - 4) ^ (1/3) = 5 * (2 * x + 4) ^ (1/3)

-- State the theorem
theorem equation_has_three_real_solutions :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    equation x₁ ∧ equation x₂ ∧ equation x₃ :=
sorry

end NUMINAMATH_CALUDE_equation_has_three_real_solutions_l3834_383450


namespace NUMINAMATH_CALUDE_complex_division_product_l3834_383499

/-- Given (2+3i)/i = a+bi, where a and b are real numbers and i is the imaginary unit, prove that ab = 6 -/
theorem complex_division_product (a b : ℝ) : (Complex.I : ℂ)⁻¹ * (2 + 3 * Complex.I) = a + b * Complex.I → a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_division_product_l3834_383499


namespace NUMINAMATH_CALUDE_drama_club_organization_l3834_383477

theorem drama_club_organization (participants : ℕ) (girls : ℕ) (boys : ℕ) : 
  participants = girls + boys →
  girls > (85 * participants) / 100 →
  boys ≥ 2 →
  participants ≥ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_drama_club_organization_l3834_383477


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l3834_383467

/-- Given two varieties of rice mixed in a specific ratio to obtain a mixture with a known cost,
    this theorem proves that the cost of the first variety can be determined. -/
theorem rice_mixture_cost
  (cost_second : ℝ)  -- Cost per kg of the second variety of rice
  (ratio : ℝ)        -- Ratio of the first variety to the second in the mixture
  (cost_mixture : ℝ) -- Cost per kg of the resulting mixture
  (h1 : cost_second = 8.75)
  (h2 : ratio = 0.8333333333333334)
  (h3 : cost_mixture = 7.50)
  : ∃ (cost_first : ℝ), 
    cost_first * (ratio / (1 + ratio)) + cost_second * (1 / (1 + ratio)) = cost_mixture ∧ 
    cost_first = 7.25 := by
  sorry


end NUMINAMATH_CALUDE_rice_mixture_cost_l3834_383467


namespace NUMINAMATH_CALUDE_subtraction_of_negative_l3834_383435

theorem subtraction_of_negative : 4 - (-7) = 11 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_l3834_383435


namespace NUMINAMATH_CALUDE_equation_solution_l3834_383415

theorem equation_solution : ∃! y : ℝ, 5 * (y + 2) + 9 = 3 * (1 - y) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3834_383415


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l3834_383456

/-- The first element of the nth set -/
def first_element (n : ℕ) : ℕ := 1 + (n - 1) * n / 2

/-- The last element of the nth set -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem: The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l3834_383456


namespace NUMINAMATH_CALUDE_function_characterization_l3834_383455

theorem function_characterization (a : ℝ) (f : ℝ → ℝ) 
  (h : ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → 
    a * f (x / y) + a * f (x / z) - f x * f ((y + z) / 2) ≥ a^2) :
  ∀ (x : ℝ), x ≠ 0 → f x = a := by
sorry

end NUMINAMATH_CALUDE_function_characterization_l3834_383455


namespace NUMINAMATH_CALUDE_remove_number_for_target_average_l3834_383409

def original_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def removed_number : ℕ := 5

def target_average : ℚ := 61/10

theorem remove_number_for_target_average :
  let remaining_list := original_list.filter (· ≠ removed_number)
  (remaining_list.sum : ℚ) / remaining_list.length = target_average := by
  sorry

end NUMINAMATH_CALUDE_remove_number_for_target_average_l3834_383409


namespace NUMINAMATH_CALUDE_altitude_inradius_inequality_l3834_383486

-- Define a triangle with altitudes and inradius
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  r : ℝ
  h_a_positive : h_a > 0
  h_b_positive : h_b > 0
  h_c_positive : h_c > 0
  r_positive : r > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem altitude_inradius_inequality (t : Triangle) : t.h_a + 4 * t.h_b + 9 * t.h_c > 36 * t.r := by
  sorry

end NUMINAMATH_CALUDE_altitude_inradius_inequality_l3834_383486


namespace NUMINAMATH_CALUDE_third_term_is_18_l3834_383400

def arithmetic_geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

theorem third_term_is_18 (a₁ q : ℝ) (h₁ : a₁ = 2) (h₂ : q = 3) :
  arithmetic_geometric_sequence a₁ q 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_18_l3834_383400


namespace NUMINAMATH_CALUDE_fraction_of_As_l3834_383442

theorem fraction_of_As (total_students : ℕ) (fraction_Bs fraction_Cs : ℚ) (num_Ds : ℕ) :
  total_students = 100 →
  fraction_Bs = 1/4 →
  fraction_Cs = 1/2 →
  num_Ds = 5 →
  (total_students - (fraction_Bs * total_students + fraction_Cs * total_students + num_Ds)) / total_students = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_As_l3834_383442


namespace NUMINAMATH_CALUDE_senior_citizen_tickets_l3834_383463

theorem senior_citizen_tickets (total_tickets : ℕ) (adult_price senior_price : ℚ) (total_receipts : ℚ) :
  total_tickets = 529 →
  adult_price = 25 →
  senior_price = 15 →
  total_receipts = 9745 →
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 348 :=
by sorry

end NUMINAMATH_CALUDE_senior_citizen_tickets_l3834_383463


namespace NUMINAMATH_CALUDE_quadruplet_babies_l3834_383431

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1200)
  (h_triplets_quadruplets : ∃ (t q : ℕ), t = 5 * q)
  (h_twins_triplets : ∃ (w t : ℕ), w = 2 * t)
  (h_sum : ∃ (w t q : ℕ), 2 * w + 3 * t + 4 * q = total_babies) :
  ∃ (q : ℕ), 4 * q = 123 := by
sorry

end NUMINAMATH_CALUDE_quadruplet_babies_l3834_383431


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l3834_383475

/-- A parabola and hyperbola with specific properties -/
structure ParabolaHyperbolaPair where
  -- Parabola properties
  parabola_vertex : ℝ × ℝ
  parabola_axis_through_focus : Bool
  parabola_perpendicular : Bool
  
  -- Hyperbola properties
  hyperbola_a : ℝ
  hyperbola_b : ℝ
  
  -- Intersection point
  intersection : ℝ × ℝ
  
  -- Conditions
  vertex_at_origin : parabola_vertex = (0, 0)
  axis_through_focus : parabola_axis_through_focus = true
  perpendicular_to_real_axis : parabola_perpendicular = true
  intersection_point : intersection = (3/2, Real.sqrt 6)
  hyperbola_equation : ∀ x y, x^2 / hyperbola_a^2 - y^2 / hyperbola_b^2 = 1 → 
    (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)

/-- The equations of the parabola and hyperbola given the conditions -/
theorem parabola_hyperbola_equations (ph : ParabolaHyperbolaPair) :
  (∀ x y, y^2 = 4*x ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)) ∧
  (∀ x y, x^2 / (1/4) - y^2 / (3/4) = 1 ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ => t)) :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l3834_383475


namespace NUMINAMATH_CALUDE_comparison_of_exponents_l3834_383408

theorem comparison_of_exponents :
  (1.7 ^ 2.5 < 1.7 ^ 3) ∧
  (0.8 ^ (-0.1) < 0.8 ^ (-0.2)) ∧
  (1.7 ^ 0.3 > 0.9 ^ 3.1) ∧
  ((1/3) ^ (1/3) < (1/4) ^ (1/4)) := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_exponents_l3834_383408


namespace NUMINAMATH_CALUDE_gear_r_rpm_calculation_l3834_383429

/-- The number of revolutions per minute for Gear L -/
def gear_l_rpm : ℚ := 20

/-- The time elapsed in seconds -/
def elapsed_time : ℚ := 6

/-- The additional revolutions made by Gear R compared to Gear L -/
def additional_revolutions : ℚ := 6

/-- Calculate the number of revolutions per minute for Gear R -/
def gear_r_rpm : ℚ :=
  (gear_l_rpm * elapsed_time / 60 + additional_revolutions) * 60 / elapsed_time

theorem gear_r_rpm_calculation :
  gear_r_rpm = 80 := by sorry

end NUMINAMATH_CALUDE_gear_r_rpm_calculation_l3834_383429


namespace NUMINAMATH_CALUDE_truck_rental_example_l3834_383478

/-- Calculates the total cost of renting a truck given the daily rate, per-mile rate, number of days, and miles driven. -/
def truck_rental_cost (daily_rate : ℚ) (mile_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mile_rate * miles

/-- Proves that renting a truck for $35 per day and $0.25 per mile for 3 days and 300 miles costs $180 in total. -/
theorem truck_rental_example : truck_rental_cost 35 (1/4) 3 300 = 180 := by
  sorry

end NUMINAMATH_CALUDE_truck_rental_example_l3834_383478


namespace NUMINAMATH_CALUDE_triangle_minimize_side_l3834_383416

/-- Given a triangle with area t and angle C, prove that the side c opposite to angle C 
    is minimized when the other two sides a and b are equal, and find the minimum length of c. -/
theorem triangle_minimize_side (t : ℝ) (C : ℝ) (h1 : t > 0) (h2 : 0 < C ∧ C < π) :
  ∃ (a b c : ℝ),
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (a * b * Real.sin C / 2 = t) ∧
    (∀ (a' b' c' : ℝ), 
      (a' > 0 ∧ b' > 0 ∧ c' > 0) ∧ 
      (a' * b' * Real.sin C / 2 = t) ∧ 
      (c' ^ 2 = a' ^ 2 + b' ^ 2 - 2 * a' * b' * Real.cos C) →
      c ≤ c') ∧
    (a = b) ∧
    (c = 2 * Real.sqrt (t * Real.tan (C / 2))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_minimize_side_l3834_383416


namespace NUMINAMATH_CALUDE_triangle_inequality_triangle_equality_l3834_383496

/-- The area of a triangle with sides a, b, c -/
noncomputable def A (a b c : ℝ) : ℝ := sorry

/-- Function f as defined in the problem -/
noncomputable def f (a b c : ℝ) : ℝ := Real.sqrt (A a b c)

/-- The main theorem -/
theorem triangle_inequality (a b c a' b' c' : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (ha' : 0 < a') (hb' : 0 < b') (hc' : 0 < c') :
    f a b c + f a' b' c' ≤ f (a + a') (b + b') (c + c') :=
  sorry

/-- Condition for equality -/
theorem triangle_equality (a b c a' b' c' : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (ha' : 0 < a') (hb' : 0 < b') (hc' : 0 < c') :
    f a b c + f a' b' c' = f (a + a') (b + b') (c + c') ↔ a / a' = b / b' ∧ b / b' = c / c' :=
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_triangle_equality_l3834_383496


namespace NUMINAMATH_CALUDE_inequality_region_l3834_383474

open Real

theorem inequality_region (x y : ℝ) : 
  (x^5 - 13*x^3 + 36*x) * (x^4 - 17*x^2 + 16) / 
  ((y^5 - 13*y^3 + 36*y) * (y^4 - 17*y^2 + 16)) ≥ 0 ↔ 
  y ≠ 0 ∧ y ≠ 1 ∧ y ≠ -1 ∧ y ≠ 2 ∧ y ≠ -2 ∧ y ≠ 3 ∧ y ≠ -3 ∧ y ≠ 4 ∧ y ≠ -4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_region_l3834_383474


namespace NUMINAMATH_CALUDE_hyperbola_min_focal_distance_l3834_383430

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, prove that the minimum semi-focal distance c is 4 -/
theorem hyperbola_min_focal_distance (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = c^2) →
  (a * b / c = c / 4 + 1) →
  c ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_min_focal_distance_l3834_383430


namespace NUMINAMATH_CALUDE_train_speed_l3834_383491

/-- Given a train of length 200 meters that takes 5 seconds to cross an electric pole,
    prove that its speed is 40 meters per second. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 200) (h2 : crossing_time = 5) :
  train_length / crossing_time = 40 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l3834_383491


namespace NUMINAMATH_CALUDE_five_points_on_circle_l3834_383414

-- Define a type for lines in general position
structure GeneralPositionLine where
  -- Add necessary fields

-- Define a type for points
structure Point where
  -- Add necessary fields

-- Define a type for circles
structure Circle where
  -- Add necessary fields

-- Function to get the intersection point of two lines
def lineIntersection (l1 l2 : GeneralPositionLine) : Point :=
  sorry

-- Function to get the circle passing through three points
def circleThrough3Points (p1 p2 p3 : Point) : Circle :=
  sorry

-- Function to get the intersection point of two circles
def circleIntersection (c1 c2 : Circle) : Point :=
  sorry

-- Function to check if a point lies on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  sorry

-- Main theorem
theorem five_points_on_circle 
  (l1 l2 l3 l4 l5 : GeneralPositionLine) : 
  ∃ (c : Circle),
    let s12 := circleThrough3Points (lineIntersection l3 l4) (lineIntersection l3 l5) (lineIntersection l4 l5)
    let s13 := circleThrough3Points (lineIntersection l2 l4) (lineIntersection l2 l5) (lineIntersection l4 l5)
    let s14 := circleThrough3Points (lineIntersection l2 l3) (lineIntersection l2 l5) (lineIntersection l3 l5)
    let s15 := circleThrough3Points (lineIntersection l2 l3) (lineIntersection l2 l4) (lineIntersection l3 l4)
    let s23 := circleThrough3Points (lineIntersection l1 l4) (lineIntersection l1 l5) (lineIntersection l4 l5)
    let s24 := circleThrough3Points (lineIntersection l1 l3) (lineIntersection l1 l5) (lineIntersection l3 l5)
    let s25 := circleThrough3Points (lineIntersection l1 l3) (lineIntersection l1 l4) (lineIntersection l3 l4)
    let s34 := circleThrough3Points (lineIntersection l1 l2) (lineIntersection l1 l5) (lineIntersection l2 l5)
    let s35 := circleThrough3Points (lineIntersection l1 l2) (lineIntersection l1 l4) (lineIntersection l2 l4)
    let s45 := circleThrough3Points (lineIntersection l1 l2) (lineIntersection l1 l3) (lineIntersection l2 l3)
    let a1 := circleIntersection s23 s24
    let a2 := circleIntersection s13 s14
    let a3 := circleIntersection s12 s14
    let a4 := circleIntersection s12 s13
    let a5 := circleIntersection s12 s23
    pointOnCircle a1 c ∧ 
    pointOnCircle a2 c ∧ 
    pointOnCircle a3 c ∧ 
    pointOnCircle a4 c ∧ 
    pointOnCircle a5 c :=
  sorry


end NUMINAMATH_CALUDE_five_points_on_circle_l3834_383414


namespace NUMINAMATH_CALUDE_complete_square_problems_l3834_383426

theorem complete_square_problems :
  (∀ a b : ℝ, a + b = 5 ∧ a * b = 2 → a^2 + b^2 = 21) ∧
  (∀ a b : ℝ, a + b = 10 ∧ a^2 + b^2 = 50^2 → a * b = -1200) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_problems_l3834_383426


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_for_radius_6_l3834_383443

/-- The area of the largest possible triangle inscribed in a circle,
    where one side of the triangle is a diameter of the circle. -/
def largest_inscribed_triangle_area (r : ℝ) : ℝ :=
  2 * r * r

theorem largest_inscribed_triangle_area_for_radius_6 :
  largest_inscribed_triangle_area 6 = 36 := by
  sorry

#eval largest_inscribed_triangle_area 6

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_for_radius_6_l3834_383443


namespace NUMINAMATH_CALUDE_polynomial_difference_divisibility_l3834_383457

/-- For any polynomial P with integer coefficients and any integers a and b,
    (a - b) divides (P(a) - P(b)) in ℤ. -/
theorem polynomial_difference_divisibility (P : Polynomial ℤ) (a b : ℤ) :
  (a - b) ∣ (P.eval a - P.eval b) :=
sorry

end NUMINAMATH_CALUDE_polynomial_difference_divisibility_l3834_383457


namespace NUMINAMATH_CALUDE_function_value_at_pi_third_l3834_383460

/-- Given a function f(x) = 2tan(ωx + φ) with the following properties:
    - ω > 0
    - |φ| < π/2
    - f(0) = 2√3/3
    - The period T ∈ (π/4, 3π/4)
    - (π/6, 0) is the center of symmetry of f(x)
    Prove that f(π/3) = -2√3/3 -/
theorem function_value_at_pi_third 
  (f : ℝ → ℝ) 
  (ω φ : ℝ) 
  (h1 : ∀ x, f x = 2 * Real.tan (ω * x + φ))
  (h2 : ω > 0)
  (h3 : abs φ < Real.pi / 2)
  (h4 : f 0 = 2 * Real.sqrt 3 / 3)
  (h5 : ∃ T, T ∈ Set.Ioo (Real.pi / 4) (3 * Real.pi / 4) ∧ ∀ x, f (x + T) = f x)
  (h6 : ∀ x, f (Real.pi / 3 - x) = f (Real.pi / 3 + x)) :
  f (Real.pi / 3) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_pi_third_l3834_383460


namespace NUMINAMATH_CALUDE_common_difference_is_two_l3834_383412

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_is_two
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 10)
  (h_fourth : a 4 = 7) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l3834_383412


namespace NUMINAMATH_CALUDE_solve_for_m_l3834_383497

def f (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + m

def g (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + 5*m

theorem solve_for_m :
  ∃ m : ℚ, 3 * (f m 5) = 2 * (g m 5) ∧ m = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3834_383497


namespace NUMINAMATH_CALUDE_ascending_order_l3834_383494

-- Define the variables
def a : ℕ := 2^55
def b : ℕ := 3^44
def c : ℕ := 5^33
def d : ℕ := 6^22

-- Theorem stating the ascending order
theorem ascending_order : a < d ∧ d < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ascending_order_l3834_383494


namespace NUMINAMATH_CALUDE_money_difference_l3834_383440

theorem money_difference (eric ben jack : ℕ) 
  (h1 : eric = ben - 10)
  (h2 : ben < 26)
  (h3 : jack = 26)
  (h4 : eric + ben + jack = 50) :
  jack - ben = 9 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l3834_383440


namespace NUMINAMATH_CALUDE_favorite_numbers_exist_l3834_383472

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem favorite_numbers_exist : ∃ (a b c : ℕ), 
  a * b * c = 71668 ∧ 
  a * sum_of_digits a = 10 * a ∧
  b * sum_of_digits b = 10 * b ∧
  c * sum_of_digits c = 10 * c :=
sorry

end NUMINAMATH_CALUDE_favorite_numbers_exist_l3834_383472


namespace NUMINAMATH_CALUDE_sequence_limit_l3834_383421

def x (n : ℕ) : ℚ := (2 * n - 1) / (3 * n + 5)

theorem sequence_limit : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - 2/3| < ε := by
  sorry

end NUMINAMATH_CALUDE_sequence_limit_l3834_383421


namespace NUMINAMATH_CALUDE_no_infinite_line_family_l3834_383464

theorem no_infinite_line_family :
  ¬ ∃ (k : ℕ → ℝ),
    (∀ n, k n ≠ 0) ∧
    (∀ n, k (n + 1) = (1 - 1 / k n) - (1 - k n)) ∧
    (∀ n, k n * k (n + 1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_line_family_l3834_383464


namespace NUMINAMATH_CALUDE_num_correct_propositions_is_one_l3834_383441

/-- Represents a geometric proposition -/
inductive GeometricProposition
  | ThreePointsCircle
  | EqualArcEqualAngle
  | RightTrianglesSimilar
  | RhombusesSimilar

/-- Determines if a geometric proposition is correct -/
def is_correct (prop : GeometricProposition) : Bool :=
  match prop with
  | GeometricProposition.ThreePointsCircle => false
  | GeometricProposition.EqualArcEqualAngle => true
  | GeometricProposition.RightTrianglesSimilar => false
  | GeometricProposition.RhombusesSimilar => false

/-- The list of all propositions to be evaluated -/
def all_propositions : List GeometricProposition :=
  [GeometricProposition.ThreePointsCircle,
   GeometricProposition.EqualArcEqualAngle,
   GeometricProposition.RightTrianglesSimilar,
   GeometricProposition.RhombusesSimilar]

/-- Theorem stating that the number of correct propositions is 1 -/
theorem num_correct_propositions_is_one :
  (all_propositions.filter is_correct).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_num_correct_propositions_is_one_l3834_383441


namespace NUMINAMATH_CALUDE_unfoldable_cone_ratio_l3834_383438

/-- A cone with lateral surface that forms a semicircle when unfolded -/
structure UnfoldableCone where
  /-- Radius of the base of the cone -/
  base_radius : ℝ
  /-- Length of the generatrix of the cone -/
  generatrix_length : ℝ
  /-- The lateral surface forms a semicircle when unfolded -/
  unfolded_is_semicircle : π * generatrix_length = 2 * π * base_radius

/-- 
If the lateral surface of a cone forms a semicircle when unfolded, 
then the ratio of the length of the cone's generatrix to the radius of its base is 2:1
-/
theorem unfoldable_cone_ratio (cone : UnfoldableCone) : 
  cone.generatrix_length / cone.base_radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_unfoldable_cone_ratio_l3834_383438


namespace NUMINAMATH_CALUDE_min_equation_implies_sum_l3834_383448

theorem min_equation_implies_sum (a b c d : ℝ) :
  (∀ x : ℝ, min (20 * x + 19) (19 * x + 20) = (a * x + b) - |c * x + d|) →
  a * b + c * d = 380 := by
  sorry

end NUMINAMATH_CALUDE_min_equation_implies_sum_l3834_383448


namespace NUMINAMATH_CALUDE_product_of_primes_l3834_383447

theorem product_of_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p * q = 69 →
  13 < q →
  q < 25 →
  15 < p * q →
  p * q < 70 →
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_l3834_383447


namespace NUMINAMATH_CALUDE_banana_price_reduction_l3834_383488

/-- Calculates the reduced price per dozen bananas given the original price and quantity change --/
def reduced_price_per_dozen (original_price : ℝ) (original_quantity : ℕ) : ℝ :=
  let reduced_price := 0.6 * original_price
  let new_quantity := original_quantity + 50
  let price_per_banana := 40 / new_quantity
  12 * price_per_banana

/-- Theorem stating the conditions and the result to be proved --/
theorem banana_price_reduction 
  (original_price : ℝ) 
  (original_quantity : ℕ) 
  (h1 : original_price * original_quantity = 40) 
  (h2 : 0.6 * original_price * (original_quantity + 50) = 40) :
  reduced_price_per_dozen original_price original_quantity = 3.84 :=
by sorry

#eval reduced_price_per_dozen (40 / 75) 75

end NUMINAMATH_CALUDE_banana_price_reduction_l3834_383488


namespace NUMINAMATH_CALUDE_reuschles_theorem_l3834_383493

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points A₁, B₁, C₁ on the sides of triangle ABC
variable (A₁ B₁ C₁ : ℝ × ℝ)

-- Define the condition that AA₁, BB₁, CC₁ intersect at a single point
def lines_concurrent (A B C A₁ B₁ C₁ : ℝ × ℝ) : Prop := sorry

-- Define the circumcircle of triangle A₁B₁C₁
def circumcircle (A₁ B₁ C₁ : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define points A₂, B₂, C₂ as the second intersection points
def second_intersection (A B C A₁ B₁ C₁ : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Theorem statement
theorem reuschles_theorem (A B C A₁ B₁ C₁ : ℝ × ℝ) :
  lines_concurrent A B C A₁ B₁ C₁ →
  let (A₂, B₂, C₂) := second_intersection A B C A₁ B₁ C₁
  lines_concurrent A B C A₂ B₂ C₂ := by sorry

end NUMINAMATH_CALUDE_reuschles_theorem_l3834_383493


namespace NUMINAMATH_CALUDE_part1_part2_l3834_383420

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| - 1

-- Part 1
theorem part1 (m : ℝ) : 
  (∀ x, f m x ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) → m = 2 := by sorry

-- Part 2
theorem part2 (t : ℝ) :
  (∀ x, f 2 x + f 2 (x + 5) ≥ t - 2) → t ≤ 5 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3834_383420


namespace NUMINAMATH_CALUDE_genevieve_code_lines_l3834_383454

/-- Represents the number of lines of code per debugging session -/
def lines_per_debug : ℕ := 100

/-- Represents the number of errors found per debugging session -/
def errors_per_debug : ℕ := 3

/-- Represents the total number of errors fixed so far -/
def total_errors_fixed : ℕ := 129

/-- Calculates the number of lines of code written based on the given conditions -/
def lines_of_code : ℕ := (total_errors_fixed / errors_per_debug) * lines_per_debug

/-- Theorem stating that the number of lines of code written is 4300 -/
theorem genevieve_code_lines : lines_of_code = 4300 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_code_lines_l3834_383454


namespace NUMINAMATH_CALUDE_actual_average_height_l3834_383418

/-- The number of boys in the class -/
def num_boys : ℕ := 50

/-- The initially calculated average height in cm -/
def initial_avg : ℝ := 175

/-- The incorrectly recorded heights of three boys in cm -/
def incorrect_heights : List ℝ := [155, 185, 170]

/-- The actual heights of the three boys in cm -/
def actual_heights : List ℝ := [145, 195, 160]

/-- The actual average height of the boys in the class -/
def actual_avg : ℝ := 174.8

theorem actual_average_height :
  let total_incorrect := num_boys * initial_avg
  let height_difference := (List.sum incorrect_heights) - (List.sum actual_heights)
  let total_correct := total_incorrect - height_difference
  (total_correct / num_boys) = actual_avg :=
sorry

end NUMINAMATH_CALUDE_actual_average_height_l3834_383418


namespace NUMINAMATH_CALUDE_f_lower_bound_l3834_383469

/-- Given f(x) = e^x - x^2 - 1 for all x ∈ ℝ, prove that f(x) ≥ x^2 + x for all x ∈ ℝ. -/
theorem f_lower_bound (x : ℝ) : Real.exp x - x^2 - 1 ≥ x^2 + x := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_l3834_383469


namespace NUMINAMATH_CALUDE_steven_more_apples_l3834_383471

/-- The number of apples Steven has -/
def steven_apples : ℕ := 19

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 15

/-- The difference between Steven's apples and peaches -/
def apple_peach_difference : ℤ := steven_apples - steven_peaches

theorem steven_more_apples : apple_peach_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_steven_more_apples_l3834_383471


namespace NUMINAMATH_CALUDE_rotate180_unique_l3834_383449

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a rigid transformation (isometry) in 2D space -/
def RigidTransformation := Point2D → Point2D

/-- Clockwise rotation by 180° about the origin -/
def rotate180 : RigidTransformation :=
  fun p => Point2D.mk (-p.x) (-p.y)

/-- The given points -/
def C : Point2D := Point2D.mk 3 (-2)
def D : Point2D := Point2D.mk 4 (-5)
def C' : Point2D := Point2D.mk (-3) 2
def D' : Point2D := Point2D.mk (-4) 5

/-- Statement: rotate180 is the unique isometry that maps C to C' and D to D' -/
theorem rotate180_unique : 
  (rotate180 C = C') ∧ 
  (rotate180 D = D') ∧ 
  (∀ (f : RigidTransformation), (f C = C' ∧ f D = D') → f = rotate180) :=
sorry

end NUMINAMATH_CALUDE_rotate180_unique_l3834_383449


namespace NUMINAMATH_CALUDE_trivia_game_score_l3834_383483

/-- Represents the score distribution in a trivia game --/
structure TriviaGame where
  total_members : Float
  absent_members : Float
  total_points : Float

/-- Calculates the score per member for a given trivia game --/
def score_per_member (game : TriviaGame) : Float :=
  game.total_points / (game.total_members - game.absent_members)

/-- Theorem: In the given trivia game scenario, each member scores 2.0 points --/
theorem trivia_game_score :
  let game := TriviaGame.mk 5.0 2.0 6.0
  score_per_member game = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_score_l3834_383483


namespace NUMINAMATH_CALUDE_beta_values_l3834_383480

theorem beta_values (β : ℂ) (h1 : β ≠ 1) 
  (h2 : Complex.abs (β^3 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^6 - 1) = 6 * Complex.abs (β - 1)) :
  β = Complex.I * Real.sqrt 2 ∨ β = -Complex.I * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_beta_values_l3834_383480


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3834_383402

theorem quadratic_equal_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 = 0 ∧ (∀ y : ℝ, y^2 - a*y + 1 = 0 → y = x)) → 
  a = 2 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3834_383402


namespace NUMINAMATH_CALUDE_pechkin_ate_four_tenths_l3834_383436

/-- The fraction of the cake eaten by each person -/
structure CakeFractions where
  pechkin : ℝ
  fyodor : ℝ
  matroskin : ℝ
  sharik : ℝ

/-- The conditions of the cake-eating problem -/
def cake_problem (f : CakeFractions) : Prop :=
  -- The whole cake was eaten
  f.pechkin + f.fyodor + f.matroskin + f.sharik = 1 ∧
  -- Uncle Fyodor ate half as much as Pechkin
  f.fyodor = f.pechkin / 2 ∧
  -- Cat Matroskin ate half as much as the portion of the cake that Pechkin did not eat
  f.matroskin = (1 - f.pechkin) / 2 ∧
  -- Sharik ate one-tenth of the cake
  f.sharik = 1 / 10

/-- Theorem stating that given the conditions, Pechkin ate 0.4 of the cake -/
theorem pechkin_ate_four_tenths (f : CakeFractions) :
  cake_problem f → f.pechkin = 0.4 := by sorry

end NUMINAMATH_CALUDE_pechkin_ate_four_tenths_l3834_383436


namespace NUMINAMATH_CALUDE_rectangle_long_side_l3834_383489

/-- Given a rectangle with perimeter 30 cm and short side 7 cm, prove the long side is 8 cm -/
theorem rectangle_long_side (perimeter : ℝ) (short_side : ℝ) (long_side : ℝ) : 
  perimeter = 30 ∧ short_side = 7 ∧ perimeter = 2 * (short_side + long_side) → long_side = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_long_side_l3834_383489


namespace NUMINAMATH_CALUDE_package_size_l3834_383404

/-- The number of candies Shirley ate -/
def candies_eaten : ℕ := 10

/-- The number of candies Shirley has left -/
def candies_left : ℕ := 2

/-- The number of candies in one package -/
def candies_in_package : ℕ := candies_eaten + candies_left

theorem package_size : candies_in_package = 12 := by
  sorry

end NUMINAMATH_CALUDE_package_size_l3834_383404


namespace NUMINAMATH_CALUDE_sum_of_cubes_is_twelve_l3834_383446

/-- Given real numbers a, b, and c satisfying certain conditions, 
    prove that the sum of their cubes is 12. -/
theorem sum_of_cubes_is_twelve (a b c : ℝ) 
    (sum_eq_three : a + b + c = 3)
    (sum_of_products_eq_three : a * b + a * c + b * c = 3)
    (product_eq_neg_one : a * b * c = -1) : 
  a^3 + b^3 + c^3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_is_twelve_l3834_383446


namespace NUMINAMATH_CALUDE_calculator_sum_theorem_l3834_383476

/-- The number of participants in the game --/
def num_participants : ℕ := 47

/-- The initial value of calculator A --/
def initial_A : ℤ := 2

/-- The initial value of calculator B --/
def initial_B : ℕ := 0

/-- The initial value of calculator C --/
def initial_C : ℤ := -1

/-- The initial value of calculator D --/
def initial_D : ℕ := 3

/-- The final value of calculator A after all participants have processed it --/
def final_A : ℤ := -initial_A

/-- The final value of calculator B after all participants have processed it --/
def final_B : ℕ := initial_B

/-- The final value of calculator C after all participants have processed it --/
def final_C : ℤ := -initial_C

/-- The final value of calculator D after all participants have processed it --/
noncomputable def final_D : ℕ := initial_D ^ (3 ^ num_participants)

/-- The theorem stating that the sum of the final calculator values equals 3^(3^47) - 3 --/
theorem calculator_sum_theorem :
  final_A + final_B + final_C + final_D = 3^(3^47) - 3 := by
  sorry


end NUMINAMATH_CALUDE_calculator_sum_theorem_l3834_383476


namespace NUMINAMATH_CALUDE_volleyball_ticket_sales_l3834_383437

theorem volleyball_ticket_sales (total_tickets : ℕ) (tickets_left : ℕ) : 
  total_tickets = 100 →
  tickets_left = 40 →
  ∃ (jude_tickets : ℕ),
    (jude_tickets : ℚ) + 2 * (jude_tickets : ℚ) + ((1/2 : ℚ) * (jude_tickets : ℚ) + 4) = (total_tickets - tickets_left : ℚ) ∧
    jude_tickets = 16 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_ticket_sales_l3834_383437


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3834_383405

/-- The perimeter of a triangle with vertices A(1,2), B(1,5), and C(4,5) on a Cartesian coordinate plane is 6 + 3√2. -/
theorem triangle_perimeter : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (1, 5)
  let C : ℝ × ℝ := (4, 5)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := distance A B + distance B C + distance C A
  perimeter = 6 + 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3834_383405


namespace NUMINAMATH_CALUDE_min_value_of_x_l3834_383423

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 3 + (2/3) * Real.log x) : x ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l3834_383423


namespace NUMINAMATH_CALUDE_eggs_per_basket_l3834_383410

theorem eggs_per_basket (red_eggs blue_eggs min_eggs : ℕ) 
  (h1 : red_eggs = 30)
  (h2 : blue_eggs = 42)
  (h3 : min_eggs = 5) :
  ∃ (n : ℕ), n ≥ min_eggs ∧ 
             n ∣ red_eggs ∧ 
             n ∣ blue_eggs ∧
             ∀ (m : ℕ), m ≥ min_eggs ∧ m ∣ red_eggs ∧ m ∣ blue_eggs → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l3834_383410


namespace NUMINAMATH_CALUDE_cost_of_one_plank_l3834_383465

/-- The cost of one plank given the conditions for building birdhouses -/
theorem cost_of_one_plank : 
  ∀ (plank_cost : ℝ),
  (4 * (7 * plank_cost + 20 * 0.05) = 88) →
  plank_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_plank_l3834_383465


namespace NUMINAMATH_CALUDE_f_min_value_when_a_is_one_f_inequality_solution_range_l3834_383427

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x + a|

-- Theorem 1: Minimum value of f when a = 1
theorem f_min_value_when_a_is_one :
  ∃ (min : ℝ), min = 3/2 ∧ ∀ (x : ℝ), f x 1 ≥ min :=
sorry

-- Theorem 2: Range of a for which f(x) < 5/x + a has a solution in [1, 2]
theorem f_inequality_solution_range :
  ∀ (a : ℝ), a > 0 →
    (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f x a < 5/x + a) ↔ (11/2 < a ∧ a < 9/2) :=
sorry

end NUMINAMATH_CALUDE_f_min_value_when_a_is_one_f_inequality_solution_range_l3834_383427


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l3834_383479

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

/-- Theorem stating the condition for f to have both maximum and minimum values -/
theorem f_has_max_and_min (a : ℝ) : 
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) ↔ (a < -3 ∨ a > 6) :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l3834_383479


namespace NUMINAMATH_CALUDE_f_47_mod_17_l3834_383425

def f (n : ℕ) : ℕ := 3^n + 7^n

theorem f_47_mod_17 : f 47 % 17 = 10 := by
  sorry

end NUMINAMATH_CALUDE_f_47_mod_17_l3834_383425


namespace NUMINAMATH_CALUDE_cubic_function_nonnegative_implies_parameter_bound_l3834_383434

theorem cubic_function_nonnegative_implies_parameter_bound 
  (f : ℝ → ℝ) (a : ℝ) 
  (h_def : ∀ x, f x = a * x^3 - 3 * x + 1)
  (h_nonneg : ∀ x ∈ Set.Icc 0 1, f x ≥ 0) :
  a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_nonnegative_implies_parameter_bound_l3834_383434


namespace NUMINAMATH_CALUDE_squares_four_greater_than_prime_l3834_383401

theorem squares_four_greater_than_prime :
  ∃! n : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n^2 = p + 4 :=
sorry

end NUMINAMATH_CALUDE_squares_four_greater_than_prime_l3834_383401


namespace NUMINAMATH_CALUDE_range_of_a_l3834_383492

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (2 * (1 - x^2) - 3 * x > 0 → x > a) ∧ 
  (∃ y : ℝ, y > a ∧ 2 * (1 - y^2) - 3 * y ≤ 0)) → 
  a ∈ Set.Iic (-2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3834_383492


namespace NUMINAMATH_CALUDE_parallel_perpendicular_relation_l3834_383411

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- Theorem statement
theorem parallel_perpendicular_relation 
  (m n : Line) (α β : Plane) :
  (parallel m α ∧ parallel n β ∧ perpendicular α β) → 
  (line_perpendicular m n ∨ line_parallel m n) = False := by
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_relation_l3834_383411


namespace NUMINAMATH_CALUDE_line_equations_l3834_383495

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 3 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y + 5 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, 2)

-- Define the perpendicular line l
def l (x y : ℝ) : Prop := 3 * x + 2 * y - 7 = 0

-- Define the parallel line l'
def l' (x y : ℝ) : Prop := x - 2 * y + 3 = 0

theorem line_equations :
  (∀ x y : ℝ, l₁ x y ∧ l₂ x y → (x, y) = M) →
  (∀ x y : ℝ, l x y ↔ (3 * x + 2 * y - 7 = 0 ∧ (x, y) = M ∨ l₁ x y)) →
  (∀ x y : ℝ, l' x y ↔ (x - 2 * y + 3 = 0 ∧ (x, y) = M ∨ l₃ x y)) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l3834_383495


namespace NUMINAMATH_CALUDE_percentage_not_sophomores_l3834_383487

theorem percentage_not_sophomores :
  ∀ (total juniors seniors freshmen sophomores : ℕ),
    total = 800 →
    juniors = (22 * total) / 100 →
    seniors = 160 →
    freshmen = sophomores + 48 →
    total = freshmen + sophomores + juniors + seniors →
    (100 * (total - sophomores)) / total = 74 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_sophomores_l3834_383487


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l3834_383452

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_power_of_two_dividing (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1).log 2) 0

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  ones_digit (2^(largest_power_of_two_dividing (factorial 32))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l3834_383452


namespace NUMINAMATH_CALUDE_sequence_inequality_l3834_383484

-- Define the sequence a_n
def a (n k : ℤ) : ℝ := |n - k| + |n + 2*k|

-- State the theorem
theorem sequence_inequality (k : ℤ) :
  (∀ n : ℕ, a n k ≥ a 3 k) ∧ (a 3 k = a 4 k) →
  k ≤ -2 ∨ k ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3834_383484


namespace NUMINAMATH_CALUDE_first_purchase_quantities_second_purchase_max_profit_new_selling_price_B_l3834_383433

-- Definitions based on the problem conditions
def purchase_price_A : ℝ := 30
def purchase_price_B : ℝ := 25
def selling_price_A : ℝ := 45
def selling_price_B : ℝ := 37
def total_keychains : ℕ := 30
def total_cost : ℝ := 850
def second_purchase_total : ℕ := 80
def second_purchase_max_cost : ℝ := 2200
def original_daily_sales_B : ℕ := 4
def price_reduction_effect : ℝ := 2

-- Part 1
theorem first_purchase_quantities (x y : ℕ) :
  purchase_price_A * x + purchase_price_B * y = total_cost ∧
  x + y = total_keychains →
  x = 20 ∧ y = 10 := by sorry

-- Part 2
theorem second_purchase_max_profit (m : ℕ) :
  m ≤ 40 →
  ∃ (w : ℝ), w = 3 * m + 960 ∧
  w ≤ 1080 ∧
  (m = 40 → w = 1080) := by sorry

-- Part 3
theorem new_selling_price_B (a : ℝ) :
  (a - purchase_price_B) * (78 - 2 * a) = 90 →
  a = 30 ∨ a = 34 := by sorry

end NUMINAMATH_CALUDE_first_purchase_quantities_second_purchase_max_profit_new_selling_price_B_l3834_383433


namespace NUMINAMATH_CALUDE_cone_base_radius_l3834_383413

/-- Given a cone with slant height 12 cm and central angle of unfolded lateral surface 150°, 
    the radius of its base is 5 cm. -/
theorem cone_base_radius (slant_height : ℝ) (central_angle : ℝ) : 
  slant_height = 12 → central_angle = 150 → ∃ (base_radius : ℝ), base_radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3834_383413


namespace NUMINAMATH_CALUDE_ring_diameter_theorem_l3834_383461

/-- The diameter of ring X -/
def diameter_X : ℝ := 16

/-- The fraction of ring X's surface not covered by ring Y -/
def uncovered_fraction : ℝ := 0.2098765432098765

/-- The diameter of ring Y -/
noncomputable def diameter_Y : ℝ := 14.222

/-- Theorem stating that given the diameter of ring X and the uncovered fraction,
    the diameter of ring Y is approximately 14.222 inches -/
theorem ring_diameter_theorem (ε : ℝ) (h : ε > 0) :
  ∃ (d : ℝ), abs (d - diameter_Y) < ε ∧ 
  d^2 / 4 = diameter_X^2 / 4 * (1 - uncovered_fraction) :=
sorry

end NUMINAMATH_CALUDE_ring_diameter_theorem_l3834_383461


namespace NUMINAMATH_CALUDE_common_points_characterization_l3834_383417

-- Define the square S
def S : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the set C_t
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p ∈ S | p.2 ≥ ((1 - t) / t) * p.1 + (1 - t)}

-- Define the intersection of all C_t
def CommonPoints : Set (ℝ × ℝ) :=
  ⋂ t ∈ {t | 0 < t ∧ t < 1}, C t

-- State the theorem
theorem common_points_characterization :
  ∀ p ∈ S, p ∈ CommonPoints ↔ Real.sqrt p.1 + Real.sqrt p.2 ≥ 1 := by sorry

end NUMINAMATH_CALUDE_common_points_characterization_l3834_383417


namespace NUMINAMATH_CALUDE_commuting_days_businessman_commute_l3834_383459

/-- Represents the commuting options for a businessman over a period of days. -/
structure CommutingData where
  /-- Total number of days -/
  total_days : ℕ
  /-- Number of times taking bus to work in the morning -/
  morning_bus : ℕ
  /-- Number of times coming home by bus in the afternoon -/
  afternoon_bus : ℕ
  /-- Number of train commuting segments (either morning or afternoon) -/
  train_segments : ℕ

/-- Theorem stating that given the commuting conditions, the total number of days is 32 -/
theorem commuting_days (data : CommutingData) : 
  data.morning_bus = 12 ∧ 
  data.afternoon_bus = 20 ∧ 
  data.train_segments = 15 →
  data.total_days = 32 := by
  sorry

/-- Main theorem proving the specific case -/
theorem businessman_commute : ∃ (data : CommutingData), 
  data.morning_bus = 12 ∧ 
  data.afternoon_bus = 20 ∧ 
  data.train_segments = 15 ∧
  data.total_days = 32 := by
  sorry

end NUMINAMATH_CALUDE_commuting_days_businessman_commute_l3834_383459


namespace NUMINAMATH_CALUDE_min_sum_squares_l3834_383468

theorem min_sum_squares (a b c d : ℝ) 
  (h1 : a + b = 9 / (c - d)) 
  (h2 : c + d = 25 / (a - b)) : 
  ∀ x y z w : ℝ, x^2 + y^2 + z^2 + w^2 ≥ 34 ∧ 
  (∃ a b c d : ℝ, a^2 + b^2 + c^2 + d^2 = 34 ∧ 
   a + b = 9 / (c - d) ∧ c + d = 25 / (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3834_383468


namespace NUMINAMATH_CALUDE_solve_equations_and_sum_l3834_383419

/-- Given two equations involving x and y, prove the values of x, y, and their sum. -/
theorem solve_equations_and_sum :
  ∀ (x y : ℝ),
  (0.65 * x = 0.20 * 552.50) →
  (0.35 * y = 0.30 * 867.30) →
  (x = 170) ∧ (y = 743.40) ∧ (x + y = 913.40) := by
  sorry

end NUMINAMATH_CALUDE_solve_equations_and_sum_l3834_383419


namespace NUMINAMATH_CALUDE_c_necessary_not_sufficient_l3834_383485

-- Define the proposition p
def p (x : ℝ) : Prop := x^2 - x < 0

-- Define the condition c
def c (x : ℝ) : Prop := -1 < x ∧ x < 1

-- Theorem stating that c is a necessary but not sufficient condition for p
theorem c_necessary_not_sufficient :
  (∀ x : ℝ, p x → c x) ∧ 
  (∃ x : ℝ, c x ∧ ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_c_necessary_not_sufficient_l3834_383485


namespace NUMINAMATH_CALUDE_jake_newspaper_count_l3834_383481

/-- The number of newspapers Jake delivers in a week -/
def jake_newspapers : ℕ := 234

/-- The number of newspapers Miranda delivers in a week -/
def miranda_newspapers : ℕ := 2 * jake_newspapers

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

theorem jake_newspaper_count : jake_newspapers = 234 :=
  by
    have h1 : miranda_newspapers = 2 * jake_newspapers := by rfl
    have h2 : weeks_in_month * miranda_newspapers - weeks_in_month * jake_newspapers = 936 :=
      by sorry
    sorry

end NUMINAMATH_CALUDE_jake_newspaper_count_l3834_383481


namespace NUMINAMATH_CALUDE_arun_speed_ratio_l3834_383462

/-- Represents the problem of finding the ratio of Arun's new speed to his original speed. -/
theorem arun_speed_ratio :
  let distance : ℝ := 30
  let arun_original_speed : ℝ := 5
  let anil_time := distance / anil_speed
  let arun_original_time := distance / arun_original_speed
  let arun_new_time := distance / arun_new_speed
  arun_original_time = anil_time + 2 →
  arun_new_time = anil_time - 1 →
  arun_new_speed / arun_original_speed = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_arun_speed_ratio_l3834_383462
