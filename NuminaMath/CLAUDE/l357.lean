import Mathlib

namespace NUMINAMATH_CALUDE_initial_overs_calculation_l357_35787

/-- Proves the number of initial overs in a cricket game given specific conditions --/
theorem initial_overs_calculation (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : target = 272) (h2 : initial_rate = 3.2) (h3 : required_rate = 6) 
  (h4 : remaining_overs = 40) :
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
  target = initial_rate * initial_overs + required_rate * remaining_overs :=
by
  sorry


end NUMINAMATH_CALUDE_initial_overs_calculation_l357_35787


namespace NUMINAMATH_CALUDE_cookie_sheet_length_l357_35713

/-- Given a rectangle with width 10 inches and perimeter 24 inches, prove its length is 2 inches. -/
theorem cookie_sheet_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 10 → perimeter = 24 → perimeter = 2 * (length + width) → length = 2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sheet_length_l357_35713


namespace NUMINAMATH_CALUDE_sum_of_fractions_l357_35794

theorem sum_of_fractions : 
  (1 / (1 * 2 : ℚ)) + (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + 
  (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l357_35794


namespace NUMINAMATH_CALUDE_percentage_both_correct_l357_35721

theorem percentage_both_correct (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) : 
  p_first = 0.63 → p_second = 0.50 → p_neither = 0.20 → 
  p_first + p_second - (1 - p_neither) = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_percentage_both_correct_l357_35721


namespace NUMINAMATH_CALUDE_quadratic_factorization_l357_35788

theorem quadratic_factorization (y : ℝ) (a b : ℤ) 
  (h : ∀ y, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) : 
  a - b = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l357_35788


namespace NUMINAMATH_CALUDE_head_start_calculation_l357_35770

/-- Proves that the head start given by A to B is 72 meters in a 96-meter race,
    given that A runs 4 times as fast as B and they finish at the same time. -/
theorem head_start_calculation (v_B : ℝ) (d : ℝ) 
  (h1 : v_B > 0)  -- B's speed is positive
  (h2 : 96 > d)   -- The head start is less than the total race distance
  (h3 : 96 / (4 * v_B) = (96 - d) / v_B)  -- A and B finish at the same time
  : d = 72 := by
  sorry

end NUMINAMATH_CALUDE_head_start_calculation_l357_35770


namespace NUMINAMATH_CALUDE_apple_price_36kg_l357_35792

/-- The price of apples with a two-tier pricing system -/
def apple_price (l q : ℚ) (kg : ℚ) : ℚ :=
  if kg ≤ 30 then l * kg
  else l * 30 + q * (kg - 30)

theorem apple_price_36kg (l q : ℚ) :
  (apple_price l q 33 = 360) →
  (apple_price l q 25 = 250) →
  (apple_price l q 36 = 420) :=
by sorry

end NUMINAMATH_CALUDE_apple_price_36kg_l357_35792


namespace NUMINAMATH_CALUDE_floor_ceiling_identity_l357_35764

theorem floor_ceiling_identity (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ⌊x⌋ + x - ⌈x⌉ = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_identity_l357_35764


namespace NUMINAMATH_CALUDE_envelope_addressing_equation_l357_35745

theorem envelope_addressing_equation (x : ℝ) : x > 0 → (
  let rate1 := 500 / 8
  let rate2 := 500 / x
  let combined_rate := 500 / 2
  rate1 + rate2 = combined_rate
) ↔ 1/8 + 1/x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_envelope_addressing_equation_l357_35745


namespace NUMINAMATH_CALUDE_linear_equation_solution_l357_35776

/-- Given a linear equation y = kx + b, prove the values of k and b,
    and find x for a specific y value. -/
theorem linear_equation_solution (k b : ℝ) :
  (4 * k + b = -20 ∧ -2 * k + b = 16) →
  (k = -6 ∧ b = 4) ∧
  (∀ x : ℝ, -6 * x + 4 = -8 → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l357_35776


namespace NUMINAMATH_CALUDE_angle_value_proof_l357_35739

theorem angle_value_proof (α β : Real) : 
  0 < α ∧ α < π ∧ 0 < β ∧ β < π →
  Real.tan (α - β) = 1/2 →
  Real.tan β = -1/7 →
  2*α - β = -3*π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_value_proof_l357_35739


namespace NUMINAMATH_CALUDE_sales_price_calculation_l357_35757

theorem sales_price_calculation (C G : ℝ) (h1 : G = 1.6 * C) (h2 : G = 56) :
  C + G = 91 := by
  sorry

end NUMINAMATH_CALUDE_sales_price_calculation_l357_35757


namespace NUMINAMATH_CALUDE_wholesale_price_calculation_l357_35722

/-- Proves that the wholesale price of a machine is $90 given the specified conditions -/
theorem wholesale_price_calculation (retail_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  retail_price = 120 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (wholesale_price : ℝ),
    wholesale_price = 90 ∧
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_wholesale_price_calculation_l357_35722


namespace NUMINAMATH_CALUDE_item_list_price_l357_35748

/-- The list price of an item -/
def list_price : ℝ := 40

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.15

/-- Bob's commission rate -/
def bob_rate : ℝ := 0.25

/-- Alice's commission -/
def alice_commission (x : ℝ) : ℝ := alice_rate * alice_price x

/-- Bob's commission -/
def bob_commission (x : ℝ) : ℝ := bob_rate * bob_price x

theorem item_list_price :
  alice_commission list_price = bob_commission list_price :=
by sorry

end NUMINAMATH_CALUDE_item_list_price_l357_35748


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l357_35725

/-- A geometric sequence with positive terms, where a₁ = 1 and a₁ + a₂ + a₃ = 7 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n) ∧
  a 1 = 1 ∧
  a 1 + a 2 + a 3 = 7

/-- The general term of the geometric sequence is 2^(n-1) -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l357_35725


namespace NUMINAMATH_CALUDE_percentage_relation_l357_35737

theorem percentage_relation (x y z : ℝ) 
  (h1 : x = 0.2 * y) 
  (h2 : x = 0.5 * z) : 
  z = 0.4 * y := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l357_35737


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l357_35746

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 1 * a 4 * a 7 = 27) : 
  a 3 * a 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l357_35746


namespace NUMINAMATH_CALUDE_no_solution_lcm_equation_l357_35768

theorem no_solution_lcm_equation :
  ¬∃ (n m : ℕ), Nat.lcm (n^2) m + Nat.lcm n (m^2) = 2019 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_lcm_equation_l357_35768


namespace NUMINAMATH_CALUDE_christophers_age_l357_35727

/-- Proves Christopher's age given the conditions of the problem -/
theorem christophers_age (christopher george ford : ℕ) 
  (h1 : george = christopher + 8)
  (h2 : ford = christopher - 2)
  (h3 : christopher + george + ford = 60) :
  christopher = 18 := by
  sorry

end NUMINAMATH_CALUDE_christophers_age_l357_35727


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l357_35777

/-- Given a geometric series with positive terms {a_n}, if a_1, 1/2 * a_3, and 2 * a_2 form an arithmetic sequence, then a_5 / a_3 = 3 + 2√2 -/
theorem geometric_series_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : (a 1 + 2 * a 2) / 2 = a 3 / 2) :
  a 5 / a 3 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l357_35777


namespace NUMINAMATH_CALUDE_zoo_new_species_l357_35797

theorem zoo_new_species (initial_types : ℕ) (time_per_type : ℕ) (total_time_after : ℕ) : 
  initial_types = 5 → 
  time_per_type = 6 → 
  total_time_after = 54 → 
  (initial_types + (total_time_after / time_per_type - initial_types)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_zoo_new_species_l357_35797


namespace NUMINAMATH_CALUDE_exists_valid_solution_l357_35755

def mother_charge : ℝ := 6.50
def child_charge_per_year : ℝ := 0.65
def total_bill : ℝ := 13.00

def is_valid_solution (twin_age youngest_age : ℕ) : Prop :=
  twin_age > youngest_age ∧
  mother_charge + child_charge_per_year * (2 * twin_age + youngest_age) = total_bill

theorem exists_valid_solution :
  ∃ (twin_age youngest_age : ℕ), is_valid_solution twin_age youngest_age ∧ (youngest_age = 2 ∨ youngest_age = 4) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_solution_l357_35755


namespace NUMINAMATH_CALUDE_power_calculation_l357_35711

theorem power_calculation : 3000 * (3000^3000)^2 = 3000^6001 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l357_35711


namespace NUMINAMATH_CALUDE_weather_period_days_l357_35736

/-- Represents the weather conditions over a period of time. -/
structure WeatherPeriod where
  totalRainyDays : ℕ
  clearEvenings : ℕ
  clearMornings : ℕ
  morningRainImpliesClearEvening : Unit
  eveningRainImpliesClearMorning : Unit

/-- Calculates the total number of days in the weather period. -/
def totalDays (w : WeatherPeriod) : ℕ :=
  w.totalRainyDays + (w.clearEvenings + w.clearMornings - w.totalRainyDays) / 2

/-- Theorem stating that given the specific weather conditions, the total period is 11 days. -/
theorem weather_period_days (w : WeatherPeriod)
  (h1 : w.totalRainyDays = 9)
  (h2 : w.clearEvenings = 6)
  (h3 : w.clearMornings = 7) :
  totalDays w = 11 := by
  sorry

end NUMINAMATH_CALUDE_weather_period_days_l357_35736


namespace NUMINAMATH_CALUDE_carrie_iphone_weeks_l357_35703

/-- Proves that Carrie needs to work 7 weeks to buy the iPhone -/
theorem carrie_iphone_weeks : 
  ∀ (iphone_cost trade_in_value weekly_earnings : ℕ),
    iphone_cost = 800 →
    trade_in_value = 240 →
    weekly_earnings = 80 →
    (iphone_cost - trade_in_value) / weekly_earnings = 7 := by
  sorry

end NUMINAMATH_CALUDE_carrie_iphone_weeks_l357_35703


namespace NUMINAMATH_CALUDE_common_chord_theorem_l357_35714

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 3*x - 3*y + 3 = 0

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 0

/-- The equation of the line containing the common chord -/
def common_chord_line (x y : ℝ) : Prop := x + y - 3 = 0

/-- Theorem stating the equation of the common chord and its length -/
theorem common_chord_theorem :
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord_line x y) ∧
  (∃ a b c d : ℝ, C₁ a b ∧ C₂ a b ∧ C₁ c d ∧ C₂ c d ∧
    common_chord_line a b ∧ common_chord_line c d ∧
    ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 6^(1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_theorem_l357_35714


namespace NUMINAMATH_CALUDE_elberta_amount_l357_35789

-- Define the amounts for each person
def granny_smith : ℕ := 75
def anjou : ℕ := granny_smith / 4
def elberta : ℕ := anjou + 3

-- Theorem statement
theorem elberta_amount : elberta = 22 := by
  sorry

end NUMINAMATH_CALUDE_elberta_amount_l357_35789


namespace NUMINAMATH_CALUDE_parabola_intersection_l357_35763

theorem parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 - 4 * y + 7) → k = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l357_35763


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l357_35720

/-- Proves that mixing 300 mL of 10% alcohol solution with 100 mL of 30% alcohol solution results in a 15% alcohol solution -/
theorem alcohol_mixture_proof :
  let solution_x_volume : ℝ := 300
  let solution_x_concentration : ℝ := 0.10
  let solution_y_volume : ℝ := 100
  let solution_y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.15
  let total_volume := solution_x_volume + solution_y_volume
  let total_alcohol := solution_x_volume * solution_x_concentration + solution_y_volume * solution_y_concentration
  total_alcohol / total_volume = target_concentration := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l357_35720


namespace NUMINAMATH_CALUDE_lcm_812_3214_l357_35775

theorem lcm_812_3214 : Nat.lcm 812 3214 = 1304124 := by
  sorry

end NUMINAMATH_CALUDE_lcm_812_3214_l357_35775


namespace NUMINAMATH_CALUDE_property_length_proof_l357_35728

/-- Given a rectangular property and a garden within it, prove the length of the property. -/
theorem property_length_proof (property_width : ℝ) (garden_area : ℝ) : 
  property_width = 1000 →
  garden_area = 28125 →
  ∃ (property_length : ℝ),
    property_length = 2250 ∧
    garden_area = (property_width / 8) * (property_length / 10) :=
by sorry

end NUMINAMATH_CALUDE_property_length_proof_l357_35728


namespace NUMINAMATH_CALUDE_complex_equation_solution_l357_35758

/-- Given that (1-√3i)z = √3+i, prove that z = i -/
theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I * Real.sqrt 3) * z = Real.sqrt 3 + Complex.I) : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l357_35758


namespace NUMINAMATH_CALUDE_selection_methods_l357_35783

theorem selection_methods (female_students male_students : ℕ) 
  (h1 : female_students = 3) 
  (h2 : male_students = 2) : 
  female_students + male_students = 5 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_l357_35783


namespace NUMINAMATH_CALUDE_coffee_pod_box_cost_l357_35742

/-- Calculates the cost of a box of coffee pods given vacation details --/
theorem coffee_pod_box_cost
  (vacation_days : ℕ)
  (daily_pods : ℕ)
  (pods_per_box : ℕ)
  (total_spending : ℚ)
  (h1 : vacation_days = 40)
  (h2 : daily_pods = 3)
  (h3 : pods_per_box = 30)
  (h4 : total_spending = 32)
  : (total_spending / (vacation_days * daily_pods / pods_per_box : ℚ)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_coffee_pod_box_cost_l357_35742


namespace NUMINAMATH_CALUDE_sum_x_y_equals_six_l357_35769

theorem sum_x_y_equals_six (x y : ℝ) 
  (h1 : x^2 + y^2 = 8*x + 4*y - 20) 
  (h2 : x + y = 6) : 
  x + y = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_six_l357_35769


namespace NUMINAMATH_CALUDE_subtraction_proof_l357_35706

theorem subtraction_proof : 25.705 - 3.289 = 22.416 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_proof_l357_35706


namespace NUMINAMATH_CALUDE_minimal_area_circle_equation_l357_35700

/-- The standard equation of a circle with minimal area, given that its center is on the curve y² = x
    and it is tangent to the line x + 2y + 6 = 0 -/
theorem minimal_area_circle_equation (x y : ℝ) : 
  (∃ (cx cy : ℝ), cy^2 = cx ∧ (x - cx)^2 + (y - cy)^2 = ((x + 2*y + 6) / Real.sqrt 5)^2) →
  (x - 1)^2 + (y + 1)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_minimal_area_circle_equation_l357_35700


namespace NUMINAMATH_CALUDE_triangle_altitude_proof_l357_35744

theorem triangle_altitude_proof (a b c h : ℝ) : 
  a = 13 ∧ b = 15 ∧ c = 22 →
  a + b > c ∧ a + c > b ∧ b + c > a →
  h = (30 * Real.sqrt 10) / 11 →
  (1 / 2) * c * h = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_proof_l357_35744


namespace NUMINAMATH_CALUDE_triangle_area_zero_l357_35735

theorem triangle_area_zero (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a + b + c = 6 →
  a*b + b*c + a*c = 11 →
  a*b*c = 6 →
  ∃ (s : ℝ), s*(s - a)*(s - b)*(s - c) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_zero_l357_35735


namespace NUMINAMATH_CALUDE_negation_of_all_divisible_by_five_are_even_l357_35786

theorem negation_of_all_divisible_by_five_are_even :
  (¬ ∀ n : ℤ, 5 ∣ n → Even n) ↔ (∃ n : ℤ, 5 ∣ n ∧ ¬Even n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_divisible_by_five_are_even_l357_35786


namespace NUMINAMATH_CALUDE_math_competition_problem_l357_35750

theorem math_competition_problem (p_a p_either : ℝ) (h1 : p_a = 0.6) (h2 : p_either = 0.92) :
  ∃ p_b : ℝ, p_b = 0.8 ∧ 1 - p_either = (1 - p_a) * (1 - p_b) :=
by sorry

end NUMINAMATH_CALUDE_math_competition_problem_l357_35750


namespace NUMINAMATH_CALUDE_solution_set_of_quadratic_inequality_l357_35780

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

-- State the theorem
theorem solution_set_of_quadratic_inequality :
  {x : ℝ | f x < 0} = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_quadratic_inequality_l357_35780


namespace NUMINAMATH_CALUDE_david_drive_distance_david_drive_distance_proof_l357_35701

theorem david_drive_distance : ℝ → Prop :=
  fun distance =>
    ∀ (initial_speed : ℝ) (increased_speed : ℝ) (on_time_duration : ℝ),
      initial_speed = 40 ∧
      increased_speed = initial_speed + 20 ∧
      distance = initial_speed * (on_time_duration + 1.5) ∧
      distance - initial_speed = increased_speed * (on_time_duration - 2) →
      distance = 340

-- The proof is omitted
theorem david_drive_distance_proof : david_drive_distance 340 := by sorry

end NUMINAMATH_CALUDE_david_drive_distance_david_drive_distance_proof_l357_35701


namespace NUMINAMATH_CALUDE_solution_equation_l357_35738

theorem solution_equation (m n : ℕ+) (x : ℝ) 
  (h1 : x = m + Real.sqrt n)
  (h2 : x^2 - 10*x + 1 = Real.sqrt x * (x + 1)) : 
  m + n = 55 := by
sorry

end NUMINAMATH_CALUDE_solution_equation_l357_35738


namespace NUMINAMATH_CALUDE_expression_value_at_nine_l357_35771

theorem expression_value_at_nine :
  let x : ℝ := 9
  (x^6 - 27*x^3 + 729) / (x^3 - 27) = 702 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_nine_l357_35771


namespace NUMINAMATH_CALUDE_expression_evaluation_l357_35715

theorem expression_evaluation (x : ℝ) (h : x = -2) : 
  (3 * x / (x - 1) - x / (x + 1)) * (x^2 - 1) / x = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l357_35715


namespace NUMINAMATH_CALUDE_august_tips_multiple_l357_35749

theorem august_tips_multiple (total_months : ℕ) (other_months : ℕ) (august_ratio : ℝ) :
  total_months = 7 →
  other_months = 6 →
  august_ratio = 0.5714285714285714 →
  ∃ (avg_other_months : ℝ),
    avg_other_months > 0 →
    august_ratio * (8 * avg_other_months + other_months * avg_other_months) = 8 * avg_other_months :=
by sorry

end NUMINAMATH_CALUDE_august_tips_multiple_l357_35749


namespace NUMINAMATH_CALUDE_x_eq_4_is_linear_l357_35716

/-- A linear equation with one variable is of the form ax + b = 0, where a ≠ 0 and x is the variable. -/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function f(x) = x - 4 represents the equation x = 4. -/
def f (x : ℝ) : ℝ := x - 4

theorem x_eq_4_is_linear :
  is_linear_equation_one_var f :=
sorry

end NUMINAMATH_CALUDE_x_eq_4_is_linear_l357_35716


namespace NUMINAMATH_CALUDE_probability_two_heads_in_three_flips_l357_35723

/-- A fair coin has an equal probability of landing heads or tails. -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The number of coin flips. -/
def num_flips : ℕ := 3

/-- The number of desired heads. -/
def num_heads : ℕ := 2

/-- The probability of getting exactly k successes in n trials with probability p of success on each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1-p)^(n-k)

/-- The main theorem: the probability of getting exactly 2 heads in 3 flips of a fair coin is 0.375. -/
theorem probability_two_heads_in_three_flips (p : ℝ) (h : fair_coin p) :
  binomial_probability num_flips num_heads p = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_in_three_flips_l357_35723


namespace NUMINAMATH_CALUDE_range_of_a_for_absolute_value_equation_l357_35766

theorem range_of_a_for_absolute_value_equation (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = a * x + 1) ∧ 
  (∀ y : ℝ, y > 0 → |y| ≠ a * y + 1) → 
  a > -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_absolute_value_equation_l357_35766


namespace NUMINAMATH_CALUDE_pedestrian_speeds_l357_35762

theorem pedestrian_speeds (x y : ℝ) 
  (h1 : x + y = 14)
  (h2 : (3/2) * x + (1/2) * y = 13) :
  (x = 6 ∧ y = 8) ∨ (x = 8 ∧ y = 6) := by
  sorry

end NUMINAMATH_CALUDE_pedestrian_speeds_l357_35762


namespace NUMINAMATH_CALUDE_triangle_acute_obtuse_characterization_l357_35726

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi

def Triangle.isAcute (t : Triangle) : Prop :=
  t.A < Real.pi / 2 ∧ t.B < Real.pi / 2 ∧ t.C < Real.pi / 2

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

theorem triangle_acute_obtuse_characterization (t : Triangle) :
  (t.isAcute ↔ Real.cos t.A ^ 2 + Real.cos t.B ^ 2 + Real.cos t.C ^ 2 < 1) ∧
  (t.isObtuse ↔ Real.cos t.A ^ 2 + Real.cos t.B ^ 2 + Real.cos t.C ^ 2 > 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_acute_obtuse_characterization_l357_35726


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l357_35753

theorem product_of_solutions_abs_y_eq_3_abs_y_minus_2 :
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3 * (|y₁| - 2)) ∧ (|y₂| = 3 * (|y₂| - 2)) ∧ y₁ ≠ y₂ ∧ y₁ * y₂ = -9 :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l357_35753


namespace NUMINAMATH_CALUDE_intersection_complement_problem_l357_35729

def I : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem intersection_complement_problem :
  A ∩ (I \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_problem_l357_35729


namespace NUMINAMATH_CALUDE_a_4_times_a_3_l357_35774

def a : ℕ → ℤ
  | n => if n % 2 = 1 then (-2)^n else n

theorem a_4_times_a_3 : a 4 * a 3 = -32 := by
  sorry

end NUMINAMATH_CALUDE_a_4_times_a_3_l357_35774


namespace NUMINAMATH_CALUDE_books_bought_at_yard_sale_l357_35760

def initial_books : ℕ := 35
def final_books : ℕ := 91

theorem books_bought_at_yard_sale :
  final_books - initial_books = 56 :=
by sorry

end NUMINAMATH_CALUDE_books_bought_at_yard_sale_l357_35760


namespace NUMINAMATH_CALUDE_share_purchase_price_l357_35791

/-- Calculates the purchase price of shares given dividend rate, par value, and ROI -/
theorem share_purchase_price 
  (dividend_rate : ℝ) 
  (par_value : ℝ) 
  (roi : ℝ) 
  (h1 : dividend_rate = 0.125)
  (h2 : par_value = 40)
  (h3 : roi = 0.25) : 
  (dividend_rate * par_value) / roi = 20 := by
  sorry

#check share_purchase_price

end NUMINAMATH_CALUDE_share_purchase_price_l357_35791


namespace NUMINAMATH_CALUDE_intersection_with_complement_l357_35790

def U : Finset Nat := {0, 1, 2, 3, 4}
def A : Finset Nat := {1, 2, 3}
def B : Finset Nat := {2, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l357_35790


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l357_35761

/-- The area of wrapping paper required for a rectangular box -/
def wrapping_paper_area (l w h : ℝ) : ℝ := l * w + 2 * l * h + 2 * w * h + 4 * h^2

/-- Theorem stating the area of wrapping paper required for a rectangular box -/
theorem wrapping_paper_area_theorem (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  let box_volume := l * w * h
  let paper_length := l + 2 * h
  let paper_width := w + 2 * h
  let paper_area := paper_length * paper_width
  paper_area = wrapping_paper_area l w h :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l357_35761


namespace NUMINAMATH_CALUDE_circles_intersect_example_l357_35751

/-- Two circles are intersecting if the distance between their centers is less than the sum of their radii
    and greater than the absolute difference of their radii. -/
def circles_intersect (r₁ r₂ d : ℝ) : Prop :=
  d < r₁ + r₂ ∧ d > |r₁ - r₂|

/-- Theorem: Two circles with radii 4 and 5, whose centers are 7 units apart, are intersecting. -/
theorem circles_intersect_example : circles_intersect 4 5 7 := by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_example_l357_35751


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l357_35756

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {aₙ} is a geometric sequence with common ratio q
  a 1 = 1 / 4 →                 -- a₁ = 1/4
  a 3 * a 5 = 4 * (a 4 - 1) →   -- a₃a₅ = 4(a₄ - 1)
  a 2 = 1 / 2 := by             -- a₂ = 1/2
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l357_35756


namespace NUMINAMATH_CALUDE_sons_age_l357_35793

theorem sons_age (man daughter son : ℕ) 
  (h1 : man = son + 30)
  (h2 : daughter = son - 8)
  (h3 : man + 2 = 3 * (son + 2))
  (h4 : man + 2 = 2 * (daughter + 2)) :
  son = 13 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l357_35793


namespace NUMINAMATH_CALUDE_base4_132_is_30_l357_35719

/-- Converts a number from base 4 to decimal --/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The decimal representation of 132 in base 4 --/
def m : Nat := base4ToDecimal [2, 3, 1]

theorem base4_132_is_30 : m = 30 := by
  sorry

end NUMINAMATH_CALUDE_base4_132_is_30_l357_35719


namespace NUMINAMATH_CALUDE_correct_divisor_l357_35702

theorem correct_divisor (incorrect_divisor : ℕ) (incorrect_quotient : ℕ) (correct_quotient : ℕ) :
  incorrect_divisor = 63 →
  incorrect_quotient = 24 →
  correct_quotient = 42 →
  (incorrect_divisor * incorrect_quotient) / correct_quotient = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l357_35702


namespace NUMINAMATH_CALUDE_frozen_fruit_sold_l357_35718

/-- Given an orchard's fruit sales, calculate the amount of frozen fruit sold. -/
theorem frozen_fruit_sold (total_fruit : ℕ) (fresh_fruit : ℕ) (h1 : total_fruit = 9792) (h2 : fresh_fruit = 6279) :
  total_fruit - fresh_fruit = 3513 := by
  sorry

end NUMINAMATH_CALUDE_frozen_fruit_sold_l357_35718


namespace NUMINAMATH_CALUDE_wall_building_time_relation_l357_35740

/-- Represents the time taken to build a wall given the number of workers -/
def build_time (workers : ℕ) (days : ℚ) : Prop :=
  workers * days = 180

theorem wall_building_time_relation :
  build_time 60 3 → build_time 90 2 := by
  sorry

end NUMINAMATH_CALUDE_wall_building_time_relation_l357_35740


namespace NUMINAMATH_CALUDE_find_m_l357_35741

theorem find_m : ∃ m : ℕ, 62519 * m = 624877405 ∧ m = 9995 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l357_35741


namespace NUMINAMATH_CALUDE_parking_lot_problem_l357_35732

theorem parking_lot_problem (medium_fee small_fee total_cars total_fee : ℕ)
  (h1 : medium_fee = 15)
  (h2 : small_fee = 8)
  (h3 : total_cars = 30)
  (h4 : total_fee = 324) :
  ∃ (medium_cars small_cars : ℕ),
    medium_cars + small_cars = total_cars ∧
    medium_cars * medium_fee + small_cars * small_fee = total_fee ∧
    medium_cars = 12 ∧
    small_cars = 18 :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l357_35732


namespace NUMINAMATH_CALUDE_harper_gift_cost_l357_35784

/-- Harper's gift-buying problem -/
theorem harper_gift_cost (son_teachers daughter_teachers total_spent : ℕ) 
  (h1 : son_teachers = 3)
  (h2 : daughter_teachers = 4)
  (h3 : total_spent = 70) :
  total_spent / (son_teachers + daughter_teachers) = 10 := by
  sorry

#check harper_gift_cost

end NUMINAMATH_CALUDE_harper_gift_cost_l357_35784


namespace NUMINAMATH_CALUDE_series_rationality_characterization_l357_35795

/-- Represents a sequence of coefficients for the series -/
def CoefficientSequence := ℕ → ℕ

/-- The series sum for a given coefficient sequence -/
noncomputable def SeriesSum (a : CoefficientSequence) : ℝ :=
  ∑' n, (a n : ℝ) / n.factorial

/-- Condition that all coefficients from N onwards are zero -/
def AllZeroFrom (a : CoefficientSequence) (N : ℕ) : Prop :=
  ∀ n ≥ N, a n = 0

/-- Condition that all coefficients from N onwards are n-1 -/
def AllNMinusOneFrom (a : CoefficientSequence) (N : ℕ) : Prop :=
  ∀ n ≥ N, a n = n - 1

/-- The main theorem statement -/
theorem series_rationality_characterization (a : CoefficientSequence) 
  (h : ∀ n ≥ 2, 0 ≤ a n ∧ a n ≤ n - 1) :
  (∃ q : ℚ, SeriesSum a = q) ↔ 
  (∃ N : ℕ, AllZeroFrom a N ∨ AllNMinusOneFrom a N) := by
  sorry

end NUMINAMATH_CALUDE_series_rationality_characterization_l357_35795


namespace NUMINAMATH_CALUDE_machine_parts_replacement_l357_35796

theorem machine_parts_replacement (num_machines : ℕ) (parts_per_machine : ℕ)
  (fail_rate_week1 : ℚ) (fail_rate_week2 : ℚ) (fail_rate_week3 : ℚ) :
  num_machines = 500 →
  parts_per_machine = 6 →
  fail_rate_week1 = 1/10 →
  fail_rate_week2 = 3/10 →
  fail_rate_week3 = 6/10 →
  (fail_rate_week1 + fail_rate_week2 + fail_rate_week3 = 1) →
  (num_machines * parts_per_machine * fail_rate_week3 +
   (num_machines * parts_per_machine * fail_rate_week2 * fail_rate_week3) +
   (num_machines * parts_per_machine * fail_rate_week1 * fail_rate_week2 * fail_rate_week3) : ℚ) = 1983 := by
  sorry


end NUMINAMATH_CALUDE_machine_parts_replacement_l357_35796


namespace NUMINAMATH_CALUDE_plot_perimeter_is_180_l357_35705

/-- A rectangular plot with specific dimensions and fencing cost -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencingRate : ℝ
  totalFencingCost : ℝ
  lengthWidthRelation : length = width + 10
  costRelation : totalFencingCost = fencingRate * (2 * (length + width))

/-- The perimeter of a rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.width)

/-- Theorem: The perimeter of the specific plot is 180 meters -/
theorem plot_perimeter_is_180 (plot : RectangularPlot)
  (h1 : plot.fencingRate = 6.5)
  (h2 : plot.totalFencingCost = 1170) :
  perimeter plot = 180 := by
  sorry

end NUMINAMATH_CALUDE_plot_perimeter_is_180_l357_35705


namespace NUMINAMATH_CALUDE_shoe_alteration_cost_l357_35772

theorem shoe_alteration_cost (pairs : ℕ) (total_cost : ℕ) (cost_per_shoe : ℕ) :
  pairs = 17 →
  total_cost = 986 →
  cost_per_shoe = total_cost / (pairs * 2) →
  cost_per_shoe = 29 := by
  sorry

end NUMINAMATH_CALUDE_shoe_alteration_cost_l357_35772


namespace NUMINAMATH_CALUDE_smallest_sum_l357_35798

/-- Given positive integers A, B, C, and D satisfying certain conditions,
    the smallest possible sum A + B + C + D is 43. -/
theorem smallest_sum (A B C D : ℕ+) : 
  (∃ r : ℚ, B.val - A.val = r ∧ C.val - B.val = r) →  -- arithmetic sequence condition
  (∃ q : ℚ, C.val / B.val = q ∧ D.val / C.val = q) →  -- geometric sequence condition
  C.val / B.val = 4 / 3 →                             -- given ratio
  A.val + B.val + C.val + D.val ≥ 43 :=               -- smallest possible sum
by sorry

end NUMINAMATH_CALUDE_smallest_sum_l357_35798


namespace NUMINAMATH_CALUDE_square_plus_double_eq_one_implies_double_square_plus_quad_minus_one_eq_one_l357_35752

/-- Given that a^2 + 2a = 1, prove that 2a^2 + 4a - 1 = 1 -/
theorem square_plus_double_eq_one_implies_double_square_plus_quad_minus_one_eq_one
  (a : ℝ) (h : a^2 + 2*a = 1) : 2*a^2 + 4*a - 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_double_eq_one_implies_double_square_plus_quad_minus_one_eq_one_l357_35752


namespace NUMINAMATH_CALUDE_greatest_ratio_bound_l357_35730

theorem greatest_ratio_bound (x y z u : ℕ+) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) :
  (x : ℝ) / y ≤ 3 + 2 * Real.sqrt 2 ∧ ∃ (x' y' z' u' : ℕ+), 
    x' + y' = z' + u' ∧ 
    2 * x' * y' = z' * u' ∧ 
    x' ≥ y' ∧ 
    (x' : ℝ) / y' = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_greatest_ratio_bound_l357_35730


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l357_35704

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12 → x.val + y.val ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l357_35704


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l357_35773

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_3402 : 
  largest_perfect_square_factor 3402 = 81 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l357_35773


namespace NUMINAMATH_CALUDE_president_and_vice_president_choices_l357_35781

/-- The number of ways to choose a President and a Vice-President from a group of people -/
def choosePresidentAndVicePresident (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 20 ways to choose a President and a Vice-President from a group of 5 people -/
theorem president_and_vice_president_choices :
  choosePresidentAndVicePresident 5 = 20 := by
  sorry

#eval choosePresidentAndVicePresident 5

end NUMINAMATH_CALUDE_president_and_vice_president_choices_l357_35781


namespace NUMINAMATH_CALUDE_hot_dog_buns_packages_l357_35743

/-- Calculates the number of packages of hot dog buns needed for a school picnic --/
theorem hot_dog_buns_packages (buns_per_package : ℕ) (num_classes : ℕ) (students_per_class : ℕ) (buns_per_student : ℕ) : 
  buns_per_package = 8 →
  num_classes = 4 →
  students_per_class = 30 →
  buns_per_student = 2 →
  (num_classes * students_per_class * buns_per_student + buns_per_package - 1) / buns_per_package = 30 := by
  sorry

#eval (4 * 30 * 2 + 8 - 1) / 8  -- Should output 30

end NUMINAMATH_CALUDE_hot_dog_buns_packages_l357_35743


namespace NUMINAMATH_CALUDE_sample_size_equals_selected_students_l357_35712

/-- Represents the sample size of a survey -/
def sample_size : ℕ := 1200

/-- Represents the number of students selected for the investigation -/
def selected_students : ℕ := 1200

/-- Theorem stating that the sample size is equal to the number of selected students -/
theorem sample_size_equals_selected_students : sample_size = selected_students := by
  sorry

end NUMINAMATH_CALUDE_sample_size_equals_selected_students_l357_35712


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l357_35707

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l357_35707


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l357_35754

/-- The trajectory of point P -/
def trajectory (x y : ℝ) : Prop :=
  y^2 = 4*x

/-- The condition for point P -/
def point_condition (x y : ℝ) : Prop :=
  2 * Real.sqrt ((x - 1)^2 + y^2) = 2*(x + 1)

/-- The perpendicularity condition for OM and ON -/
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem trajectory_and_intersection :
  -- Point P satisfies the condition
  ∀ x y : ℝ, point_condition x y →
  -- The trajectory is y² = 4x
  (trajectory x y) ∧
  -- For any non-zero m where y = x + m intersects the trajectory at M and N
  ∀ m : ℝ, m ≠ 0 →
    ∃ x₁ y₁ x₂ y₂ : ℝ,
      -- M and N are on the trajectory
      trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
      -- M and N are on the line y = x + m
      y₁ = x₁ + m ∧ y₂ = x₂ + m ∧
      -- OM is perpendicular to ON
      perpendicular_condition x₁ y₁ x₂ y₂ →
      -- Then m = -4
      m = -4 :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l357_35754


namespace NUMINAMATH_CALUDE_unique_integer_solution_quadratic_l357_35731

theorem unique_integer_solution_quadratic :
  ∃! a : ℤ, ∃ x : ℤ, x^2 + a*x + a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_quadratic_l357_35731


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l357_35799

theorem arithmetic_calculation : 5 * 6 - 2 * 3 + 7 * 4 + 9 * 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l357_35799


namespace NUMINAMATH_CALUDE_solution_set_f_max_integer_m_max_m_is_two_l357_35717

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |1 - 2*x|

-- Theorem for part 1
theorem solution_set_f (x : ℝ) :
  f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
sorry

-- Theorem for part 2
theorem max_integer_m (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) (h4 : f a = 3 * f b) :
  ∀ m : ℤ, (a^2 + b^2 > m) → m ≤ 2 :=
sorry

-- Theorem to prove that 2 is the maximum integer satisfying the condition
theorem max_m_is_two (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) (h4 : f a = 3 * f b) :
  ∃ m : ℤ, (∀ n : ℤ, (a^2 + b^2 > n) → n ≤ m) ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_max_integer_m_max_m_is_two_l357_35717


namespace NUMINAMATH_CALUDE_odd_function_property_l357_35765

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h : OddFunction f) :
  ∀ x : ℝ, f x + f (-x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l357_35765


namespace NUMINAMATH_CALUDE_function_bounded_l357_35778

/-- The function f(x, y) = √(4 - x² - y²) is bounded between 0 and 2 -/
theorem function_bounded (x y : ℝ) (h : x^2 + y^2 ≤ 4) :
  0 ≤ Real.sqrt (4 - x^2 - y^2) ∧ Real.sqrt (4 - x^2 - y^2) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_function_bounded_l357_35778


namespace NUMINAMATH_CALUDE_x_plus_x_squared_l357_35733

theorem x_plus_x_squared (x : ℕ) (h : x = 3) : x + (x * x) = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_x_squared_l357_35733


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_thirds_l357_35734

-- Define the polynomial
def f (x : ℝ) : ℝ := (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7)

-- Theorem statement
theorem sum_of_roots_equals_fourteen_thirds :
  ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = 14/3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_thirds_l357_35734


namespace NUMINAMATH_CALUDE_skew_lines_and_parallel_imply_not_parallel_l357_35709

/-- Represents a line in 3D space -/
structure Line3D where
  -- We don't need to specify the internal structure of a line
  -- as we're only interested in their relationships

/-- Two lines are skew if they do not intersect and are not parallel -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry

theorem skew_lines_and_parallel_imply_not_parallel
  (a b c : Line3D)
  (h1 : are_skew a b)
  (h2 : are_parallel a c) :
  ¬(are_parallel b c) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_and_parallel_imply_not_parallel_l357_35709


namespace NUMINAMATH_CALUDE_circle_equation_proof_l357_35710

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

/-- The equation of the resulting circle -/
def resultCircle (x y : ℝ) : Prop := 9*x^2 + 9*y^2 - 14*x + 4*y = 0

theorem circle_equation_proof :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → resultCircle x y) ∧
  resultCircle 1 (-1) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l357_35710


namespace NUMINAMATH_CALUDE_right_triangle_sets_l357_35785

theorem right_triangle_sets :
  -- Set A
  (5^2 + 12^2 = 13^2) ∧
  -- Set B
  ((Real.sqrt 2)^2 + (Real.sqrt 3)^2 = (Real.sqrt 5)^2) ∧
  -- Set C
  (3^2 + (Real.sqrt 7)^2 = 4^2) ∧
  -- Set D
  (2^2 + 3^2 ≠ 4^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l357_35785


namespace NUMINAMATH_CALUDE_texas_tech_game_profit_l357_35708

/-- Represents the discount tiers for t-shirt sales -/
inductive DiscountTier
  | NoDiscount
  | MediumDiscount
  | HighDiscount

/-- Calculates the discount tier based on the number of t-shirts sold -/
def getDiscountTier (numSold : ℕ) : DiscountTier :=
  if numSold ≤ 50 then DiscountTier.NoDiscount
  else if numSold ≤ 100 then DiscountTier.MediumDiscount
  else DiscountTier.HighDiscount

/-- Calculates the profit per t-shirt based on the discount tier -/
def getProfitPerShirt (tier : DiscountTier) (fullPrice : ℕ) : ℕ :=
  match tier with
  | DiscountTier.NoDiscount => fullPrice
  | DiscountTier.MediumDiscount => fullPrice - 5
  | DiscountTier.HighDiscount => fullPrice - 10

/-- Theorem: The money made from selling t-shirts during the Texas Tech game is $1092 -/
theorem texas_tech_game_profit (totalSold arkansasSold fullPrice : ℕ) 
    (h1 : totalSold = 186)
    (h2 : arkansasSold = 172)
    (h3 : fullPrice = 78) :
    let texasTechSold := totalSold - arkansasSold
    let tier := getDiscountTier texasTechSold
    let profitPerShirt := getProfitPerShirt tier fullPrice
    texasTechSold * profitPerShirt = 1092 := by
  sorry

end NUMINAMATH_CALUDE_texas_tech_game_profit_l357_35708


namespace NUMINAMATH_CALUDE_min_value_f_l357_35782

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x + 1) * (cos x + 1) / (sin x * cos x)

theorem min_value_f :
  ∀ x ∈ Set.Ioo 0 (π/2), f x ≥ 3 + 2 * sqrt 2 ∧
  ∃ x₀ ∈ Set.Ioo 0 (π/2), f x₀ = 3 + 2 * sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_l357_35782


namespace NUMINAMATH_CALUDE_segment_division_ratio_l357_35747

/-- Given a line segment AC and points B and D on it, this theorem proves
    that if B divides AC in a 2:1 ratio and D divides AB in a 3:2 ratio,
    then D divides AC in a 2:3 ratio. -/
theorem segment_division_ratio (A B C D : ℝ) :
  (B - A) / (C - B) = 2 / 1 →
  (D - A) / (B - D) = 3 / 2 →
  (D - A) / (C - D) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_segment_division_ratio_l357_35747


namespace NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l357_35724

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l357_35724


namespace NUMINAMATH_CALUDE_water_evaporation_per_day_l357_35759

/-- Given a glass of water with initial amount, evaporation period, and total evaporation percentage,
    calculate the amount of water evaporated per day. -/
theorem water_evaporation_per_day 
  (initial_amount : ℝ) 
  (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) 
  (h1 : initial_amount = 10)
  (h2 : evaporation_period = 20)
  (h3 : evaporation_percentage = 4) : 
  (initial_amount * evaporation_percentage / 100) / evaporation_period = 0.02 := by
  sorry

#check water_evaporation_per_day

end NUMINAMATH_CALUDE_water_evaporation_per_day_l357_35759


namespace NUMINAMATH_CALUDE_set_membership_properties_l357_35779

theorem set_membership_properties (M P : Set α) (h_nonempty : M.Nonempty) 
  (h_not_subset : ¬(M ⊆ P)) : 
  (∃ x, x ∈ M ∧ x ∉ P) ∧ (∃ y, y ∈ M ∧ y ∈ P) := by
  sorry

end NUMINAMATH_CALUDE_set_membership_properties_l357_35779


namespace NUMINAMATH_CALUDE_journey_end_day_l357_35767

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the next day of the week -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

/-- Calculates the arrival day given a starting day and journey duration in hours -/
def arrivalDay (startDay : Day) (journeyHours : Nat) : Day :=
  let daysPassed := journeyHours / 24
  (List.range daysPassed).foldl (fun d _ => nextDay d) startDay

/-- Theorem: A 28-hour journey starting on Tuesday will end on Wednesday -/
theorem journey_end_day :
  arrivalDay Day.Tuesday 28 = Day.Wednesday := by
  sorry


end NUMINAMATH_CALUDE_journey_end_day_l357_35767
