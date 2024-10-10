import Mathlib

namespace trajectory_of_point_P_l3885_388592

/-- The trajectory of a point P on the curve ρcos θ + 2ρsin θ = 3, where 0 ≤ θ ≤ π/4 and ρ > 0,
    is a line segment with endpoints (1,1) and (3,0). -/
theorem trajectory_of_point_P (θ : ℝ) (ρ : ℝ) (h1 : 0 ≤ θ) (h2 : θ ≤ π/4) (h3 : ρ > 0)
  (h4 : ρ * Real.cos θ + 2 * ρ * Real.sin θ = 3) :
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
  ρ * Real.cos θ = 3 - 2 * t ∧
  ρ * Real.sin θ = t :=
by sorry

end trajectory_of_point_P_l3885_388592


namespace square_area_above_line_l3885_388540

/-- The fraction of a square's area above a line -/
def fractionAboveLine (p1 p2 v1 v2 v3 v4 : ℝ × ℝ) : ℚ :=
  sorry

/-- The main theorem -/
theorem square_area_above_line :
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 1)
  let v1 : ℝ × ℝ := (2, 1)
  let v2 : ℝ × ℝ := (5, 1)
  let v3 : ℝ × ℝ := (5, 4)
  let v4 : ℝ × ℝ := (2, 4)
  fractionAboveLine p1 p2 v1 v2 v3 v4 = 2/3 :=
sorry

end square_area_above_line_l3885_388540


namespace train_passengers_l3885_388569

theorem train_passengers (initial_passengers : ℕ) : 
  initial_passengers = 288 →
  let after_first := initial_passengers * 2 / 3 + 280
  let after_second := after_first / 2 + 12
  after_second = 248 := by
sorry

end train_passengers_l3885_388569


namespace actual_distance_traveled_l3885_388557

/-- The actual distance traveled by a person, given two walking speeds and additional distance information. -/
theorem actual_distance_traveled (slow_speed fast_speed : ℝ) (additional_distance : ℝ) 
  (h1 : slow_speed = 5)
  (h2 : fast_speed = 10)
  (h3 : additional_distance = 20)
  (h4 : ∀ d : ℝ, d / slow_speed = (d + additional_distance) / fast_speed) :
  ∃ d : ℝ, d = 20 := by
sorry

end actual_distance_traveled_l3885_388557


namespace min_wednesday_birthdays_l3885_388531

/-- Given a company with 61 employees, where the number of employees with birthdays on Wednesday
    is greater than the number on any other day, and all other days have an equal number of birthdays,
    the minimum number of employees with birthdays on Wednesday is 13. -/
theorem min_wednesday_birthdays (total_employees : ℕ) (wednesday_birthdays : ℕ) 
  (other_day_birthdays : ℕ) : 
  total_employees = 61 →
  wednesday_birthdays > other_day_birthdays →
  total_employees = wednesday_birthdays + 6 * other_day_birthdays →
  wednesday_birthdays ≥ 13 :=
by sorry

end min_wednesday_birthdays_l3885_388531


namespace f_value_l3885_388545

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 5 * Real.pi / 2) * Real.cos (3 * Real.pi / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (Real.pi - α))

theorem f_value (α : Real) 
  (h1 : Real.cos (α + 3 * Real.pi / 2) = 1 / 5)
  (h2 : 0 < α - Real.pi / 2 ∧ α - Real.pi / 2 < Real.pi / 2) : 
  f α = 2 * Real.sqrt 6 / 5 := by
sorry

end f_value_l3885_388545


namespace different_suit_combinations_l3885_388558

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := 13

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 4

/-- Theorem stating the number of ways to choose 4 cards of different suits from a standard deck -/
theorem different_suit_combinations : 
  (number_of_suits.choose cards_to_choose) * (cards_per_suit ^ cards_to_choose) = 28561 := by
  sorry

end different_suit_combinations_l3885_388558


namespace total_rectangles_area_2_l3885_388559

-- Define the structure of the figure
structure Figure where
  patterns : List String
  small_square_side : ℕ

-- Define a rectangle in the figure
structure Rectangle where
  width : ℕ
  height : ℕ

-- Function to calculate the area of a rectangle
def rectangle_area (r : Rectangle) : ℕ :=
  r.width * r.height

-- Function to count rectangles with area 2 in a specific pattern
def count_rectangles_area_2 (pattern : String) : ℕ :=
  match pattern with
  | "2" => 10
  | "0" => 12
  | "1" => 4
  | "4" => 8
  | _ => 0

-- Theorem stating the total number of rectangles with area 2
theorem total_rectangles_area_2 (fig : Figure) 
  (h1 : fig.small_square_side = 1) 
  (h2 : fig.patterns = ["2", "0", "1", "4"]) : 
  (fig.patterns.map count_rectangles_area_2).sum = 34 := by
  sorry

end total_rectangles_area_2_l3885_388559


namespace radius_of_circle_M_l3885_388595

/-- Definition of Circle M -/
def CircleM (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

/-- Theorem: The radius of Circle M is 5 -/
theorem radius_of_circle_M : ∃ (h k r : ℝ), r = 5 ∧ 
  ∀ (x y : ℝ), CircleM x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end radius_of_circle_M_l3885_388595


namespace complement_union_intersection_equivalence_l3885_388556

-- Define the sets U, M, and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | -1 ≤ x ∧ x < 5}

-- State the theorem
theorem complement_union_intersection_equivalence :
  ∀ x : ℝ, x ∈ (U \ M) ∪ (M ∩ N) ↔ x < 5 := by sorry

end complement_union_intersection_equivalence_l3885_388556


namespace equation_solution_range_l3885_388522

theorem equation_solution_range (x m : ℝ) : 
  ((2 * x + m) / (x - 1) = 1) → 
  (x > 0) → 
  (x ≠ 1) → 
  (m < -1) :=
by
  sorry

end equation_solution_range_l3885_388522


namespace debt_payment_average_l3885_388566

theorem debt_payment_average (n : ℕ) (first_payment second_payment : ℚ) : 
  n = 40 →
  first_payment = 410 →
  second_payment = first_payment + 65 →
  (20 * first_payment + 20 * second_payment) / n = 442.50 := by
  sorry

end debt_payment_average_l3885_388566


namespace product_remainder_mod_nine_l3885_388546

theorem product_remainder_mod_nine : (2156 * 4427 * 9313) % 9 = 1 := by
  sorry

end product_remainder_mod_nine_l3885_388546


namespace mary_height_to_grow_l3885_388593

/-- The problem of calculating how much Mary needs to grow to ride Kingda Ka -/
theorem mary_height_to_grow (min_height brother_height : ℝ) (h1 : min_height = 140) 
  (h2 : brother_height = 180) : 
  min_height - (2/3 * brother_height) = 20 := by
  sorry

end mary_height_to_grow_l3885_388593


namespace eugene_pencils_count_l3885_388503

/-- The total number of pencils Eugene has after receiving additional pencils -/
def total_pencils (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Eugene's total pencils equals the sum of his initial pencils and additional pencils -/
theorem eugene_pencils_count : 
  total_pencils 51 6 = 57 := by
  sorry

end eugene_pencils_count_l3885_388503


namespace polygon_with_special_angle_property_l3885_388553

theorem polygon_with_special_angle_property (n : ℕ) 
  (h : (n - 2) * 180 = 2 * 360) : n = 6 := by
  sorry

end polygon_with_special_angle_property_l3885_388553


namespace symmetric_point_of_M_l3885_388584

/-- The symmetric point of (x, y) with respect to the x-axis is (x, -y) -/
def symmetricPointXAxis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Given M(-2, -3), its symmetric point with respect to the x-axis is (-2, 3) -/
theorem symmetric_point_of_M : 
  let M : ℝ × ℝ := (-2, -3)
  symmetricPointXAxis M = (-2, 3) := by sorry

end symmetric_point_of_M_l3885_388584


namespace train_length_l3885_388599

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length (train_speed : Real) (bridge_crossing_time : Real) (bridge_length : Real) :
  train_speed = 72 * 1000 / 3600 ∧ 
  bridge_crossing_time = 12.099 ∧ 
  bridge_length = 132 →
  train_speed * bridge_crossing_time - bridge_length = 110 :=
by sorry

end train_length_l3885_388599


namespace smallest_n_for_probability_threshold_l3885_388591

def P (n : ℕ) : ℚ := 3 / ((n + 1) * (n + 2) * (n + 3))

theorem smallest_n_for_probability_threshold : 
  ∀ k : ℕ, k ≥ 1 → (P k < 1 / 3015 ↔ k ≥ 19) :=
by sorry

end smallest_n_for_probability_threshold_l3885_388591


namespace roses_remaining_l3885_388538

/-- Given 3 dozen roses, prove that after giving half away and removing one-third of the remaining flowers, 12 flowers are left. -/
theorem roses_remaining (initial_roses : ℕ) (dozen : ℕ) (half : ℕ → ℕ) (third : ℕ → ℕ) : 
  initial_roses = 3 * dozen → 
  dozen = 12 →
  half n = n / 2 →
  third n = n / 3 →
  third (initial_roses - half initial_roses) = 12 := by
sorry

end roses_remaining_l3885_388538


namespace smaller_number_proof_l3885_388598

theorem smaller_number_proof (x y m : ℝ) 
  (h1 : x - y = 9)
  (h2 : x + y = 46)
  (h3 : x = m * y) : 
  min x y = 18.5 := by
  sorry

end smaller_number_proof_l3885_388598


namespace sams_cows_l3885_388504

theorem sams_cows (C : ℕ) : 
  (C / 2 + 5 = C - 4) → C = 18 := by
  sorry

end sams_cows_l3885_388504


namespace oplus_neg_two_three_l3885_388570

def oplus (a b : ℝ) : ℝ := a * (a - b) + 1

theorem oplus_neg_two_three : oplus (-2) 3 = 11 := by sorry

end oplus_neg_two_three_l3885_388570


namespace expected_rainfall_theorem_l3885_388523

/-- Weather forecast for a day --/
structure DailyForecast where
  sunny_prob : ℝ
  light_rain_prob : ℝ
  heavy_rain_prob : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculate expected rainfall for a single day --/
def expected_daily_rainfall (f : DailyForecast) : ℝ :=
  f.light_rain_prob * f.light_rain_amount + f.heavy_rain_prob * f.heavy_rain_amount

/-- Calculate expected rainfall for a week --/
def expected_weekly_rainfall (f : DailyForecast) (days : ℕ) : ℝ :=
  (expected_daily_rainfall f) * days

/-- The weather forecast for the week --/
def weekly_forecast : DailyForecast :=
  { sunny_prob := 0.30
  , light_rain_prob := 0.35
  , heavy_rain_prob := 0.35
  , light_rain_amount := 3
  , heavy_rain_amount := 8 }

/-- The number of days in the forecast --/
def forecast_days : ℕ := 7

/-- Theorem: The expected rainfall for the week is approximately 26.9 inches --/
theorem expected_rainfall_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |expected_weekly_rainfall weekly_forecast forecast_days - 26.9| < ε := by
  sorry

end expected_rainfall_theorem_l3885_388523


namespace unique_prime_base_l3885_388543

theorem unique_prime_base : ∃! (n : ℕ), n ≥ 2 ∧ Nat.Prime (n^4 + 1) := by
  sorry

end unique_prime_base_l3885_388543


namespace markus_family_ages_l3885_388544

theorem markus_family_ages :
  ∀ (grandson_age son_age markus_age : ℕ),
    son_age = 2 * grandson_age →
    markus_age = 2 * son_age →
    grandson_age + son_age + markus_age = 140 →
    grandson_age = 20 := by
  sorry

end markus_family_ages_l3885_388544


namespace michael_and_brothers_ages_l3885_388567

/-- The ages of Michael and his brothers satisfy the given conditions and their combined age is 28. -/
theorem michael_and_brothers_ages :
  ∀ (michael_age older_brother_age younger_brother_age : ℕ),
    younger_brother_age = 5 →
    older_brother_age = 3 * younger_brother_age →
    older_brother_age = 1 + 2 * (michael_age - 1) →
    michael_age + older_brother_age + younger_brother_age = 28 :=
by
  sorry


end michael_and_brothers_ages_l3885_388567


namespace pythagorean_pattern_solution_for_eleven_l3885_388554

theorem pythagorean_pattern (n : ℕ) : 
  (2*n + 1)^2 + (2*n^2 + 2*n)^2 = (2*n^2 + 2*n + 1)^2 := by sorry

theorem solution_for_eleven : 
  let n : ℕ := 5
  (2*n^2 + 2*n + 1) = 61 := by sorry

end pythagorean_pattern_solution_for_eleven_l3885_388554


namespace money_distribution_l3885_388587

theorem money_distribution (a b c : ℤ) 
  (total : a + b + c = 500)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 310) :
  c = 10 := by
  sorry

end money_distribution_l3885_388587


namespace complex_subtraction_l3885_388576

theorem complex_subtraction (z₁ z₂ : ℂ) (h1 : z₁ = 7 - 6*I) (h2 : z₂ = 4 - 7*I) :
  z₁ - z₂ = 3 + I := by
  sorry

end complex_subtraction_l3885_388576


namespace evaluate_expression_l3885_388516

theorem evaluate_expression : 
  Real.sqrt (9 / 4) - Real.sqrt (8 / 9) + Real.sqrt 1 = (15 - 4 * Real.sqrt 2) / 6 := by
  sorry

end evaluate_expression_l3885_388516


namespace jacob_already_twice_as_old_l3885_388572

/-- Proves that Jacob is already twice as old as his brother -/
theorem jacob_already_twice_as_old (jacob_age : ℕ) (brother_age : ℕ) 
  (h1 : jacob_age = 18) 
  (h2 : jacob_age = 2 * brother_age) : 
  jacob_age = 2 * brother_age := by
  sorry

end jacob_already_twice_as_old_l3885_388572


namespace student_mistake_difference_l3885_388526

theorem student_mistake_difference : (5/6 : ℚ) * 576 - (5/16 : ℚ) * 576 = 300 := by
  sorry

end student_mistake_difference_l3885_388526


namespace computer_price_increase_l3885_388511

theorem computer_price_increase (c : ℝ) (h : 2 * c = 540) : 
  (351 - c) / c * 100 = 30 := by
  sorry

end computer_price_increase_l3885_388511


namespace arrange_75510_eq_48_l3885_388518

/-- The number of ways to arrange the digits of 75,510 to form a 5-digit number not beginning with '0' -/
def arrange_75510 : ℕ :=
  let digits : List ℕ := [7, 5, 5, 1, 0]
  let total_digits := digits.length
  let non_zero_digits := digits.filter (· ≠ 0)
  let zero_count := total_digits - non_zero_digits.length
  let non_zero_permutations := Nat.factorial non_zero_digits.length / 
    (Nat.factorial 2 * Nat.factorial (non_zero_digits.length - 2))
  (total_digits - 1) * non_zero_permutations

theorem arrange_75510_eq_48 : arrange_75510 = 48 := by
  sorry

end arrange_75510_eq_48_l3885_388518


namespace system_solution_l3885_388573

theorem system_solution :
  let f (x y : ℝ) := y^2 - (x^3 - 3*x^2 + 2*x)
  let g (x y : ℝ) := x^2 - (y^3 - 3*y^2 + 2*y)
  ∀ x y : ℝ, f x y = 0 ∧ g x y = 0 ↔
    (x = 0 ∧ y = 0) ∨
    (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2) ∨
    (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2) :=
by sorry

end system_solution_l3885_388573


namespace horner_method_V₂_l3885_388529

-- Define the polynomial coefficients
def a₄ : ℤ := 3
def a₃ : ℤ := 5
def a₂ : ℤ := 6
def a₁ : ℤ := 79
def a₀ : ℤ := -8

-- Define the x value
def x : ℤ := -4

-- Define Horner's method steps
def V₀ : ℤ := a₄
def V₁ : ℤ := x * V₀ + a₃
def V₂ : ℤ := x * V₁ + a₂

-- Theorem statement
theorem horner_method_V₂ : V₂ = 34 := by
  sorry

end horner_method_V₂_l3885_388529


namespace rectangular_prism_surface_area_increase_l3885_388568

theorem rectangular_prism_surface_area_increase 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  let original_surface_area := 2 * (a * b + b * c + a * c)
  let new_surface_area := 2 * ((1.8 * a) * (1.8 * b) + (1.8 * b) * (1.8 * c) + (1.8 * c) * (1.8 * a))
  (new_surface_area - original_surface_area) / original_surface_area = 2.24 := by
  sorry

end rectangular_prism_surface_area_increase_l3885_388568


namespace right_angled_triangle_k_values_l3885_388548

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

def AB : ℝ × ℝ := (2, 1)
def AC (k : ℝ) : ℝ × ℝ := (3, k)

def is_right_angled (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem right_angled_triangle_k_values :
  ∃ k₁ k₂ : ℝ, k₁ ≠ k₂ ∧ 
  (∀ k : ℝ, (is_right_angled AB (AC k) ∨ 
             is_right_angled AB (AC k - AB) ∨ 
             is_right_angled (AC k - AB) (AC k)) 
  ↔ (k = k₁ ∨ k = k₂)) :=
sorry

end right_angled_triangle_k_values_l3885_388548


namespace price_decrease_l3885_388577

/-- Given a 24% decrease in price resulting in a cost of Rs. 684, prove that the original price was Rs. 900. -/
theorem price_decrease (original_price : ℝ) : 
  (original_price * (1 - 0.24) = 684) → original_price = 900 := by
  sorry

end price_decrease_l3885_388577


namespace train_ride_nap_time_l3885_388597

theorem train_ride_nap_time (total_time reading_time eating_time movie_time : ℕ) 
  (h1 : total_time = 9)
  (h2 : reading_time = 2)
  (h3 : eating_time = 1)
  (h4 : movie_time = 3) :
  total_time - (reading_time + eating_time + movie_time) = 3 := by
  sorry

end train_ride_nap_time_l3885_388597


namespace problem_solution_l3885_388564

theorem problem_solution : 
  (9^100 : ℕ) % 8 = 1 ∧ (2012^2012 : ℕ) % 10 = 6 := by
sorry

end problem_solution_l3885_388564


namespace volunteers_from_third_grade_l3885_388574

/-- Calculates the number of volunteers to be recruited from a specific grade --/
def volunteersFromGrade (totalStudents : ℕ) (gradeStudents : ℕ) (totalVolunteers : ℕ) : ℕ :=
  (gradeStudents * totalVolunteers) / totalStudents

/-- Represents the problem of calculating volunteers from the third grade --/
theorem volunteers_from_third_grade 
  (totalStudents : ℕ) 
  (firstGradeStudents : ℕ) 
  (secondGradeStudents : ℕ) 
  (thirdGradeStudents : ℕ) 
  (totalVolunteers : ℕ) :
  totalStudents = firstGradeStudents + secondGradeStudents + thirdGradeStudents →
  totalStudents = 2040 →
  firstGradeStudents = 680 →
  secondGradeStudents = 850 →
  thirdGradeStudents = 510 →
  totalVolunteers = 12 →
  volunteersFromGrade totalStudents thirdGradeStudents totalVolunteers = 3 :=
by sorry

end volunteers_from_third_grade_l3885_388574


namespace minibus_children_count_l3885_388585

theorem minibus_children_count (total_seats : ℕ) (full_seats : ℕ) (children_per_full_seat : ℕ) (children_per_remaining_seat : ℕ) :
  total_seats = 7 →
  full_seats = 5 →
  children_per_full_seat = 3 →
  children_per_remaining_seat = 2 →
  full_seats * children_per_full_seat + (total_seats - full_seats) * children_per_remaining_seat = 19 :=
by sorry

end minibus_children_count_l3885_388585


namespace max_gcd_of_consecutive_bn_l3885_388530

theorem max_gcd_of_consecutive_bn (n : ℕ) : Nat.gcd (2^n - 1) (2^(n+1) - 1) = 1 := by
  sorry

end max_gcd_of_consecutive_bn_l3885_388530


namespace square_perimeter_l3885_388541

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 144 →
  area = side ^ 2 →
  perimeter = 4 * side →
  perimeter = 48 := by
  sorry

end square_perimeter_l3885_388541


namespace perpendicular_line_equation_l3885_388542

/-- Given a line L1 with equation 3x + 2y - 7 = 0, prove that the line L2 passing through
    the point (-1, 2) and perpendicular to L1 has the equation 2x - 3y + 8 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) : 
  (3 * x + 2 * y - 7 = 0) →  -- equation of L1
  (2 * (-1) - 3 * 2 + 8 = 0) ∧  -- L2 passes through (-1, 2)
  (3 * 2 + 2 * 3 = 0) →  -- perpendicularity condition
  (2 * x - 3 * y + 8 = 0)  -- equation of L2
  := by sorry

end perpendicular_line_equation_l3885_388542


namespace square_sum_power_of_two_l3885_388582

theorem square_sum_power_of_two (n : ℕ) : 
  (∃ m : ℕ, 2^6 + 2^9 + 2^n = m^2) → n = 10 := by
sorry

end square_sum_power_of_two_l3885_388582


namespace ratio_proof_l3885_388575

/-- Given two positive integers with specific properties, prove their ratio -/
theorem ratio_proof (A B : ℕ+) (h1 : A = 48) (h2 : Nat.lcm A B = 432) :
  (A : ℚ) / B = 1 / (4.5 : ℚ) := by
  sorry

end ratio_proof_l3885_388575


namespace ellipse_properties_l3885_388547

-- Define the ellipse G
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / 4 = 1

-- Define the foci
def left_focus (c : ℝ) : ℝ × ℝ := (-c, 0)
def right_focus (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define a point on the ellipse
def point_on_ellipse (a : ℝ) (M : ℝ × ℝ) : Prop := ellipse a M.1 M.2

-- Define perpendicularity condition
def perpendicular (M F₁ F₂ : ℝ × ℝ) : Prop :=
  (M.1 - F₂.1) * (F₂.1 - F₁.1) + (M.2 - F₂.2) * (F₂.2 - F₁.2) = 0

-- Define the distance difference condition
def distance_diff (M F₁ F₂ : ℝ × ℝ) (a : ℝ) : Prop :=
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) -
  Real.sqrt ((M.2 - F₂.1)^2 + (M.2 - F₂.2)^2) = 4*a/3

-- Define the theorem
theorem ellipse_properties (a c : ℝ) (M : ℝ × ℝ) 
  (h_a_pos : a > 0)
  (h_on_ellipse : point_on_ellipse a M)
  (h_perp : perpendicular M (left_focus c) (right_focus c))
  (h_dist : distance_diff M (left_focus c) (right_focus c) a) :
  (∀ x y, ellipse a x y ↔ x^2 / 12 + y^2 / 4 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    ellipse a A.1 A.2 ∧ 
    ellipse a B.1 B.2 ∧
    B.2 - A.2 = B.1 - A.1 ∧ 
    (let P : ℝ × ℝ := (-3, 2);
     let S := (B.1 - A.1) * (P.2 - A.2) - (B.2 - A.2) * (P.1 - A.1);
     S * S / 2 = 9/2)) := by sorry

end ellipse_properties_l3885_388547


namespace winner_for_10_winner_for_12_winner_for_15_winner_for_30_l3885_388501

/-- Represents the outcome of the game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Represents the game state -/
structure GameState where
  n : Nat
  circled : List Nat

/-- Checks if two numbers are relatively prime -/
def isRelativelyPrime (a b : Nat) : Bool :=
  Nat.gcd a b = 1

/-- Checks if a number can be circled given the current game state -/
def canCircle (state : GameState) (num : Nat) : Bool :=
  num ≤ state.n &&
  num ∉ state.circled &&
  state.circled.all (isRelativelyPrime num)

/-- Determines the winner of the game given the initial value of N -/
def determineWinner (n : Nat) : GameOutcome :=
  sorry

/-- Theorem stating the game outcome for N = 10 -/
theorem winner_for_10 : determineWinner 10 = GameOutcome.FirstPlayerWins := by sorry

/-- Theorem stating the game outcome for N = 12 -/
theorem winner_for_12 : determineWinner 12 = GameOutcome.FirstPlayerWins := by sorry

/-- Theorem stating the game outcome for N = 15 -/
theorem winner_for_15 : determineWinner 15 = GameOutcome.SecondPlayerWins := by sorry

/-- Theorem stating the game outcome for N = 30 -/
theorem winner_for_30 : determineWinner 30 = GameOutcome.FirstPlayerWins := by sorry

end winner_for_10_winner_for_12_winner_for_15_winner_for_30_l3885_388501


namespace power_of_two_plus_two_eq_rational_square_l3885_388515

theorem power_of_two_plus_two_eq_rational_square (r : ℚ) :
  (∃ z : ℤ, 2^z + 2 = r^2) ↔ (r = 2 ∨ r = -2 ∨ r = 3/2 ∨ r = -3/2) :=
by sorry

end power_of_two_plus_two_eq_rational_square_l3885_388515


namespace arithmetic_geometric_sequence_problem_l3885_388537

theorem arithmetic_geometric_sequence_problem :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    b - a = c - b ∧
    a + b + c = 15 ∧
    (a + 2) * (c + 13) = (b + 5)^2 ∧
    a = 3 ∧ b = 5 ∧ c = 7 := by
  sorry

end arithmetic_geometric_sequence_problem_l3885_388537


namespace students_liking_both_desserts_l3885_388583

theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (like_apple_pie : ℕ) 
  (like_chocolate_cake : ℕ) 
  (like_neither : ℕ) 
  (h1 : total_students = 50)
  (h2 : like_apple_pie = 25)
  (h3 : like_chocolate_cake = 20)
  (h4 : like_neither = 15) :
  like_apple_pie + like_chocolate_cake - (total_students - like_neither) = 10 := by
  sorry

end students_liking_both_desserts_l3885_388583


namespace smaller_mold_radius_prove_smaller_mold_radius_l3885_388565

/-- The radius of smaller hemisphere-shaped molds when jelly from a larger hemisphere
    is evenly distributed -/
theorem smaller_mold_radius (large_radius : ℝ) (num_small_molds : ℕ) : ℝ :=
  let large_volume := (2 / 3) * Real.pi * large_radius ^ 3
  let small_radius := (large_volume / (num_small_molds * ((2 / 3) * Real.pi))) ^ (1 / 3)
  small_radius

/-- Prove that the radius of each smaller mold is 1 / (2^(2/3)) feet -/
theorem prove_smaller_mold_radius :
  smaller_mold_radius 2 64 = 1 / (2 ^ (2 / 3)) := by
  sorry

end smaller_mold_radius_prove_smaller_mold_radius_l3885_388565


namespace obtuse_triangle_count_l3885_388580

/-- A triangle with sides 5, 12, and k is obtuse -/
def isObtuse (k : ℕ) : Prop :=
  (k > 5 ∧ k > 12 ∧ k^2 > 5^2 + 12^2) ∨
  (12 > 5 ∧ 12 > k ∧ 12^2 > 5^2 + k^2) ∨
  (5 > 12 ∧ 5 > k ∧ 5^2 > 12^2 + k^2)

/-- The triangle with sides 5, 12, and k is valid (satisfies triangle inequality) -/
def isValidTriangle (k : ℕ) : Prop :=
  k + 5 > 12 ∧ k + 12 > 5 ∧ 5 + 12 > k

theorem obtuse_triangle_count :
  ∃! (s : Finset ℕ), (∀ k ∈ s, k > 0 ∧ isValidTriangle k ∧ isObtuse k) ∧ s.card = 6 :=
sorry

end obtuse_triangle_count_l3885_388580


namespace option_a_same_function_option_b_different_function_option_c_different_domain_option_d_same_function_l3885_388555

-- Option A
theorem option_a_same_function (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

-- Option B
theorem option_b_different_function : ∃ x : ℤ, 2*x + 1 ≠ 2*x - 1 := by sorry

-- Option C
def domain_f (x : ℝ) : Prop := x^2 ≥ 9
def domain_g (x : ℝ) : Prop := x ≥ 3

theorem option_c_different_domain : domain_f ≠ domain_g := by sorry

-- Option D
theorem option_d_same_function (x t : ℝ) (h : x = t) : x^2 - 2*x - 1 = t^2 - 2*t - 1 := by sorry

end option_a_same_function_option_b_different_function_option_c_different_domain_option_d_same_function_l3885_388555


namespace tan_double_angle_l3885_388588

theorem tan_double_angle (α β : Real) 
  (h1 : Real.tan (α + β) = 7)
  (h2 : Real.tan (α - β) = 1) :
  Real.tan (2 * α) = -4/3 := by
sorry

end tan_double_angle_l3885_388588


namespace cat_cafe_theorem_l3885_388590

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 2 * cool_cats

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := 3 * paw_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw -/
def total_cats : ℕ := meow_cats + paw_cats

theorem cat_cafe_theorem : total_cats = 40 := by
  sorry

end cat_cafe_theorem_l3885_388590


namespace balloon_distribution_l3885_388581

theorem balloon_distribution (yellow_balloons : ℕ) (black_balloon_difference : ℕ) (num_schools : ℕ) : 
  yellow_balloons = 3414 →
  black_balloon_difference = 1762 →
  num_schools = 10 →
  (yellow_balloons + (yellow_balloons + black_balloon_difference)) / num_schools = 859 :=
by sorry

end balloon_distribution_l3885_388581


namespace subtract_like_terms_l3885_388509

theorem subtract_like_terms (a : ℝ) : 7 * a - 3 * a = 4 * a := by
  sorry

end subtract_like_terms_l3885_388509


namespace greg_travel_distance_l3885_388521

/-- Greg's travel problem -/
theorem greg_travel_distance :
  let distance_to_market : ℝ := 30
  let time_from_market : ℝ := 30 / 60  -- 30 minutes converted to hours
  let speed_from_market : ℝ := 20
  let distance_from_market : ℝ := time_from_market * speed_from_market
  distance_to_market + distance_from_market = 40 := by
  sorry

end greg_travel_distance_l3885_388521


namespace f_monotonicity_and_range_l3885_388500

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - x

theorem f_monotonicity_and_range (a : ℝ) :
  (a ≥ 1/8 → ∀ x > 0, StrictMono (f a)) ∧
  (∀ x ≥ 1, f a x ≥ 0 ↔ a ≥ -1) :=
sorry

end f_monotonicity_and_range_l3885_388500


namespace alex_shirts_l3885_388517

theorem alex_shirts (alex joe ben : ℕ) 
  (h1 : joe = alex + 3) 
  (h2 : ben = joe + 8) 
  (h3 : ben = 15) : 
  alex = 4 := by
sorry

end alex_shirts_l3885_388517


namespace geometric_sequence_first_term_l3885_388560

theorem geometric_sequence_first_term
  (a : ℝ)  -- first term of the sequence
  (r : ℝ)  -- common ratio
  (h1 : a * r^2 = 27)  -- third term is 27
  (h2 : a * r^3 = 81)  -- fourth term is 81
  : a = 3 :=
by sorry

end geometric_sequence_first_term_l3885_388560


namespace percentage_equality_l3885_388549

theorem percentage_equality : (0.375 / 100) * 41000 = 153.75 := by sorry

end percentage_equality_l3885_388549


namespace halfway_fraction_l3885_388596

theorem halfway_fraction (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) :
  (a + b) / 2 = 19/24 := by sorry

end halfway_fraction_l3885_388596


namespace at_least_two_sums_divisible_by_p_l3885_388510

def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem at_least_two_sums_divisible_by_p (p a b c d : ℕ) (hp : p > 2) (hprime : Nat.Prime p)
  (ha : ¬ p ∣ a) (hb : ¬ p ∣ b) (hc : ¬ p ∣ c) (hd : ¬ p ∣ d)
  (h : ∀ r : ℕ, ¬ p ∣ r → 
    fractional_part (r * a / p) + fractional_part (r * b / p) + 
    fractional_part (r * c / p) + fractional_part (r * d / p) = 2) :
  (∃ (x y : ℕ × ℕ), x ≠ y ∧ 
    (x ∈ [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]) ∧
    (y ∈ [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]) ∧
    p ∣ (x.1 + x.2) ∧ p ∣ (y.1 + y.2)) :=
by sorry

end at_least_two_sums_divisible_by_p_l3885_388510


namespace inequality_proof_l3885_388562

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) :
  a^2 * b * c + b^2 * d * a + c^2 * d * a + d^2 * b * c ≤ 4 := by
  sorry

end inequality_proof_l3885_388562


namespace pentagon_area_half_decagon_area_l3885_388535

/-- The area of a pentagon formed by connecting every second vertex of a regular decagon
    is half the area of the decagon. -/
theorem pentagon_area_half_decagon_area (n : ℝ) (h : n > 0) :
  ∃ (m : ℝ), m > 0 ∧ m / n = 1 / 2 := by
  sorry

end pentagon_area_half_decagon_area_l3885_388535


namespace isosceles_right_triangle_not_regular_polygon_l3885_388502

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- The triangle has two equal angles -/
  has_two_equal_angles : Bool
  /-- The triangle has a right angle -/
  has_right_angle : Bool
  /-- All isosceles right triangles are similar -/
  always_similar : Bool
  /-- The triangle has two equal sides -/
  has_two_equal_sides : Bool

/-- A regular polygon -/
structure RegularPolygon where
  /-- All sides are equal -/
  equilateral : Bool
  /-- All angles are equal -/
  equiangular : Bool

/-- Theorem: Isosceles right triangles are not regular polygons -/
theorem isosceles_right_triangle_not_regular_polygon (t : IsoscelesRightTriangle) : 
  ¬∃(p : RegularPolygon), (t.has_two_equal_angles ∧ t.has_right_angle ∧ t.always_similar ∧ t.has_two_equal_sides) → 
  (p.equilateral ∧ p.equiangular) := by
  sorry

end isosceles_right_triangle_not_regular_polygon_l3885_388502


namespace circle_s_radius_l3885_388534

/-- Triangle XYZ with given side lengths -/
structure Triangle :=
  (xy : ℝ)
  (xz : ℝ)
  (yz : ℝ)

/-- Circle with given radius -/
structure Circle :=
  (radius : ℝ)

/-- Theorem stating the radius of circle S in the given triangle configuration -/
theorem circle_s_radius (t : Triangle) (r : Circle) (s : Circle) :
  t.xy = 120 →
  t.xz = 120 →
  t.yz = 80 →
  r.radius = 20 →
  -- Circle R is tangent to XZ and YZ
  -- Circle S is externally tangent to R and tangent to XY and YZ
  -- No point of circle S lies outside of triangle XYZ
  s.radius = 56 - 8 * Real.sqrt 21 := by
  sorry

end circle_s_radius_l3885_388534


namespace division_of_hundred_by_quarter_l3885_388579

theorem division_of_hundred_by_quarter : (100 : ℝ) / 0.25 = 400 := by
  sorry

end division_of_hundred_by_quarter_l3885_388579


namespace certain_number_problem_l3885_388536

theorem certain_number_problem : 
  ∃ x : ℝ, (3500 - (x / 20.50) = 3451.2195121951218) ∧ (x = 1000) := by
  sorry

end certain_number_problem_l3885_388536


namespace shirt_tie_combinations_l3885_388552

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of ties available. -/
def num_ties : ℕ := 7

/-- The number of specific shirt-tie pairs that cannot be worn together. -/
def num_restricted_pairs : ℕ := 3

/-- The total number of possible shirt-tie combinations. -/
def total_combinations : ℕ := num_shirts * num_ties

/-- The number of allowable shirt-tie combinations. -/
def allowable_combinations : ℕ := total_combinations - num_restricted_pairs

theorem shirt_tie_combinations : allowable_combinations = 53 := by
  sorry

end shirt_tie_combinations_l3885_388552


namespace square_reciprocal_sum_l3885_388527

theorem square_reciprocal_sum (p : ℝ) (h : p + 1/p = 10) :
  p^2 + 1/p^2 + 6 = 104 := by
  sorry

end square_reciprocal_sum_l3885_388527


namespace ellipse_eccentricity_l3885_388539

def complex_equation (z : ℂ) : Prop :=
  (z - 2) * (z^2 + z + 2) * (z^2 + 5*z + 8) = 0

def is_root (z : ℂ) : Prop :=
  complex_equation z

def ellipse_through_roots (e : ℝ) : Prop :=
  ∃ (a b : ℝ) (h : ℂ), 
    a > 0 ∧ b > 0 ∧
    ∀ (z : ℂ), is_root z → 
      (z.re - h.re)^2 / a^2 + (z.im - h.im)^2 / b^2 = 1 ∧
    e = Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity : 
  ellipse_through_roots (Real.sqrt (1/5)) :=
sorry

end ellipse_eccentricity_l3885_388539


namespace f_neg_one_value_l3885_388513

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = x^2 + x

theorem f_neg_one_value (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f_nonneg f) : f (-1) = -2 := by
  sorry

end f_neg_one_value_l3885_388513


namespace f_always_positive_l3885_388594

def f (x : ℝ) : ℝ := x^8 - x^5 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, f x > 0 := by
  sorry

end f_always_positive_l3885_388594


namespace football_practice_hours_l3885_388525

/-- Given a football team's practice schedule and a week with one missed day,
    calculate the total practice hours for the week. -/
theorem football_practice_hours (practice_hours_per_day : ℕ) (days_in_week : ℕ) (missed_days : ℕ) : 
  practice_hours_per_day = 5 → days_in_week = 7 → missed_days = 1 →
  (days_in_week - missed_days) * practice_hours_per_day = 30 := by
sorry

end football_practice_hours_l3885_388525


namespace rhombus_properties_l3885_388578

/-- Properties of a rhombus with given area and one diagonal --/
theorem rhombus_properties (area : ℝ) (d1 : ℝ) (d2 : ℝ) (θ : ℝ) 
  (h1 : area = 432)
  (h2 : d1 = 36)
  (h3 : area = (d1 * d2) / 2)
  (h4 : θ = 2 * Real.arccos (2 / 3)) :
  d2 = 24 ∧ θ = 2 * Real.arccos (2 / 3) := by
  sorry


end rhombus_properties_l3885_388578


namespace jogging_average_l3885_388506

theorem jogging_average (days_short : ℕ) (days_long : ℕ) (minutes_short : ℕ) (minutes_long : ℕ) 
  (target_average : ℕ) (total_days : ℕ) :
  days_short = 6 →
  days_long = 4 →
  minutes_short = 80 →
  minutes_long = 105 →
  target_average = 100 →
  total_days = 11 →
  (days_short * minutes_short + days_long * minutes_long + 
   (target_average * total_days - (days_short * minutes_short + days_long * minutes_long))) / total_days = target_average :=
by sorry

end jogging_average_l3885_388506


namespace evaluate_expression_l3885_388524

theorem evaluate_expression : 6 - 9 * (10 - 4^2) * 5 = 276 := by
  sorry

end evaluate_expression_l3885_388524


namespace point_coordinates_l3885_388563

/-- A point in the first quadrant with given distances to axes -/
structure FirstQuadrantPoint where
  m : ℝ
  n : ℝ
  first_quadrant : m > 0 ∧ n > 0
  x_axis_distance : n = 5
  y_axis_distance : m = 3

/-- Theorem: The coordinates of the point are (3,5) -/
theorem point_coordinates (P : FirstQuadrantPoint) : P.m = 3 ∧ P.n = 5 := by
  sorry

end point_coordinates_l3885_388563


namespace fish_tank_capacity_l3885_388519

/-- The capacity of a fish tank given pouring rate, duration, and remaining volume --/
theorem fish_tank_capacity
  (pour_rate : ℚ)  -- Pouring rate in gallons per second
  (pour_duration : ℕ)  -- Pouring duration in minutes
  (remaining_volume : ℕ)  -- Remaining volume to fill the tank in gallons
  (h1 : pour_rate = 1 / 20)  -- 1 gallon every 20 seconds
  (h2 : pour_duration = 6)  -- Poured for 6 minutes
  (h3 : remaining_volume = 32)  -- 32 more gallons needed
  : ℕ :=
by
  sorry

#check fish_tank_capacity

end fish_tank_capacity_l3885_388519


namespace positive_sum_geq_two_l3885_388550

theorem positive_sum_geq_two (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 3) : 
  a + b ≥ 2 := by
  sorry

end positive_sum_geq_two_l3885_388550


namespace undominated_implies_favorite_toy_l3885_388586

/-- A type representing children -/
def Child : Type := Nat

/-- A type representing toys -/
def Toy : Type := Nat

/-- A type representing a preference ordering of toys for a child -/
def Preference := Toy → Toy → Prop

/-- A type representing a distribution of toys to children -/
def Distribution := Child → Toy

/-- Predicate indicating if a toy is preferred over another for a given child's preference -/
def IsPreferred (pref : Preference) (t1 t2 : Toy) : Prop := pref t1 t2 ∧ ¬pref t2 t1

/-- Predicate indicating if a distribution is dominated by another -/
def Dominates (prefs : Child → Preference) (d1 d2 : Distribution) : Prop :=
  ∀ c : Child, IsPreferred (prefs c) (d1 c) (d2 c) ∨ d1 c = d2 c

/-- Predicate indicating if a toy is the favorite for a child -/
def IsFavorite (pref : Preference) (t : Toy) : Prop :=
  ∀ t' : Toy, t ≠ t' → IsPreferred pref t t'

theorem undominated_implies_favorite_toy
  (n : Nat)
  (prefs : Child → Preference)
  (d : Distribution)
  (h_strict : ∀ c : Child, ∀ t1 t2 : Toy, t1 ≠ t2 → (IsPreferred (prefs c) t1 t2 ∨ IsPreferred (prefs c) t2 t1))
  (h_undominated : ∀ d' : Distribution, ¬Dominates prefs d' d) :
  ∃ c : Child, IsFavorite (prefs c) (d c) :=
sorry

end undominated_implies_favorite_toy_l3885_388586


namespace black_fraction_after_changes_l3885_388589

/-- Represents the fraction of the triangle that remains black after each change. -/
def black_fraction_after_change : ℚ := 8/9

/-- Represents the fraction of the triangle that is always black (the central triangle). -/
def always_black_fraction : ℚ := 1/9

/-- Represents the number of changes applied to the triangle. -/
def num_changes : ℕ := 4

/-- Theorem stating the fractional part of the original area that remains black after the changes. -/
theorem black_fraction_after_changes :
  (black_fraction_after_change ^ num_changes) * (1 - always_black_fraction) + always_black_fraction = 39329/59049 := by
  sorry

end black_fraction_after_changes_l3885_388589


namespace water_tank_capacity_l3885_388512

theorem water_tank_capacity : 
  ∀ (tank_capacity : ℝ),
  (0.75 * tank_capacity - 0.4 * tank_capacity = 36) →
  ⌈tank_capacity⌉ = 103 := by
sorry

end water_tank_capacity_l3885_388512


namespace max_sum_is_four_l3885_388551

-- Define the system of inequalities and conditions
def system (x y : ℕ) : Prop :=
  5 * x + 10 * y ≤ 30 ∧ 2 * x - y ≤ 3

-- Theorem statement
theorem max_sum_is_four :
  ∃ (x y : ℕ), system x y ∧ x + y = 4 ∧
  ∀ (a b : ℕ), system a b → a + b ≤ 4 :=
sorry

end max_sum_is_four_l3885_388551


namespace larger_number_proof_l3885_388507

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 1365)
  (h2 : y = 4 * x + 15) : 
  y = 1815 := by
  sorry

end larger_number_proof_l3885_388507


namespace sector_area_l3885_388528

theorem sector_area (θ : Real) (arc_length : Real) (area : Real) : 
  θ = π / 3 →  -- 60° in radians
  arc_length = 2 * π → 
  area = 6 * π :=
by
  sorry

end sector_area_l3885_388528


namespace randy_initial_biscuits_l3885_388533

/-- The number of biscuits Randy's father gave him -/
def father_gift : ℕ := 13

/-- The number of biscuits Randy's mother gave him -/
def mother_gift : ℕ := 15

/-- The number of biscuits Randy's brother ate -/
def brother_ate : ℕ := 20

/-- The number of biscuits Randy is left with -/
def remaining_biscuits : ℕ := 40

/-- Randy's initial number of biscuits -/
def initial_biscuits : ℕ := 32

theorem randy_initial_biscuits :
  initial_biscuits + father_gift + mother_gift - brother_ate = remaining_biscuits :=
by sorry

end randy_initial_biscuits_l3885_388533


namespace count_power_functions_l3885_388532

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = x^k

def f₁ (x : ℝ) : ℝ := x^3
def f₂ (x : ℝ) : ℝ := 4*x^2
def f₃ (x : ℝ) : ℝ := x^5 + 1
def f₄ (x : ℝ) : ℝ := (x-1)^2
def f₅ (x : ℝ) : ℝ := x

theorem count_power_functions : 
  (is_power_function f₁ ∧ ¬is_power_function f₂ ∧ ¬is_power_function f₃ ∧ 
   ¬is_power_function f₄ ∧ is_power_function f₅) :=
by sorry

end count_power_functions_l3885_388532


namespace sqrt_inequality_l3885_388561

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) : Real.sqrt (x - 3) + Real.sqrt (x - 2) > Real.sqrt (x - 4) + x - 1 := by
  sorry

end sqrt_inequality_l3885_388561


namespace liam_and_sisters_ages_l3885_388514

theorem liam_and_sisters_ages (a b : ℕ+) (h1 : a < b) (h2 : a * b * b = 72) : 
  a + b + b = 14 := by
sorry

end liam_and_sisters_ages_l3885_388514


namespace train_passengers_l3885_388505

theorem train_passengers (initial : ℕ) 
  (h1 : initial + 17 - 29 - 27 + 35 = 116) : initial = 120 := by
  sorry

end train_passengers_l3885_388505


namespace at_least_one_not_divisible_l3885_388508

theorem at_least_one_not_divisible (a b c d : ℕ) (h : a * d - b * c > 1) :
  ¬(a * d - b * c ∣ a) ∨ ¬(a * d - b * c ∣ b) ∨ ¬(a * d - b * c ∣ c) ∨ ¬(a * d - b * c ∣ d) :=
by sorry

end at_least_one_not_divisible_l3885_388508


namespace circle_has_zero_radius_l3885_388571

/-- The equation of a circle with radius 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 10*x + y^2 - 4*y + 29 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-5, 2)

theorem circle_has_zero_radius :
  ∀ x y : ℝ, circle_equation x y ↔ (x, y) = circle_center :=
by sorry

end circle_has_zero_radius_l3885_388571


namespace parabola_equation_correct_l3885_388520

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : Line

/-- The equation of a parabola in general form -/
def parabola_equation (p : Parabola) (x y : ℝ) : Prop :=
  x^2 - 2*x*y + y^2 - 12*x - 16*y + 78 = 0

theorem parabola_equation_correct (p : Parabola) :
  p.focus = Point.mk 4 5 →
  p.directrix = Line.mk 1 1 (-2) →
  ∀ x y : ℝ, (x^2 - 2*x*y + y^2 - 12*x - 16*y + 78 = 0) ↔
    (((x - 4)^2 + (y - 5)^2) = ((x + y - 2)^2 / 2)) :=
by sorry

end parabola_equation_correct_l3885_388520
