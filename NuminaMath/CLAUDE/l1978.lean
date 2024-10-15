import Mathlib

namespace NUMINAMATH_CALUDE_negation_existential_equivalence_l1978_197867

theorem negation_existential_equivalence (f : ℝ → ℝ) :
  (¬ ∃ x₀ : ℝ, f x₀ < 0) ↔ (∀ x : ℝ, f x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_equivalence_l1978_197867


namespace NUMINAMATH_CALUDE_missing_number_in_mean_l1978_197863

theorem missing_number_in_mean (known_numbers : List ℝ) (mean : ℝ) : 
  known_numbers = [1, 22, 23, 24, 26, 27, 2] ∧ 
  mean = 20 ∧ 
  (List.sum known_numbers + 35) / 8 = mean →
  35 = 8 * mean - List.sum known_numbers :=
by
  sorry

#check missing_number_in_mean

end NUMINAMATH_CALUDE_missing_number_in_mean_l1978_197863


namespace NUMINAMATH_CALUDE_greatest_common_factor_48_180_240_l1978_197852

theorem greatest_common_factor_48_180_240 : Nat.gcd 48 (Nat.gcd 180 240) = 12 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_48_180_240_l1978_197852


namespace NUMINAMATH_CALUDE_initial_fish_count_l1978_197816

def days_in_three_weeks : ℕ := 21

def koi_added_per_day : ℕ := 2
def goldfish_added_per_day : ℕ := 5

def final_koi_count : ℕ := 227
def final_goldfish_count : ℕ := 200

def total_koi_added : ℕ := days_in_three_weeks * koi_added_per_day
def total_goldfish_added : ℕ := days_in_three_weeks * goldfish_added_per_day

def initial_koi_count : ℕ := final_koi_count - total_koi_added
def initial_goldfish_count : ℕ := final_goldfish_count - total_goldfish_added

theorem initial_fish_count :
  initial_koi_count + initial_goldfish_count = 280 := by
  sorry

end NUMINAMATH_CALUDE_initial_fish_count_l1978_197816


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1978_197846

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = 7) : a^2 + 1/a^2 = 47 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1978_197846


namespace NUMINAMATH_CALUDE_bookstore_sales_ratio_l1978_197805

theorem bookstore_sales_ratio :
  -- Initial conditions
  let initial_inventory : ℕ := 743
  let saturday_instore : ℕ := 37
  let saturday_online : ℕ := 128
  let sunday_online_increase : ℕ := 34
  let shipment : ℕ := 160
  let final_inventory : ℕ := 502

  -- Define Sunday in-store sales
  let sunday_instore : ℕ := initial_inventory - final_inventory + shipment - 
    (saturday_instore + saturday_online + sunday_online_increase)

  -- Theorem statement
  (sunday_instore : ℚ) / (saturday_instore : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_ratio_l1978_197805


namespace NUMINAMATH_CALUDE_trisha_total_distance_l1978_197839

/-- The total distance Trisha walked during her vacation in New York City -/
def total_distance (hotel_to_postcard postcard_to_tshirt tshirt_to_hotel : ℝ) : ℝ :=
  hotel_to_postcard + postcard_to_tshirt + tshirt_to_hotel

/-- Theorem stating that Trisha's total walking distance is 0.89 miles -/
theorem trisha_total_distance :
  total_distance 0.11 0.11 0.67 = 0.89 := by sorry

end NUMINAMATH_CALUDE_trisha_total_distance_l1978_197839


namespace NUMINAMATH_CALUDE_union_determines_m_l1978_197826

def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {2, 3}

theorem union_determines_m (m : ℝ) (h : A m ∪ B = {1, 2, 3}) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_determines_m_l1978_197826


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1978_197829

theorem arithmetic_calculations :
  (3.21 - 1.05 - 1.95 = 0.21) ∧
  (15 - (2.95 + 8.37) = 3.68) ∧
  (14.6 * 2 - 0.6 * 2 = 28) ∧
  (0.25 * 1.25 * 32 = 10) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1978_197829


namespace NUMINAMATH_CALUDE_stating_regular_polygon_triangle_counts_l1978_197817

variable (n : ℕ)

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- The number of right-angled triangles in a regular polygon with 2n sides -/
def num_right_triangles (n : ℕ) : ℕ := 2*n*(n-1)

/-- The number of acute-angled triangles in a regular polygon with 2n sides -/
def num_acute_triangles (n : ℕ) : ℕ := n*(n-1)*(n-2)/3

/-- 
Theorem stating the number of right-angled and acute-angled triangles 
in a regular polygon with 2n sides
-/
theorem regular_polygon_triangle_counts (n : ℕ) (p : RegularPolygon n) :
  (num_right_triangles n = 2*n*(n-1)) ∧ 
  (num_acute_triangles n = n*(n-1)*(n-2)/3) := by
  sorry


end NUMINAMATH_CALUDE_stating_regular_polygon_triangle_counts_l1978_197817


namespace NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l1978_197800

theorem factorization_m_squared_minus_3m (m : ℝ) : m^2 - 3*m = m*(m-3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l1978_197800


namespace NUMINAMATH_CALUDE_correct_num_technicians_l1978_197865

/-- The number of technicians in a workshop -/
def num_technicians : ℕ := 5

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 15

/-- The average salary of all workers -/
def avg_salary_all : ℕ := 700

/-- The average salary of technicians -/
def avg_salary_technicians : ℕ := 800

/-- The average salary of non-technicians -/
def avg_salary_others : ℕ := 650

/-- Theorem stating that the number of technicians is correct given the conditions -/
theorem correct_num_technicians :
  num_technicians = 5 ∧
  num_technicians ≤ total_workers ∧
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_others) / total_workers = avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_correct_num_technicians_l1978_197865


namespace NUMINAMATH_CALUDE_clara_total_earnings_l1978_197855

/-- Represents a staff member at the cake shop -/
structure Staff :=
  (name : String)
  (hourlyRate : ℝ)
  (holidayBonus : ℝ)

/-- Calculates the total earnings for a staff member -/
def totalEarnings (s : Staff) (hoursWorked : ℝ) : ℝ :=
  s.hourlyRate * hoursWorked + s.holidayBonus

/-- Theorem: Clara's total earnings for the 2-month period -/
theorem clara_total_earnings :
  let clara : Staff := { name := "Clara", hourlyRate := 13, holidayBonus := 60 }
  let standardHours : ℝ := 20 * 8  -- 20 hours per week for 8 weeks
  let vacationHours : ℝ := 20 * 1.5  -- 10 days vacation (1.5 weeks)
  let claraHours : ℝ := standardHours - vacationHours
  totalEarnings clara claraHours = 1750 := by
  sorry

end NUMINAMATH_CALUDE_clara_total_earnings_l1978_197855


namespace NUMINAMATH_CALUDE_min_socks_for_pair_l1978_197809

/-- Represents the number of socks of each color in the drawer -/
def socksPerColor : ℕ := 24

/-- Represents the total number of colors of socks in the drawer -/
def numColors : ℕ := 2

/-- Represents the minimum number of socks that must be picked to guarantee a pair of the same color -/
def minSocksToPick : ℕ := 3

/-- Theorem stating that picking 3 socks guarantees at least one pair of the same color,
    and this is the minimum number required -/
theorem min_socks_for_pair :
  (∀ (picked : Finset ℕ), picked.card = minSocksToPick → 
    ∃ (color : Fin numColors), (picked.filter (λ sock => sock % numColors = color)).card ≥ 2) ∧
  (∀ (n : ℕ), n < minSocksToPick → 
    ∃ (picked : Finset ℕ), picked.card = n ∧ 
      ∀ (color : Fin numColors), (picked.filter (λ sock => sock % numColors = color)).card < 2) :=
sorry

end NUMINAMATH_CALUDE_min_socks_for_pair_l1978_197809


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equal_area_l1978_197833

theorem rectangle_perimeter_equal_area (x y : ℕ) : 
  x > 0 ∧ y > 0 → 2 * x + 2 * y = x * y → (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) ∨ (x = 4 ∧ y = 4) := by
  sorry

#check rectangle_perimeter_equal_area

end NUMINAMATH_CALUDE_rectangle_perimeter_equal_area_l1978_197833


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l1978_197819

theorem fraction_sum_theorem (a b c x y z : ℝ) 
  (h1 : x/a + y/b + z/c = 4) 
  (h2 : a/x + b/y + c/z = 3) 
  (h3 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 + 6*(x*y*z)/(a*b*c) = 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l1978_197819


namespace NUMINAMATH_CALUDE_smallest_k_divisible_by_nine_l1978_197811

theorem smallest_k_divisible_by_nine (k : ℕ) : k = 2024 ↔ 
  k > 2019 ∧ 
  (∀ m : ℕ, m > 2019 ∧ m < k → ¬(9 ∣ (m * (m + 1) / 2))) ∧ 
  (9 ∣ (k * (k + 1) / 2)) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_divisible_by_nine_l1978_197811


namespace NUMINAMATH_CALUDE_initial_group_size_l1978_197889

theorem initial_group_size (total_groups : Nat) (students_left : Nat) (remaining_students : Nat) :
  total_groups = 3 →
  students_left = 2 →
  remaining_students = 22 →
  ∃ initial_group_size : Nat, 
    initial_group_size * total_groups - students_left = remaining_students ∧
    initial_group_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_group_size_l1978_197889


namespace NUMINAMATH_CALUDE_hyperbola_branch_from_condition_l1978_197827

/-- The set of points forming one branch of a hyperbola -/
def HyperbolaBranch : Set (ℝ × ℝ) :=
  {P | ∃ (x y : ℝ), P = (x, y) ∧ 
    Real.sqrt ((x + 3)^2 + y^2) - Real.sqrt ((x - 3)^2 + y^2) = 4}

/-- Theorem stating that the given condition forms one branch of a hyperbola -/
theorem hyperbola_branch_from_condition :
  ∃ (F₁ F₂ : ℝ × ℝ), F₁ = (-3, 0) ∧ F₂ = (3, 0) ∧
  HyperbolaBranch = {P | |P.1 - F₁.1| - |P.1 - F₂.1| = 4} :=
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_branch_from_condition_l1978_197827


namespace NUMINAMATH_CALUDE_inequality_proof_l1978_197854

theorem inequality_proof (a : ℝ) (h : a ≠ 2) :
  (1 : ℝ) / (a^2 - 4*a + 4) > 2 / (a^3 - 8) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1978_197854


namespace NUMINAMATH_CALUDE_joyce_final_egg_count_l1978_197880

/-- Calculates the final number of eggs Joyce has after a series of transactions -/
def final_egg_count (initial_eggs : ℝ) (received_eggs : ℝ) (traded_eggs : ℝ) (given_away_eggs : ℝ) : ℝ :=
  initial_eggs + received_eggs - traded_eggs - given_away_eggs

/-- Proves that Joyce ends up with 9 eggs given the initial conditions and transactions -/
theorem joyce_final_egg_count :
  final_egg_count 8 3.5 0.5 2 = 9 := by sorry

end NUMINAMATH_CALUDE_joyce_final_egg_count_l1978_197880


namespace NUMINAMATH_CALUDE_sum_of_squares_equality_l1978_197828

theorem sum_of_squares_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 + b^2 + c^2 = a*b + b*c + c*a) :
  (a^2*b^2)/((a^2+b*c)*(b^2+a*c)) + (a^2*c^2)/((a^2+b*c)*(c^2+a*b)) + (b^2*c^2)/((b^2+a*c)*(c^2+a*b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equality_l1978_197828


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l1978_197847

theorem arithmetic_sequence_squares (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_arithmetic : ∃ d : ℝ, b / (c + a) - a / (b + c) = d ∧ c / (a + b) - b / (c + a) = d) :
  ∃ d' : ℝ, b^2 - a^2 = d' ∧ c^2 - b^2 = d' :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l1978_197847


namespace NUMINAMATH_CALUDE_time_per_regular_letter_l1978_197892

-- Define the given conditions
def days_between_letters : ℕ := 3
def minutes_per_page_regular : ℕ := 10
def minutes_per_page_long : ℕ := 20
def total_minutes_long_letter : ℕ := 80
def total_pages_per_month : ℕ := 24
def days_in_month : ℕ := 30

-- Define the theorem
theorem time_per_regular_letter :
  let pages_long_letter := total_minutes_long_letter / minutes_per_page_long
  let pages_regular_letters := total_pages_per_month - pages_long_letter
  let total_minutes_regular_letters := pages_regular_letters * minutes_per_page_regular
  let num_regular_letters := days_in_month / days_between_letters
  total_minutes_regular_letters / num_regular_letters = 20 := by
  sorry

end NUMINAMATH_CALUDE_time_per_regular_letter_l1978_197892


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l1978_197831

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l1978_197831


namespace NUMINAMATH_CALUDE_inequality_condition_l1978_197818

theorem inequality_condition (a b : ℝ) : 
  (a * |a + b| < |a| * (a + b)) ↔ (a < 0 ∧ b > -a) := by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1978_197818


namespace NUMINAMATH_CALUDE_team_loss_percentage_l1978_197869

theorem team_loss_percentage
  (win_loss_ratio : ℚ)
  (total_games : ℕ)
  (h1 : win_loss_ratio = 8 / 5)
  (h2 : total_games = 52) :
  (loss_percentage : ℚ) →
  loss_percentage = 38 / 100 :=
by sorry

end NUMINAMATH_CALUDE_team_loss_percentage_l1978_197869


namespace NUMINAMATH_CALUDE_fish_per_bowl_l1978_197866

theorem fish_per_bowl (total_bowls : ℕ) (total_fish : ℕ) (h1 : total_bowls = 261) (h2 : total_fish = 6003) :
  total_fish / total_bowls = 23 :=
by sorry

end NUMINAMATH_CALUDE_fish_per_bowl_l1978_197866


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1978_197821

theorem hyperbola_condition (k : ℝ) : 
  (∃ x y : ℝ, x^2 / (2 - k) + y^2 / (k - 1) = 1) ∧ 
  (∀ x y : ℝ, x^2 / (2 - k) + y^2 / (k - 1) = 1 → 
    ∃ a b : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∨ (y^2 / a^2 - x^2 / b^2 = 1)) →
  k < 1 ∨ k > 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1978_197821


namespace NUMINAMATH_CALUDE_square_root_problem_l1978_197857

theorem square_root_problem (a b : ℝ) 
  (h1 : 3^2 = a + 7)
  (h2 : 2^3 = 2*b + 2) :
  ∃ (x : ℝ), x^2 = 3*a + b ∧ (x = 3 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1978_197857


namespace NUMINAMATH_CALUDE_wreath_problem_l1978_197822

/-- Represents the number of flowers in a wreath -/
structure Wreath where
  dandelions : ℕ
  cornflowers : ℕ
  daisies : ℕ

/-- The problem statement -/
theorem wreath_problem (masha katya : Wreath) : 
  (masha.dandelions + masha.cornflowers + masha.daisies + 
   katya.dandelions + katya.cornflowers + katya.daisies = 70) →
  (masha.dandelions = (5 * (masha.dandelions + masha.cornflowers + masha.daisies)) / 9) →
  (katya.daisies = (7 * (katya.dandelions + katya.cornflowers + katya.daisies)) / 17) →
  (masha.dandelions = katya.dandelions) →
  (masha.daisies = katya.daisies) →
  (masha.cornflowers = 2 ∧ katya.cornflowers = 0) :=
by sorry

end NUMINAMATH_CALUDE_wreath_problem_l1978_197822


namespace NUMINAMATH_CALUDE_prob_b_greater_a_l1978_197843

-- Define the sets for a and b
def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {1, 2, 3}

-- Define the event space
def Ω : Finset (ℕ × ℕ) := A.product B

-- Define the favorable event (b > a)
def E : Finset (ℕ × ℕ) := Ω.filter (fun p => p.2 > p.1)

-- Theorem statement
theorem prob_b_greater_a :
  (E.card : ℚ) / Ω.card = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_b_greater_a_l1978_197843


namespace NUMINAMATH_CALUDE_security_guard_schedule_l1978_197883

structure Guard where
  id : Nat
  hours : Nat

def valid_schedule (g2 g3 g4 g5 : Guard) : Prop :=
  g2.id = 2 ∧ g3.id = 3 ∧ g4.id = 4 ∧ g5.id = 5 ∧
  g2.hours + g3.hours + g4.hours + g5.hours = 6 ∧
  g2.hours ≤ 2 ∧
  g3.hours ≤ 3 ∧
  g4.hours = g5.hours + 1 ∧
  g5.hours > 0

theorem security_guard_schedule :
  ∃ (g2 g3 g4 g5 : Guard), valid_schedule g2 g3 g4 g5 :=
sorry

end NUMINAMATH_CALUDE_security_guard_schedule_l1978_197883


namespace NUMINAMATH_CALUDE_divisible_by_512_l1978_197897

theorem divisible_by_512 (n : ℤ) (h : Odd n) :
  ∃ k : ℤ, n^12 - n^8 - n^4 + 1 = 512 * k := by sorry

end NUMINAMATH_CALUDE_divisible_by_512_l1978_197897


namespace NUMINAMATH_CALUDE_power_product_three_six_l1978_197825

theorem power_product_three_six : (3^5 * 6^5 : ℕ) = 34012224 := by
  sorry

end NUMINAMATH_CALUDE_power_product_three_six_l1978_197825


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_increase_l1978_197803

/-- The area increase of an equilateral triangle -/
theorem equilateral_triangle_area_increase :
  ∀ (s : ℝ),
  s > 0 →
  s^2 * Real.sqrt 3 / 4 = 100 * Real.sqrt 3 →
  let new_s := s + 3
  let new_area := new_s^2 * Real.sqrt 3 / 4
  let initial_area := 100 * Real.sqrt 3
  new_area - initial_area = 32.25 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_increase_l1978_197803


namespace NUMINAMATH_CALUDE_radical_product_simplification_l1978_197845

theorem radical_product_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (48 * x) * Real.sqrt (3 * x) * Real.sqrt (50 * x) = 60 * x * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l1978_197845


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_max_value_is_five_l1978_197871

theorem max_value_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by
  sorry

theorem max_value_achieved : 
  ∃ x : ℝ, x^2 + |2*x - 6| = 5 :=
by
  sorry

theorem max_value_is_five : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ 5) ∧ 
  (∃ x : ℝ, x^2 + |2*x - 6| = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_max_value_is_five_l1978_197871


namespace NUMINAMATH_CALUDE_one_meeting_before_completion_l1978_197808

/-- Represents the number of meetings between two runners on a circular track. -/
def number_of_meetings (circumference : ℝ) (speed1 speed2 : ℝ) : ℕ :=
  sorry

/-- Theorem stating that under given conditions, the runners meet once before completing a lap. -/
theorem one_meeting_before_completion :
  let circumference : ℝ := 300
  let speed1 : ℝ := 7
  let speed2 : ℝ := 3
  number_of_meetings circumference speed1 speed2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_meeting_before_completion_l1978_197808


namespace NUMINAMATH_CALUDE_working_partner_receives_8160_l1978_197838

/-- Calculates the money received by the working partner in a business partnership --/
def money_received_by_working_partner (a_investment : ℕ) (b_investment : ℕ) (management_fee_percent : ℕ) (total_profit : ℕ) : ℕ :=
  let management_fee := (management_fee_percent * total_profit) / 100
  let remaining_profit := total_profit - management_fee
  let total_investment := a_investment + b_investment
  let a_share := (a_investment * remaining_profit) / total_investment
  management_fee + a_share

/-- Theorem stating that under given conditions, the working partner receives 8160 rs --/
theorem working_partner_receives_8160 :
  money_received_by_working_partner 5000 1000 10 9600 = 8160 := by
  sorry

#eval money_received_by_working_partner 5000 1000 10 9600

end NUMINAMATH_CALUDE_working_partner_receives_8160_l1978_197838


namespace NUMINAMATH_CALUDE_frozen_yoghurt_cartons_l1978_197884

/-- Represents the number of cartons of ice cream Caleb bought -/
def ice_cream_cartons : ℕ := 10

/-- Represents the cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 4

/-- Represents the cost of one carton of frozen yoghurt in dollars -/
def yoghurt_cost : ℕ := 1

/-- Represents the difference in dollars between ice cream and frozen yoghurt spending -/
def spending_difference : ℕ := 36

/-- Theorem stating that the number of frozen yoghurt cartons Caleb bought is 4 -/
theorem frozen_yoghurt_cartons : ℕ := by
  sorry

end NUMINAMATH_CALUDE_frozen_yoghurt_cartons_l1978_197884


namespace NUMINAMATH_CALUDE_sin_2023pi_over_6_l1978_197848

theorem sin_2023pi_over_6 : Real.sin (2023 * Real.pi / 6) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_2023pi_over_6_l1978_197848


namespace NUMINAMATH_CALUDE_cube_root_of_three_cubed_l1978_197830

theorem cube_root_of_three_cubed (b : ℝ) : b^3 = 3 → b = 3^(1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_three_cubed_l1978_197830


namespace NUMINAMATH_CALUDE_at_least_three_positive_and_negative_l1978_197881

theorem at_least_three_positive_and_negative (a : Fin 12 → ℝ) 
  (h : ∀ i : Fin 11, a (i + 1) * (a i - a (i + 1) + a (i + 2)) < 0) :
  (∃ i j k : Fin 12, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 0 < a i ∧ 0 < a j ∧ 0 < a k) ∧
  (∃ i j k : Fin 12, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i < 0 ∧ a j < 0 ∧ a k < 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_three_positive_and_negative_l1978_197881


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l1978_197882

theorem perfect_square_quadratic (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + m*x + 16 = y^2) → (m = 8 ∨ m = -8) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l1978_197882


namespace NUMINAMATH_CALUDE_record_storage_cost_l1978_197899

/-- A record storage problem -/
theorem record_storage_cost (box_length box_width box_height : ℝ)
  (total_volume : ℝ) (cost_per_box : ℝ) :
  box_length = 15 →
  box_width = 12 →
  box_height = 10 →
  total_volume = 1080000 →
  cost_per_box = 0.2 →
  (total_volume / (box_length * box_width * box_height)) * cost_per_box = 120 := by
  sorry

end NUMINAMATH_CALUDE_record_storage_cost_l1978_197899


namespace NUMINAMATH_CALUDE_range_of_a_range_of_f_when_a_is_2_l1978_197890

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- Part 1: Range of a when f(x) ≥ 0 for all x ∈ ℝ
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 0) → a ∈ Set.Icc (-2) 2 :=
sorry

-- Part 2: Range of f(x) when a = 2 and x ∈ [0, 3]
theorem range_of_f_when_a_is_2 : 
  Set.image (f 2) (Set.Icc 0 3) = Set.Icc 0 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_f_when_a_is_2_l1978_197890


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l1978_197820

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 4 → 
  (10 * x + y) - (10 * y + x) = 36 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l1978_197820


namespace NUMINAMATH_CALUDE_vector_perpendicular_l1978_197870

/-- Given plane vectors a and b, prove that (a - b) is perpendicular to b -/
theorem vector_perpendicular (a b : ℝ × ℝ) (ha : a = (2, 0)) (hb : b = (1, 1)) :
  (a - b) • b = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l1978_197870


namespace NUMINAMATH_CALUDE_sector_central_angle_l1978_197802

/-- Given a circle sector with radius 10 cm and perimeter 45 cm, 
    the central angle of the sector is 2.5 radians. -/
theorem sector_central_angle (r : ℝ) (p : ℝ) (h1 : r = 10) (h2 : p = 45) :
  (p - 2 * r) / r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1978_197802


namespace NUMINAMATH_CALUDE_general_position_lines_regions_l1978_197801

/-- 
A configuration of lines in general position.
-/
structure GeneralPositionLines where
  n : ℕ
  no_parallel : True  -- Represents the condition that no two lines are parallel
  no_concurrent : True -- Represents the condition that no three lines are concurrent

/-- 
The number of regions created by n lines in general position.
-/
def num_regions (lines : GeneralPositionLines) : ℕ :=
  1 + (lines.n * (lines.n + 1)) / 2

/-- 
Theorem: n lines in general position divide a plane into 1 + (1/2) * n * (n + 1) regions.
-/
theorem general_position_lines_regions (lines : GeneralPositionLines) :
  num_regions lines = 1 + (lines.n * (lines.n + 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_general_position_lines_regions_l1978_197801


namespace NUMINAMATH_CALUDE_jack_money_per_can_l1978_197885

def bottles_recycled : ℕ := 80
def cans_recycled : ℕ := 140
def total_money : ℚ := 15
def money_per_bottle : ℚ := 1/10

theorem jack_money_per_can :
  (total_money - (bottles_recycled : ℚ) * money_per_bottle) / (cans_recycled : ℚ) = 5/100 := by
  sorry

end NUMINAMATH_CALUDE_jack_money_per_can_l1978_197885


namespace NUMINAMATH_CALUDE_factorization_equality_l1978_197806

theorem factorization_equality (x a : ℝ) : 4*x - x*a^2 = x*(2-a)*(2+a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1978_197806


namespace NUMINAMATH_CALUDE_equal_chord_lengths_l1978_197841

-- Define the circle equation
def circle_equation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

-- Define the condition D^2 ≠ E^2 > 4F
def condition (D E F : ℝ) : Prop :=
  D^2 ≠ E^2 ∧ E^2 > 4*F

-- Theorem statement
theorem equal_chord_lengths (D E F : ℝ) 
  (h : condition D E F) : 
  ∃ (chord_x chord_y : ℝ), 
    (∀ (x y : ℝ), circle_equation x y D E F → 
      (x = chord_x/2 ∨ x = -chord_x/2) ∨ (y = chord_y/2 ∨ y = -chord_y/2)) ∧
    chord_x = chord_y :=
sorry

end NUMINAMATH_CALUDE_equal_chord_lengths_l1978_197841


namespace NUMINAMATH_CALUDE_lecture_scheduling_l1978_197861

-- Define the number of lecturers
def n : ℕ := 7

-- Theorem statement
theorem lecture_scheduling (n : ℕ) (h : n = 7) : 
  (n! : ℕ) / 2 = 2520 :=
sorry

end NUMINAMATH_CALUDE_lecture_scheduling_l1978_197861


namespace NUMINAMATH_CALUDE_sine_sum_constant_l1978_197874

theorem sine_sum_constant (α : Real) :
  (Real.sin α) ^ 2 + (Real.sin (α + 60 * π / 180)) ^ 2 + (Real.sin (α + 120 * π / 180)) ^ 2 =
  (Real.sin (α - 60 * π / 180)) ^ 2 + (Real.sin α) ^ 2 + (Real.sin (α + 60 * π / 180)) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_sine_sum_constant_l1978_197874


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l1978_197896

/-- Given a curve y = e^(ax), prove that if its tangent line at (0,1) is perpendicular to the line x + 2y + 1 = 0, then a = 2. -/
theorem tangent_line_perpendicular (a : ℝ) : 
  (∀ x, deriv (fun x => Real.exp (a * x)) x = a * Real.exp (a * x)) →
  (fun x => Real.exp (a * x)) 0 = 1 →
  (deriv (fun x => Real.exp (a * x))) 0 = (-1 / (2 : ℝ))⁻¹ →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l1978_197896


namespace NUMINAMATH_CALUDE_last_digit_sum_l1978_197814

theorem last_digit_sum (n : ℕ) : 
  (2^2 + 20^20 + 200^200 + 2006^2006) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_sum_l1978_197814


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1978_197824

theorem unknown_number_proof (x : ℝ) :
  (x + 48 / 69) * 69 = 1980 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1978_197824


namespace NUMINAMATH_CALUDE_f_monotone_increasing_F_lower_bound_l1978_197844

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - 4 * a * Real.exp x - 2 * a * x

def g (a : ℝ) (x : ℝ) : ℝ := x^2 + 5 * a^2

def F (a : ℝ) (x : ℝ) : ℝ := f a x + g a x

-- Theorem 1: f is monotonically increasing when a ≤ 0
theorem f_monotone_increasing (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) ↔ a ≤ 0 :=
sorry

-- Theorem 2: F has a lower bound
theorem F_lower_bound (a : ℝ) (x : ℝ) :
  F a x ≥ 4 * (1 - Real.log 2)^2 / 5 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_F_lower_bound_l1978_197844


namespace NUMINAMATH_CALUDE_hope_project_protractors_l1978_197891

theorem hope_project_protractors :
  ∀ (x y z : ℕ),
  x > 31 →
  z > 33 →
  10 * x + 15 * y + 20 * z = 1710 →
  8 * x + 2 * y + 8 * z = 664 →
  6 * x + 7 * y + 10 * z = 870 :=
by
  sorry

end NUMINAMATH_CALUDE_hope_project_protractors_l1978_197891


namespace NUMINAMATH_CALUDE_no_real_solutions_l1978_197836

theorem no_real_solutions : 
  ∀ x : ℝ, (2 * x^2 - 3 * x + 5)^2 + 1 ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1978_197836


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_minus_1_l1978_197859

theorem infinitely_many_primes_4k_minus_1 : 
  ∃ (S : Set Nat), (∀ n ∈ S, Nat.Prime n ∧ ∃ k, n = 4*k - 1) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_minus_1_l1978_197859


namespace NUMINAMATH_CALUDE_correct_answer_is_ten_l1978_197893

theorem correct_answer_is_ten (x : ℝ) (h : 3 * x = 90) : x / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_is_ten_l1978_197893


namespace NUMINAMATH_CALUDE_container_capacity_l1978_197860

theorem container_capacity (container_capacity : ℝ) 
  (h1 : 8 = 0.2 * container_capacity) 
  (num_containers : ℕ := 40) : 
  num_containers * container_capacity = 1600 := by
sorry

end NUMINAMATH_CALUDE_container_capacity_l1978_197860


namespace NUMINAMATH_CALUDE_part_I_part_II_l1978_197834

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := { x | 0 < 2*x + a ∧ 2*x + a ≤ 3 }
def B : Set ℝ := { x | -1/2 < x ∧ x < 2 }

-- Part I
theorem part_I : 
  (Set.univ \ B) ∪ (A 1) = { x | x ≤ 1 ∨ x ≥ 2 } := by sorry

-- Part II
theorem part_II : 
  ∀ a : ℝ, (A a) ∩ B = A a ↔ -1 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l1978_197834


namespace NUMINAMATH_CALUDE_jacket_sale_price_l1978_197887

/-- Proves that the price of each jacket after noon was $18.95 given the sale conditions --/
theorem jacket_sale_price (total_jackets : ℕ) (price_before_noon : ℚ) (total_receipts : ℚ) (jackets_sold_after_noon : ℕ) :
  total_jackets = 214 →
  price_before_noon = 31.95 →
  total_receipts = 5108.30 →
  jackets_sold_after_noon = 133 →
  (total_receipts - (total_jackets - jackets_sold_after_noon : ℚ) * price_before_noon) / jackets_sold_after_noon = 18.95 := by
  sorry

end NUMINAMATH_CALUDE_jacket_sale_price_l1978_197887


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1978_197875

theorem complex_fraction_equality : (5 * Complex.I) / (2 - Complex.I) = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1978_197875


namespace NUMINAMATH_CALUDE_integral_condition_implies_b_value_l1978_197837

open MeasureTheory Measure Set Real
open intervalIntegral

theorem integral_condition_implies_b_value (b : ℝ) :
  (∫ x in (-1)..0, (2 * x + b)) = 2 →
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_condition_implies_b_value_l1978_197837


namespace NUMINAMATH_CALUDE_bisecting_centers_form_line_l1978_197898

/-- Two non-overlapping circles in a plane -/
structure TwoCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  R₁ : ℝ
  R₂ : ℝ
  h_positive : R₁ > 0 ∧ R₂ > 0
  h_non_overlapping : Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2) > R₁ + R₂

/-- A point that is the center of a circle bisecting both given circles -/
def BisectingCenter (tc : TwoCircles) (X : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    r^2 = (X.1 - tc.O₁.1)^2 + (X.2 - tc.O₁.2)^2 + tc.R₁^2 ∧
    r^2 = (X.1 - tc.O₂.1)^2 + (X.2 - tc.O₂.2)^2 + tc.R₂^2

/-- The locus of bisecting centers forms a straight line -/
theorem bisecting_centers_form_line (tc : TwoCircles) :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧
    (∀ X : ℝ × ℝ, BisectingCenter tc X ↔ a * X.1 + b * X.2 + c = 0) ∧
    (a * (tc.O₂.1 - tc.O₁.1) + b * (tc.O₂.2 - tc.O₁.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_bisecting_centers_form_line_l1978_197898


namespace NUMINAMATH_CALUDE_work_earnings_equality_l1978_197876

theorem work_earnings_equality (t : ℝ) 
  (my_hours : ℝ := t - 4)
  (my_rate : ℝ := 3*t - 7)
  (bob_hours : ℝ := 3*t - 12)
  (bob_rate : ℝ := t - 6)
  (h : my_hours * my_rate = bob_hours * bob_rate) : 
  t = 44 := by
sorry

end NUMINAMATH_CALUDE_work_earnings_equality_l1978_197876


namespace NUMINAMATH_CALUDE_arlo_books_count_l1978_197850

theorem arlo_books_count (total_stationery : ℕ) (book_ratio pen_ratio : ℕ) (h1 : total_stationery = 400) (h2 : book_ratio = 7) (h3 : pen_ratio = 3) : 
  (book_ratio * total_stationery) / (book_ratio + pen_ratio) = 280 := by
sorry

end NUMINAMATH_CALUDE_arlo_books_count_l1978_197850


namespace NUMINAMATH_CALUDE_ben_basketball_boxes_ben_basketball_boxes_correct_l1978_197879

theorem ben_basketball_boxes (basketball_cards_per_box : ℕ) 
                              (baseball_boxes : ℕ) 
                              (baseball_cards_per_box : ℕ) 
                              (cards_given_away : ℕ) 
                              (cards_left : ℕ) : ℕ :=
  let total_baseball_cards := baseball_boxes * baseball_cards_per_box
  let total_cards_before := cards_given_away + cards_left
  let basketball_boxes := (total_cards_before - total_baseball_cards) / basketball_cards_per_box
  basketball_boxes

#check ben_basketball_boxes 10 5 8 58 22 = 4

theorem ben_basketball_boxes_correct : ben_basketball_boxes 10 5 8 58 22 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ben_basketball_boxes_ben_basketball_boxes_correct_l1978_197879


namespace NUMINAMATH_CALUDE_factor_expression_l1978_197853

theorem factor_expression (x y : ℝ) : 286 * x^2 * y + 143 * x = 143 * x * (2 * x * y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1978_197853


namespace NUMINAMATH_CALUDE_f_unique_zero_g_max_increasing_param_l1978_197835

noncomputable section

def f (x : ℝ) := (x - 2) * Real.log x + 2 * x - 3

def g (a : ℝ) (x : ℝ) := (x - a) * Real.log x + a * (x - 1) / x

theorem f_unique_zero :
  ∃! x : ℝ, x ≥ 1 ∧ f x = 0 :=
sorry

theorem g_max_increasing_param :
  (∃ (a : ℤ), ∀ x ≥ 1, Monotone (g a)) ∧
  (∀ a : ℤ, a > 6 → ∃ x ≥ 1, ¬Monotone (g a)) :=
sorry

end NUMINAMATH_CALUDE_f_unique_zero_g_max_increasing_param_l1978_197835


namespace NUMINAMATH_CALUDE_snack_pack_distribution_l1978_197858

/-- Given the number of pretzels, goldfish, suckers, and kids, calculate the number of items per baggie -/
def items_per_baggie (pretzels : ℕ) (goldfish_multiplier : ℕ) (suckers : ℕ) (kids : ℕ) : ℕ :=
  (pretzels + pretzels * goldfish_multiplier + suckers) / kids

/-- Theorem: Given 64 pretzels, 4 times as many goldfish, 32 suckers, and 16 kids, each baggie will contain 22 items -/
theorem snack_pack_distribution :
  items_per_baggie 64 4 32 16 = 22 := by
  sorry

end NUMINAMATH_CALUDE_snack_pack_distribution_l1978_197858


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_inradius_and_area_l1978_197878

theorem triangle_perimeter_from_inradius_and_area :
  ∀ (r A p : ℝ),
  r > 0 →
  A > 0 →
  r = 2.5 →
  A = 60 →
  A = r * (p / 2) →
  p = 48 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_inradius_and_area_l1978_197878


namespace NUMINAMATH_CALUDE_test_probabilities_l1978_197862

theorem test_probabilities (p_A p_B p_C : ℝ) 
  (h_A : p_A = 0.8) (h_B : p_B = 0.6) (h_C : p_C = 0.5) : 
  p_A * p_B * p_C = 0.24 ∧ 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_test_probabilities_l1978_197862


namespace NUMINAMATH_CALUDE_cube_cube_squared_power_calculation_l1978_197872

theorem cube_cube_squared (a b : ℕ) : (a^3 * b^3)^2 = (a * b)^6 := by sorry

theorem power_calculation : (3^3 * 4^3)^2 = 2985984 := by sorry

end NUMINAMATH_CALUDE_cube_cube_squared_power_calculation_l1978_197872


namespace NUMINAMATH_CALUDE_ordered_pair_solution_l1978_197823

theorem ordered_pair_solution : ∃ (x y : ℤ), 
  Real.sqrt (16 - 12 * Real.cos (40 * π / 180)) = ↑x + ↑y * (1 / Real.cos (40 * π / 180)) ∧ 
  (x, y) = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_ordered_pair_solution_l1978_197823


namespace NUMINAMATH_CALUDE_tablecloth_diameter_is_ten_l1978_197807

/-- The diameter of a circular tablecloth with a given radius --/
def tablecloth_diameter (radius : ℝ) : ℝ := 2 * radius

/-- Theorem: The diameter of a circular tablecloth with a radius of 5 feet is 10 feet --/
theorem tablecloth_diameter_is_ten :
  tablecloth_diameter 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tablecloth_diameter_is_ten_l1978_197807


namespace NUMINAMATH_CALUDE_basketballs_in_boxes_l1978_197813

theorem basketballs_in_boxes 
  (total_basketballs : ℕ) 
  (basketballs_per_bag : ℕ) 
  (bags_per_box : ℕ) 
  (h1 : total_basketballs = 720) 
  (h2 : basketballs_per_bag = 8) 
  (h3 : bags_per_box = 6) : 
  (total_basketballs / (basketballs_per_bag * bags_per_box)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_basketballs_in_boxes_l1978_197813


namespace NUMINAMATH_CALUDE_goods_train_speed_l1978_197849

/-- The speed of the goods train given the conditions of the problem -/
theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (passing_time : ℝ) 
  (goods_train_length : ℝ) 
  (h1 : man_train_speed = 20) 
  (h2 : passing_time = 9) 
  (h3 : goods_train_length = 0.28) : 
  ∃ (goods_train_speed : ℝ), goods_train_speed = 92 := by
  sorry

end NUMINAMATH_CALUDE_goods_train_speed_l1978_197849


namespace NUMINAMATH_CALUDE_tangent_circles_distance_l1978_197815

/-- The distance between the centers of two tangent circles with radii 1 and 7 is either 6 or 8. -/
theorem tangent_circles_distance (r₁ r₂ d : ℝ) : 
  r₁ = 1 → r₂ = 7 → (d = |r₁ - r₂| ∨ d = r₁ + r₂) → d = 6 ∨ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_distance_l1978_197815


namespace NUMINAMATH_CALUDE_matrix_power_four_l1978_197810

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -2; 2, 1]

theorem matrix_power_four :
  A ^ 4 = !![(-7 : ℤ), 24; -24, 7] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l1978_197810


namespace NUMINAMATH_CALUDE_coefficient_a5_equals_6_l1978_197856

theorem coefficient_a5_equals_6 
  (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, x^6 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6) →
  a₅ = 6 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a5_equals_6_l1978_197856


namespace NUMINAMATH_CALUDE_apple_probabilities_l1978_197840

structure ApplePlot where
  name : String
  first_grade_ratio : ℚ
  production_ratio : ℕ

def plot_a : ApplePlot := ⟨"A", 3/4, 2⟩
def plot_b : ApplePlot := ⟨"B", 3/5, 5⟩
def plot_c : ApplePlot := ⟨"C", 4/5, 3⟩

def total_production : ℕ := plot_a.production_ratio + plot_b.production_ratio + plot_c.production_ratio

theorem apple_probabilities :
  (plot_a.production_ratio : ℚ) / total_production = 1/5 ∧
  (plot_a.production_ratio * plot_a.first_grade_ratio +
   plot_b.production_ratio * plot_b.first_grade_ratio +
   plot_c.production_ratio * plot_c.first_grade_ratio) / total_production = 69/100 ∧
  (plot_a.production_ratio * plot_a.first_grade_ratio) /
  (plot_a.production_ratio * plot_a.first_grade_ratio +
   plot_b.production_ratio * plot_b.first_grade_ratio +
   plot_c.production_ratio * plot_c.first_grade_ratio) = 5/23 := by
  sorry


end NUMINAMATH_CALUDE_apple_probabilities_l1978_197840


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1978_197804

/-- An arithmetic sequence with its sum function and properties -/
structure ArithmeticSequence where
  /-- The general term of the sequence -/
  a : ℕ → ℝ
  /-- The sum of the first n terms -/
  S : ℕ → ℝ
  /-- The sum of the first 4 terms is 0 -/
  sum_4 : S 4 = 0
  /-- The 5th term is 5 -/
  term_5 : a 5 = 5
  /-- The sequence is arithmetic -/
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The general term of the arithmetic sequence is 2n - 5 -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n - 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1978_197804


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1978_197886

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1978_197886


namespace NUMINAMATH_CALUDE_matt_trading_profit_l1978_197894

/-- Represents the profit made from trading baseball cards -/
def tradingProfit (initialCardCount : ℕ) (initialCardValue : ℕ) 
                  (tradedCardCount : ℕ) (receivedCardValues : List ℕ) : ℤ :=
  let initialValue := initialCardCount * initialCardValue
  let tradedValue := tradedCardCount * initialCardValue
  let receivedValue := receivedCardValues.sum
  (receivedValue : ℤ) - (tradedValue : ℤ)

/-- Theorem stating that Matt's trading profit is $3 -/
theorem matt_trading_profit :
  tradingProfit 8 6 2 [2, 2, 2, 9] = 3 := by
  sorry

#eval tradingProfit 8 6 2 [2, 2, 2, 9]

end NUMINAMATH_CALUDE_matt_trading_profit_l1978_197894


namespace NUMINAMATH_CALUDE_angle_A_value_max_value_angle_B_at_max_l1978_197864

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Side lengths
variable (S : ℝ) -- Area of the triangle

-- Define the conditions
axiom triangle_condition : a^2 = b^2 + c^2 + Real.sqrt 3 * a * b
axiom side_a_value : a = Real.sqrt 3

-- Define the theorems to be proved
theorem angle_A_value : A = 5 * Real.pi / 6 :=
sorry

theorem max_value : 
  ∃ (max : ℝ), ∀ (B C : ℝ), S + 3 * Real.cos B * Real.cos C ≤ max ∧ 
  ∃ (B₀ C₀ : ℝ), S + 3 * Real.cos B₀ * Real.cos C₀ = max ∧ max = 3 :=
sorry

theorem angle_B_at_max : 
  ∃ (B₀ C₀ : ℝ), S + 3 * Real.cos B₀ * Real.cos C₀ = 3 ∧ B₀ = Real.pi / 12 :=
sorry

end

end NUMINAMATH_CALUDE_angle_A_value_max_value_angle_B_at_max_l1978_197864


namespace NUMINAMATH_CALUDE_brownie_cutting_l1978_197812

theorem brownie_cutting (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 15)
  (h3 : piece_length = 3)
  (h4 : piece_width = 4) : 
  pan_length * pan_width - (pan_length * pan_width / (piece_length * piece_width)) * (piece_length * piece_width) = 0 :=
by sorry

end NUMINAMATH_CALUDE_brownie_cutting_l1978_197812


namespace NUMINAMATH_CALUDE_find_n_l1978_197842

theorem find_n : ∃ n : ℤ, 3^3 - 5 = 2^5 + n ∧ n = -10 := by sorry

end NUMINAMATH_CALUDE_find_n_l1978_197842


namespace NUMINAMATH_CALUDE_square_difference_equals_product_l1978_197832

theorem square_difference_equals_product (x y : ℚ) 
  (sum_eq : x + y = 8/15) 
  (diff_eq : x - y = 2/15) : 
  x^2 - y^2 = 16/225 := by
sorry

end NUMINAMATH_CALUDE_square_difference_equals_product_l1978_197832


namespace NUMINAMATH_CALUDE_congruence_problem_l1978_197873

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 4 ∧ n ≡ -1458 [ZMOD 5] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1978_197873


namespace NUMINAMATH_CALUDE_viggo_payment_l1978_197888

/-- Represents the denomination of the other bills used by Viggo --/
def other_denomination : ℕ := sorry

/-- The total amount spent on the shirt --/
def total_spent : ℕ := 80

/-- The number of other denomination bills used --/
def num_other_bills : ℕ := 2

/-- The denomination of the $20 bills --/
def twenty_bill : ℕ := 20

/-- The number of $20 bills used --/
def num_twenty_bills : ℕ := num_other_bills + 1

theorem viggo_payment :
  (num_twenty_bills * twenty_bill) + (num_other_bills * other_denomination) = total_spent ∧
  other_denomination = 10 := by sorry

end NUMINAMATH_CALUDE_viggo_payment_l1978_197888


namespace NUMINAMATH_CALUDE_baseball_tickets_sold_l1978_197877

theorem baseball_tickets_sold (fair_tickets : ℕ) (baseball_tickets : ℕ) : 
  fair_tickets = 25 → 
  fair_tickets = 2 * baseball_tickets + 6 → 
  baseball_tickets = 9 := by
sorry

end NUMINAMATH_CALUDE_baseball_tickets_sold_l1978_197877


namespace NUMINAMATH_CALUDE_quadratic_root_transform_l1978_197851

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots x₁ and x₂,
    this theorem proves the equations with transformed roots. -/
theorem quadratic_root_transform (a b c : ℝ) (x₁ x₂ : ℝ) 
  (hroot : a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :
  (∃ y₁ y₂ : ℝ, y₁ = 1/x₁^3 ∧ y₂ = 1/x₂^3 ∧ 
    c^3 * y₁^2 + (b^3 - 3*a*b*c) * y₁ + a^3 = 0 ∧
    c^3 * y₂^2 + (b^3 - 3*a*b*c) * y₂ + a^3 = 0) ∧
  (∃ z₁ z₂ : ℝ, z₁ = (x₁ - x₂)^2 ∧ z₂ = (x₁ + x₂)^2 ∧
    a^4 * z₁^2 + 2*a^2*(2*a*c - b^2) * z₁ + b^2*(b^2 - 4*a*c) = 0 ∧
    a^4 * z₂^2 + 2*a^2*(2*a*c - b^2) * z₂ + b^2*(b^2 - 4*a*c) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transform_l1978_197851


namespace NUMINAMATH_CALUDE_square_area_not_tripled_when_side_tripled_l1978_197868

theorem square_area_not_tripled_when_side_tripled (s : ℝ) (h : s > 0) :
  (3 * s)^2 ≠ 3 * s^2 := by sorry

end NUMINAMATH_CALUDE_square_area_not_tripled_when_side_tripled_l1978_197868


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1978_197895

theorem nested_fraction_evaluation :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1978_197895
