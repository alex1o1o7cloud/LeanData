import Mathlib

namespace NUMINAMATH_CALUDE_prudence_sleep_is_200_l2428_242846

/-- Represents Prudence's sleep schedule and calculates total sleep over 4 weeks -/
def prudence_sleep : ℕ :=
  let weekday_sleep : ℕ := 5 * 6  -- 5 nights of 6 hours each
  let weekend_sleep : ℕ := 2 * 9  -- 2 nights of 9 hours each
  let nap_sleep : ℕ := 2 * 1      -- 2 days of 1 hour nap each
  let weekly_sleep : ℕ := weekday_sleep + weekend_sleep + nap_sleep
  4 * weekly_sleep                -- 4 weeks

/-- Theorem stating that Prudence's total sleep over 4 weeks is 200 hours -/
theorem prudence_sleep_is_200 : prudence_sleep = 200 := by
  sorry

#eval prudence_sleep  -- This will evaluate to 200

end NUMINAMATH_CALUDE_prudence_sleep_is_200_l2428_242846


namespace NUMINAMATH_CALUDE_shopping_tax_percentage_l2428_242882

theorem shopping_tax_percentage (total : ℝ) (h_total_pos : total > 0) : 
  let clothing_percent : ℝ := 0.50
  let food_percent : ℝ := 0.20
  let other_percent : ℝ := 0.30
  let clothing_tax_rate : ℝ := 0.04
  let food_tax_rate : ℝ := 0
  let other_tax_rate : ℝ := 0.08
  let clothing_amount := clothing_percent * total
  let food_amount := food_percent * total
  let other_amount := other_percent * total
  let clothing_tax := clothing_tax_rate * clothing_amount
  let food_tax := food_tax_rate * food_amount
  let other_tax := other_tax_rate * other_amount
  let total_tax := clothing_tax + food_tax + other_tax
  total_tax / total = 0.0440 :=
by sorry

end NUMINAMATH_CALUDE_shopping_tax_percentage_l2428_242882


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l2428_242809

theorem polynomial_value_theorem (f : ℝ → ℝ) :
  (∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  (|f 1| = 12 ∧ |f 2| = 12 ∧ |f 3| = 12 ∧ |f 5| = 12 ∧ |f 6| = 12 ∧ |f 7| = 12) →
  |f 0| = 72 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l2428_242809


namespace NUMINAMATH_CALUDE_number_equation_proof_l2428_242831

theorem number_equation_proof (x : ℝ) (N : ℝ) : 
  x = 32 → 
  N - (23 - (15 - x)) = 12 * 2 / (1 / 2) → 
  N = 88 := by
sorry

end NUMINAMATH_CALUDE_number_equation_proof_l2428_242831


namespace NUMINAMATH_CALUDE_llama_accessible_area_l2428_242880

/-- Represents a rectangular shed -/
structure Shed :=
  (length : ℝ)
  (width : ℝ)

/-- Calculates the area accessible to a llama tied to the corner of a shed -/
def accessible_area (s : Shed) (leash_length : ℝ) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem stating the accessible area for a llama tied to a 2m by 4m shed with a 4m leash -/
theorem llama_accessible_area :
  let s : Shed := ⟨4, 2⟩
  let leash_length : ℝ := 4
  accessible_area s leash_length = 13 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_llama_accessible_area_l2428_242880


namespace NUMINAMATH_CALUDE_set_equality_l2428_242836

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l2428_242836


namespace NUMINAMATH_CALUDE_reading_time_calculation_l2428_242876

def total_reading_time (total_chapters : ℕ) (reading_time_per_chapter : ℕ) : ℕ :=
  let chapters_read := total_chapters - (total_chapters / 3)
  (chapters_read * reading_time_per_chapter) / 60

theorem reading_time_calculation :
  total_reading_time 31 20 = 7 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l2428_242876


namespace NUMINAMATH_CALUDE_john_unintended_texts_l2428_242837

/-- The number of text messages John receives per week that are not intended for him -/
def unintended_texts_per_week (old_daily_texts old_daily_texts_from_friends new_daily_texts days_per_week : ℕ) : ℕ :=
  (new_daily_texts - old_daily_texts) * days_per_week

/-- Proof that John receives 245 unintended text messages per week -/
theorem john_unintended_texts :
  let old_daily_texts : ℕ := 20
  let new_daily_texts : ℕ := 55
  let days_per_week : ℕ := 7
  unintended_texts_per_week old_daily_texts old_daily_texts new_daily_texts days_per_week = 245 :=
by sorry

end NUMINAMATH_CALUDE_john_unintended_texts_l2428_242837


namespace NUMINAMATH_CALUDE_molecular_weight_8_moles_Al2O3_l2428_242872

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Aluminum atoms in one molecule of Al2O3 -/
def num_Al_atoms : ℕ := 2

/-- The number of Oxygen atoms in one molecule of Al2O3 -/
def num_O_atoms : ℕ := 3

/-- The number of moles of Al2O3 -/
def num_moles : ℕ := 8

/-- The molecular weight of Al2O3 in g/mol -/
def molecular_weight_Al2O3 : ℝ := 
  num_Al_atoms * atomic_weight_Al + num_O_atoms * atomic_weight_O

theorem molecular_weight_8_moles_Al2O3 : 
  num_moles * molecular_weight_Al2O3 = 815.68 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_8_moles_Al2O3_l2428_242872


namespace NUMINAMATH_CALUDE_point_circle_relationship_l2428_242820

theorem point_circle_relationship :
  ∀ θ : ℝ,
  let P : ℝ × ℝ := (5 * Real.cos θ, 4 * Real.sin θ)
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 25}
  P ∈ C ∨ (P.1^2 + P.2^2 < 25) :=
by sorry

end NUMINAMATH_CALUDE_point_circle_relationship_l2428_242820


namespace NUMINAMATH_CALUDE_circle_center_point_is_center_l2428_242893

/-- The center of a circle given by the equation x^2 - 6x + y^2 + 8y - 16 = 0 is (3, -4) -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 6*x + y^2 + 8*y - 16 = 0) ↔ ((x - 3)^2 + (y + 4)^2 = 9) :=
by sorry

/-- The point (3, -4) is the center of the circle -/
theorem point_is_center : 
  ∃ (r : ℝ), ∀ (x y : ℝ), x^2 - 6*x + y^2 + 8*y - 16 = 0 ↔ (x - 3)^2 + (y + 4)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_point_is_center_l2428_242893


namespace NUMINAMATH_CALUDE_system_solution_correct_l2428_242871

theorem system_solution_correct (x y : ℝ) : 
  x = 2 ∧ y = 0 → (x - 2*y = 2 ∧ 2*x + y = 4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_correct_l2428_242871


namespace NUMINAMATH_CALUDE_expression_value_l2428_242858

theorem expression_value : 
  let a : ℕ := 2017
  let b : ℕ := 2016
  let c : ℕ := 2015
  ((a^2 + b^2)^2 - c^2 - 4 * a^2 * b^2) / (a^2 + c - b^2) = 2018 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2428_242858


namespace NUMINAMATH_CALUDE_painting_cost_conversion_l2428_242883

/-- Given exchange rates and the cost of a painting in Namibian dollars, 
    prove its cost in Euros -/
theorem painting_cost_conversion 
  (usd_to_nam : ℝ) 
  (usd_to_eur : ℝ) 
  (painting_cost_nam : ℝ) 
  (h1 : usd_to_nam = 7) 
  (h2 : usd_to_eur = 0.9) 
  (h3 : painting_cost_nam = 140) : 
  painting_cost_nam / usd_to_nam * usd_to_eur = 18 := by
  sorry

end NUMINAMATH_CALUDE_painting_cost_conversion_l2428_242883


namespace NUMINAMATH_CALUDE_regular_price_is_100_l2428_242892

/-- The regular price of one bag -/
def regular_price : ℝ := 100

/-- The promotional price of the fourth bag -/
def fourth_bag_price : ℝ := 5

/-- The total cost for four bags -/
def total_cost : ℝ := 305

/-- Theorem stating that the regular price of one bag is $100 -/
theorem regular_price_is_100 :
  3 * regular_price + fourth_bag_price = total_cost :=
by sorry

end NUMINAMATH_CALUDE_regular_price_is_100_l2428_242892


namespace NUMINAMATH_CALUDE_cattle_breeder_milk_production_l2428_242843

/-- Calculates the weekly milk production for a given number of cows and daily milk production per cow. -/
def weekly_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) : ℕ :=
  num_cows * milk_per_cow_per_day * 7

/-- Proves that 52 cows producing 1000 oz of milk per day will produce 364,000 oz of milk per week. -/
theorem cattle_breeder_milk_production :
  weekly_milk_production 52 1000 = 364000 := by
  sorry

#eval weekly_milk_production 52 1000

end NUMINAMATH_CALUDE_cattle_breeder_milk_production_l2428_242843


namespace NUMINAMATH_CALUDE_inequality_solution_l2428_242856

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 0}
  else if 0 < a ∧ a < 1 then {x | (1 - Real.sqrt (1 - a^2)) / a < x ∧ x < (1 + Real.sqrt (1 - a^2)) / a}
  else if a ≥ 1 then ∅
  else if -1 < a ∧ a < 0 then {x | x > (1 - Real.sqrt (1 - a^2)) / a ∨ x < (1 + Real.sqrt (1 - a^2)) / a}
  else if a = -1 then {x | x ≠ 1 / a}
  else Set.univ

theorem inequality_solution (a : ℝ) :
  {x : ℝ | a * x^2 - 2 * x + a < 0} = solution_set a := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2428_242856


namespace NUMINAMATH_CALUDE_inverse_composition_l2428_242815

-- Define the function f
def f : ℕ → ℕ
| 2 => 8
| 3 => 15
| 4 => 24
| 5 => 35
| 6 => 48
| _ => 0  -- For other inputs, we'll define it as 0

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 8 => 2
| 15 => 3
| 24 => 4
| 35 => 5
| 48 => 6
| _ => 0  -- For other inputs, we'll define it as 0

-- State the theorem
theorem inverse_composition :
  f_inv (f_inv 48 * f_inv 8 - f_inv 24) = 2 :=
by sorry

end NUMINAMATH_CALUDE_inverse_composition_l2428_242815


namespace NUMINAMATH_CALUDE_petya_vasya_game_l2428_242885

theorem petya_vasya_game (k : ℚ) : ∃ (a b c : ℚ), 
  ∃ (x y : ℂ), x ≠ y ∧ 
  (x^3 + a*x^2 + b*x + c = 0) ∧ 
  (y^3 + a*y^2 + b*y + c = 0) ∧ 
  (x - y = 2014 ∨ y - x = 2014) :=
sorry

end NUMINAMATH_CALUDE_petya_vasya_game_l2428_242885


namespace NUMINAMATH_CALUDE_adults_average_age_is_22_l2428_242847

/-- Represents the programming bootcamp group -/
structure BootcampGroup where
  totalMembers : ℕ
  averageAge : ℕ
  girlsCount : ℕ
  boysCount : ℕ
  adultsCount : ℕ
  girlsAverageAge : ℕ
  boysAverageAge : ℕ

/-- Calculates the average age of adults in the bootcamp group -/
def adultsAverageAge (group : BootcampGroup) : ℕ :=
  ((group.totalMembers * group.averageAge) - 
   (group.girlsCount * group.girlsAverageAge) - 
   (group.boysCount * group.boysAverageAge)) / group.adultsCount

/-- Theorem stating that the average age of adults is 22 years -/
theorem adults_average_age_is_22 (group : BootcampGroup) 
  (h1 : group.totalMembers = 50)
  (h2 : group.averageAge = 20)
  (h3 : group.girlsCount = 25)
  (h4 : group.boysCount = 20)
  (h5 : group.adultsCount = 5)
  (h6 : group.girlsAverageAge = 18)
  (h7 : group.boysAverageAge = 22) :
  adultsAverageAge group = 22 := by
  sorry


end NUMINAMATH_CALUDE_adults_average_age_is_22_l2428_242847


namespace NUMINAMATH_CALUDE_divisibility_by_36_l2428_242898

theorem divisibility_by_36 (n : ℤ) (h1 : n ≥ 5) (h2 : ¬ 2 ∣ n) (h3 : ¬ 3 ∣ n) : 
  36 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_36_l2428_242898


namespace NUMINAMATH_CALUDE_circle_equation_l2428_242822

/-- A circle passing through points A(0, -6) and B(1, -5) with center C on the line x-y+1=0 
    has the standard equation (x + 3)^2 + (y + 2)^2 = 25 -/
theorem circle_equation (C : ℝ × ℝ) : 
  (C.1 - C.2 + 1 = 0) →  -- C lies on the line x-y+1=0
  ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 = ((1 : ℝ) - C.1)^2 + ((-5 : ℝ) - C.2)^2 →  -- C is equidistant from A and B
  ∀ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ↔ (x - C.1)^2 + (y - C.2)^2 = ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2428_242822


namespace NUMINAMATH_CALUDE_deepak_age_l2428_242839

theorem deepak_age (arun_age deepak_age : ℕ) : 
  arun_age / deepak_age = 2 / 3 →
  arun_age + 5 = 25 →
  deepak_age = 30 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2428_242839


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2428_242865

theorem fraction_evaluation (a b : ℝ) (ha : a = 7) (hb : b = 3) :
  5 / (a + b) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2428_242865


namespace NUMINAMATH_CALUDE_jose_profit_share_l2428_242850

/-- Calculates the share of profit for an investor given the total profit and investments --/
def calculate_profit_share (total_profit : ℚ) (investment1 : ℚ) (months1 : ℕ) (investment2 : ℚ) (months2 : ℕ) : ℚ :=
  let total_investment := investment1 * months1 + investment2 * months2
  let share_ratio := (investment2 * months2) / total_investment
  share_ratio * total_profit

/-- Proves that Jose's share of the profit is 3500 given the problem conditions --/
theorem jose_profit_share :
  let tom_investment : ℚ := 3000
  let jose_investment : ℚ := 4500
  let tom_months : ℕ := 12
  let jose_months : ℕ := 10
  let total_profit : ℚ := 6300
  calculate_profit_share total_profit tom_investment tom_months jose_investment jose_months = 3500 := by
  sorry


end NUMINAMATH_CALUDE_jose_profit_share_l2428_242850


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2428_242866

theorem cube_volume_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_area_ratio : a^2 / b^2 = 9 / 25) : 
  b^3 / a^3 = 125 / 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2428_242866


namespace NUMINAMATH_CALUDE_tri_divisible_iff_l2428_242884

/-- A polynomial is tri-divisible if 3 divides f(k) for any integer k -/
def TriDivisible (f : Polynomial ℤ) : Prop :=
  ∀ k : ℤ, (3 : ℤ) ∣ (f.eval k)

/-- The necessary and sufficient condition for a polynomial to be tri-divisible -/
theorem tri_divisible_iff (f : Polynomial ℤ) :
  TriDivisible f ↔ ∃ (Q : Polynomial ℤ) (a b c : ℤ),
    f = (X - 1) * (X - 2) * X * Q + 3 * (a * X^2 + b * X + c) :=
sorry

end NUMINAMATH_CALUDE_tri_divisible_iff_l2428_242884


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l2428_242881

theorem rectangle_area_theorem : ∃ (x y : ℝ), 
  (x + 3) * (y - 1) = x * y ∧ 
  (x - 3) * (y + 2) = x * y ∧ 
  x * y = 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l2428_242881


namespace NUMINAMATH_CALUDE_employee_count_proof_l2428_242886

/-- The number of employees in the room -/
def num_employees : ℕ := 25000

/-- The initial percentage of managers as a rational number -/
def initial_manager_percentage : ℚ := 99 / 100

/-- The final percentage of managers as a rational number -/
def final_manager_percentage : ℚ := 98 / 100

/-- The number of managers that leave the room -/
def managers_leaving : ℕ := 250

theorem employee_count_proof :
  (initial_manager_percentage * num_employees : ℚ) - managers_leaving = 
  final_manager_percentage * num_employees :=
by sorry

end NUMINAMATH_CALUDE_employee_count_proof_l2428_242886


namespace NUMINAMATH_CALUDE_only_hexagonal_prism_no_circular_cross_section_l2428_242838

-- Define the types of geometric shapes
inductive GeometricShape
  | Sphere
  | Cone
  | Cylinder
  | HexagonalPrism

-- Define a property for shapes that can have circular cross-sections
def has_circular_cross_section (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => true
  | GeometricShape.Cone => true
  | GeometricShape.Cylinder => true
  | GeometricShape.HexagonalPrism => false

-- Theorem statement
theorem only_hexagonal_prism_no_circular_cross_section :
  ∀ (shape : GeometricShape),
    ¬(has_circular_cross_section shape) ↔ shape = GeometricShape.HexagonalPrism :=
by
  sorry

end NUMINAMATH_CALUDE_only_hexagonal_prism_no_circular_cross_section_l2428_242838


namespace NUMINAMATH_CALUDE_seat_3_9_description_l2428_242844

/-- Represents a seat in a movie theater -/
structure TheaterSeat where
  row : ℕ
  seat : ℕ

/-- Interprets a pair of natural numbers as a theater seat -/
def interpretSeat (p : ℕ × ℕ) : TheaterSeat :=
  { row := p.1, seat := p.2 }

/-- Describes a theater seat as a string -/
def describeSeat (s : TheaterSeat) : String :=
  s.row.repr ++ "th row, " ++ s.seat.repr ++ "th seat"

theorem seat_3_9_description :
  describeSeat (interpretSeat (3, 9)) = "3rd row, 9th seat" :=
sorry

end NUMINAMATH_CALUDE_seat_3_9_description_l2428_242844


namespace NUMINAMATH_CALUDE_square_distance_l2428_242828

theorem square_distance (small_perimeter : ℝ) (large_area : ℝ) :
  small_perimeter = 8 →
  large_area = 36 →
  let small_side := small_perimeter / 4
  let large_side := Real.sqrt large_area
  let leg1 := large_side
  let leg2 := large_side - 2 * small_side
  Real.sqrt (leg1^2 + leg2^2) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_square_distance_l2428_242828


namespace NUMINAMATH_CALUDE_ordered_pairs_count_l2428_242855

theorem ordered_pairs_count : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 32) (Finset.product (Finset.range 33) (Finset.range 33))).card ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l2428_242855


namespace NUMINAMATH_CALUDE_johns_computer_cost_l2428_242848

/-- The total cost of John's computer setup -/
def total_cost (computer_cost peripherals_cost original_video_card_cost upgraded_video_card_cost : ℝ) : ℝ :=
  computer_cost + peripherals_cost + (upgraded_video_card_cost - original_video_card_cost)

/-- Theorem stating the total cost of John's computer setup -/
theorem johns_computer_cost :
  let computer_cost : ℝ := 1500
  let peripherals_cost : ℝ := computer_cost / 5
  let original_video_card_cost : ℝ := 300
  let upgraded_video_card_cost : ℝ := 2 * original_video_card_cost
  total_cost computer_cost peripherals_cost original_video_card_cost upgraded_video_card_cost = 2100 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_computer_cost_l2428_242848


namespace NUMINAMATH_CALUDE_largest_n_for_equation_solution_exists_for_two_l2428_242854

theorem largest_n_for_equation : 
  ∀ n : ℕ+, n > 2 → 
  ¬∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12 :=
by sorry

theorem solution_exists_for_two :
  ∃ x y z : ℕ+, 2^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_solution_exists_for_two_l2428_242854


namespace NUMINAMATH_CALUDE_min_value_fraction_l2428_242801

theorem min_value_fraction (x : ℝ) (h : x > -1) : 
  x^2 / (x + 1) ≥ 0 ∧ ∃ y > -1, y^2 / (y + 1) = 0 := by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2428_242801


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2428_242826

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age : ℝ) 
  (group1_size group2_size group3_size : Nat) 
  (group1_avg group2_avg group3_avg : ℝ) :
  total_students = 15 →
  avg_age = 15 →
  group1_size = 5 →
  group2_size = 6 →
  group3_size = 3 →
  group1_avg = 13 →
  group2_avg = 15 →
  group3_avg = 17 →
  ∃ (fifteenth_student_age : ℝ),
    fifteenth_student_age = 19 ∧
    (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg + fifteenth_student_age) / total_students = avg_age :=
by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l2428_242826


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l2428_242806

/-- Given a point in rectangular coordinates (3, -8, 6) with corresponding
    spherical coordinates (ρ, θ, φ), this theorem proves the rectangular
    coordinates of the point with spherical coordinates (ρ, θ + π/4, -φ). -/
theorem spherical_coordinate_transformation (ρ θ φ : ℝ) :
  3 = ρ * Real.sin φ * Real.cos θ →
  -8 = ρ * Real.sin φ * Real.sin θ →
  6 = ρ * Real.cos φ →
  ∃ (x y : ℝ),
    x = -ρ * Real.sin φ * (Real.sqrt 2 / 2 * Real.cos θ - Real.sqrt 2 / 2 * Real.sin θ) ∧
    y = -ρ * Real.sin φ * (Real.sqrt 2 / 2 * Real.sin θ + Real.sqrt 2 / 2 * Real.cos θ) ∧
    6 = ρ * Real.cos φ :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l2428_242806


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l2428_242864

theorem greatest_integer_inequality (y : ℤ) : 
  (5 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l2428_242864


namespace NUMINAMATH_CALUDE_sqrt_difference_comparison_l2428_242859

theorem sqrt_difference_comparison (m : ℝ) (hm : m > 1) :
  Real.sqrt m - Real.sqrt (m - 1) > Real.sqrt (m + 1) - Real.sqrt m := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_comparison_l2428_242859


namespace NUMINAMATH_CALUDE_max_min_difference_r_l2428_242862

theorem max_min_difference_r (p q r : ℝ) 
  (sum_condition : p + q + r = 5)
  (sum_squares_condition : p^2 + q^2 + r^2 = 27) :
  ∃ (r_max r_min : ℝ),
    (∀ r' : ℝ, p + q + r' = 5 ∧ p^2 + q^2 + r'^2 = 27 → r' ≤ r_max) ∧
    (∀ r' : ℝ, p + q + r' = 5 ∧ p^2 + q^2 + r'^2 = 27 → r' ≥ r_min) ∧
    r_max - r_min = 8 * Real.sqrt 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_r_l2428_242862


namespace NUMINAMATH_CALUDE_max_M_is_five_l2428_242827

/-- Definition of I_k -/
def I (k : ℕ) : ℕ := 10^(k+2) + 25

/-- Definition of M(k) -/
def M (k : ℕ) : ℕ := (I k).factors.count 2

/-- Theorem: The maximum value of M(k) for k > 0 is 5 -/
theorem max_M_is_five : ∃ (k : ℕ), k > 0 ∧ M k = 5 ∧ ∀ (j : ℕ), j > 0 → M j ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_M_is_five_l2428_242827


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2428_242818

theorem hyperbola_eccentricity (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0)
  (h_equilateral : b / a = Real.sqrt 3 / 3) :
  let e := Real.sqrt (1 + b^2 / a^2)
  e = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2428_242818


namespace NUMINAMATH_CALUDE_election_votes_calculation_l2428_242877

theorem election_votes_calculation (total_votes : ℕ) : 
  (total_votes : ℚ) * (55 / 100) - (total_votes : ℚ) * (30 / 100) = 174 →
  total_votes = 696 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l2428_242877


namespace NUMINAMATH_CALUDE_complex_division_result_l2428_242812

theorem complex_division_result : 
  let i := Complex.I
  (3 + i) / (1 + i) = 2 - i := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l2428_242812


namespace NUMINAMATH_CALUDE_green_area_growth_rate_l2428_242867

theorem green_area_growth_rate :
  ∀ x : ℝ, (1 + x)^2 = 1.44 → x = 0.2 :=
by
  sorry

end NUMINAMATH_CALUDE_green_area_growth_rate_l2428_242867


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_value_l2428_242873

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qinJiushao (f : ℝ → ℝ) (x : ℝ) : ℕ → ℝ
  | 0 => x + 3
  | 1 => qinJiushao f x 0 * x - 1
  | 2 => qinJiushao f x 1 * x
  | 3 => qinJiushao f x 2 * x + 2
  | 4 => qinJiushao f x 3 * x - 1
  | _ => 0

/-- The polynomial f(x) = x^5 + 3x^4 - x^3 + 2x - 1 -/
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - x^3 + 2*x - 1

theorem qin_jiushao_v3_value :
  qinJiushao f 2 2 = 18 := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_value_l2428_242873


namespace NUMINAMATH_CALUDE_alpha_values_l2428_242833

/-- Given a function f where f(α) = 4, prove that α is either -4 or 2 -/
theorem alpha_values (f : ℝ → ℝ) (α : ℝ) (h : f α = 4) : α = -4 ∨ α = 2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_values_l2428_242833


namespace NUMINAMATH_CALUDE_nine_rings_five_classes_l2428_242889

/-- Represents the number of classes in a school day based on bell rings --/
def number_of_classes (total_rings : ℕ) : ℕ :=
  let completed_classes := (total_rings - 1) / 2
  completed_classes + 1

/-- Theorem stating that 9 total bell rings corresponds to 5 classes --/
theorem nine_rings_five_classes : number_of_classes 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_nine_rings_five_classes_l2428_242889


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2428_242894

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2428_242894


namespace NUMINAMATH_CALUDE_first_character_lines_l2428_242869

/-- The number of lines for each character in Jerry's skit script --/
structure ScriptLines where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of Jerry's skit script --/
def script_conditions (lines : ScriptLines) : Prop :=
  lines.first = lines.second + 8 ∧
  lines.third = 2 ∧
  lines.second = 6 + 3 * lines.third

/-- Theorem stating that the first character has 20 lines --/
theorem first_character_lines (lines : ScriptLines) 
  (h : script_conditions lines) : lines.first = 20 := by
  sorry

#check first_character_lines

end NUMINAMATH_CALUDE_first_character_lines_l2428_242869


namespace NUMINAMATH_CALUDE_airline_capacity_is_2482_l2428_242875

/-- Calculates the number of passengers an airline can accommodate daily --/
def airline_capacity (small_planes medium_planes large_planes : ℕ)
  (small_rows small_seats small_flights small_occupancy : ℕ)
  (medium_rows medium_seats medium_flights medium_occupancy : ℕ)
  (large_rows large_seats large_flights large_occupancy : ℕ) : ℕ :=
  let small_capacity := small_planes * small_rows * small_seats * small_flights * small_occupancy / 100
  let medium_capacity := medium_planes * medium_rows * medium_seats * medium_flights * medium_occupancy / 100
  let large_capacity := large_planes * large_rows * large_seats * large_flights * large_occupancy / 100
  small_capacity + medium_capacity + large_capacity

/-- The airline's daily passenger capacity is 2482 --/
theorem airline_capacity_is_2482 :
  airline_capacity 2 2 1 15 6 3 80 25 8 2 90 35 10 4 95 = 2482 := by
  sorry

end NUMINAMATH_CALUDE_airline_capacity_is_2482_l2428_242875


namespace NUMINAMATH_CALUDE_elective_schemes_count_l2428_242897

/-- The number of courses offered by the school -/
def total_courses : ℕ := 10

/-- The number of conflicting courses (A, B, C) -/
def conflicting_courses : ℕ := 3

/-- The number of courses each student must elect -/
def courses_to_elect : ℕ := 3

/-- The number of different elective schemes available for a student -/
def elective_schemes : ℕ := Nat.choose (total_courses - conflicting_courses) courses_to_elect +
                             conflicting_courses * Nat.choose (total_courses - conflicting_courses) (courses_to_elect - 1)

theorem elective_schemes_count :
  elective_schemes = 98 :=
sorry

end NUMINAMATH_CALUDE_elective_schemes_count_l2428_242897


namespace NUMINAMATH_CALUDE_empty_chests_count_l2428_242888

/-- Represents a nested chest system -/
structure ChestSystem where
  total_chests : ℕ
  non_empty_chests : ℕ
  hNonEmpty : non_empty_chests = 2006
  hTotal : total_chests = 10 * non_empty_chests + 1

/-- The number of empty chests in the system -/
def empty_chests (cs : ChestSystem) : ℕ :=
  cs.total_chests - (cs.non_empty_chests + 1)

/-- Theorem stating the number of empty chests in the given system -/
theorem empty_chests_count (cs : ChestSystem) : empty_chests cs = 18054 := by
  sorry

end NUMINAMATH_CALUDE_empty_chests_count_l2428_242888


namespace NUMINAMATH_CALUDE_anthony_pencils_l2428_242800

theorem anthony_pencils (initial final added : ℕ) 
  (h1 : added = 56)
  (h2 : final = 65)
  (h3 : final = initial + added) :
  initial = 9 := by
sorry

end NUMINAMATH_CALUDE_anthony_pencils_l2428_242800


namespace NUMINAMATH_CALUDE_vitya_older_probability_l2428_242841

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The probability that Vitya is at least one day older than Masha -/
def probability_vitya_older (june_days : ℕ) : ℚ :=
  (june_days * (june_days - 1) / 2) / (june_days * june_days)

theorem vitya_older_probability :
  probability_vitya_older june_days = 29 / 60 := by
  sorry

end NUMINAMATH_CALUDE_vitya_older_probability_l2428_242841


namespace NUMINAMATH_CALUDE_sisters_sandcastle_height_l2428_242825

/-- Given the height of Miki's sandcastle and the difference in height between
    the two sandcastles, calculate the height of her sister's sandcastle. -/
theorem sisters_sandcastle_height
  (miki_height : ℝ)
  (height_difference : ℝ)
  (h1 : miki_height = 0.8333333333333334)
  (h2 : height_difference = 0.3333333333333333) :
  miki_height - height_difference = 0.5 := by
sorry

end NUMINAMATH_CALUDE_sisters_sandcastle_height_l2428_242825


namespace NUMINAMATH_CALUDE_self_inverse_matrix_l2428_242842

/-- A 2x2 matrix is its own inverse if and only if p = 15/2 and q = -4 -/
theorem self_inverse_matrix (p q : ℚ) :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, p; -2, q]
  (A * A = 1) ↔ p = 15/2 ∧ q = -4 := by
  sorry

end NUMINAMATH_CALUDE_self_inverse_matrix_l2428_242842


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2428_242817

theorem polynomial_factorization (a b c d : ℤ) :
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + x^3 - 5*x^2 + x - 6) →
  a + b + c + d = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2428_242817


namespace NUMINAMATH_CALUDE_negative_sqrt_ten_less_than_negative_three_l2428_242849

theorem negative_sqrt_ten_less_than_negative_three :
  -Real.sqrt 10 < -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_ten_less_than_negative_three_l2428_242849


namespace NUMINAMATH_CALUDE_min_people_for_hundred_chairs_is_minimum_people_l2428_242821

/-- The number of chairs in the circle -/
def num_chairs : ℕ := 100

/-- A function that calculates the minimum number of people needed -/
def min_people (chairs : ℕ) : ℕ :=
  (chairs + 2) / 3

/-- The theorem stating the minimum number of people for 100 chairs -/
theorem min_people_for_hundred_chairs :
  min_people num_chairs = 34 := by
  sorry

/-- The theorem proving that this is indeed the minimum -/
theorem is_minimum_people (n : ℕ) :
  n < min_people num_chairs →
  ∃ (m : ℕ), m > 2 ∧ m < num_chairs ∧
  ∀ (i : ℕ), i < n → ∃ (j : ℕ), j < n ∧ j ≠ i ∧
  (m + i) % num_chairs = (m + j) % num_chairs := by
  sorry

end NUMINAMATH_CALUDE_min_people_for_hundred_chairs_is_minimum_people_l2428_242821


namespace NUMINAMATH_CALUDE_unique_triple_solution_l2428_242868

theorem unique_triple_solution : 
  ∀ (a b p : ℕ+), 
    Nat.Prime p.val → 
    (a.val + b.val : ℕ) ^ p.val = p.val ^ a.val + p.val ^ b.val → 
    a = 1 ∧ b = 1 ∧ p = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l2428_242868


namespace NUMINAMATH_CALUDE_am_gm_difference_bound_l2428_242845

theorem am_gm_difference_bound (a : ℝ) (h : 0 < a) :
  let b := a + 1
  let am := (a + b) / 2
  let gm := Real.sqrt (a * b)
  am - gm < (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_am_gm_difference_bound_l2428_242845


namespace NUMINAMATH_CALUDE_birth_year_problem_l2428_242860

theorem birth_year_problem (x : ℕ) : 
  (1850 ≤ x^2 - 2*x + 1) ∧ (x^2 - 2*x + 1 < 1900) →
  (x^2 - x + 1 - (x^2 - 2*x + 1) = x) →
  x^2 - 2*x + 1 = 1849 := by
sorry

end NUMINAMATH_CALUDE_birth_year_problem_l2428_242860


namespace NUMINAMATH_CALUDE_doritos_distribution_l2428_242805

theorem doritos_distribution (total_bags : ℕ) (doritos_fraction : ℚ) (num_piles : ℕ) : 
  total_bags = 80 →
  doritos_fraction = 1/4 →
  num_piles = 4 →
  (total_bags : ℚ) * doritos_fraction / num_piles = 5 := by
  sorry

end NUMINAMATH_CALUDE_doritos_distribution_l2428_242805


namespace NUMINAMATH_CALUDE_max_blocks_fit_l2428_242802

/-- Represents the dimensions of a rectangular box or block -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box or block given its dimensions -/
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the box and block dimensions -/
def box : Dimensions := ⟨4, 3, 2⟩
def block : Dimensions := ⟨1, 1, 2⟩

/-- Calculates the maximum number of blocks that can fit in the box based on volume -/
def max_blocks_by_volume : ℕ :=
  volume box / volume block

/-- Calculates the maximum number of blocks that can fit in the box based on physical arrangement -/
def max_blocks_by_arrangement : ℕ :=
  (box.length / block.length) * (box.width / block.width)

theorem max_blocks_fit :
  max_blocks_by_volume = 12 ∧ max_blocks_by_arrangement = 12 :=
sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l2428_242802


namespace NUMINAMATH_CALUDE_hearts_then_king_probability_l2428_242896

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)
  (suits : ∀ c ∈ cards, c.1 ∈ Finset.range 4)
  (ranks : ∀ c ∈ cards, c.2 ∈ Finset.range 13)

/-- The probability of drawing a specific sequence of cards from a shuffled deck -/
def draw_probability (d : Deck) (seq : List (Nat × Nat)) : ℚ :=
  sorry

/-- Hearts suit is represented by 0 -/
def hearts : Nat := 0

/-- King rank is represented by 12 (0-indexed) -/
def king : Nat := 12

theorem hearts_then_king_probability :
  ∀ d : Deck, 
    draw_probability d [(hearts, 0), (hearts, 1), (hearts, 2), (hearts, 3), (0, king)] = 286 / 124900 := by
  sorry

end NUMINAMATH_CALUDE_hearts_then_king_probability_l2428_242896


namespace NUMINAMATH_CALUDE_quadratic_solution_symmetry_l2428_242814

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_solution_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f := QuadraticFunction a b c
  f (-5) = f 1 → f 2 = 0 → ∃ n : ℝ, f 3 = n ∧ (∀ x : ℝ, f x = n ↔ x = 3 ∨ x = -7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_symmetry_l2428_242814


namespace NUMINAMATH_CALUDE_range_of_a_for_non_negative_f_l2428_242829

/-- The range of a for which f(x) = x³ - x² - 2a has a non-negative value in (-∞, a] -/
theorem range_of_a_for_non_negative_f (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ≤ a ∧ x₀^3 - x₀^2 - 2*a ≥ 0) ↔ a ∈ Set.Icc (-1) 0 ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_non_negative_f_l2428_242829


namespace NUMINAMATH_CALUDE_no_solution_exists_l2428_242863

open Function Set

-- Define the property that a function must satisfy
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = f x - y

-- State the theorem
theorem no_solution_exists :
  ¬ ∃ f : ℝ → ℝ, Continuous f ∧ SatisfiesFunctionalEquation f :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2428_242863


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l2428_242891

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.35 * x + 245 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l2428_242891


namespace NUMINAMATH_CALUDE_solve_quadratic_coefficients_l2428_242830

-- Define the universal set U
def U : Set ℤ := {2, 3, 5}

-- Define the set A
def A (b c : ℤ) : Set ℤ := {x ∈ U | x^2 + b*x + c = 0}

-- Define the theorem
theorem solve_quadratic_coefficients :
  ∀ b c : ℤ, (U \ A b c = {2}) → (b = -8 ∧ c = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_coefficients_l2428_242830


namespace NUMINAMATH_CALUDE_simple_interest_time_proof_l2428_242874

/-- The simple interest rate per annum -/
def simple_interest_rate : ℚ := 8 / 100

/-- The principal amount for simple interest -/
def simple_principal : ℚ := 1750.000000000002

/-- The principal amount for compound interest -/
def compound_principal : ℚ := 4000

/-- The compound interest rate per annum -/
def compound_interest_rate : ℚ := 10 / 100

/-- The time period for compound interest in years -/
def compound_time : ℕ := 2

/-- Function to calculate compound interest -/
def compound_interest (p : ℚ) (r : ℚ) (t : ℕ) : ℚ :=
  p * ((1 + r) ^ t - 1)

/-- The time period for simple interest in years -/
def simple_time : ℕ := 3

theorem simple_interest_time_proof :
  simple_principal * simple_interest_rate * simple_time =
  (1 / 2) * compound_interest compound_principal compound_interest_rate compound_time :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_time_proof_l2428_242874


namespace NUMINAMATH_CALUDE_table_arrangement_l2428_242851

theorem table_arrangement (total_tables : Nat) (num_rows : Nat) 
  (tables_per_row : Nat) (leftover : Nat) : 
  total_tables = 74 → num_rows = 8 → 
  tables_per_row = total_tables / num_rows →
  leftover = total_tables % num_rows →
  tables_per_row = 9 ∧ leftover = 2 := by
  sorry

end NUMINAMATH_CALUDE_table_arrangement_l2428_242851


namespace NUMINAMATH_CALUDE_henry_tic_tac_toe_wins_l2428_242879

theorem henry_tic_tac_toe_wins 
  (total_games : ℕ) 
  (losses : ℕ) 
  (draws : ℕ) 
  (h1 : total_games = 14) 
  (h2 : losses = 2) 
  (h3 : draws = 10) : 
  total_games - losses - draws = 2 := by
  sorry

end NUMINAMATH_CALUDE_henry_tic_tac_toe_wins_l2428_242879


namespace NUMINAMATH_CALUDE_single_intersection_l2428_242861

/-- The parabola function -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 2 * y + 4

/-- The line function -/
def line (k : ℝ) : ℝ := k

/-- Theorem stating the condition for single intersection -/
theorem single_intersection (k : ℝ) : 
  (∃! y, parabola y = line k) ↔ k = 13/3 := by sorry

end NUMINAMATH_CALUDE_single_intersection_l2428_242861


namespace NUMINAMATH_CALUDE_tanya_work_days_l2428_242857

/-- Given Sakshi can do a piece of work in 20 days and Tanya is 25% more efficient than Sakshi,
    prove that Tanya will take 16 days to do the same piece of work. -/
theorem tanya_work_days (sakshi_days : ℕ) (tanya_efficiency : ℚ) :
  sakshi_days = 20 →
  tanya_efficiency = 125 / 100 →
  (sakshi_days : ℚ) / tanya_efficiency = 16 := by
  sorry

end NUMINAMATH_CALUDE_tanya_work_days_l2428_242857


namespace NUMINAMATH_CALUDE_f_value_at_2_f_equals_f_horner_f_2_equals_62_l2428_242804

/-- The polynomial function f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2*x^4 + 3*x^3 + 5*x - 4

/-- Horner's method representation of f(x) -/
def f_horner (x : ℝ) : ℝ := x*(x*(x*(2*x + 3)) + 5) - 4

theorem f_value_at_2 : f 2 = 62 := by sorry

theorem f_equals_f_horner : ∀ x, f x = f_horner x := by sorry

theorem f_2_equals_62 : f_horner 2 = 62 := by sorry

end NUMINAMATH_CALUDE_f_value_at_2_f_equals_f_horner_f_2_equals_62_l2428_242804


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l2428_242840

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, y = a * x - 2 ∨ y = (a + 2) * x + 1) →
  (a * (a + 2) = -1) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l2428_242840


namespace NUMINAMATH_CALUDE_seventh_term_is_10_4_l2428_242819

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℝ
  -- Common difference of the sequence
  d : ℝ
  -- Sum of first four terms is 20
  sum_first_four : a + (a + d) + (a + 2*d) + (a + 3*d) = 20
  -- Fifth term is 8
  fifth_term : a + 4*d = 8

/-- The seventh term of the arithmetic sequence is 10.4 -/
theorem seventh_term_is_10_4 (seq : ArithmeticSequence) : 
  seq.a + 6*seq.d = 10.4 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_10_4_l2428_242819


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l2428_242870

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l2428_242870


namespace NUMINAMATH_CALUDE_expression_evaluation_l2428_242890

theorem expression_evaluation :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * Real.sqrt (3^2 + 1^2) = 3280 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2428_242890


namespace NUMINAMATH_CALUDE_circle_properties_l2428_242834

/-- A circle with center on the line y = -4x and tangent to x + y - 1 = 0 at (3, -2) -/
def special_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 4)^2 = 8

/-- The line y = -4x -/
def center_line (x y : ℝ) : Prop := y = -4 * x

/-- The line x + y - 1 = 0 -/
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The point P(3, -2) -/
def point_P : ℝ × ℝ := (3, -2)

theorem circle_properties :
  ∃ (cx cy : ℝ),
    center_line cx cy ∧
    special_circle cx cy ∧
    tangent_line (point_P.1) (point_P.2) ∧
    (∀ (x y : ℝ), tangent_line x y → ((x - cx)^2 + (y - cy)^2 ≥ 8)) ∧
    ((point_P.1 - cx)^2 + (point_P.2 - cy)^2 = 8) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l2428_242834


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l2428_242835

theorem sum_of_x_and_y_equals_two (x y : ℝ) 
  (eq1 : 2 * x + 3 * y = 6)
  (eq2 : 3 * x + 2 * y = 4) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l2428_242835


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2428_242878

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (9/2) 101 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2428_242878


namespace NUMINAMATH_CALUDE_ten_streets_intersections_l2428_242887

/-- The number of intersections created by n non-parallel streets -/
def intersections (n : ℕ) : ℕ := n.choose 2

/-- The theorem stating that 10 non-parallel streets create 45 intersections -/
theorem ten_streets_intersections : intersections 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_streets_intersections_l2428_242887


namespace NUMINAMATH_CALUDE_factor_theorem_quadratic_l2428_242895

theorem factor_theorem_quadratic (t : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, 4*x^2 - 8*x + 3 = (x - t) * p x) ↔ (t = 1.5 ∨ t = 0.5) :=
by sorry

end NUMINAMATH_CALUDE_factor_theorem_quadratic_l2428_242895


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l2428_242853

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 2 = 0

-- Define the point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the line l (we'll use the point-slope form)
def line_l (x y : ℝ) (m : ℝ) : Prop := y - point_P.2 = m * (x - point_P.1)

-- State the theorem
theorem circle_and_tangent_line :
  ∃ (m : ℝ),
    -- The line l passes through P(1,1) and is tangent to C
    (∀ (x y : ℝ), line_l x y m → (circle_equation x y → x = y)) ∧
    -- The radius of C is √2
    (∃ (c_x c_y : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - c_x)^2 + (y - c_y)^2 = 2) ∧
    -- The equation of l is x - y = 0
    (∀ (x y : ℝ), line_l x y m ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l2428_242853


namespace NUMINAMATH_CALUDE_divisor_problem_l2428_242808

theorem divisor_problem (w : ℤ) (x : ℤ) :
  (∃ k : ℤ, w = 13 * k) →
  (∃ m : ℤ, w + 3 = x * m) →
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l2428_242808


namespace NUMINAMATH_CALUDE_exactly_one_two_defective_mutually_exclusive_at_least_one_defective_all_genuine_mutually_exclusive_mutually_exclusive_pairs_l2428_242807

/-- Represents the number of genuine items in the box -/
def genuine_items : ℕ := 4

/-- Represents the number of defective items in the box -/
def defective_items : ℕ := 3

/-- Represents the number of items randomly selected -/
def selected_items : ℕ := 2

/-- Represents the event "Exactly one defective item" -/
def exactly_one_defective : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Represents the event "Exactly two defective items" -/
def exactly_two_defective : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Represents the event "At least one defective item" -/
def at_least_one_defective : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Represents the event "All are genuine" -/
def all_genuine : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Theorem stating that "Exactly one defective item" and "Exactly two defective items" are mutually exclusive -/
theorem exactly_one_two_defective_mutually_exclusive :
  exactly_one_defective ∩ exactly_two_defective = ∅ := sorry

/-- Theorem stating that "At least one defective item" and "All are genuine" are mutually exclusive -/
theorem at_least_one_defective_all_genuine_mutually_exclusive :
  at_least_one_defective ∩ all_genuine = ∅ := sorry

/-- Main theorem proving that only the specified pairs of events are mutually exclusive -/
theorem mutually_exclusive_pairs :
  (exactly_one_defective ∩ exactly_two_defective = ∅) ∧
  (at_least_one_defective ∩ all_genuine = ∅) ∧
  (exactly_one_defective ∩ at_least_one_defective ≠ ∅) ∧
  (exactly_two_defective ∩ at_least_one_defective ≠ ∅) ∧
  (exactly_one_defective ∩ all_genuine ≠ ∅) ∧
  (exactly_two_defective ∩ all_genuine ≠ ∅) := sorry

end NUMINAMATH_CALUDE_exactly_one_two_defective_mutually_exclusive_at_least_one_defective_all_genuine_mutually_exclusive_mutually_exclusive_pairs_l2428_242807


namespace NUMINAMATH_CALUDE_ball_bearing_savings_l2428_242832

/-- Calculates the savings when buying ball bearings during a sale with bulk discount --/
theorem ball_bearing_savings
  (num_machines : ℕ)
  (bearings_per_machine : ℕ)
  (regular_price : ℚ)
  (sale_price : ℚ)
  (bulk_discount : ℚ)
  (h1 : num_machines = 10)
  (h2 : bearings_per_machine = 30)
  (h3 : regular_price = 1)
  (h4 : sale_price = 3/4)
  (h5 : bulk_discount = 1/5)
  : (num_machines * bearings_per_machine * regular_price) -
    (num_machines * bearings_per_machine * sale_price * (1 - bulk_discount)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ball_bearing_savings_l2428_242832


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l2428_242852

theorem sin_cos_sum_equals_sqrt2_over_2 :
  Real.sin (63 * π / 180) * Real.cos (18 * π / 180) +
  Real.cos (63 * π / 180) * Real.cos (108 * π / 180) =
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l2428_242852


namespace NUMINAMATH_CALUDE_middle_group_frequency_is_32_l2428_242823

/-- Represents a frequency distribution histogram -/
structure Histogram where
  num_rectangles : ℕ
  sample_size : ℕ
  middle_rectangle_area : ℝ
  other_rectangles_area : ℝ

/-- The frequency of the middle group in the histogram -/
def middle_group_frequency (h : Histogram) : ℕ :=
  h.sample_size / 2

/-- Theorem: The frequency of the middle group is 32 under given conditions -/
theorem middle_group_frequency_is_32 (h : Histogram) 
  (h_num_rectangles : h.num_rectangles = 11)
  (h_sample_size : h.sample_size = 160)
  (h_area_equality : h.middle_rectangle_area = h.other_rectangles_area) :
  middle_group_frequency h = 32 := by
  sorry

end NUMINAMATH_CALUDE_middle_group_frequency_is_32_l2428_242823


namespace NUMINAMATH_CALUDE_power_product_equals_five_l2428_242803

theorem power_product_equals_five (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_five_l2428_242803


namespace NUMINAMATH_CALUDE_quadratic_touches_x_axis_l2428_242899

/-- A quadratic function that touches the x-axis -/
def touches_x_axis (a : ℝ) : Prop :=
  ∃ x : ℝ, 2 * x^2 - 8 * x + a = 0 ∧
  ∀ y : ℝ, 2 * y^2 - 8 * y + a ≥ 0

/-- The value of 'a' for which the quadratic function touches the x-axis is 8 -/
theorem quadratic_touches_x_axis :
  ∃! a : ℝ, touches_x_axis a ∧ a = 8 :=
sorry

end NUMINAMATH_CALUDE_quadratic_touches_x_axis_l2428_242899


namespace NUMINAMATH_CALUDE_popsicle_stick_count_l2428_242813

/-- The number of popsicle sticks Steve has -/
def steve_sticks : ℕ := 12

/-- The number of popsicle sticks Sid has -/
def sid_sticks : ℕ := 2 * steve_sticks

/-- The number of popsicle sticks Sam has -/
def sam_sticks : ℕ := 3 * sid_sticks

/-- The total number of popsicle sticks -/
def total_sticks : ℕ := steve_sticks + sid_sticks + sam_sticks

theorem popsicle_stick_count : total_sticks = 108 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_count_l2428_242813


namespace NUMINAMATH_CALUDE_ball_hexagons_l2428_242824

/-- A ball made of hexagons and pentagons -/
structure Ball where
  pentagons : ℕ
  hexagons : ℕ
  pentagon_hexagon_edges : ℕ
  hexagon_pentagon_edges : ℕ

/-- Theorem: A ball with 12 pentagons has 20 hexagons -/
theorem ball_hexagons (b : Ball) 
  (h1 : b.pentagons = 12)
  (h2 : b.pentagon_hexagon_edges = 5)
  (h3 : b.hexagon_pentagon_edges = 3) :
  b.hexagons = 20 := by
  sorry

#check ball_hexagons

end NUMINAMATH_CALUDE_ball_hexagons_l2428_242824


namespace NUMINAMATH_CALUDE_tom_lake_crossing_cost_l2428_242816

/-- The cost of hiring an assistant for crossing a lake back and forth -/
def lake_crossing_cost (one_way_time : ℕ) (hourly_rate : ℕ) : ℕ :=
  2 * one_way_time * hourly_rate

/-- Theorem: The cost for Tom to hire an assistant for crossing the lake back and forth is $80 -/
theorem tom_lake_crossing_cost :
  lake_crossing_cost 4 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_tom_lake_crossing_cost_l2428_242816


namespace NUMINAMATH_CALUDE_negation_of_set_implication_l2428_242810

theorem negation_of_set_implication (A B : Set α) :
  ¬(A ∪ B = A → A ∩ B = B) ↔ (A ∪ B ≠ A → A ∩ B ≠ B) :=
sorry

end NUMINAMATH_CALUDE_negation_of_set_implication_l2428_242810


namespace NUMINAMATH_CALUDE_domain_all_reals_l2428_242811

theorem domain_all_reals (k : ℝ) :
  (∀ x : ℝ, (-7 * x^2 - 4 * x + k ≠ 0)) ↔ k < -4/7 := by sorry

end NUMINAMATH_CALUDE_domain_all_reals_l2428_242811
