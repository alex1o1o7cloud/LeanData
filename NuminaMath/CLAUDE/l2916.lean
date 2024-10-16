import Mathlib

namespace NUMINAMATH_CALUDE_intersection_A_B_l2916_291617

-- Define sets A and B
def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | (x - 1) / (x - 4) < 0}

-- State the theorem
theorem intersection_A_B :
  ∀ x : ℝ, x ∈ A ∩ B ↔ 3 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2916_291617


namespace NUMINAMATH_CALUDE_deepak_present_age_l2916_291627

/-- Represents the ages of two people with a given ratio --/
structure AgeRatio where
  x : ℕ
  rahul_age : ℕ := 4 * x
  deepak_age : ℕ := 3 * x

/-- The theorem stating Deepak's present age given the conditions --/
theorem deepak_present_age (ar : AgeRatio) 
  (h1 : ar.rahul_age + 6 = 50) : 
  ar.deepak_age = 33 := by
  sorry

#check deepak_present_age

end NUMINAMATH_CALUDE_deepak_present_age_l2916_291627


namespace NUMINAMATH_CALUDE_cone_base_area_l2916_291671

-- Define the cone
structure Cone where
  lateral_surface_area : ℝ
  base_radius : ℝ

-- Define the properties of the cone
def is_valid_cone (c : Cone) : Prop :=
  c.lateral_surface_area = 2 * Real.pi ∧
  c.lateral_surface_area = Real.pi * c.base_radius * c.base_radius

-- Theorem statement
theorem cone_base_area (c : Cone) (h : is_valid_cone c) :
  Real.pi * c.base_radius^2 = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_base_area_l2916_291671


namespace NUMINAMATH_CALUDE_wilsons_theorem_and_square_l2916_291687

theorem wilsons_theorem_and_square (p : Nat) (hp : p > 1) :
  (((Nat.factorial (p - 1) + 1) % p = 0) ↔ Nat.Prime p) ∧
  (Nat.Prime p → (Nat.factorial (p - 1))^2 % p = 1) ∧
  (¬Nat.Prime p → (Nat.factorial (p - 1))^2 % p = 0) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_and_square_l2916_291687


namespace NUMINAMATH_CALUDE_thursday_five_times_in_july_l2916_291612

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- A month with its number of days and list of dates -/
structure Month :=
  (numDays : Nat)
  (dates : List Date)

def june : Month := sorry
def july : Month := sorry

/-- Counts the number of occurrences of a specific day in a month -/
def countDayInMonth (m : Month) (d : DayOfWeek) : Nat := sorry

theorem thursday_five_times_in_july 
  (h1 : june.numDays = 30)
  (h2 : july.numDays = 31)
  (h3 : countDayInMonth june DayOfWeek.Tuesday = 5) :
  countDayInMonth july DayOfWeek.Thursday = 5 := by
  sorry

end NUMINAMATH_CALUDE_thursday_five_times_in_july_l2916_291612


namespace NUMINAMATH_CALUDE_tan_C_minus_pi_4_max_area_l2916_291664

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.a * t.b + t.c^2

-- Part I
theorem tan_C_minus_pi_4 (t : Triangle) (h : satisfiesCondition t) :
  Real.tan (t.C - π/4) = 2 - Real.sqrt 3 := by
  sorry

-- Part II
theorem max_area (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.c = Real.sqrt 3) :
  (∀ s : Triangle, satisfiesCondition s → s.c = Real.sqrt 3 →
    t.a * t.b * Real.sin t.C / 2 ≥ s.a * s.b * Real.sin s.C / 2) →
  t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_C_minus_pi_4_max_area_l2916_291664


namespace NUMINAMATH_CALUDE_square_problem_l2916_291669

/-- Square with side length 800 -/
structure Square :=
  (side : ℝ)
  (is_800 : side = 800)

/-- Point on the side of the square -/
structure PointOnSide :=
  (x : ℝ)
  (in_range : 0 ≤ x ∧ x ≤ 800)

/-- Expression of the form p + q√r -/
structure SurdExpression :=
  (p q r : ℕ)
  (r_not_perfect_square : ∀ (n : ℕ), n > 1 → ¬(r.gcd (n^2) > 1))

/-- Main theorem -/
theorem square_problem (S : Square) (E F : PointOnSide) (BF : SurdExpression) :
  S.side = 800 →
  E.x < F.x →
  F.x - E.x = 300 →
  Real.cos (60 * π / 180) * (F.x - 400) = Real.sin (60 * π / 180) * 400 →
  800 - F.x = BF.p + BF.q * Real.sqrt BF.r →
  BF.p + BF.q + BF.r = 334 := by
  sorry

end NUMINAMATH_CALUDE_square_problem_l2916_291669


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2916_291621

def set_A : Set ℝ := { x | (x - 2) / (x + 5) < 0 }

def set_B : Set ℝ := { x | x^2 - 2*x - 3 ≥ 0 }

theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ -5 < x ∧ x ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2916_291621


namespace NUMINAMATH_CALUDE_shyam_weight_increase_l2916_291686

theorem shyam_weight_increase 
  (ram_original : ℝ) 
  (shyam_original : ℝ) 
  (ram_shyam_ratio : ram_original / shyam_original = 4 / 5)
  (ram_increase : ℝ) 
  (ram_increase_percent : ram_increase = 0.1 * ram_original)
  (total_new : ℝ) 
  (total_new_value : total_new = 82.8)
  (total_increase : ℝ) 
  (total_increase_percent : total_increase = 0.15 * (ram_original + shyam_original))
  (total_new_eq : total_new = ram_original + ram_increase + shyam_original + (shyam_original * x))
  : x = 0.19 := by
  sorry

#check shyam_weight_increase

end NUMINAMATH_CALUDE_shyam_weight_increase_l2916_291686


namespace NUMINAMATH_CALUDE_function_property_l2916_291618

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ p q, f (p + q) = f p * f q) 
  (h2 : f 1 = 3) : 
  (f 1^2 + f 2) / f 1 + 
  (f 2^2 + f 4) / f 3 + 
  (f 3^2 + f 6) / f 5 + 
  (f 4^2 + f 8) / f 7 + 
  (f 5^2 + f 10) / f 9 = 30 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2916_291618


namespace NUMINAMATH_CALUDE_min_value_theorem_l2916_291681

theorem min_value_theorem (x y : ℝ) (h : x * y > 0) :
  ∃ (min_val : ℝ), min_val = 4 - 2 * Real.sqrt 2 ∧
  ∀ (z : ℝ), y / (x + y) + 2 * x / (2 * x + y) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2916_291681


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2916_291649

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Theorem: Maximum number of soap boxes in a carton -/
theorem max_soap_boxes_in_carton (carton soap : BoxDimensions)
    (h_carton : carton = ⟨25, 42, 60⟩)
    (h_soap : soap = ⟨7, 6, 10⟩) :
    (boxVolume carton) / (boxVolume soap) = 150 := by
  sorry


end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2916_291649


namespace NUMINAMATH_CALUDE_output_is_fifteen_l2916_291660

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 2
  if step1 > 18 then step1 - 5 else step1 + 8

theorem output_is_fifteen : function_machine 10 = 15 := by sorry

end NUMINAMATH_CALUDE_output_is_fifteen_l2916_291660


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l2916_291647

/-- The length of each identical rectangle forming PQRS, rounded to the nearest integer -/
def rectangle_length : ℕ :=
  37

theorem rectangle_length_proof (area_PQRS : ℝ) (num_rectangles : ℕ) (PQ_ratio : ℝ) (RS_ratio : ℝ) :
  area_PQRS = 6000 →
  num_rectangles = 6 →
  PQ_ratio = 4 →
  RS_ratio = 3 →
  rectangle_length = 37 := by
  sorry

#check rectangle_length_proof

end NUMINAMATH_CALUDE_rectangle_length_proof_l2916_291647


namespace NUMINAMATH_CALUDE_contractor_payment_l2916_291670

/-- A contractor's payment calculation --/
theorem contractor_payment
  (total_days : ℕ)
  (work_pay : ℚ)
  (fine : ℚ)
  (absent_days : ℕ)
  (h1 : total_days = 30)
  (h2 : work_pay = 25)
  (h3 : fine = 7.5)
  (h4 : absent_days = 4)
  : (total_days - absent_days : ℚ) * work_pay - (absent_days : ℚ) * fine = 620 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_l2916_291670


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2916_291638

theorem quadratic_roots_sum_of_squares (a b c : ℝ) : 
  (∀ x, x^2 - 7*x + c = 0 ↔ x = a ∨ x = b) →
  a^2 + b^2 = 17 →
  c = 16 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2916_291638


namespace NUMINAMATH_CALUDE_eight_power_fifteen_divided_by_sixtyfour_power_seven_l2916_291607

theorem eight_power_fifteen_divided_by_sixtyfour_power_seven :
  8^15 / 64^7 = 8 := by sorry

end NUMINAMATH_CALUDE_eight_power_fifteen_divided_by_sixtyfour_power_seven_l2916_291607


namespace NUMINAMATH_CALUDE_function_positive_implies_m_bound_l2916_291698

open Real

theorem function_positive_implies_m_bound (m : ℝ) :
  (∀ x : ℝ, x > 0 → (Real.exp x / x - m * x) > 0) →
  m < Real.exp 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_function_positive_implies_m_bound_l2916_291698


namespace NUMINAMATH_CALUDE_initial_hours_were_eight_l2916_291666

/-- Represents the highway construction scenario -/
structure HighwayConstruction where
  initial_workforce : ℕ
  total_length : ℕ
  initial_duration : ℕ
  partial_duration : ℕ
  partial_completion : ℚ
  additional_workforce : ℕ
  new_daily_hours : ℕ

/-- Calculates the initial daily working hours -/
def calculate_initial_hours (scenario : HighwayConstruction) : ℚ :=
  (scenario.new_daily_hours * (scenario.initial_workforce + scenario.additional_workforce) * scenario.partial_duration * (1 - scenario.partial_completion)) /
  (scenario.initial_workforce * scenario.partial_duration * scenario.partial_completion)

/-- Theorem stating that the initial daily working hours were 8 -/
theorem initial_hours_were_eight (scenario : HighwayConstruction) 
  (h1 : scenario.initial_workforce = 100)
  (h2 : scenario.total_length = 2)
  (h3 : scenario.initial_duration = 50)
  (h4 : scenario.partial_duration = 25)
  (h5 : scenario.partial_completion = 1/3)
  (h6 : scenario.additional_workforce = 60)
  (h7 : scenario.new_daily_hours = 10) :
  calculate_initial_hours scenario = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_hours_were_eight_l2916_291666


namespace NUMINAMATH_CALUDE_square_sum_equals_six_l2916_291684

theorem square_sum_equals_six (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_six_l2916_291684


namespace NUMINAMATH_CALUDE_angle_measure_l2916_291644

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l2916_291644


namespace NUMINAMATH_CALUDE_chicken_problem_l2916_291632

/-- The problem of calculating the difference in number of chickens bought by John and Ray -/
theorem chicken_problem (chicken_cost : ℕ) (john_extra : ℕ) (ray_less : ℕ) (ray_chickens : ℕ) :
  chicken_cost = 3 →
  john_extra = 15 →
  ray_less = 18 →
  ray_chickens = 10 →
  (john_extra + ray_less + ray_chickens * chicken_cost) / chicken_cost - ray_chickens = 11 :=
by sorry

end NUMINAMATH_CALUDE_chicken_problem_l2916_291632


namespace NUMINAMATH_CALUDE_bee_count_l2916_291662

theorem bee_count (initial_bees additional_bees : ℕ) : 
  initial_bees = 16 → additional_bees = 10 → initial_bees + additional_bees = 26 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l2916_291662


namespace NUMINAMATH_CALUDE_three_equal_products_exist_l2916_291630

/-- Represents a 3x3 table filled with numbers from 1 to 9 --/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if all numbers in the table are unique --/
def all_unique (t : Table) : Prop :=
  ∀ i j i' j', t i j = t i' j' → (i = i' ∧ j = j')

/-- Calculates the product of a row --/
def row_product (t : Table) (i : Fin 3) : ℕ :=
  ((t i 0).val + 1) * ((t i 1).val + 1) * ((t i 2).val + 1)

/-- Calculates the product of a column --/
def col_product (t : Table) (j : Fin 3) : ℕ :=
  ((t 0 j).val + 1) * ((t 1 j).val + 1) * ((t 2 j).val + 1)

/-- Checks if at least three products are equal --/
def three_equal_products (t : Table) : Prop :=
  ∃ p : ℕ, (
    (row_product t 0 = p ∧ row_product t 1 = p ∧ row_product t 2 = p) ∨
    (row_product t 0 = p ∧ row_product t 1 = p ∧ col_product t 0 = p) ∨
    (row_product t 0 = p ∧ row_product t 1 = p ∧ col_product t 1 = p) ∨
    (row_product t 0 = p ∧ row_product t 1 = p ∧ col_product t 2 = p) ∨
    (row_product t 0 = p ∧ row_product t 2 = p ∧ col_product t 0 = p) ∨
    (row_product t 0 = p ∧ row_product t 2 = p ∧ col_product t 1 = p) ∨
    (row_product t 0 = p ∧ row_product t 2 = p ∧ col_product t 2 = p) ∨
    (row_product t 0 = p ∧ col_product t 0 = p ∧ col_product t 1 = p) ∨
    (row_product t 0 = p ∧ col_product t 0 = p ∧ col_product t 2 = p) ∨
    (row_product t 0 = p ∧ col_product t 1 = p ∧ col_product t 2 = p) ∨
    (row_product t 1 = p ∧ row_product t 2 = p ∧ col_product t 0 = p) ∨
    (row_product t 1 = p ∧ row_product t 2 = p ∧ col_product t 1 = p) ∨
    (row_product t 1 = p ∧ row_product t 2 = p ∧ col_product t 2 = p) ∨
    (row_product t 1 = p ∧ col_product t 0 = p ∧ col_product t 1 = p) ∨
    (row_product t 1 = p ∧ col_product t 0 = p ∧ col_product t 2 = p) ∨
    (row_product t 1 = p ∧ col_product t 1 = p ∧ col_product t 2 = p) ∨
    (row_product t 2 = p ∧ col_product t 0 = p ∧ col_product t 1 = p) ∨
    (row_product t 2 = p ∧ col_product t 0 = p ∧ col_product t 2 = p) ∨
    (row_product t 2 = p ∧ col_product t 1 = p ∧ col_product t 2 = p) ∨
    (col_product t 0 = p ∧ col_product t 1 = p ∧ col_product t 2 = p)
  )

theorem three_equal_products_exist :
  ∃ t : Table, all_unique t ∧ three_equal_products t :=
by sorry

end NUMINAMATH_CALUDE_three_equal_products_exist_l2916_291630


namespace NUMINAMATH_CALUDE_laptop_lighter_than_tote_l2916_291603

/-- Represents the weights of various items in pounds -/
structure Weights where
  karens_tote : ℝ
  kevins_empty_briefcase : ℝ
  kevins_umbrella : ℝ
  kevins_laptop : ℝ
  kevins_work_papers : ℝ

/-- Conditions given in the problem -/
def problem_conditions (w : Weights) : Prop :=
  w.karens_tote = 8 ∧
  w.karens_tote = 2 * w.kevins_empty_briefcase ∧
  w.kevins_umbrella = w.kevins_empty_briefcase / 2 ∧
  w.kevins_empty_briefcase + w.kevins_laptop + w.kevins_work_papers + w.kevins_umbrella = 2 * w.karens_tote ∧
  w.kevins_work_papers = (w.kevins_empty_briefcase + w.kevins_laptop + w.kevins_work_papers) / 6

theorem laptop_lighter_than_tote (w : Weights) (h : problem_conditions w) :
  w.kevins_laptop < w.karens_tote ∧ w.karens_tote - w.kevins_laptop = 1/3 := by
  sorry

#check laptop_lighter_than_tote

end NUMINAMATH_CALUDE_laptop_lighter_than_tote_l2916_291603


namespace NUMINAMATH_CALUDE_fruit_prices_l2916_291679

theorem fruit_prices (x y z f : ℝ) 
  (h1 : x + y + z + f = 45)
  (h2 : f = 3 * x)
  (h3 : z = x + y) :
  y + z = 9 := by
sorry

end NUMINAMATH_CALUDE_fruit_prices_l2916_291679


namespace NUMINAMATH_CALUDE_remaining_pie_portion_l2916_291635

-- Define the pie as 100%
def whole_pie : ℚ := 1

-- Carlos's share
def carlos_share : ℚ := 0.6

-- Maria takes half of the remainder
def maria_share_ratio : ℚ := 1/2

-- Theorem to prove
theorem remaining_pie_portion : 
  let remainder_after_carlos := whole_pie - carlos_share
  let maria_share := maria_share_ratio * remainder_after_carlos
  let final_remainder := remainder_after_carlos - maria_share
  final_remainder = 0.2 := by
sorry

end NUMINAMATH_CALUDE_remaining_pie_portion_l2916_291635


namespace NUMINAMATH_CALUDE_book_cost_price_l2916_291654

theorem book_cost_price (cost : ℝ) : 
  (1.15 * cost - 1.10 * cost = 120) → cost = 2400 :=
by sorry

end NUMINAMATH_CALUDE_book_cost_price_l2916_291654


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_expression_value_at_negative_quarter_l2916_291659

theorem algebraic_expression_simplification (a : ℝ) :
  (a - 2)^2 + (a + 1) * (a - 1) - 2 * a * (a - 3) = 2 * a + 3 :=
by sorry

theorem expression_value_at_negative_quarter :
  let a : ℝ := -1/4
  (a - 2)^2 + (a + 1) * (a - 1) - 2 * a * (a - 3) = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_expression_value_at_negative_quarter_l2916_291659


namespace NUMINAMATH_CALUDE_largest_fraction_sum_l2916_291605

theorem largest_fraction_sum : 
  let sum1 := (3 : ℚ) / 10 + 2 / 20
  let sum2 := (1 : ℚ) / 6 + 1 / 8
  let sum3 := (1 : ℚ) / 5 + 2 / 15
  let sum4 := (1 : ℚ) / 7 + 4 / 21
  let sum5 := (2 : ℚ) / 9 + 3 / 18
  sum1 > sum2 ∧ sum1 > sum3 ∧ sum1 > sum4 ∧ sum1 > sum5 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_sum_l2916_291605


namespace NUMINAMATH_CALUDE_sum_of_divisors_540_has_4_prime_factors_l2916_291608

-- Define the number we're working with
def n : ℕ := 540

-- Define the sum of positive divisors function
noncomputable def sum_of_divisors (m : ℕ) : ℕ := sorry

-- Define a function to count distinct prime factors
noncomputable def count_distinct_prime_factors (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_divisors_540_has_4_prime_factors :
  count_distinct_prime_factors (sum_of_divisors n) = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_540_has_4_prime_factors_l2916_291608


namespace NUMINAMATH_CALUDE_multiply_95_105_l2916_291692

theorem multiply_95_105 : 95 * 105 = 9975 := by
  sorry

end NUMINAMATH_CALUDE_multiply_95_105_l2916_291692


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2916_291696

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2916_291696


namespace NUMINAMATH_CALUDE_total_towels_folded_per_hour_l2916_291665

/-- Represents the number of towels a person can fold in one hour -/
def towels_per_hour (towels : ℕ) (minutes : ℕ) : ℕ :=
  (60 / minutes) * towels

/-- Proves that Jane, Kyla, and Anthony can fold 87 towels together in one hour -/
theorem total_towels_folded_per_hour :
  let jane_rate := towels_per_hour 3 5
  let kyla_rate := towels_per_hour 5 10
  let anthony_rate := towels_per_hour 7 20
  jane_rate + kyla_rate + anthony_rate = 87 := by
  sorry

#eval towels_per_hour 3 5 + towels_per_hour 5 10 + towels_per_hour 7 20

end NUMINAMATH_CALUDE_total_towels_folded_per_hour_l2916_291665


namespace NUMINAMATH_CALUDE_orange_boxes_l2916_291640

theorem orange_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 42) (h2 : oranges_per_box = 6) :
  total_oranges / oranges_per_box = 7 :=
by sorry

end NUMINAMATH_CALUDE_orange_boxes_l2916_291640


namespace NUMINAMATH_CALUDE_sets_equality_l2916_291676

def A : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 1}
def B : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4*b + 5}

theorem sets_equality : A = B := by sorry

end NUMINAMATH_CALUDE_sets_equality_l2916_291676


namespace NUMINAMATH_CALUDE_soap_discount_theorem_l2916_291626

/-- The original price of a bar of soap in yuan -/
def original_price : ℝ := 2

/-- The discount rate for the first method (applied to all bars except the first) -/
def discount_rate1 : ℝ := 0.3

/-- The discount rate for the second method (applied to all bars) -/
def discount_rate2 : ℝ := 0.2

/-- The cost of n bars using the first discount method -/
def cost1 (n : ℕ) : ℝ := original_price + (n - 1) * original_price * (1 - discount_rate1)

/-- The cost of n bars using the second discount method -/
def cost2 (n : ℕ) : ℝ := n * original_price * (1 - discount_rate2)

/-- The minimum number of bars needed for the first method to provide more discount -/
def min_bars : ℕ := 4

theorem soap_discount_theorem :
  ∀ n : ℕ, n ≥ min_bars → cost1 n < cost2 n ∧
  ∀ m : ℕ, m < min_bars → cost1 m ≥ cost2 m :=
sorry

end NUMINAMATH_CALUDE_soap_discount_theorem_l2916_291626


namespace NUMINAMATH_CALUDE_circumcenter_on_median_l2916_291655

variable {A B C O H P Q : ℂ}

/-- The triangle ABC is acute -/
def is_acute_triangle (A B C : ℂ) : Prop := sorry

/-- O is the circumcenter of triangle ABC -/
def is_circumcenter (O A B C : ℂ) : Prop := sorry

/-- H is the orthocenter of triangle ABC -/
def is_orthocenter (H A B C : ℂ) : Prop := sorry

/-- P is the intersection of OA and the altitude from B -/
def is_P_intersection (P O A B C : ℂ) : Prop := sorry

/-- Q is the intersection of OA and the altitude from C -/
def is_Q_intersection (Q O A B C : ℂ) : Prop := sorry

/-- X is the circumcenter of triangle PQH -/
def is_PQH_circumcenter (X P Q H : ℂ) : Prop := sorry

/-- M is the midpoint of BC -/
def is_midpoint_BC (M B C : ℂ) : Prop := sorry

/-- Three points are collinear -/
def collinear (X Y Z : ℂ) : Prop := sorry

theorem circumcenter_on_median 
  (h_acute : is_acute_triangle A B C)
  (h_O : is_circumcenter O A B C)
  (h_H : is_orthocenter H A B C)
  (h_P : is_P_intersection P O A B C)
  (h_Q : is_Q_intersection Q O A B C) :
  ∃ (X M : ℂ), is_PQH_circumcenter X P Q H ∧ 
               is_midpoint_BC M B C ∧ 
               collinear A X M :=
sorry

end NUMINAMATH_CALUDE_circumcenter_on_median_l2916_291655


namespace NUMINAMATH_CALUDE_find_x_l2916_291624

theorem find_x (x y z : ℝ) 
  (h1 : x * y / (x + y) = 4)
  (h2 : x * z / (x + z) = 5)
  (h3 : y * z / (y + z) = 6)
  : x = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2916_291624


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_sizes_l2916_291648

def total_population : ℕ := 300
def top_class_size : ℕ := 30
def experimental_class_size : ℕ := 90
def regular_class_size : ℕ := 180
def total_sample_size : ℕ := 30

def stratum_sample_size (stratum_size : ℕ) : ℕ :=
  (stratum_size * total_sample_size) / total_population

theorem stratified_sampling_correct_sizes :
  stratum_sample_size top_class_size = 3 ∧
  stratum_sample_size experimental_class_size = 9 ∧
  stratum_sample_size regular_class_size = 18 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_sizes_l2916_291648


namespace NUMINAMATH_CALUDE_general_admission_ticket_cost_l2916_291616

theorem general_admission_ticket_cost
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (general_admission_tickets : ℕ)
  (student_ticket_cost : ℕ)
  (h1 : total_tickets = 525)
  (h2 : total_revenue = 2876)
  (h3 : general_admission_tickets = 388)
  (h4 : student_ticket_cost = 4) :
  ∃ (general_admission_cost : ℕ),
    general_admission_cost * general_admission_tickets +
    student_ticket_cost * (total_tickets - general_admission_tickets) =
    total_revenue ∧
    general_admission_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_general_admission_ticket_cost_l2916_291616


namespace NUMINAMATH_CALUDE_julia_tag_game_l2916_291668

theorem julia_tag_game (tuesday_kids : ℕ) (extra_monday_kids : ℕ) : 
  tuesday_kids = 14 → extra_monday_kids = 8 → tuesday_kids + extra_monday_kids = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l2916_291668


namespace NUMINAMATH_CALUDE_expression_evaluation_inequality_system_solution_l2916_291633

-- Part 1
theorem expression_evaluation :
  Real.sqrt 12 + |Real.sqrt 3 - 2| - 2 * Real.tan (60 * π / 180) + (1/3)⁻¹ = 5 - Real.sqrt 3 := by
  sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (x + 3 * (x - 2) ≥ 2 ∧ (1 + 2 * x) / 3 > x - 1) ↔ (2 ≤ x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_inequality_system_solution_l2916_291633


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l2916_291625

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 9*n + 20 ≤ 0 ∧ n = 5 ∧ ∀ (m : ℤ), m^2 - 9*m + 20 ≤ 0 → m ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l2916_291625


namespace NUMINAMATH_CALUDE_sets_partition_l2916_291600

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define a property for primes greater than 2013
def IsPrimeGreaterThan2013 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n > 2013

-- Define the property for the special difference condition
def SpecialDifference (A B : Set ℕ) : Prop :=
  ∀ (x y : ℕ), x ∈ PositiveIntegers → y ∈ PositiveIntegers →
    IsPrimeGreaterThan2013 (x - y) →
    ((x ∈ A ∧ y ∈ B) ∨ (x ∈ B ∧ y ∈ A))

theorem sets_partition (A B : Set ℕ) :
  (A ∪ B = PositiveIntegers) →
  (A ∩ B = ∅) →
  SpecialDifference A B →
  ((∀ n : ℕ, n ∈ A ↔ n ∈ PositiveIntegers ∧ Even n) ∧
   (∀ n : ℕ, n ∈ B ↔ n ∈ PositiveIntegers ∧ Odd n)) :=
by sorry

end NUMINAMATH_CALUDE_sets_partition_l2916_291600


namespace NUMINAMATH_CALUDE_expression_simplification_l2916_291613

theorem expression_simplification (x : ℝ) :
  2*x*(4*x^2 - 3*x + 1) - 7*(2*x^2 - 3*x + 4) = 8*x^3 - 20*x^2 + 23*x - 28 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2916_291613


namespace NUMINAMATH_CALUDE_sin_beta_value_l2916_291697

theorem sin_beta_value (α β : Real) 
  (h : Real.sin α * Real.cos (α - β) - Real.cos α * Real.sin (α - β) = 4/5) : 
  Real.sin β = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l2916_291697


namespace NUMINAMATH_CALUDE_least_multiple_of_17_greater_than_450_l2916_291641

theorem least_multiple_of_17_greater_than_450 :
  ∀ n : ℕ, n > 0 ∧ 17 ∣ n ∧ n > 450 → n ≥ 459 :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_of_17_greater_than_450_l2916_291641


namespace NUMINAMATH_CALUDE_third_number_is_41_l2916_291661

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

end NUMINAMATH_CALUDE_third_number_is_41_l2916_291661


namespace NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l2916_291677

def is_monic_cubic (q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, q x = x^3 + a*x^2 + b*x + c

theorem cubic_polynomial_uniqueness (q : ℝ → ℂ) :
  is_monic_cubic (λ x : ℝ ↦ (q x).re) →
  q (5 - 3*I) = 0 →
  q 0 = -80 →
  ∀ x, q x = x^3 - 10*x^2 + 40*x - 80 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l2916_291677


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l2916_291636

theorem simplify_sqrt_product : 
  Real.sqrt (3 * 5) * Real.sqrt (5^2 * 3^3) = 45 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l2916_291636


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2916_291680

theorem imaginary_part_of_z (z : ℂ) : z = Complex.I * (-1 + 2 * Complex.I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2916_291680


namespace NUMINAMATH_CALUDE_remaining_money_calculation_l2916_291678

def euro_to_dollar : ℝ := 1.183
def pound_to_dollar : ℝ := 1.329
def yen_to_dollar : ℝ := 0.009
def real_to_dollar : ℝ := 0.193

def conversion_fee_rate : ℝ := 0.015
def sales_tax_rate : ℝ := 0.08
def transportation_fee : ℝ := 12
def gift_wrapping_fee : ℝ := 7.5
def spending_fraction : ℝ := 0.75

def initial_euro : ℝ := 25
def initial_pound : ℝ := 50
def initial_dollar : ℝ := 35
def initial_yen : ℝ := 8000
def initial_real : ℝ := 60
def initial_savings : ℝ := 105

theorem remaining_money_calculation :
  let total_dollars := initial_euro * euro_to_dollar +
                       initial_pound * pound_to_dollar +
                       initial_dollar +
                       initial_yen * yen_to_dollar +
                       initial_real * real_to_dollar +
                       initial_savings
  let after_conversion_fee := total_dollars * (1 - conversion_fee_rate)
  let after_fixed_fees := after_conversion_fee - transportation_fee - gift_wrapping_fee
  let spent_amount := after_fixed_fees * spending_fraction
  let tax_amount := spent_amount * sales_tax_rate
  let remaining_amount := after_fixed_fees - (spent_amount + tax_amount)
  remaining_amount = 73.82773125 := by sorry

end NUMINAMATH_CALUDE_remaining_money_calculation_l2916_291678


namespace NUMINAMATH_CALUDE_sets_and_domains_l2916_291639

-- Define the sets A, B, and C
def A : Set ℝ := {x | |x - 1| ≥ 1}
def B : Set ℝ := {x | x < -1 ∨ x ≥ 1}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a + 1}

-- State the theorem
theorem sets_and_domains (a : ℝ) (h : a < 1) :
  (A ∩ B = {x | x < -1 ∨ x ≥ 2}) ∧
  ((Set.univ \ (A ∪ B)) = {x | 0 < x ∧ x < 1}) ∧
  (C a ⊆ B → (a ≤ -2 ∨ (1/2 ≤ a ∧ a < 1))) :=
by sorry

end NUMINAMATH_CALUDE_sets_and_domains_l2916_291639


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2916_291622

theorem fixed_point_on_line (k : ℝ) : k * 2 + 0 - 2 * k = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2916_291622


namespace NUMINAMATH_CALUDE_incorrect_equation_l2916_291634

/-- A repeating decimal with non-repeating part N and repeating part R -/
structure RepeatingDecimal where
  N : ℕ  -- non-repeating part
  R : ℕ  -- repeating part
  t : ℕ  -- number of digits in N
  u : ℕ  -- number of digits in R
  t_pos : t > 0
  u_pos : u > 0

/-- The value of the repeating decimal -/
noncomputable def RepeatingDecimal.value (M : RepeatingDecimal) : ℝ :=
  (M.N : ℝ) / 10^M.t + (M.R : ℝ) / (10^M.t * (10^M.u - 1))

/-- The theorem stating that the equation in option D is incorrect -/
theorem incorrect_equation (M : RepeatingDecimal) :
  ¬(10^M.t * (10^M.u - 1) * M.value = (M.R : ℝ) * ((M.N : ℝ) - 1)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_equation_l2916_291634


namespace NUMINAMATH_CALUDE_no_valid_solution_l2916_291604

theorem no_valid_solution : ¬∃ (Y : ℕ), Y > 0 ∧ 2*Y + Y + 3*Y = 14 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_solution_l2916_291604


namespace NUMINAMATH_CALUDE_skill_testing_question_l2916_291623

theorem skill_testing_question : 5 * (10 - 6) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_skill_testing_question_l2916_291623


namespace NUMINAMATH_CALUDE_conor_carrot_count_l2916_291637

/-- Represents the number of vegetables Conor can chop in a day -/
structure DailyVegetables where
  eggplants : ℕ
  carrots : ℕ
  potatoes : ℕ

/-- Represents Conor's weekly vegetable chopping -/
def WeeklyVegetables (d : DailyVegetables) (workDays : ℕ) : ℕ :=
  workDays * (d.eggplants + d.carrots + d.potatoes)

/-- Theorem stating the number of carrots Conor can chop in a day -/
theorem conor_carrot_count :
  ∀ (d : DailyVegetables),
    d.eggplants = 12 →
    d.potatoes = 8 →
    WeeklyVegetables d 4 = 116 →
    d.carrots = 9 := by
  sorry


end NUMINAMATH_CALUDE_conor_carrot_count_l2916_291637


namespace NUMINAMATH_CALUDE_rachel_score_l2916_291620

/-- Rachel's video game scoring system -/
structure GameScore where
  points_per_treasure : ℕ
  treasures_level1 : ℕ
  treasures_level2 : ℕ

/-- Calculate the total score for Rachel's game -/
def total_score (game : GameScore) : ℕ :=
  game.points_per_treasure * (game.treasures_level1 + game.treasures_level2)

/-- Theorem: Rachel's total score is 63 points -/
theorem rachel_score :
  ∀ (game : GameScore),
  game.points_per_treasure = 9 →
  game.treasures_level1 = 5 →
  game.treasures_level2 = 2 →
  total_score game = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_rachel_score_l2916_291620


namespace NUMINAMATH_CALUDE_hyuksu_meat_consumption_l2916_291663

/-- The amount of meat Hyuksu ate yesterday in kilograms -/
def meat_yesterday : ℝ := 2.6

/-- The amount of meat Hyuksu ate today in kilograms -/
def meat_today : ℝ := 5.98

/-- The total amount of meat Hyuksu ate in two days in kilograms -/
def total_meat : ℝ := meat_yesterday + meat_today

theorem hyuksu_meat_consumption : total_meat = 8.58 := by
  sorry

end NUMINAMATH_CALUDE_hyuksu_meat_consumption_l2916_291663


namespace NUMINAMATH_CALUDE_problem_solution_l2916_291614

theorem problem_solution (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 2/x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2916_291614


namespace NUMINAMATH_CALUDE_total_rocks_in_border_l2916_291645

/-- The number of rocks in Mrs. Hilt's garden border -/
def garden_border (placed : ℝ) (additional : ℝ) : ℝ :=
  placed + additional

/-- Theorem stating the total number of rocks in the completed border -/
theorem total_rocks_in_border :
  garden_border 125.0 64.0 = 189.0 := by
  sorry

end NUMINAMATH_CALUDE_total_rocks_in_border_l2916_291645


namespace NUMINAMATH_CALUDE_quadratic_equation_m_range_l2916_291653

theorem quadratic_equation_m_range (m : ℝ) :
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m^2 - 4) * x^2 + (2 - m) * x + 1 = a * x^2 + b * x + c) ↔
  m ≠ 2 ∧ m ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_range_l2916_291653


namespace NUMINAMATH_CALUDE_three_toys_picked_l2916_291628

def toy_count : ℕ := 4

def probability_yo_yo_and_ball (n : ℕ) : ℚ :=
  if n < 2 then 0
  else (Nat.choose 2 (n - 2) : ℚ) / (Nat.choose toy_count n : ℚ)

theorem three_toys_picked :
  ∃ (n : ℕ), n ≤ toy_count ∧ probability_yo_yo_and_ball n = 1/2 ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_three_toys_picked_l2916_291628


namespace NUMINAMATH_CALUDE_average_visitors_theorem_l2916_291601

/-- The average number of visitors on Sundays -/
def sunday_visitors : ℕ := 540

/-- The average number of visitors on other days -/
def other_day_visitors : ℕ := 240

/-- The number of days in the month -/
def days_in_month : ℕ := 30

/-- The number of Sundays in the month -/
def sundays_in_month : ℕ := 5

/-- The number of other days in the month -/
def other_days_in_month : ℕ := days_in_month - sundays_in_month

/-- The average number of visitors per day in the month -/
def average_visitors_per_day : ℚ :=
  (sunday_visitors * sundays_in_month + other_day_visitors * other_days_in_month) / days_in_month

theorem average_visitors_theorem :
  average_visitors_per_day = 290 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_theorem_l2916_291601


namespace NUMINAMATH_CALUDE_no_solution_with_vasyas_correction_l2916_291673

theorem no_solution_with_vasyas_correction (r : ℝ) : ¬ ∃ (a h : ℝ),
  (0 < r) ∧                           -- radius is positive
  (0 < a) ∧ (0 < h) ∧                 -- base and height are positive
  (a ≤ 2*r) ∧                         -- base is at most diameter
  (h < 2*r) ∧                         -- height is less than diameter
  (a + h = 2*Real.pi*r) :=            -- sum equals circumference (Vasya's condition)
by
  sorry

end NUMINAMATH_CALUDE_no_solution_with_vasyas_correction_l2916_291673


namespace NUMINAMATH_CALUDE_overtime_rate_is_five_l2916_291688

/-- Calculates the overtime pay rate given daily wage, total earnings, days worked, and overtime hours. -/
def overtime_pay_rate (daily_wage : ℚ) (total_earnings : ℚ) (days_worked : ℕ) (overtime_hours : ℕ) : ℚ :=
  (total_earnings - daily_wage * days_worked) / overtime_hours

/-- Proves that given the conditions, the overtime pay rate is $5 per hour. -/
theorem overtime_rate_is_five :
  let daily_wage : ℚ := 150
  let total_earnings : ℚ := 770
  let days_worked : ℕ := 5
  let overtime_hours : ℕ := 4
  overtime_pay_rate daily_wage total_earnings days_worked overtime_hours = 5 := by
  sorry

#eval overtime_pay_rate 150 770 5 4

end NUMINAMATH_CALUDE_overtime_rate_is_five_l2916_291688


namespace NUMINAMATH_CALUDE_complex_expression_equals_81_l2916_291656

theorem complex_expression_equals_81 :
  3 * ((-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_81_l2916_291656


namespace NUMINAMATH_CALUDE_halloween_candy_proof_l2916_291695

/-- Represents the number of candy pieces Debby's sister had -/
def sisters_candy : ℕ := 42

theorem halloween_candy_proof :
  let debbys_candy : ℕ := 32
  let eaten_candy : ℕ := 35
  let remaining_candy : ℕ := 39
  debbys_candy + sisters_candy - eaten_candy = remaining_candy :=
by
  sorry

#check halloween_candy_proof

end NUMINAMATH_CALUDE_halloween_candy_proof_l2916_291695


namespace NUMINAMATH_CALUDE_baseball_card_pages_l2916_291631

theorem baseball_card_pages (cards_per_page new_cards old_cards : ℕ) 
  (h1 : cards_per_page = 3)
  (h2 : new_cards = 3)
  (h3 : old_cards = 9) :
  (new_cards + old_cards) / cards_per_page = 4 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_pages_l2916_291631


namespace NUMINAMATH_CALUDE_same_gender_probability_l2916_291629

/-- The probability of selecting two students of the same gender from a group of 3 male and 2 female students -/
theorem same_gender_probability (male_students female_students : ℕ) 
  (h1 : male_students = 3)
  (h2 : female_students = 2) : 
  (Nat.choose male_students 2 + Nat.choose female_students 2) / Nat.choose (male_students + female_students) 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_gender_probability_l2916_291629


namespace NUMINAMATH_CALUDE_sofie_total_distance_l2916_291672

/-- Represents the side lengths of the pentagon-shaped track in meters -/
def track_sides : List ℕ := [25, 35, 20, 40, 30]

/-- Calculates the perimeter of the track in meters -/
def track_perimeter : ℕ := track_sides.sum

/-- The number of initial laps Sofie runs -/
def initial_laps : ℕ := 2

/-- The number of additional laps Sofie runs -/
def additional_laps : ℕ := 5

/-- Theorem stating the total distance Sofie runs -/
theorem sofie_total_distance :
  initial_laps * track_perimeter + additional_laps * track_perimeter = 1050 := by
  sorry

end NUMINAMATH_CALUDE_sofie_total_distance_l2916_291672


namespace NUMINAMATH_CALUDE_parabola_axis_l2916_291643

/-- The equation of the axis of the parabola y = x^2 -/
theorem parabola_axis (x y : ℝ) : 
  (y = x^2) → (∃ (axis : ℝ → ℝ), axis y = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_l2916_291643


namespace NUMINAMATH_CALUDE_back_seat_capacity_is_eleven_l2916_291602

/-- Represents the seating capacity of a bus with specific arrangements -/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  total_capacity : Nat

/-- Calculates the number of people who can sit at the back seat of the bus -/
def back_seat_capacity (bus : BusSeating) : Nat :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating the back seat capacity of the given bus configuration -/
theorem back_seat_capacity_is_eleven : ∃ (bus : BusSeating),
  bus.left_seats = 15 ∧
  bus.right_seats = bus.left_seats - 3 ∧
  bus.people_per_seat = 3 ∧
  bus.total_capacity = 92 ∧
  back_seat_capacity bus = 11 := by
  sorry

end NUMINAMATH_CALUDE_back_seat_capacity_is_eleven_l2916_291602


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l2916_291650

theorem max_value_of_fraction (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ (max : ℝ), max = 7/5 ∧ ∀ x y, 
    x + y - 2 ≥ 0 → y - x - 1 ≤ 0 → x ≤ 1 → 
    (x + 2*y) / (2*x + y) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l2916_291650


namespace NUMINAMATH_CALUDE_xyz_less_than_one_l2916_291675

theorem xyz_less_than_one (x y z : ℝ) 
  (h1 : 2 * x > y^2 + z^2)
  (h2 : 2 * y > x^2 + z^2)
  (h3 : 2 * z > y^2 + x^2) : 
  x * y * z < 1 := by
  sorry

end NUMINAMATH_CALUDE_xyz_less_than_one_l2916_291675


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l2916_291652

/-- A geometric progression with sum to infinity 8 and sum of first three terms 7 has first term 4 -/
theorem geometric_progression_first_term :
  ∀ (a r : ℝ),
  (a / (1 - r) = 8) →  -- sum to infinity
  (a + a*r + a*r^2 = 7) →  -- sum of first three terms
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l2916_291652


namespace NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l2916_291694

-- Define the function f(x)
def f (x : ℝ) : ℝ := (15 * x + (15 * x + 17) ^ (1/3)) ^ (1/3)

-- State the theorem
theorem unique_solution_cube_root_equation :
  ∃! x : ℝ, f x = 18 ∧ x = 387 := by sorry

end NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l2916_291694


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l2916_291699

def batman_price : ℚ := 13.60
def superman_price : ℚ := 5.06
def batman_discount : ℚ := 0.10
def superman_discount : ℚ := 0.05
def sales_tax : ℚ := 0.08
def game1_price : ℚ := 7.25
def game2_price : ℚ := 12.50

def total_spent : ℚ :=
  let batman_discounted := batman_price * (1 - batman_discount)
  let superman_discounted := superman_price * (1 - superman_discount)
  let batman_with_tax := batman_discounted * (1 + sales_tax)
  let superman_with_tax := superman_discounted * (1 + sales_tax)
  batman_with_tax + superman_with_tax + game1_price + game2_price

theorem total_spent_is_correct : total_spent = 38.16 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l2916_291699


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2916_291611

/-- Two lines in the xy-plane --/
structure TwoLines :=
  (a : ℝ)

/-- The condition for two lines to be parallel --/
def are_parallel (lines : TwoLines) : Prop :=
  lines.a^2 - lines.a = 2

/-- The statement that a=2 is sufficient but not necessary for the lines to be parallel --/
theorem sufficient_not_necessary :
  (∃ (lines : TwoLines), lines.a = 2 → are_parallel lines) ∧
  (∃ (lines : TwoLines), are_parallel lines ∧ lines.a ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2916_291611


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l2916_291689

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 25) :
  (train_length + bridge_length) / crossing_time = 16 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l2916_291689


namespace NUMINAMATH_CALUDE_ninety_eight_squared_l2916_291674

theorem ninety_eight_squared : (100 - 2)^2 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_ninety_eight_squared_l2916_291674


namespace NUMINAMATH_CALUDE_additional_wax_needed_l2916_291690

theorem additional_wax_needed (total_wax : ℕ) (available_wax : ℕ) (h1 : total_wax = 353) (h2 : available_wax = 331) :
  total_wax - available_wax = 22 := by
  sorry

end NUMINAMATH_CALUDE_additional_wax_needed_l2916_291690


namespace NUMINAMATH_CALUDE_greek_yogurt_cost_per_pack_l2916_291682

theorem greek_yogurt_cost_per_pack 
  (total_packs : ℕ)
  (expired_percentage : ℚ)
  (total_refund : ℚ)
  (h1 : total_packs = 80)
  (h2 : expired_percentage = 40 / 100)
  (h3 : total_refund = 384) :
  total_refund / (expired_percentage * total_packs) = 12 := by
sorry

end NUMINAMATH_CALUDE_greek_yogurt_cost_per_pack_l2916_291682


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2916_291619

/-- Given a line l symmetric to the line 2x - 3y + 4 = 0 with respect to x = 1,
    prove that the equation of l is 2x + 3y - 8 = 0 -/
theorem symmetric_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (2 - x, y) ∈ {(x, y) | 2*x - 3*y + 4 = 0}) →
  l = {(x, y) | 2*x + 3*y - 8 = 0} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2916_291619


namespace NUMINAMATH_CALUDE_test_scores_analysis_l2916_291691

def benchmark : ℝ := 85

def deviations : List ℝ := [8, -3, 12, -7, -10, -4, -8, 1, 0, 10]

def actual_scores : List ℝ := deviations.map (λ x => benchmark + x)

theorem test_scores_analysis :
  let max_score := actual_scores.maximum
  let min_score := actual_scores.minimum
  let avg_score := benchmark + (deviations.sum / deviations.length)
  (max_score = 97 ∧ min_score = 75) ∧ avg_score = 84.9 := by
  sorry

end NUMINAMATH_CALUDE_test_scores_analysis_l2916_291691


namespace NUMINAMATH_CALUDE_smallest_y_in_geometric_sequence_125_l2916_291667

/-- A geometric sequence of three positive integers with product 125 -/
structure GeometricSequence125 where
  x : ℕ+
  y : ℕ+
  z : ℕ+
  geometric : ∃ (r : ℚ), y = x * r ∧ z = y * r
  product : x * y * z = 125

/-- The smallest possible value of y in a geometric sequence of three positive integers with product 125 -/
theorem smallest_y_in_geometric_sequence_125 : 
  ∀ (seq : GeometricSequence125), seq.y ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_y_in_geometric_sequence_125_l2916_291667


namespace NUMINAMATH_CALUDE_at_least_two_solved_five_l2916_291610

/-- The number of problems in the competition -/
def num_problems : ℕ := 6

/-- The structure representing a participant in the competition -/
structure Participant where
  solved : Finset (Fin num_problems)

/-- The type of the competition -/
structure Competition where
  participants : Finset Participant
  pair_solved : ∀ (i j : Fin num_problems), i ≠ j →
    (participants.filter (λ p => i ∈ p.solved ∧ j ∈ p.solved)).card >
    (2 * participants.card) / 5
  no_all_solved : ∀ p : Participant, p ∈ participants → p.solved.card < num_problems

/-- The main theorem -/
theorem at_least_two_solved_five (comp : Competition) :
  (comp.participants.filter (λ p => p.solved.card = num_problems - 1)).card ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_solved_five_l2916_291610


namespace NUMINAMATH_CALUDE_x_values_l2916_291658

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^2 + 1/y = 13) (h2 : y^2 + 1/x = 8) :
  x = Real.sqrt 13 ∨ x = -Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_x_values_l2916_291658


namespace NUMINAMATH_CALUDE_integral_equals_two_minus_three_ln_three_l2916_291615

/-- Given that the solution set of the inequality 1 - 3/(x+a) < 0 is (-1,2),
    prove that the integral from 0 to 2 of (1 - 3/(x+a)) dx equals 2 - 3 * ln 3 -/
theorem integral_equals_two_minus_three_ln_three 
  (a : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 2 = {x : ℝ | 1 - 3 / (x + a) < 0}) : 
  ∫ x in (0:ℝ)..2, (1 - 3 / (x + a)) = 2 - 3 * Real.log 3 := by
  sorry

#check integral_equals_two_minus_three_ln_three

end NUMINAMATH_CALUDE_integral_equals_two_minus_three_ln_three_l2916_291615


namespace NUMINAMATH_CALUDE_last_digit_is_four_l2916_291606

/-- Represents a 5-digit number of the form 5228□ -/
def five_digit_number (last_digit : ℕ) : ℕ := 52280 + last_digit

/-- Checks if a number is a multiple of 6 -/
def is_multiple_of_six (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

theorem last_digit_is_four :
  ∀ d : ℕ, d < 10 →
    is_multiple_of_six (five_digit_number d) →
    d = 4 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_is_four_l2916_291606


namespace NUMINAMATH_CALUDE_base_conversion_problem_l2916_291685

theorem base_conversion_problem (c d : ℕ) :
  (c ≤ 9 ∧ d ≤ 9) →  -- c and d are base-10 digits
  (5 * 8^2 + 4 * 8^1 + 3 * 8^0 = 300 + 10 * c + d) →  -- 543₈ = 3cd₁₀
  (c * d) / 12 = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l2916_291685


namespace NUMINAMATH_CALUDE_opposites_sum_to_zero_l2916_291642

theorem opposites_sum_to_zero (a b : ℚ) (h : a = -b) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_to_zero_l2916_291642


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2916_291683

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 3 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 3 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2916_291683


namespace NUMINAMATH_CALUDE_train_crossing_time_l2916_291657

theorem train_crossing_time (train_length : ℝ) (platform1_length : ℝ) (platform2_length : ℝ) (time2 : ℝ) :
  train_length = 230 →
  platform1_length = 130 →
  platform2_length = 250 →
  time2 = 20 →
  let speed := (train_length + platform2_length) / time2
  let time1 := (train_length + platform1_length) / speed
  time1 = 15 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2916_291657


namespace NUMINAMATH_CALUDE_product_mod_nine_l2916_291651

theorem product_mod_nine : (98 * 102) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_nine_l2916_291651


namespace NUMINAMATH_CALUDE_inequality_proof_l2916_291609

theorem inequality_proof (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) 
  (h5 : a * d = b * c) : 
  (a - d)^2 ≥ 4*d + 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2916_291609


namespace NUMINAMATH_CALUDE_collinear_vectors_problem_l2916_291646

/-- Given vectors a, b, and c in ℝ², prove that if a + b is collinear with c, then the y-coordinate of c is 1. -/
theorem collinear_vectors_problem (a b c : ℝ × ℝ) 
    (ha : a = (1, 2))
    (hb : b = (1, -3))
    (hc : c.1 = -2) 
    (h_collinear : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 + b.1, a.2 + b.2) = (k * c.1, k * c.2)) :
  c.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_problem_l2916_291646


namespace NUMINAMATH_CALUDE_xyz_problem_l2916_291693

theorem xyz_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 132)
  (h2 : y * (z + x) = 152)
  (h3 : x * y * z = 160) :
  z * (x + y) = 131.92 := by
  sorry

end NUMINAMATH_CALUDE_xyz_problem_l2916_291693
